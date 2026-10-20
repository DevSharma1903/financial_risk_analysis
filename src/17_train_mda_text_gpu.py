"""GPU-only, selected-case-control MD&A experiment using current proxy labels.

This is an exploratory balanced-cohort comparison, not a natural-population
evaluation. The source MD&A corpus was historically selected and 50/50 sampled.
Its old `y` label is never loaded here; prepared `y_proxy` is the sole target.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import re
import sys
import numpy as np
import pandas as pd
import research_v5_core as core


def atomic_json(path,obj):
    temp=Path(str(path)+'.part')
    temp.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    temp.replace(path)


def atomic_npy(path,values):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.part.npy')
    np.save(temp,np.asarray(values))
    temp.replace(path)


def stable_signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def issuer_holdout_mask(cik, fraction=.20, salt='mda-expanded-cik-v1'):
    """Stable issuer split; no company can cross model selection and holdout."""
    if not 0 < fraction < 1:
        raise ValueError('Issuer holdout fraction must be strictly between zero and one.')
    limit=int(fraction*(2**64))
    return np.asarray([
        int.from_bytes(hashlib.sha256(f'{salt}:{str(value).lstrip("0") or "0"}'.encode()).digest()[:8],
                       'big') < limit for value in cik],dtype=bool)


def matched_cohort(part,seed):
    rows=[]
    part=part.copy()
    part['filed_year']=pd.to_datetime(part.filed_date).dt.year
    for _,group in part.groupby(['filed_year','form'],sort=True):
        positive=group[group.y_amendment_proxy.eq(1)]
        negative=group[group.y_amendment_proxy.eq(0)]
        count=min(len(positive),len(negative))
        if count:
            rows.extend([positive.sample(count,random_state=seed),
                         negative.sample(count,random_state=seed)])
    if not rows:
        raise ValueError('No positive/negative year-form strata can be matched.')
    return pd.concat(rows).sample(frac=1,random_state=seed).reset_index(drop=True)


def balanced_train(part,seed):
    part=part.copy()
    part['filed_year']=pd.to_datetime(part.filed_date).dt.year
    cohorts=[]
    # Match within filing year and form so the model cannot gain from the
    # selected corpus' changing case/control composition across time or forms.
    for _,group in part.groupby(['filed_year','form'],sort=True):
        positive=group[group.y_amendment_proxy.eq(1)]
        negative=group[group.y_amendment_proxy.eq(0)]
        count=min(len(positive),len(negative))
        if count:
            cohorts.extend([positive.sample(count,random_state=seed),
                            negative.sample(count,random_state=seed)])
    if not cohorts:
        raise ValueError('No positive/negative year-form training strata can be matched.')
    return pd.concat(cohorts).sample(frac=1,random_state=seed).reset_index(drop=True)


def columns_for(train,with_tabular):
    categories=[c for c in ('form','fp','sic2') if c in train]
    # The legacy corpus includes a `y` column whose labels disagree with the
    # current accession-joined proxy. Never allow it, or common label aliases,
    # to enter predictors.
    excluded=core.EXCLUDED|set(categories)|{
        'mda_text','y_proxy','y','legacy_y','label','label_source',
        'adsh_amendment','filed_amendment','amendment_url'}
    numeric=[c for c in train if c not in excluded and
             pd.api.types.is_numeric_dtype(train[c]) and
             (with_tabular or c.startswith(('text_emb_','text_style_')) or c=='mda_word_count')]
    if not with_tabular:
        categories=[c for c in ('form',) if c in train]
    if not numeric:
        raise ValueError('The model has no permitted numeric features.')
    return numeric,categories


def matrices(train,predict,numeric,categorical):
    category_vocab=pd.get_dummies(train[categorical].fillna('unknown').astype(str),dtype='float32')
    columns=category_vocab.columns.tolist()
    def convert(frame):
        values=frame[numeric].replace([np.inf,-np.inf],np.nan).astype('float32')
        flags=values.isna().astype('float32').add_prefix('missing__')
        cat=pd.get_dummies(frame[categorical].fillna('unknown').astype(str),dtype='float32')
        cat=cat.reindex(columns=columns,fill_value=0)
        return pd.concat([values,flags,cat],axis=1)
    return convert(train),convert(predict)


def previous_text_links(frame):
    """Map each filing to the latest strictly earlier same-issuer/form filing."""
    required={'adsh','cik','form','filed_date'}
    if required-set(frame):
        raise ValueError(f'Missing fields for prior-text links: {sorted(required-set(frame))}')
    meta=frame[['adsh','cik','form','filed_date']].copy()
    meta['filed_date']=pd.to_datetime(meta.filed_date,errors='raise')
    meta['cik']=meta.cik.astype(str).str.lstrip('0').replace('','0')
    if meta.adsh.duplicated().any():
        raise ValueError('Prior-text mapping requires unique accessions.')
    daily=(meta.sort_values(['cik','form','filed_date','adsh'],kind='stable')
           .drop_duplicates(['cik','form','filed_date'],keep='first').copy())
    daily['prior_adsh']=daily.groupby(['cik','form'],sort=False).adsh.shift(1)
    return meta[['adsh','cik','form','filed_date']].merge(
        daily[['cik','form','filed_date','prior_adsh']],
        on=['cik','form','filed_date'],how='left',validate='many_to_one')


def text_change_features(frame,batch_size=512):
    """Compute point-in-time lexical similarity to each issuer's prior same-form MD&A."""
    from sklearn.feature_extraction.text import HashingVectorizer
    if 'mda_text' not in frame:
        raise ValueError('MD&A text is required for change features.')
    links=previous_text_links(frame)
    text_by_id=dict(zip(frame.adsh.astype(str),frame.mda_text.fillna('').astype(str)))
    current=frame.mda_text.fillna('').astype(str).tolist()
    prior_ids=links.prior_adsh.fillna('').astype(str).tolist()
    previous=[text_by_id.get(adsh,'') for adsh in prior_ids]
    vectorizer=HashingVectorizer(n_features=2**16,ngram_range=(1,2),norm='l2',
        alternate_sign=False,lowercase=True,strip_accents='unicode',
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z0-9]+\b',dtype=np.float32)
    similarity=np.zeros(len(frame),dtype=np.float32)
    available=np.zeros(len(frame),dtype=np.int8)
    for start in range(0,len(frame),batch_size):
        stop=min(start+batch_size,len(frame))
        left=current[start:stop]; right=previous[start:stop]
        mask=np.asarray([bool(a.strip()) and bool(b.strip()) for a,b in zip(left,right)])
        if mask.any():
            a=vectorizer.transform([value for value,keep in zip(left,mask) if keep])
            b=vectorizer.transform([value for value,keep in zip(right,mask) if keep])
            indices=start+np.flatnonzero(mask)
            similarity[indices]=np.asarray(a.multiply(b).sum(axis=1)).reshape(-1)
            available[indices]=1
    result=links[['adsh','prior_adsh']].copy()
    result['text_prior_available']=available
    result['text_pairwise_cosine']=similarity
    result['text_pairwise_novelty']=np.where(available==1,1.0-similarity,0.0).astype(np.float32)
    if 'mda_word_count' in frame:
        word_counts=pd.Series(frame.mda_word_count.to_numpy(),index=frame.adsh.astype(str))
    else:
        counts=np.fromiter((len(value.split()) for value in frame.mda_text.fillna('').astype(str)),
                           dtype=np.int32,count=len(frame))
        word_counts=pd.Series(counts,index=frame.adsh.astype(str))
    prior_counts=links.prior_adsh.map(word_counts).fillna(0).to_numpy(dtype=np.float32)
    current_counts=np.asarray([len(value.split()) for value in current],dtype=np.float32)
    result['text_word_count_delta']=current_counts-prior_counts
    result['text_word_count_ratio']=current_counts/np.maximum(prior_counts,1.0)
    return result


def add_text_embedding_changes(frame,embeddings):
    """Add embedding deltas from the strictly earlier same-form filing when available."""
    links=previous_text_links(frame)
    region_names=('beginning','middle','ending')
    region_cols=[[f'text_emb_{region}_{i}' for i in range(384)] for region in region_names]
    required=['adsh']+[column for columns in region_cols for column in columns]
    if set(required)-set(embeddings):
        raise ValueError('Embedding table lacks expected region vectors.')
    current=embeddings[required].copy()
    current_mean=np.zeros((len(current),384),dtype=np.float32)
    for columns in region_cols:
        current_mean+=current[columns].to_numpy(dtype=np.float32)
    current_mean/=len(region_cols)
    current_ids=current.adsh.astype(str).to_numpy()
    prior_table=pd.DataFrame(current_mean,columns=[f'prior_{i}' for i in range(384)])
    prior_table.insert(0,'prior_adsh',current_ids)
    linked=links[['adsh','prior_adsh']].merge(prior_table,on='prior_adsh',how='left',validate='many_to_one')
    prior=np.column_stack([linked[f'prior_{i}'].to_numpy(dtype=np.float32) for i in range(384)])
    available=(linked.prior_adsh.notna().to_numpy() & np.isfinite(prior).all(axis=1) &
               (np.linalg.norm(prior,axis=1)>0) & (np.linalg.norm(current_mean,axis=1)>0))
    prior[~available]=0.0
    delta=current_mean-prior
    result=pd.DataFrame(delta,columns=[f'text_emb_delta_{i}' for i in range(384)])
    result.insert(0,'adsh',current_ids)
    result['text_embedding_prior_available']=available.astype(np.int8)
    return frame[['adsh']].merge(result,on='adsh',how='left',validate='one_to_one')


def fit_xgb(train,predict,numeric,categorical,depth,seed,fixed=None,valid=None,model_params=None):
    from xgboost import XGBClassifier
    x,xp=matrices(train,predict,numeric,categorical)
    y=train.y_amendment_proxy.to_numpy(dtype=np.int8)
    if len(np.unique(y))!=2:
        raise ValueError('Both labels are required for fitting.')
    params={'max_depth':depth,'learning_rate':.025,'min_child_weight':5,'subsample':.85,
            'colsample_bytree':.85,'reg_lambda':8,'reg_alpha':.2}
    params.update(model_params or {})
    model=XGBClassifier(n_estimators=fixed or 1600,**params,
        tree_method='hist',device='cuda',n_jobs=4,objective='binary:logistic',eval_metric='aucpr',
        random_state=seed,**({'early_stopping_rounds':120} if fixed is None else {}))
    if fixed is None:
        if valid is None:
            raise ValueError('Chronological validation is required for early stopping.')
        _,v=matrices(train,valid,numeric,categorical)
        model.fit(x,y,eval_set=[(v,valid.y_amendment_proxy.to_numpy(dtype=np.int8))],verbose=False)
        iterations=int(model.best_iteration)+1
    else:
        model.fit(x,y,verbose=False); iterations=int(fixed)
    device=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
    if not device.startswith('cuda'):
        raise RuntimeError(f'XGBoost used {device}; this notebook requires GPU fitting.')
    probabilities=model.predict_proba(xp)[:,1]
    return probabilities,iterations,model,x.columns.tolist()


def _rank_groups(frame):
    part=frame.copy()
    part['_rank_year']=pd.to_datetime(part.filed_date).dt.year
    order=np.lexsort((part.adsh.astype(str).to_numpy(),part.form.astype(str).to_numpy(),
                      part._rank_year.to_numpy()))
    ordered=part.iloc[order]
    sizes=ordered.groupby(['_rank_year','form'],sort=False).size().astype(int).tolist()
    labels=ordered.y_amendment_proxy.to_numpy(dtype=np.int8)
    start=0
    for size in sizes:
        if size<2 or len(np.unique(labels[start:start+size]))<2:
            raise ValueError('Pairwise ranking needs both labels in every year-form query group.')
        start+=size
    return order,sizes


def fit_pairwise_ranker(train,predict,numeric,categorical,depth,seed,fixed=None,valid=None):
    """GPU XGBoost pairwise ranking within filing-year/form matched queries."""
    from xgboost import XGBRanker
    x,xp=matrices(train,predict,numeric,categorical)
    train_order,train_groups=_rank_groups(train)
    y=train.iloc[train_order].y_amendment_proxy.to_numpy(dtype=np.int8)
    params={'max_depth':depth,'learning_rate':.03,'min_child_weight':6,'subsample':.8,
            'colsample_bytree':.8,'reg_lambda':12,'reg_alpha':.3}
    ranker=XGBRanker(n_estimators=fixed or 1600,**params,tree_method='hist',device='cuda',
        n_jobs=4,objective='rank:pairwise',eval_metric='ndcg',random_state=seed,
        **({'early_stopping_rounds':120} if fixed is None else {}))
    if fixed is None:
        if valid is None:
            raise ValueError('Pairwise ranker early stopping requires a matched validation cohort.')
        _,v=matrices(train,valid,numeric,categorical)
        valid_order,valid_groups=_rank_groups(valid)
        vy=valid.iloc[valid_order].y_amendment_proxy.to_numpy(dtype=np.int8)
        ranker.fit(x.iloc[train_order],y,group=train_groups,
                   eval_set=[(v.iloc[valid_order],vy)],eval_group=[valid_groups],verbose=False)
        iterations=int(ranker.best_iteration)+1
    else:
        ranker.fit(x.iloc[train_order],y,group=train_groups,verbose=False)
        iterations=int(fixed)
    device=json.loads(ranker.get_booster().save_config())['learner']['generic_param']['device']
    if not device.startswith('cuda'):
        raise RuntimeError(f'XGBoost ranker used {device}; GPU fitting is required.')
    margin=ranker.predict(xp)
    probabilities=1.0/(1.0+np.exp(-np.clip(margin,-30,30)))
    return probabilities,iterations,ranker,x.columns.tolist()


def fit_tfidf_xgb(train,predict,numeric,categorical,depth,seed,fixed=None,valid=None,
                  save_path=None,model_params=None):
    """Fit sparse word/character TF-IDF plus tabular features on CUDA XGBoost."""
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    from xgboost import XGBClassifier
    vectorizer=FeatureUnion([
        ('word',TfidfVectorizer(analyzer='word',ngram_range=(1,2),min_df=3,
            max_features=40000,sublinear_tf=True,dtype=np.float32,strip_accents='unicode')),
        ('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=4,
            max_features=30000,sublinear_tf=True,dtype=np.float32,strip_accents='unicode'))])
    train_text=train.mda_text.fillna('').astype(str).tolist()
    predict_text=predict.mda_text.fillna('').astype(str).tolist()
    text_x=vectorizer.fit_transform(train_text)
    text_p=vectorizer.transform(predict_text)
    dense_x,dense_p=matrices(train,predict,numeric,categorical)
    x=sparse.hstack([text_x,sparse.csr_matrix(dense_x.to_numpy(dtype=np.float32))],format='csr')
    xp=sparse.hstack([text_p,sparse.csr_matrix(dense_p.to_numpy(dtype=np.float32))],format='csr')
    params={'max_depth':depth,'learning_rate':.04,'min_child_weight':8,'subsample':.8,
            'colsample_bytree':.6,'reg_lambda':16,'reg_alpha':.5}
    params.update(model_params or {})
    model=XGBClassifier(n_estimators=fixed or 1400,**params,tree_method='hist',device='cuda',
        n_jobs=4,objective='binary:logistic',eval_metric='aucpr',random_state=seed,
        **({'early_stopping_rounds':120} if fixed is None else {}))
    if fixed is None:
        if valid is None:
            raise ValueError('Sparse text XGBoost early stopping requires a time validation cohort.')
        text_v=vectorizer.transform(valid.mda_text.fillna('').astype(str).tolist())
        _,dense_v=matrices(train,valid,numeric,categorical)
        v=sparse.hstack([text_v,sparse.csr_matrix(dense_v.to_numpy(dtype=np.float32))],format='csr')
        model.fit(x,train.y_amendment_proxy.to_numpy(dtype=np.int8),
                  eval_set=[(v,valid.y_amendment_proxy.to_numpy(dtype=np.int8))],verbose=False)
        iterations=int(model.best_iteration)+1
    else:
        model.fit(x,train.y_amendment_proxy.to_numpy(dtype=np.int8),verbose=False)
        iterations=int(fixed)
    device=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
    if not device.startswith('cuda'):
        raise RuntimeError(f'Sparse XGBoost used {device}; GPU fitting is required.')
    if save_path is not None:
        import joblib
        joblib.dump(vectorizer,Path(str(save_path)+'.tfidf.joblib'))
    return model.predict_proba(xp)[:,1],iterations,model,[]


def fit_mlp(train_x,y,predict_x,seed,valid_x=None,valid_y=None,epochs=None,save_path=None):
    """Fit a small dense classifier with early stopping on an inner time slice."""
    import torch
    from sklearn.metrics import average_precision_score
    if not torch.cuda.is_available():
        raise RuntimeError('The MLP requires CUDA; CPU fallback is disabled.')
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    a=train_x.to_numpy(dtype=np.float32); b=predict_x.to_numpy(dtype=np.float32)
    usable=~np.isnan(a).all(axis=0)
    median=np.zeros(a.shape[1],dtype=np.float32)
    median[usable]=np.nanmedian(a[:,usable],axis=0)
    a=np.where(np.isnan(a),median,a); b=np.where(np.isnan(b),median,b)
    mean=a.mean(axis=0); std=np.maximum(a.std(axis=0),1e-3)
    a=np.clip((a-mean)/std,-10,10); b=np.clip((b-mean)/std,-10,10)
    if valid_x is not None:
        v=valid_x.to_numpy(dtype=np.float32)
        v=np.where(np.isnan(v),median,v); v=np.clip((v-mean)/std,-10,10)
    device=torch.device('cuda')
    model=torch.nn.Sequential(
        torch.nn.Linear(a.shape[1],256),torch.nn.LayerNorm(256),torch.nn.GELU(),torch.nn.Dropout(.35),
        torch.nn.Linear(256,128),torch.nn.LayerNorm(128),torch.nn.GELU(),torch.nn.Dropout(.2),
        torch.nn.Linear(128,1)).to(device)
    xa=torch.tensor(a,device=device); ya=torch.tensor(y,dtype=torch.float32,device=device)
    xb=torch.tensor(b,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=.02)
    best=-np.inf; best_epoch=0; best_state=None; stale=0
    def probabilities(matrix):
        model.eval()
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(model(matrix[i:i+2048])).flatten().cpu().numpy()
                                   for i in range(0,len(matrix),2048)])
    for epoch in range(epochs or 100):
        model.train(); order=torch.randperm(len(xa),device=device)
        for start in range(0,len(order),256):
            indices=order[start:start+256]
            loss=torch.nn.functional.binary_cross_entropy_with_logits(
                model(xa[indices]).flatten(),ya[indices])
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
        if epochs is None:
            if valid_x is None or valid_y is None or len(np.unique(valid_y))<2:
                raise ValueError('MLP early stopping requires both classes in an inner validation slice.')
            score=average_precision_score(valid_y,probabilities(torch.tensor(v,device=device)))
            if score>best+1e-5:
                best=score; best_epoch=epoch+1; stale=0
                best_state={name:value.detach().clone() for name,value in model.state_dict().items()}
            else:
                stale+=1
            if stale>=12: break
    if epochs is None:
        if best_state is None:
            raise RuntimeError('MLP did not retain an early-stopping checkpoint.')
        model.load_state_dict(best_state)
        chosen=best_epoch
    else:
        chosen=int(epochs)
    probabilities_out=probabilities(xb)
    if save_path is not None:
        save_path=Path(save_path); save_path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'state_dict':{name:value.detach().cpu() for name,value in model.state_dict().items()},
                    'feature_columns':train_x.columns.tolist(),'median':median,'mean':mean,'std':std,
                    'seed':int(seed),'epochs':int(chosen),'hidden_layers':[256,128]},save_path)
    del model,xa,ya,xb
    torch.cuda.empty_cache()
    return probabilities_out,chosen


def specialist_model_base(base,form):
    return Path(base).with_name(Path(base).name+'_'+str(form).replace('-',''))


def fit_model(config,train,predict,numeric,categorical,seed,fixed=None,valid=None,save_path=None):
    if config['family']=='form_specialists':
        probabilities=np.full(len(predict),np.nan,dtype=np.float32)
        iterations=[]; columns=[]
        for form in sorted(set(train.form.astype(str)) & set(predict.form.astype(str))):
            train_mask=train.form.astype(str).eq(form)
            predict_mask=predict.form.astype(str).eq(form).to_numpy()
            valid_part=valid.loc[valid.form.astype(str).eq(form)].copy() if valid is not None else None
            base=specialist_model_base(save_path,form) if save_path is not None else None
            specialized={**config,'family':'xgb'}
            pred,count,feature_columns=fit_model(
                specialized,train.loc[train_mask].copy(),predict.loc[predict_mask].copy(),
                numeric,categorical,seed,fixed=fixed,valid=valid_part,save_path=base)
            probabilities[predict_mask]=pred
            iterations.append(count); columns.extend(f'{form}:{name}' for name in feature_columns)
        if not np.isfinite(probabilities).all():
            raise ValueError('Form specialists did not score every requested filing.')
        return probabilities,int(np.median(iterations)),columns
    if config['family']=='xgb':
        probabilities,iterations,model,columns=fit_xgb(
            train,predict,numeric,categorical,config['depth'],seed,fixed=fixed,valid=valid,
            model_params=config.get('xgb_params'))
        if save_path is not None:
            model.save_model(str(save_path.with_suffix('.json')))
        return probabilities,iterations,columns
    if config['family']=='ranker':
        probabilities,iterations,model,columns=fit_pairwise_ranker(
            train,predict,numeric,categorical,config['depth'],seed,fixed=fixed,valid=valid)
        if save_path is not None:
            model.save_model(str(save_path.with_suffix('.json')))
        return probabilities,iterations,columns
    if config['family']=='tfidf_xgb':
        probabilities,iterations,model,columns=fit_tfidf_xgb(
            train,predict,numeric,categorical,config['depth'],seed,fixed=fixed,valid=valid,
            save_path=save_path,model_params=config.get('xgb_params'))
        if save_path is not None:
            model.save_model(str(save_path.with_suffix('.json')))
        return probabilities,iterations,columns
    x,xp=matrices(train,predict,numeric,categorical)
    vx=None; vy=None
    if fixed is None:
        if valid is None:
            raise ValueError('Chronological inner validation is required for MLP early stopping.')
        _,vx=matrices(train,valid,numeric,categorical)
        vy=valid.y_amendment_proxy.to_numpy(dtype=np.int8)
    probabilities,iterations=fit_mlp(x,train.y_amendment_proxy.to_numpy(dtype=np.int8),xp,
                                      seed,valid_x=vx,valid_y=vy,epochs=fixed,
                                      save_path=save_path.with_suffix('.pt') if save_path else None)
    if save_path is not None:
        save_path.with_suffix('.json').write_text(json.dumps(
            {'feature_columns':x.columns.tolist(),'seed':int(seed),'epochs':int(iterations),
             'device':'cuda','family':'mlp'},indent=2),encoding='utf-8')
    return probabilities,iterations,x.columns.tolist()


def score_report(y,p,threshold):
    from sklearn.metrics import average_precision_score,roc_auc_score,matthews_corrcoef
    y=np.asarray(y,dtype=np.int8); p=np.asarray(p,dtype=float)
    threshold_array=np.asarray(threshold,dtype=float)
    if threshold_array.ndim and threshold_array.shape!=p.shape:
        raise ValueError('Per-row thresholds must align with prediction scores.')
    pred=p>=threshold_array
    threshold_report=(float(threshold_array) if threshold_array.ndim==0 else 'row_specific')
    tp=int(np.sum(pred&(y==1))); fp=int(np.sum(pred&(y==0)))
    fn=int(np.sum(~pred&(y==1))); tn=int(np.sum(~pred&(y==0)))
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
    f1=2*precision*recall/max(precision+recall,1e-12)
    accuracy=(tp+tn)/len(y)
    return {'n':len(y),'positive_count':int(y.sum()),'prevalence_in_matched_sample':float(y.mean()),
            'threshold':threshold_report,'accuracy':accuracy,'precision':precision,'recall':recall,
            'f1':f1,'balanced_accuracy':(recall+tn/max(tn+fp,1))/2,
            'pr_auc':float(average_precision_score(y,p)),'roc_auc':float(roc_auc_score(y,p)),
            'mcc':float(matthews_corrcoef(y,pred)),'tn_fp_fn_tp':[tn,fp,fn,tp],
            'all_four_ge_80':bool(min(accuracy,precision,recall,f1)>=.80),
            'all_four_ge_85':bool(min(accuracy,precision,recall,f1)>=.85)}


def issuer_cluster_confidence_intervals(frame,probability,threshold,draws=400,seed=2026):
    """Issuer-cluster bootstrap intervals for thresholded matched-cohort metrics."""
    probability=np.asarray(probability,dtype=float)
    if len(frame)!=len(probability) or frame.cik.isna().any():
        raise ValueError('Cluster intervals require one finite score and CIK per row.')
    y=frame.y_amendment_proxy.to_numpy(dtype=np.int8)
    threshold_array=np.asarray(threshold,dtype=float)
    if threshold_array.ndim and threshold_array.shape!=probability.shape:
        raise ValueError('Per-row bootstrap thresholds must align with prediction scores.')
    pred=probability>=threshold_array
    counts=pd.DataFrame({'cik':frame.cik.astype(str).to_numpy(),
        'tp':pred&(y==1),'fp':pred&(y==0),'tn':(~pred)&(y==0),'fn':(~pred)&(y==1)})
    clusters=counts.groupby('cik',sort=False)[['tp','fp','tn','fn']].sum().to_numpy(dtype=np.int64)
    if len(clusters)<2 or draws<1:
        raise ValueError('Cluster intervals need at least two issuers and one bootstrap draw.')
    rng=np.random.default_rng(seed)
    estimates=[]
    for _ in range(draws):
        tp,fp,tn,fn=clusters[rng.integers(0,len(clusters),len(clusters))].sum(axis=0)
        precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
        f1=2*precision*recall/max(precision+recall,1e-12)
        estimates.append([(tp+tn)/max(tp+fp+tn+fn,1),precision,recall,f1])
    bounds=np.quantile(np.asarray(estimates),[.025,.975],axis=0)
    return {'method':'issuer-cluster bootstrap percentile interval','draws':int(draws),
            'clusters':int(len(clusters)),
            'ci_95':{name:[float(bounds[0,i]),float(bounds[1,i])]
                     for i,name in enumerate(('accuracy','precision','recall','f1'))}}


def calibrated_form_thresholds(calibration,probability,global_threshold,min_per_class=30):
    """Calibrate 10-K/10-Q cutoffs on 2023 only, with a minimum-count fallback."""
    probability=np.asarray(probability,dtype=float)
    if len(calibration)!=len(probability):
        raise ValueError('Calibration rows and scores differ in length.')
    result={}
    forms=calibration.form.astype(str).to_numpy()
    labels=calibration.y_amendment_proxy.to_numpy(dtype=np.int8)
    for form in sorted(np.unique(forms)):
        mask=forms==form
        counts=np.bincount(labels[mask],minlength=2)
        if np.min(counts)<min_per_class:
            result[form]={'threshold':float(global_threshold),'source':'global_fallback',
                          'calibration_rows':int(mask.sum()),
                          'positive_count':int(counts[1]),'negative_count':int(counts[0])}
        else:
            selected=core.threshold_choice(labels[mask],probability[mask])
            result[form]={**selected,'source':'form_specific_2023_calibration',
                          'calibration_rows':int(mask.sum()),
                          'positive_count':int(counts[1]),'negative_count':int(counts[0])}
    return result


def thresholds_for_frame(frame,threshold_map,global_threshold):
    return np.asarray([threshold_map.get(str(form),{}).get('threshold',global_threshold)
                       for form in frame.form],dtype=float)


def embed_text(frame,cache_dir,device):
    import torch
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import model_info
    cache_dir=Path(cache_dir); cache_dir.mkdir(parents=True,exist_ok=True)
    revision=model_info('sentence-transformers/all-MiniLM-L6-v2').sha
    encoder_record={'model':'sentence-transformers/all-MiniLM-L6-v2','revision':revision,
                    'license':'Apache-2.0','source':'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2',
                    'pooling':'Up to 20 chunks sampled evenly across the full extracted section; beginning, middle, ending means and document-level max pooling'}
    (cache_dir/'encoder.json').write_text(json.dumps(encoder_record,indent=2),encoding='utf-8')
    cache_key=hashlib.sha256(json.dumps([core.file_hash(frame.attrs['dataset_path']),
                                        revision,'mda-full-section-streamed-even-chunks-regions-max-v5'],sort_keys=True).encode()).hexdigest()[:24]
    path=cache_dir/f'{cache_key}.parquet'
    if path.exists():
        cached=pd.read_parquet(path)
        if cached.adsh.tolist()==frame.adsh.tolist():
            return cached,revision
    encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',revision=revision,
                                device=device,trust_remote_code=False)
    if not str(encoder.device).startswith('cuda'):
        raise RuntimeError(f'MiniLM is on {encoder.device}; this experiment requires GPU embeddings.')
    # Stream at most a few thousand 160-word chunks at once. Materializing
    # token lists/chunks for every filing can exceed Colab RAM on the full
    # manifest even when the GPU itself has enough memory.
    vectors=np.zeros((len(frame),3,384),dtype=np.float32)
    counts=np.zeros((len(frame),3),dtype=np.float32)
    chunk_max=np.full((len(frame),384),-np.inf,dtype=np.float32)
    chunks=[]; owners=[]; regions=[]; processed=0
    def flush_chunk_batch():
        nonlocal processed
        if not chunks:
            return
        encoded=encoder.encode(chunks,batch_size=64,show_progress_bar=False,
                              convert_to_numpy=True,normalize_embeddings=True)
        owner_ids=np.asarray(owners,dtype=np.int64); region_ids=np.asarray(regions,dtype=np.int64)
        np.add.at(vectors,(owner_ids,region_ids),encoded)
        np.add.at(counts,(owner_ids,region_ids),1)
        np.maximum.at(chunk_max,owner_ids,encoded)
        processed+=len(chunks)
        print(f'Embedded {processed:,} MD&A text chunks on GPU.',flush=True)
        chunks.clear(); owners.clear(); regions.clear()
    for owner,text in enumerate(frame.mda_text.fillna('').astype(str)):
        tokens=text.split()
        chunk_count=(len(tokens)+159)//160
        if not chunk_count:
            continue
        starts=list(range(0,len(tokens),160))
        selected=(np.unique(np.linspace(0,len(starts)-1,num=20,dtype=int)).tolist()
                  if len(starts)>20 else list(range(len(starts))))
        edge_count=max(1,(chunk_count+3)//4)
        for chunk_index in selected:
            start=starts[chunk_index]
            part=' '.join(tokens[start:start+160])
            if not part:
                continue
            chunks.append(part); owners.append(owner)
            regions.append(0 if chunk_index<edge_count else
                           2 if chunk_index>=chunk_count-edge_count else 1)
            if len(chunks)>=4096:
                flush_chunk_batch()
    flush_chunk_batch()
    vectors/=np.maximum(counts[:,:,None],1)
    chunk_max[~np.isfinite(chunk_max)]=0.0
    pieces=[frame[['adsh']].reset_index(drop=True)]
    for region,name in enumerate(('beginning','middle','ending')):
        pieces.append(pd.DataFrame(vectors[:,region,:],
                                   columns=[f'text_emb_{name}_{i}' for i in range(384)]))
    pieces.append(pd.DataFrame(chunk_max,columns=[f'text_emb_chunk_max_{i}' for i in range(384)]))
    result=pd.concat(pieces,axis=1)
    temp=path.with_suffix('.part.parquet'); result.to_parquet(temp,index=False); temp.replace(path)
    del encoder; torch.cuda.empty_cache()
    return result,revision


def text_style_features(frame):
    """Transparent style counts with bounded per-document token memory."""
    texts=frame.mda_text.fillna('').astype(str).tolist()
    pattern=re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)?%?")
    groups={
        'uncertainty':{'may','might','could','approximately','estimate','estimated','uncertain','uncertainty'},
        'forward_looking':{'expect','expects','expected','anticipate','anticipates','believe','plan','plans','intend','intends'},
        'negation':{'no','not','never','neither','nor','without','unable'},
        'contrast':{'however','although','whereas','despite','nevertheless','conversely'} }
    known_counts=(frame.mda_word_count.to_numpy(dtype=np.float32)
                  if 'mda_word_count' in frame else None)
    values=[]
    for i,text in enumerate(texts):
        words=text.split()
        if len(words)>2000:
            middle=len(words)//2
            selected=words[:700]+words[middle-350:middle+350]+words[-600:]
            full_count=known_counts[i] if known_counts is not None else len(words)
        else:
            selected=words
            full_count=known_counts[i] if known_counts is not None else len(words)
        tokens=pattern.findall(' '.join(selected))
        lowered=[token.lower() for token in tokens]
        denom=max(len(tokens),1)
        row={
            'text_style_unique_token_fraction':len(set(lowered))/denom,
            'text_style_mean_token_length':float(np.mean([len(token) for token in tokens])) if tokens else 0.0,
            'text_style_long_token_fraction':sum(len(token)>=10 for token in tokens)/denom,
            'text_style_numeric_token_fraction':sum(any(ch.isdigit() for ch in token) for token in tokens)/denom,
            'text_style_mean_sentence_length':float(full_count)/max(text.count('.')+text.count('!')+text.count('?'),1),
        }
        for name,vocabulary in groups.items():
            row[f'text_style_{name}_per_1000']=1000*sum(token in vocabulary for token in lowered)/denom
        values.append(row)
    result=pd.DataFrame(values,index=frame.index)
    # Transparent generic language groups; no external/private lexicon is used.
    return result.astype('float32')


def prior_filing_features(frame):
    """Build same-issuer, same-form changes from the complete prepared panel."""
    columns=[name for name in core.RATIOS if name in frame]
    if not columns:
        return frame[['adsh']].copy()
    keys=['cik','form','filed_date']
    daily=(frame.groupby(keys,as_index=False,dropna=False)[columns].mean()
           .sort_values(keys,kind='stable').reset_index(drop=True))
    previous=daily.groupby(['cik','form'],sort=False)[columns].shift(1)
    engineered=daily[keys].copy()
    for name in columns:
        prior=previous[name].to_numpy(dtype='float64')
        current=daily[name].to_numpy(dtype='float64')
        delta=current-prior
        engineered[f'prior_filing_{name}']=prior
        engineered[f'change_prior_filing_{name}']=delta
        engineered[f'rel_change_prior_filing_{name}']=np.clip(
            delta/(np.abs(prior)+1e-3),-10,10)
    mapping=frame[['adsh']+keys].merge(engineered,on=keys,how='left',validate='many_to_one')
    return mapping.drop(columns=keys)


def run(root,text_path=None,experiment_name='mda_text_expanded'):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Select a Colab NVIDIA GPU runtime before running this experiment.')
    root=Path(root)
    text_path=Path(text_path) if text_path else root/'data/colab_mda_text_case_control.csv.gz'
    proxy_path=root/'data/colab_amendment_proxy.csv.gz'
    if not text_path.is_file() or not proxy_path.is_file():
        raise FileNotFoundError('Upload both prepared CSV files to Drive under data/.')
    gpu_name=torch.cuda.get_device_name(0)
    print('Confirmed GPU:',gpu_name,flush=True)
    runtime_versions={'python':sys.version.split()[0],**{
        name:importlib.metadata.version(dist) for name,dist in
        [('torch','torch'),('xgboost','xgboost-cu12'),('scikit_learn','scikit-learn'),
         ('sentence_transformers','sentence-transformers'),('numpy','numpy'),('pandas','pandas')]}}
    runtime_signature={**runtime_versions,'gpu':gpu_name}
    text=pd.read_csv(text_path,compression='gzip',dtype={'adsh':str,'cik':str,'form':str},low_memory=False)
    text['filed_date']=pd.to_datetime(text.filed_date,errors='raise')
    text['mda_text']=text.mda_text.fillna('').astype(str)
    if 'mda_word_count' not in text:
        text['mda_word_count']=text.mda_text.str.split().str.len()
    if 'text_available' not in text:
        text['text_available']=text.mda_word_count.ge(200).astype('int8')
    text['y_amendment_proxy']=text.y_proxy.astype('int8')
    # Reuse current as-filed features only for the selected MD&A accessions.
    wanted=set(text.adsh); parts=[]; history_parts=[]
    proxy_row_count=0; proxy_positive_count=0
    history_columns=[name for name in core.RATIOS if name in pd.read_csv(
        proxy_path,compression='gzip',nrows=0).columns]
    for chunk in pd.read_csv(proxy_path,compression='gzip',dtype={'adsh':str,'cik':str},
                             low_memory=False,chunksize=50000):
        proxy_row_count+=len(chunk)
        proxy_positive_count+=int(chunk.y_amendment_proxy.sum())
        keep=chunk[chunk.adsh.isin(wanted)]
        if len(keep): parts.append(keep)
        history_parts.append(chunk[['adsh','cik','form','filed_date',*history_columns]])
    tabular=pd.concat(parts,ignore_index=True)
    if tabular.adsh.duplicated().any() or set(tabular.adsh)!=wanted:
        raise ValueError('MD&A accessions do not match the current prepared data.')
    aligned=text[['adsh','form','filed_date','y_amendment_proxy']].merge(
        tabular[['adsh','form','filed_date','y_amendment_proxy']],on='adsh',
        how='inner',validate='one_to_one',suffixes=('_text','_proxy'))
    if not aligned.form_text.eq(aligned.form_proxy).all() or not aligned.y_amendment_proxy_text.eq(aligned.y_amendment_proxy_proxy).all():
        raise ValueError('Filing metadata or relabeling mismatch.')
    if not aligned.filed_date_text.eq(aligned.filed_date_proxy).all():
        raise ValueError('Filing dates differ between the text sample and proxy data.')
    dataset_hash=core.file_hash(proxy_path); text_hash=core.file_hash(text_path)
    text.attrs['dataset_path']=text_path
    embeddings,revision=embed_text(text,root/'evidence_v5/mda_embeddings',device='cuda')
    style=text_style_features(text)
    lexical_change=text_change_features(text)
    sample=text.merge(embeddings,on='adsh',validate='one_to_one').merge(
        tabular.drop(columns=['cik','form','filed_date','y_amendment_proxy']),on='adsh',
        how='inner',validate='one_to_one',suffixes=('','_proxy'))
    sample=pd.concat([sample.reset_index(drop=True),style.reset_index(drop=True),
                      lexical_change.drop(columns='adsh').reset_index(drop=True)],axis=1)
    embedding_change=add_text_embedding_changes(sample,embeddings)
    sample=sample.merge(embedding_change,on='adsh',how='left',validate='one_to_one')
    history=pd.concat(history_parts,ignore_index=True)
    history['filed_date']=pd.to_datetime(history.filed_date,errors='raise')
    sample=sample.merge(prior_filing_features(history),on='adsh',how='left',validate='one_to_one')
    # Use labels and dates checked against the current CSV above.
    sample['y_amendment_proxy']=sample.y_proxy.astype('int8')
    sample['filed_date']=pd.to_datetime(sample.filed_date,errors='raise')
    sample['fp']=sample.fp.fillna('unknown').astype(str)
    sample['sic2']=sample.sic2.fillna('unknown').astype(str)
    for name in ('y_amendment_proxy','cik'):
        if name not in sample: raise ValueError(f'Missing required field {name}')
    # A fixed CIK hash keeps issuers out of every CV fit and out of final model
    # selection; their later filings provide a separate cold-start comparison.
    heldout_issuer=issuer_holdout_mask(sample.cik)
    selection_sample=sample.loc[~heldout_issuer].copy()
    configs=[{'name':'text_xgb_d3','family':'xgb','depth':3,'tabular':False},
             {'name':'text_xgb_d5','family':'xgb','depth':5,'tabular':False},
             {'name':'text_plus_financial_xgb_d3','family':'xgb','depth':3,'tabular':True},
             {'name':'text_plus_financial_xgb_d5','family':'xgb','depth':5,'tabular':True},
             {'name':'text_plus_financial_xgb_d2_regularized','family':'xgb','depth':2,'tabular':True,
              'xgb_params':{'min_child_weight':8,'subsample':.8,'colsample_bytree':.7,
                            'reg_lambda':20,'reg_alpha':.5}},
             {'name':'text_plus_financial_xgb_d4','family':'xgb','depth':4,'tabular':True,
              'xgb_params':{'learning_rate':.02,'min_child_weight':6,'colsample_bytree':.8,
                            'reg_lambda':12,'reg_alpha':.3}},
             {'name':'text_plus_financial_xgb_d6_regularized','family':'xgb','depth':6,'tabular':True,
              'xgb_params':{'min_child_weight':10,'subsample':.8,'colsample_bytree':.7,
                            'reg_lambda':16,'reg_alpha':.5}},
             {'name':'text_plus_financial_xgb_d2_form_specialists','family':'form_specialists','depth':2,'tabular':True,
              'xgb_params':{'min_child_weight':8,'subsample':.8,'colsample_bytree':.7,
                            'reg_lambda':20,'reg_alpha':.5}},
             {'name':'text_plus_financial_pairwise_ranker','family':'ranker','depth':3,'tabular':True},
             {'name':'text_plus_financial_word_char_tfidf_xgb','family':'tfidf_xgb','depth':3,'tabular':True,
              'xgb_params':{'min_child_weight':10,'subsample':.8,'colsample_bytree':.6,
                            'reg_lambda':16,'reg_alpha':.5}},
             {'name':'text_mlp','family':'mlp','depth':0,'tabular':False},
             {'name':'text_plus_financial_mlp','family':'mlp','depth':0,'tabular':True}]
    # The early-stopping year is inside each training window; the later year
    # remains untouched until outer scoring.
    folds=[(2016,2018),(2018,2020),(2020,2022)]
    cv={}; leaderboard=[]
    from sklearn.metrics import average_precision_score
    trainer_hash=core.file_hash(__file__)
    core_hash=core.file_hash(core.__file__)
    cv_cache_root=root/'evidence_v5'/'mda_cv_cache'
    for config in configs:
        numeric,categorical=columns_for(sample,config['tabular'])
        preds=[]; iterations=[]; fold_scores=[]
        for fit_year,valid_year in folds:
            first_year=max(2009,fit_year-7)
            fit_rows=selection_sample[selection_sample.filed_date.between(f'{first_year}-01-01',f'{fit_year-1}-12-31')].copy()
            stop_pool=selection_sample[selection_sample.filed_date.between(f'{fit_year}-01-01',f'{fit_year}-12-31')]
            if fit_rows.empty or stop_pool.empty:
                raise ValueError(f'Empty inner time split for {fit_year} -> {valid_year}.')
            validation=matched_cohort(selection_sample[selection_sample.filed_date.dt.year.eq(valid_year)],707)
            train=balanced_train(fit_rows,42)
            stop=matched_cohort(stop_pool,707)
            fold_dir=cv_cache_root/config['name']/f'{fit_year}_{valid_year}'
            probability_path=fold_dir/'probabilities.npy'; metadata_path=fold_dir/'complete.json'
            signature=stable_signature({'config':config,'seed':42,'train':train.adsh.tolist(),
                'inner_stop':stop.adsh.tolist(),'outer_score':validation.adsh.tolist(),
                'proxy_sha256':dataset_hash,'text_sha256':text_hash,
                'trainer_sha256':trainer_hash,'core_sha256':core_hash,
                'runtime':runtime_signature})
            cached=None
            if probability_path.is_file() and metadata_path.is_file():
                metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
                if metadata.get('signature')==signature:
                    candidate=np.load(probability_path)
                    if candidate.shape==(len(validation),) and np.isfinite(candidate).all():
                        cached=(candidate,int(metadata['iterations']))
            if cached is not None:
                p,count=cached
                print(f"Resume cached CV member {config['name']} {fit_year}->{valid_year}.",flush=True)
            else:
                p,count,_=fit_model(config,train,validation,numeric,categorical,42,valid=stop)
                atomic_npy(probability_path,p)
                atomic_json(metadata_path,{'signature':signature,'iterations':int(count),
                                           'gpu':torch.cuda.get_device_name(0),
                                           'outer_year':valid_year,'inner_stop_year':fit_year})
            preds.append(p); iterations.append(count)
            fold_scores.append(float(average_precision_score(validation.y_amendment_proxy,p)))
            print(f"{config['name']} fit through {fit_year-1}, stop on {fit_year}, score on {valid_year}: AP={fold_scores[-1]:.4f}, iterations={count}",flush=True)
        row={**config,'cv_balanced_pr_auc':float(np.mean(fold_scores)),
             'fold_scores':fold_scores,'iterations':iterations,
             'inner_stop_years':[year for year,_ in folds],
             'outer_score_years':[year for _,year in folds],
             'final_iterations':int(np.median(iterations))}
        leaderboard.append(row); cv[config['name']]=preds
        print(row,flush=True)
    def cv_score(names):
        scores=[]
        for i,(_,year) in enumerate(folds):
            part=matched_cohort(selection_sample[selection_sample.filed_date.dt.year.eq(year)],707)
            combined=np.mean([cv[n][i] for n in names],axis=0)
            scores.append(average_precision_score(part.y_amendment_proxy,combined))
        return float(np.mean(scores))
    ranked=sorted(cv,key=lambda n:cv_score([n]),reverse=True)
    candidates=[ranked[:1],ranked[:min(3,len(ranked))]]
    selected=max(candidates,key=cv_score)
    selection={'members':selected,'cv_balanced_pr_auc':cv_score(selected),
               'single_model_ranking':{n:cv_score([n]) for n in ranked}}
    print('Selected from forward CV:',selection,flush=True)
    calibration=matched_cohort(selection_sample[selection_sample.filed_date.dt.year.eq(2023)],43)
    development_pool=sample[sample.filed_date.between('2024-10-01','2025-09-30')].copy()
    development=development_pool
    development=matched_cohort(development,44)
    scores={}
    for name in selected:
        config=next(c for c in configs if c['name']==name)
        numeric,categorical=columns_for(sample,config['tabular'])
        train=selection_sample[selection_sample.filed_date.between('2014-01-01','2021-12-31')]
        bags=[]
        artifacts=root/'evidence_v5'/'mda_models'; artifacts.mkdir(parents=True,exist_ok=True)
        for seed in core.SEEDS:
            bag=balanced_train(train,seed)
            predict_rows=pd.concat([calibration,development_pool])
            fixed=next(r['final_iterations'] for r in leaderboard if r['name']==name)
            model_base=artifacts/f'{name}_{seed}'
            model_file=model_base.with_suffix('.pt' if config['family']=='mlp' else '.json')
            vectorizer_file=Path(str(model_base)+'.tfidf.joblib')
            if config['family']=='form_specialists':
                specialist_files=[specialist_model_base(model_base,form).with_suffix('.json')
                                  for form in ('10-K','10-Q')]
                model_complete=all(path.is_file() for path in specialist_files)
            else:
                model_complete=(model_file.is_file() and
                                (config['family']!='tfidf_xgb' or vectorizer_file.is_file()))
            member_dir=root/'evidence_v5'/'mda_member_cache'/name/str(seed)
            probability_path=member_dir/'probabilities.npy'; metadata_path=member_dir/'complete.json'
            signature=stable_signature({'config':config,'seed':seed,'fixed':fixed,
                'train':bag.adsh.tolist(),'predict':predict_rows.adsh.tolist(),
                'proxy_sha256':dataset_hash,'text_sha256':text_hash,
                'trainer_sha256':trainer_hash,'core_sha256':core_hash,
                'runtime':runtime_signature})
            cached=None
            if probability_path.is_file() and metadata_path.is_file() and model_complete:
                metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
                if metadata.get('signature')==signature:
                    candidate=np.load(probability_path)
                    if candidate.shape==(len(predict_rows),) and np.isfinite(candidate).all():
                        cached=(candidate,int(metadata['iterations']))
            if cached is not None:
                p,count=cached
                print(f'Resume cached final member {name}, seed {seed}.',flush=True)
            else:
                print(f'Final GPU fit {name}, seed {seed}, {len(bag):,} case-control rows.',flush=True)
                p,count,_=fit_model(config,bag,predict_rows,numeric,categorical,seed,fixed=fixed,
                                    save_path=model_base)
                atomic_npy(probability_path,p)
                atomic_json(metadata_path,{'signature':signature,'iterations':int(count),
                                           'gpu':torch.cuda.get_device_name(0),
                                           'family':config['family'],'train_rows':len(bag)})
            bags.append(p)
        scores[name]=np.mean(bags,axis=0)
    ensemble=np.mean([scores[name] for name in selected],axis=0)
    calibration_scores=ensemble[:len(calibration)]
    development_pool_scores=ensemble[len(calibration):]
    score_lookup=pd.Series(development_pool_scores,index=development_pool.adsh)
    development_scores=development.adsh.map(score_lookup).to_numpy(dtype=float)
    threshold=core.threshold_choice(calibration.y_amendment_proxy,calibration_scores)
    form_threshold_selection=calibrated_form_thresholds(
        calibration,calibration_scores,threshold['threshold'])
    calibration_form_thresholds=thresholds_for_frame(
        calibration,form_threshold_selection,threshold['threshold'])
    development_form_thresholds=thresholds_for_frame(
        development,form_threshold_selection,threshold['threshold'])
    sample_sensitivity=[]
    for seed in range(100,120):
        cohort=matched_cohort(development_pool,seed)
        cohort_scores=cohort.adsh.map(score_lookup).to_numpy(dtype=float)
        sample_sensitivity.append({'seed':seed,**score_report(
            cohort.y_amendment_proxy,cohort_scores,threshold['threshold']),
            'form_calibrated':score_report(cohort.y_amendment_proxy,cohort_scores,
                thresholds_for_frame(cohort,form_threshold_selection,threshold['threshold']))})
    sensitivity_metrics=('accuracy','precision','recall','f1','pr_auc','roc_auc')
    sensitivity_summary={name:{'mean':float(np.mean([r[name] for r in sample_sensitivity])),
                               'std':float(np.std([r[name] for r in sample_sensitivity],ddof=1)),
                               'min':float(np.min([r[name] for r in sample_sensitivity])),
                               'max':float(np.max([r[name] for r in sample_sensitivity]))}
                         for name in sensitivity_metrics}
    form_sensitivity_summary={name:{'mean':float(np.mean([r['form_calibrated'][name] for r in sample_sensitivity])),
                                    'std':float(np.std([r['form_calibrated'][name] for r in sample_sensitivity],ddof=1)),
                                    'min':float(np.min([r['form_calibrated'][name] for r in sample_sensitivity])),
                                    'max':float(np.max([r['form_calibrated'][name] for r in sample_sensitivity]))}
                              for name in sensitivity_metrics}
    issuer_development_pool=development_pool.loc[
        issuer_holdout_mask(development_pool.cik)].copy()
    issuer_development=None
    if not issuer_development_pool.empty:
        try:
            issuer_development=matched_cohort(issuer_development_pool,44)
            issuer_development_scores=issuer_development.adsh.map(score_lookup).to_numpy(dtype=float)
            issuer_disjoint_metrics=score_report(issuer_development.y_amendment_proxy,
                                                 issuer_development_scores,threshold['threshold'])
            issuer_disjoint_form_metrics=score_report(
                issuer_development.y_amendment_proxy,issuer_development_scores,
                thresholds_for_frame(issuer_development,form_threshold_selection,threshold['threshold']))
        except ValueError as exc:
            issuer_disjoint_metrics={'status':'insufficient_year_form_matches','detail':str(exc),
                                     'eligible_rows':len(issuer_development_pool)}
            issuer_disjoint_form_metrics={'status':'insufficient_year_form_matches','detail':str(exc),
                                          'eligible_rows':len(issuer_development_pool)}
    else:
        issuer_disjoint_metrics={'status':'no_heldout_issuers_in_development_period','eligible_rows':0}
        issuer_disjoint_form_metrics={'status':'no_heldout_issuers_in_development_period','eligible_rows':0}
    development_ci=issuer_cluster_confidence_intervals(
        development,development_scores,threshold['threshold'])
    issuer_development_ci=(issuer_cluster_confidence_intervals(
        issuer_development,issuer_development_scores,threshold['threshold'],seed=2027)
        if issuer_development is not None and 'n' in issuer_disjoint_metrics else None)
    development_form_ci=issuer_cluster_confidence_intervals(
        development,development_scores,development_form_thresholds,seed=2028)
    issuer_development_form_ci=(issuer_cluster_confidence_intervals(
        issuer_development,issuer_development_scores,
        thresholds_for_frame(issuer_development,form_threshold_selection,threshold['threshold']),seed=2029)
        if issuer_development is not None and 'n' in issuer_disjoint_form_metrics else None)
    metrics={'evaluation_status':'Previously inspected development case-control period; issuer-disjoint subset is an exploratory cold-start check; no natural-prevalence metric is estimated',
        'target':'Current FSDS same-period 365-day amendment proxy joined by accession; old corpus y excluded',
        'sampling_warning':'The MD&A corpus is an outcome-selected, approximately balanced sample. These results apply only to the matched text-bearing case-control cohort.',
        'gpu':gpu_name,'dataset_sha256':dataset_hash,
        'text_dataset_sha256':text_hash,
        'code_sha256':{'trainer':core.file_hash(__file__),'core':core.file_hash(core.__file__)},
        'runtime_versions':runtime_versions,
        'text_rows':len(text),'text_available_rows':int(text.text_available.sum()),
        'text_status_counts':(text.text_status.fillna('unknown').value_counts().to_dict()
                              if 'text_status' in text else {'provided_corpus':len(text)}),
        'text_positive':int(text.y_amendment_proxy.sum()),
        'full_proxy_rows':proxy_row_count,'full_proxy_positive_count':proxy_positive_count,
        'legacy_y_used':False,
        'selection':selection,'cv_leaderboard':leaderboard,
        'calibration':score_report(calibration.y_amendment_proxy,calibration_scores,threshold['threshold']),
        'calibration_form_specific_thresholds':{
            'thresholds':form_threshold_selection,
            'metrics':score_report(calibration.y_amendment_proxy,calibration_scores,
                                   calibration_form_thresholds),
            'rule':'Each 10-K/10-Q threshold is calibrated on 2023 only; groups below 30 examples per class use the global calibration threshold.'},
        'development':score_report(development.y_amendment_proxy,development_scores,threshold['threshold']),
        'development_form_specific_thresholds':score_report(
            development.y_amendment_proxy,development_scores,development_form_thresholds),
        'development_issuer_cluster_ci':development_ci,
        'development_form_threshold_issuer_cluster_ci':development_form_ci,
        'issuer_disjoint_development':issuer_disjoint_metrics,
        'issuer_disjoint_form_specific_thresholds':issuer_disjoint_form_metrics,
        'issuer_disjoint_issuer_cluster_ci':issuer_development_ci,
        'issuer_disjoint_form_threshold_issuer_cluster_ci':issuer_development_form_ci,
        'issuer_holdout':{'salt':'mda-expanded-cik-v1','fraction':.20,
                          'heldout_ciks':int(sample.loc[heldout_issuer,'cik'].nunique()),
                          'selection_ciks':int(sample.loc[~heldout_issuer,'cik'].nunique()),
                          'selection_accessions':len(selection_sample),
                          'interpretation':'Model selection and fitting exclude these CIKs. The period overlaps previously inspected development dates, so this is not an untouched prospective test.'},
        'development_control_sampling_sensitivity':{
            'description':'Twenty year-form matched 1:1 draws from the same text-bearing development pool; the calibrated threshold is kept fixed.',
            'draws':sample_sensitivity,'summary':sensitivity_summary,
            'form_specific_threshold_summary':form_sensitivity_summary},
        'threshold_selection':threshold,
        'development_rows_before_matching':int(sample.filed_date.between('2024-10-01','2025-09-30').sum()),
        'development_text_rows_after_matching':len(development),
        'text_only_encoder':'sentence-transformers/all-MiniLM-L6-v2; frozen; GPU embeddings; beginning/middle/ending means and chunk max pooling',
        'text_style_features':'Word count, lexical diversity, token/sentence length, numeric density, and transparent generic language-group rates; computed from supplied MD&A text',
        'text_change_features':'HashingVectorizer word unigram/bigram similarity and novelty plus embedding deltas from the strictly earlier same-issuer/form filing when both texts are available',
        'new_model_families':['GPU pairwise XGBoost ranking by year/form query','GPU XGBoost over train-fold-only word/character TF-IDF plus structured features'],
        'encoder_revision':revision,'encoder_license':'Apache-2.0',
        'cv_interpretation':'The chronological folds and leaderboard are used for model/ensemble selection; their scores are not an untouched estimate after trying multiple candidates.',
        'tabular_reference':'The matching-cohort v5 scores are saved from the existing v5 development predictions when available.'}
    # Compare the v5 balanced-score threshold on the exact same matched rows.
    prior_candidates=list(root.glob('colab_outputs/research_v5_*/development_predictions.csv.gz'))
    prior_candidates+=list(root.glob('research_v5_*/development_predictions.csv.gz'))
    prior_candidates+=list(root.glob('research_v5_*/research_v5_*/development_predictions.csv.gz'))
    prior=sorted(prior_candidates,key=lambda p:p.stat().st_mtime,reverse=True)
    if prior:
        previous=pd.read_csv(prior[0],compression='gzip',usecols=['adsh','balanced_score'])
        common=development[['adsh','y_amendment_proxy']].merge(previous,on='adsh',how='inner',validate='one_to_one')
        if len(common)==len(development):
            previous_metrics=json.loads((prior[0].parent/'metrics.json').read_text(encoding='utf-8'))
            old_threshold=previous_metrics['balanced']['threshold_selection']['threshold']
            metrics['v5_tabular_on_same_matched_development']=score_report(
                common.y_amendment_proxy,common.balanced_score,old_threshold)
            metrics['v5_tabular_threshold_note']='The prior V5 calibration threshold is applied unchanged; this is a selected case-control text cohort, not the full population.'
    out=root/'colab_outputs'/f'{experiment_name}_{hashlib.sha256((dataset_hash+text_hash+trainer_hash).encode()).hexdigest()[:12]}'
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'metrics.json',metrics)
    prediction=development[['adsh','cik','form','filed_date','y_amendment_proxy']].copy()
    prediction['text_ensemble_score']=development_scores
    prediction['text_ensemble_prediction']=(development_scores>=threshold['threshold']).astype('int8')
    prediction['form_calibrated_threshold']=development_form_thresholds
    prediction['form_calibrated_prediction']=(development_scores>=development_form_thresholds).astype('int8')
    prediction.to_csv(out/'development_predictions_case_control.csv.gz',index=False,compression='gzip')
    if issuer_development is not None and 'n' in issuer_disjoint_metrics:
        issuer_prediction=issuer_development[['adsh','cik','form','filed_date','y_amendment_proxy']].copy()
        issuer_prediction['text_ensemble_score']=issuer_development_scores
        issuer_prediction['text_ensemble_prediction']=(issuer_development_scores>=threshold['threshold']).astype('int8')
        issuer_form_thresholds=thresholds_for_frame(
            issuer_development,form_threshold_selection,threshold['threshold'])
        issuer_prediction['form_calibrated_threshold']=issuer_form_thresholds
        issuer_prediction['form_calibrated_prediction']=(
            issuer_development_scores>=issuer_form_thresholds).astype('int8')
        issuer_prediction.to_csv(out/'issuer_disjoint_predictions_case_control.csv.gz',index=False,compression='gzip')
    pd.DataFrame(leaderboard).to_csv(out/'cv_leaderboard.csv',index=False)
    print(json.dumps({'calibration':metrics['calibration'],'development':metrics['development'],
                      'same_cohort_v5':metrics.get('v5_tabular_on_same_matched_development'),
                      'output':str(out)},indent=2),flush=True)
    return out
