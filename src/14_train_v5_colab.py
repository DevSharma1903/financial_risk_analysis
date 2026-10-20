"""GPU-only research comparison. Previously examined cohorts are development data.

No execution on import. The generated Colab notebook calls run().
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import time
import numpy as np
import pandas as pd
import research_v5_core as core
import sec_v5_evidence as evidence


def atomic_json(path, obj):
    temp=Path(str(path)+'.part')
    temp.write_text(json.dumps(obj,indent=2),encoding='utf-8'); temp.replace(path)


def metric_report(y,p,threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float); pred=p>=threshold
    tp=int(np.sum(pred & (y==1))); fp=int(np.sum(pred & (y==0)))
    fn=int(np.sum(~pred & (y==1))); tn=int(np.sum(~pred & (y==0)))
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
    f1=2*tp/max(2*tp+fp+fn,1); accuracy=(tp+tn)/len(y)
    return {'n':len(y),'prevalence':float(y.mean()),'threshold':float(threshold),
            'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1,
            'balanced_accuracy':(recall+tn/max(tn+fp,1))/2,
            'pr_auc':float(average_precision_score(y,p)),
            'roc_auc':float(roc_auc_score(y,p)),
            'mcc':float(matthews_corrcoef(y,pred)), 'tn_fp_fn_tp':[tn,fp,fn,tp],
            'all_four_metrics_ge_80':bool(min(accuracy,precision,recall,f1)>=.80),
            'all_four_metrics_ge_85':bool(min(accuracy,precision,recall,f1)>=.85),
            'always_negative_accuracy':float(1-y.mean())}


def cluster_intervals(part, p, threshold, draws=400):
    y=part[core.TARGET].to_numpy(); pred=p>=threshold
    rows=pd.DataFrame({'cik':part.cik.to_numpy(),'tp':pred & (y==1),
                       'fp':pred & (y==0),'tn':~pred & (y==0),'fn':~pred & (y==1)})
    groups=rows.groupby('cik')[['tp','fp','tn','fn']].sum().to_numpy()
    rng=np.random.default_rng(2026); values=[]
    for _ in range(draws):
        tp,fp,tn,fn=groups[rng.integers(0,len(groups),len(groups))].sum(axis=0)
        values.append([tp/max(tp+fp,1),tp/max(tp+fn,1),2*tp/max(2*tp+fp+fn,1),
                       (tp+tn)/max(tp+fp+tn+fn,1)])
    limits=np.quantile(values,[.025,.975],axis=0)
    return {name:[float(limits[0,i]),float(limits[1,i])] for i,name in
            enumerate(['precision','recall','f1','accuracy'])}


def candidate_configs(use_text):
    configs=[
        dict(name='v4_features_reference',family='xgb',depth=4,bags=1,window=30,base=True,fixed=201),
        dict(name='history_bagged_xgb',family='xgb',depth=4,bags=3,window=8),
        dict(name='recent_bagged_xgb',family='xgb',depth=3,bags=3,window=4),
        dict(name='history_bagged_cat',family='cat',depth=5,bags=3,window=8),
        dict(name='form_specialists',family='xgb',depth=4,bags=3,window=8,specialists=True),
        dict(name='all_controls_weighted',family='xgb',depth=4,bags=1,window=8,all_controls=True),
        dict(name='neural_tabular',family='mlp',depth=0,bags=1,window=8),
    ]
    if use_text:
        configs += [dict(name='disclosure_xgb',family='xgb',depth=4,bags=3,window=8,text=True),
                    dict(name='disclosure_neural',family='mlp',depth=0,bags=1,window=8,text=True)]
    return configs


def feature_columns(df, base_columns, config):
    allowed=base_columns if config.get('base') else list(df.columns)
    cat=[c for c in ('form','fp','sic2') if c in allowed]
    num=[c for c in allowed if c not in core.EXCLUDED and c not in cat
         and pd.api.types.is_numeric_dtype(df[c])
         and (config.get('text',False) or not c.startswith('text_'))]
    if not num or set(num)&core.EXCLUDED:
        raise ValueError('Invalid model feature list.')
    return num,cat


def matrices(train, predict, numeric, categorical):
    # Category vocabularies and missingness columns are determined by training only.
    cat=pd.get_dummies(train[categorical].fillna('unknown').astype(str),dtype='float32')
    columns=cat.columns.tolist()
    def transform(part):
        numeric_part=part[numeric].replace([np.inf,-np.inf],np.nan).astype('float32')
        cats=pd.get_dummies(part[categorical].fillna('unknown').astype(str),dtype='float32').reindex(columns=columns,fill_value=0)
        return pd.concat([numeric_part,numeric_part.isna().astype('float32').add_prefix('missing__'),cats],axis=1)
    return transform(train), transform(predict)


def mlp_fit(train_x,y,predict_x,valid_y,folder,seed,iterations=None):
    import torch
    from sklearn.metrics import average_precision_score
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    a=train_x.to_numpy(dtype=np.float32); b=predict_x.to_numpy(dtype=np.float32)
    usable=~np.isnan(a).all(axis=0)
    median=np.zeros(a.shape[1],dtype=np.float32)
    median[usable]=np.nanmedian(a[:,usable],axis=0)
    a=np.where(np.isnan(a),median,a); b=np.where(np.isnan(b),median,b)
    mean=a.mean(axis=0); std=np.maximum(a.std(axis=0),1e-3)
    a=np.clip((a-mean)/std,-10,10); b=np.clip((b-mean)/std,-10,10)
    np.savez_compressed(folder/'normalization.npz',median=median,mean=mean,std=std)
    device=torch.device('cuda')
    model=torch.nn.Sequential(torch.nn.Linear(a.shape[1],256),torch.nn.LayerNorm(256),
          torch.nn.GELU(),torch.nn.Dropout(.3),torch.nn.Linear(256,128),
          torch.nn.GELU(),torch.nn.Dropout(.2),torch.nn.Linear(128,1)).to(device)
    xa=torch.tensor(a,device=device); ya=torch.tensor(y,dtype=torch.float32,device=device)
    xb=torch.tensor(b,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    best=-np.inf; chosen=0; stale=0
    def probabilities():
        model.eval()
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(model(xb[i:i+2048])).flatten().cpu().numpy()
                                   for i in range(0,len(xb),2048)])
    for epoch in range(iterations or 100):
        model.train(); order=torch.randperm(len(xa),device=device)
        for start in range(0,len(order),256):
            indices=order[start:start+256]
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(xa[indices]).flatten(),ya[indices])
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
        if iterations is None:
            score=average_precision_score(valid_y,probabilities())
            if score>best+1e-5:
                best=score; chosen=epoch+1; stale=0
                torch.save(model.state_dict(),folder/'weights.pt')
            else:
                stale+=1
            if stale>=12:
                break
    if iterations is None:
        model.load_state_dict(torch.load(folder/'weights.pt',map_location=device,weights_only=True))
    else:
        chosen=iterations; torch.save(model.state_dict(),folder/'weights.pt')
    p=probabilities()
    del model,xa,ya,xb
    torch.cuda.empty_cache()
    return p,chosen,'cuda'


def fit_member(config, train, predict, numeric, categorical, folder, seed, fixed=None):
    """A completed member is resumed only within a source/data fingerprinted run."""
    folder.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps([config,train.adsh.tolist(),predict.adsh.tolist(),
                                        numeric,categorical,seed,fixed],sort_keys=True).encode()).hexdigest()
    meta_path=folder/'complete.json'; probability_path=folder/'probabilities.npy'
    if meta_path.exists() and probability_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['signature']==signature:
            print(f'Resume completed member: {folder}',flush=True)
            return np.load(probability_path),meta['iterations'],meta
    print(f'GPU member {folder.name}: {len(train):,} training rows',flush=True)
    y=train[core.TARGET].to_numpy()
    valid_y=predict[core.TARGET].to_numpy() if fixed is None else None
    if len(np.unique(y))!=2:
        raise ValueError('Both training classes are required.')
    started=time.time()
    if config['family']=='cat':
        from catboost import CatBoostClassifier
        x=train[numeric+categorical].copy(); xp=predict[numeric+categorical].copy()
        x[numeric]=x[numeric].replace([np.inf,-np.inf],np.nan)
        xp[numeric]=xp[numeric].replace([np.inf,-np.inf],np.nan)
        for c in categorical:
            x[c]=x[c].fillna('unknown').astype(str); xp[c]=xp[c].fillna('unknown').astype(str)
        model=CatBoostClassifier(iterations=fixed or 1600,depth=config['depth'],learning_rate=.03,
               l2_leaf_reg=12,task_type='GPU',devices='0',loss_function='Logloss',
               random_seed=seed,allow_writing_files=False,verbose=False)
        opts={}
        if fixed is None:
            stop=core.balanced(predict,seed=707)
            opts={'eval_set':(xp.loc[stop.index],stop[core.TARGET]),'early_stopping_rounds':120}
        model.fit(x,y,cat_features=categorical,**opts)
        count=fixed or model.get_best_iteration()+1
        p=model.predict_proba(xp)[:,1]
        gpu=model.get_all_params().get('task_type')
        if gpu!='GPU':
            raise RuntimeError('CatBoost did not report GPU execution.')
        model.save_model(str(folder/'model.cbm')); columns=x.columns.tolist()
    else:
        x,xp=matrices(train,predict,numeric,categorical); columns=x.columns.tolist()
        if config['family']=='xgb':
            from xgboost import XGBClassifier
            positive_weight=float((y==0).sum()/max((y==1).sum(),1)) if config.get('all_controls') else 1.0
            model=XGBClassifier(n_estimators=fixed or 1600,max_depth=config['depth'],learning_rate=.025,
                min_child_weight=5,subsample=.85,colsample_bytree=.85,reg_lambda=8,reg_alpha=.2,
                scale_pos_weight=positive_weight,tree_method='hist',device='cuda',n_jobs=4,
                objective='binary:logistic',eval_metric='aucpr',
                early_stopping_rounds=120 if fixed is None else None,random_state=seed)
            # Balanced validation is used for early stopping; natural scores are reported separately.
            if fixed is None:
                stop=core.balanced(predict,seed=707)
                model.fit(x,y,eval_set=[(xp.loc[stop.index],stop[core.TARGET])],verbose=False)
                count=int(model.best_iteration)+1
            else:
                model.fit(x,y,verbose=False); count=int(fixed)
            gpu=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
            if not gpu.startswith('cuda'):
                raise RuntimeError(f'XGBoost fell back to {gpu}; this experiment requires GPU fitting.')
            p=model.predict_proba(xp)[:,1]; model.save_model(str(folder/'model.json'))
        else:
            p,count,gpu=mlp_fit(x,y,xp,valid_y,folder,seed,iterations=fixed)
    if not np.isfinite(p).all():
        raise ValueError('Model returned nonfinite probabilities.')
    meta={'signature':signature,'iterations':int(count),'gpu':gpu,'seconds':time.time()-started,
          'train_rows':len(train),'train_positives':int(y.sum()),'columns':columns,
          'numeric':numeric,'categorical':categorical,'config':config,'seed':seed}
    temp=folder/'probabilities.part.npy'; np.save(temp,p); temp.replace(probability_path)
    atomic_json(meta_path,meta)
    return p,int(count),meta


def fit_ensemble(config, df, fit_year, predict, base_columns, folder, fixed=None):
    pool=core.training_pool(df,fit_year,config['window'])
    if pool.filed_date.max()+pd.Timedelta(days=365)>=predict.filed_date.min():
        raise ValueError('Training outcome windows overlap the prediction cohort.')
    numeric,categorical=feature_columns(df,base_columns,config)
    predictions=[]; counts=[]; metadata=[]
    for bag,seed in enumerate(core.SEEDS[:config['bags']]):
        train=pool if config.get('all_controls') else core.balanced(pool,seed)
        member=np.full(len(predict),np.nan)
        groups=['10-K','10-Q'] if config.get('specialists') else ['both']
        for group in groups:
            tr=train if group=='both' else train[train.form.eq(group)]
            mask=np.ones(len(predict),dtype=bool) if group=='both' else predict.form.eq(group).to_numpy()
            sub=predict.loc[mask]
            p,count,meta=fit_member(config,tr,sub,numeric,categorical,folder/f'bag{bag}_{group}',seed,
                                     fixed=config.get('fixed',fixed))
            member[mask]=p; counts.append(count); metadata.append(meta)
        if not np.isfinite(member).all():
            raise ValueError('A specialist failed to score its cohort.')
        predictions.append(member)
    return np.mean(predictions,axis=0),counts,metadata


def select_ensemble(predictions,fold_parts,population):
    from sklearn.metrics import average_precision_score
    def mean_score(names):
        scores=[]
        for j,part in enumerate(fold_parts):
            selected=core.balanced(part,707).index if population=='balanced' else part.index
            mask=part.index.isin(selected)
            p=np.mean([predictions[name][j] for name in names],axis=0)
            scores.append(average_precision_score(part.loc[mask,core.TARGET],p[mask]))
        return float(np.mean(scores))
    ranked=sorted(predictions,key=lambda name:mean_score([name]),reverse=True)
    choices=[ranked[:1],ranked[:min(3,len(ranked))]]
    selected=max(choices,key=mean_score)
    return {'members':selected,'cv_pr_auc':mean_score(selected),
            'single_model_ranking':{name:mean_score([name]) for name in ranked}}


def run(root, use_text=False):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Select a Colab GPU runtime. CUDA is required before any model fit.')
    root=Path(root); path=root/'data/colab_amendment_proxy.csv.gz'
    if not path.exists():
        path=root/'colab_amendment_proxy.csv.gz'
    print('Confirmed CUDA GPU:',torch.cuda.get_device_name(0),flush=True)
    df=core.load_data(path); base_columns=df.columns.tolist()
    source_hashes=[core.file_hash(__file__),core.file_hash(core.__file__),core.file_hash(evidence.__file__)]
    index_paths=[root/'evidence_v5/indexes'/f'{year}Q{quarter}.parquet'
                 for year in range(2009,2026) for quarter in range(1,5)
                 if year<2025 or quarter<=3]
    missing=[p.name for p in index_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f'Run the evidence notebook first. Missing index quarters: {missing}')
    history_hashes=[core.file_hash(p) for p in index_paths]
    packages=('torch','xgboost','catboost','pandas','numpy','scikit-learn','pyarrow')
    if use_text:
        packages+=('sentence-transformers','transformers','huggingface-hub')
    versions={name:importlib.metadata.version(name) for name in packages}
    signatures={'dataset':core.file_hash(path),'code':source_hashes,'indexes':history_hashes,
                'text':use_text,'versions':versions,'gpu':torch.cuda.get_device_name(0)}
    print('Adding strictly historical outcomes, financial trends, peer comparisons and SEC events.',flush=True)
    df=core.add_financial_history(df)
    df=core.add_mature_history(df)
    history=evidence.load_sec_history(root/'evidence_v5/indexes')
    df=core.add_event_history(df,history)
    del history
    text_coverage=None
    if use_text:
        manifest=core.text_manifest(df)
        features=evidence.gpu_text_features(manifest,root/'evidence_v5/original_text.sqlite',root/'evidence_v5/embeddings')
        # Missing documents stay in every evaluation cohort with explicit availability.
        df=df.merge(features,on='adsh',how='left',validate='one_to_one')
        for c in features:
            if c!='adsh':
                df[c]=df[c].fillna(0)
        signatures['encoder']=core.file_hash(root/'evidence_v5/embeddings/encoder.json')
        signatures['text_features']=hashlib.sha256(pd.util.hash_pandas_object(features,index=False).values.tobytes()).hexdigest()
        text_coverage=df[df.adsh.isin(manifest.adsh)].groupby(['filed_year',core.TARGET]).text_available.agg(['count','mean']).reset_index().to_dict('records')
    digest=hashlib.sha256(json.dumps(signatures,sort_keys=True).encode()).hexdigest()[:16]
    out=root/'colab_outputs'/f'research_v5_{digest}'; out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'provenance.json',{'signatures':signatures,'versions':versions,
           'gpu':torch.cuda.get_device_name(0),'text_coverage':text_coverage,
           'evaluation_status':'2024Q4-2025Q3 already inspected; development comparison, not an untouched test',
           'labels':'Existing FSDS same-period amendment proxy unchanged; master indexes enrich predictors only',
           'history_limit':'SEC index snapshots may incorporate later corrections/removals; historical archive starts 2009',
           'sampling':'Each bag is 1:1 matched by year/form; all-controls arm balances total class weight',
           'created_utc':pd.Timestamp.now(tz='UTC').isoformat()})
    print('Resumable run directory:',out,flush=True)
    configs=candidate_configs(use_text); cv_predictions={}; leaderboard=[]; counts_by_name={}
    fold_parts=[core.date_part(df,f'{year}-01-01',f'{year}-12-31') for _,year in core.FOLDS]
    from sklearn.metrics import average_precision_score
    for config in configs:
        predictions=[]; counts=[]; balanced_scores=[]; natural_scores=[]
        for (fit_year,valid_year),part in zip(core.FOLDS,fold_parts):
            print(f'Fit {config["name"]}: through {fit_year}, validation {valid_year}',flush=True)
            p,iterations,_=fit_ensemble(config,df,fit_year,part,base_columns,
                                        out/'cv'/config['name']/str(valid_year))
            predictions.append(p); counts.extend(iterations)
            mask=part.index.isin(core.balanced(part,707).index)
            balanced_scores.append(float(average_precision_score(part.loc[mask,core.TARGET],p[mask])))
            natural_scores.append(float(average_precision_score(part[core.TARGET],p)))
        count=int(np.median(counts))
        eligible=config['family']=='mlp' or count>=25
        row={**config,'cv_balanced_pr_auc':float(np.mean(balanced_scores)),
             'cv_natural_pr_auc':float(np.mean(natural_scores)),'fold_balanced_pr_auc':balanced_scores,
             'fold_natural_pr_auc':natural_scores,'iterations':counts,'final_iterations':count,
             'eligible':eligible}
        leaderboard.append(row); print(row,flush=True)
        atomic_json(out/'cv_progress.json',leaderboard)
        if eligible:
            cv_predictions[config['name']]=predictions; counts_by_name[config['name']]=count
    if not cv_predictions:
        raise RuntimeError('No eligible models survived the CV checks.')
    choices={pop:select_ensemble(cv_predictions,fold_parts,pop) for pop in ('balanced','natural')}
    atomic_json(out/'selected_before_evaluation.json',choices)
    cal=core.date_part(df,'2023-01-01','2023-09-30')
    dev=core.date_part(df,'2024-10-01','2025-09-30')
    # All training labels mature before calibration; all calibration labels mature before development.
    if not (pd.Timestamp('2021-12-31')+pd.Timedelta(days=365)<cal.filed_date.min()
            and cal.filed_date.max()+pd.Timedelta(days=365)<dev.filed_date.min()):
        raise ValueError('Temporal embargo is insufficient for the 365-day label.')
    predict=pd.concat([cal,dev]); scores={}
    selected_names=set(sum([c['members'] for c in choices.values()],[]))
    for config in configs:
        if config['name'] not in selected_names:
            continue
        print('Fixed-iteration final fit:',config['name'],flush=True)
        p,_,_=fit_ensemble(config,df,2021,predict,base_columns,out/'evaluation_models'/config['name'],
                           fixed=counts_by_name[config['name']])
        scores[config['name']]=p
    results={'evaluation_status':'Previously inspected development cohort; no pristine final score claimed',
             'target':'FSDS same-period 365-day amendment proxy, not a confirmed restatement',
             'selection':choices,'cv_leaderboard':leaderboard,'text_coverage':text_coverage}
    predictions=dev[['adsh','cik','form','filed_date',core.TARGET]].copy()
    for pop in ('balanced','natural'):
        p=np.mean([scores[name] for name in choices[pop]['members']],axis=0)
        pc,pd_=p[:len(cal)],p[len(cal):]
        cal_mask=cal.index.isin(core.balanced(cal,43).index) if pop=='balanced' else np.ones(len(cal),bool)
        dev_mask=dev.index.isin(core.balanced(dev,44).index) if pop=='balanced' else np.ones(len(dev),bool)
        selection=core.threshold_choice(cal.loc[cal_mask,core.TARGET],pc[cal_mask])
        threshold=selection['threshold']
        results[pop]={'threshold_selection':selection,
                      'calibration':metric_report(cal.loc[cal_mask,core.TARGET],pc[cal_mask],threshold),
                      'development':metric_report(dev.loc[dev_mask,core.TARGET],pd_[dev_mask],threshold),
                      'issuer_bootstrap_95pct':cluster_intervals(dev.loc[dev_mask],pd_[dev_mask],threshold)}
        predictions[f'{pop}_score']=pd_; predictions[f'{pop}_prediction']=(pd_>=threshold).astype(int)
        predictions[f'in_{pop}_evaluation']=dev_mask
        # Applying the balanced model to all rows is reported explicitly as well.
        if pop=='balanced':
            results['balanced_model_on_natural_population']=metric_report(dev[core.TARGET],pd_,threshold)
    predictions.to_csv(out/'development_predictions.csv.gz',index=False,compression='gzip')
    for year,part in zip([y for _,y in core.FOLDS],fold_parts):
        j=[y for _,y in core.FOLDS].index(year)
        exported=part[['adsh','cik','form','filed_date',core.TARGET]].copy()
        for name in cv_predictions:
            exported[f'score_{name}']=cv_predictions[name][j]
        exported.to_csv(out/f'cv_predictions_{year}.csv.gz',index=False,compression='gzip')
    pd.DataFrame(leaderboard).to_csv(out/'cv_leaderboard.csv',index=False)
    atomic_json(out/'metrics.json',results)
    print(json.dumps({pop:results[pop] for pop in ('balanced','natural')},indent=2))
    print('Finished. Return metrics.json, cv_leaderboard.csv, provenance.json and prediction CSVs from',out)
    return out


if __name__=='__main__':
    run('/content/drive/MyDrive/financial-risk-detector')
