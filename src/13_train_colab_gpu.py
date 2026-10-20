"""Colab-only GPU experiment for the 365-day SEC amendment proxy.

Run from the companion notebook after uploading the schema-v4 CSV to Drive.
All model fits use CUDA. Thresholds and any ensemble decision are set before
the untouched 2024Q4-2025Q3 final cohort is scored.
"""

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


ROOT = Path('/content/drive/MyDrive/financial-risk-detector')
DATASET = ROOT / 'data' / 'colab_amendment_proxy.csv.gz'
OUT = ROOT / 'colab_outputs' / 'schema_v4_final'
TARGET = 'y_amendment_proxy'
SEED = 2026

if not torch.cuda.is_available():
    raise RuntimeError('A Colab CUDA GPU is required. Select Runtime > Change runtime type > GPU and restart.')
print('CUDA GPU:', torch.cuda.get_device_name(0))
if not DATASET.is_file():
    DATASET = ROOT / 'colab_amendment_proxy.csv.gz'
if not DATASET.is_file():
    raise FileNotFoundError(f'Upload schema-v4 data to {DATASET}')
df = pd.read_csv(DATASET, compression='gzip', low_memory=False)
needed = {TARGET, 'adsh', 'cik', 'form', 'filed_date', 'filed_year', 'feature_schema_version',
          'prior_amendment_count', 'prior_amendments_365d', 'label_observation_cutoff'}
if needed - set(df):
    raise ValueError(f'Missing required columns: {sorted(needed-set(df))}')
if not pd.to_numeric(df.feature_schema_version, errors='coerce').eq(4).all():
    raise ValueError('Dataset schema must be 4. Re-run src/09_prepare_colab_dataset.py after refreshing SEC quarters.')
df['filed_date'] = pd.to_datetime(df.filed_date, errors='raise')
if df.adsh.duplicated().any() or df[TARGET].isna().any() or not set(df[TARGET].unique()) <= {0, 1}:
    raise ValueError('Accessions must be unique and labels must be binary without missing values.')
cutoffs = pd.to_datetime(df.label_observation_cutoff, errors='raise')
if cutoffs.nunique() != 1 or cutoffs.iloc[0] < pd.Timestamp('2026-09-30'):
    raise ValueError('Final cohort is not mature. Add 2025Q4-2026Q3 SEC quarters, rerun preparation, and reupload.')

categorical = [c for c in ('form', 'fp', 'sic2') if c in df]
excluded = {'adsh', 'cik', 'period', 'period_date', 'filed', 'filed_date', 'filed_year',
            'sic', 'afs', 'wksi', 'days_to_amendment', 'feature_schema_version',
            'label_observation_cutoff', TARGET,
            *categorical}
numeric = [c for c in df if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
assert numeric and 'prior_amendment_count' in numeric

def select_dates(start=None, end=None):
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df.filed_date >= pd.Timestamp(start)
    if end is not None:
        mask &= df.filed_date <= pd.Timestamp(end)
    return df.loc[mask].copy()


def balanced(part, seed):
    positives = part[part[TARGET].eq(1)]
    negatives = part[part[TARGET].eq(0)]
    if positives.empty:
        raise ValueError('A cohort has no amendment-positive original filings.')
    chosen = []
    for (year, form), group in positives.groupby(['filed_year', 'form']):
        pool = negatives[negatives.filed_year.eq(year) & negatives.form.eq(form)]
        if len(pool) < len(group):
            raise ValueError(f'Insufficient {year} {form} controls')
        chosen.extend(pool.sample(len(group), random_state=seed, replace=False).index)
    result = pd.concat([positives, negatives.loc[chosen]]).sample(frac=1, random_state=seed)
    if not result.adsh.is_unique or len(result) != 2 * len(positives):
        raise AssertionError('Control sampling failed.')
    return result


fold_specs = [(2016, 2018), (2018, 2020), (2020, 2022)]
folds = []
for fit_end, valid_year in fold_specs:
    fit = balanced(select_dates(end=f'{fit_end}-12-31'), SEED+fit_end)
    valid = balanced(select_dates(f'{valid_year}-01-01', f'{valid_year}-12-31'), SEED+valid_year)
    folds.append((fit, valid))
train_natural = select_dates(end='2021-12-31')
cal_natural = select_dates('2023-01-01', '2023-09-30')
holdout_natural = select_dates('2024-10-01', '2025-09-30')
if min(map(len, (train_natural, cal_natural, holdout_natural))) == 0:
    raise ValueError('Missing train, calibration, or final holdout rows.')
train = balanced(train_natural, 42)
cal = balanced(cal_natural, 43)
holdout = balanced(holdout_natural, 44)
for name, part in [('train',train), ('calibration',cal), ('balanced_holdout',holdout),
                   ('natural_holdout',holdout_natural)]:
    print(f'{name}: {len(part):,} rows; {part[TARGET].mean():.3%} positives')


def cat_frame(part):
    x = part[numeric + categorical].copy()
    x[numeric] = x[numeric].replace([np.inf, -np.inf], np.nan)
    for col in categorical:
        x[col] = x[col].fillna('unknown').astype(str)
    return x


def xgb_frames(fit_part, *parts):
    fit_num = fit_part[numeric].replace([np.inf, -np.inf], np.nan)
    fit_cat = pd.get_dummies(fit_part[categorical].fillna('unknown').astype(str), dtype='uint8')
    def build(part):
        num = part[numeric].replace([np.inf, -np.inf], np.nan)
        cat = pd.get_dummies(part[categorical].fillna('unknown').astype(str), dtype='uint8')
        cat = cat.reindex(columns=fit_cat.columns, fill_value=0)
        missing = num.isna().astype('uint8').add_prefix('missing__')
        return pd.concat([num, cat, missing], axis=1)
    train_x = build(fit_part)
    return [train_x] + [build(part).reindex(columns=train_x.columns, fill_value=0) for part in parts]


def make_model(config, trees, seed, early_stop=False):
    if config['family'] == 'xgboost':
        return XGBClassifier(
            n_estimators=int(trees), max_depth=int(config['depth']), learning_rate=0.025,
            min_child_weight=5, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=8.0, reg_alpha=0.2, objective='binary:logistic',
            eval_metric='aucpr', early_stopping_rounds=100 if early_stop else None,
            tree_method='hist', device='cuda', random_state=seed, n_jobs=4)
    return CatBoostClassifier(
        iterations=int(trees), depth=int(config['depth']), learning_rate=0.03,
        l2_leaf_reg=8.0, random_seed=seed, loss_function='Logloss',
        eval_metric='Logloss', task_type='GPU', devices='0',
        allow_writing_files=False, verbose=False)


def fit_cv(config, fit, valid):
    model = make_model(config, 1800, SEED, early_stop=True)
    if config['family'] == 'xgboost':
        x_fit, x_valid = xgb_frames(fit, valid)
        model.fit(x_fit, fit[TARGET], eval_set=[(x_valid, valid[TARGET])], verbose=False)
        p = model.predict_proba(x_valid)[:, 1]
        count = int(model.best_iteration) + 1
    else:
        x_fit, x_valid = cat_frame(fit), cat_frame(valid)
        cat_indices = [x_fit.columns.get_loc(c) for c in categorical]
        model.fit(x_fit, fit[TARGET], cat_features=cat_indices,
                  eval_set=(x_valid, valid[TARGET]), early_stopping_rounds=100, verbose=False)
        p = model.predict_proba(x_valid)[:, 1]
        count = int(model.get_best_iteration()) + 1
    return float(average_precision_score(valid[TARGET], p)), count


candidates = [{'family':fam, 'depth':depth} for fam, depths in
              [('xgboost', [2, 3, 4, 5, 6]), ('catboost', [4, 5, 6, 7, 8])]
              for depth in depths]
leaderboard = []
for config in candidates:
    started = time.time()
    scores, counts = zip(*(fit_cv(config, fit, valid) for fit, valid in folds))
    row = {**config, 'fold_pr_auc':list(scores), 'fold_tree_counts':list(counts),
           'mean_pr_auc':float(np.mean(scores)), 'median_trees':int(np.median(counts)),
           'seconds':round(time.time()-started, 1)}
    leaderboard.append(row)
    print(row, flush=True)
leaderboard.sort(key=lambda row: row['mean_pr_auc'], reverse=True)
selected = leaderboard[0]
tree_count = selected['median_trees']
if tree_count < 25:
    raise RuntimeError(f'CV selected only {tree_count} trees. Review the folds/data; refusing a degenerate final fit.')
print('Selected before calibration and holdout:', selected)


def fit_fixed(config, fit, trees):
    model = make_model(config, trees, SEED, early_stop=False)
    if config['family'] == 'xgboost':
        x = xgb_frames(fit)[0]
        model.fit(x, fit[TARGET], verbose=False)
        return model, x.columns.tolist()
    x = cat_frame(fit)
    model.fit(x, fit[TARGET], cat_features=[x.columns.get_loc(c) for c in categorical], verbose=False)
    return model, x.columns.tolist()


model, model_columns = fit_fixed(selected, train, tree_count)
if selected['family'] == 'xgboost' and model.get_booster().num_boosted_rounds() != tree_count:
    raise AssertionError('Saved model tree count differs from CV-selected tree count.')


def probabilities(model, config, fit_reference, *parts):
    if config['family'] == 'xgboost':
        frames = xgb_frames(fit_reference, *parts)[1:]
        return [model.predict_proba(frame)[:,1] for frame in frames]
    return [model.predict_proba(cat_frame(part))[:,1] for part in parts]


p_cal, p_cal_natural, p_holdout, p_holdout_natural = probabilities(
    model, selected, train, cal, cal_natural, holdout, holdout_natural)


def best_f1_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    if not len(thresholds):
        raise ValueError('No usable calibration threshold.')
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


balanced_threshold = best_f1_threshold(cal[TARGET], p_cal)
natural_threshold = best_f1_threshold(cal_natural[TARGET], p_cal_natural)


def evaluate(y, p, threshold):
    y = np.asarray(y, dtype=int)
    predictions = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0,1]).ravel()
    return {'n':len(y), 'prevalence':float(np.mean(y)), 'threshold':threshold,
            'accuracy':float(accuracy_score(y, predictions)),
            'precision':float(precision_score(y, predictions, zero_division=0)),
            'recall':float(recall_score(y, predictions, zero_division=0)),
            'f1':float(f1_score(y, predictions, zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y, predictions)),
            'mcc':float(matthews_corrcoef(y, predictions)),
            'pr_auc':float(average_precision_score(y, p)),
            'roc_auc':float(roc_auc_score(y, p)),
            'brier_raw_balanced_score':float(brier_score_loss(y, p)),
            'tn_fp_fn_tp':[int(tn),int(fp),int(fn),int(tp)]}


def subset_metrics(part, p, mask, threshold):
    if not np.any(mask) or part.loc[mask, TARGET].nunique() < 2:
        return {'n':int(np.sum(mask)), 'note':'subgroup has fewer than two classes'}
    return evaluate(part.loc[mask, TARGET], p[mask], threshold)


seen_issuers = holdout_natural.cik.astype(str).isin(set(train.cik.astype(str))).to_numpy()
report = {
    'target':'same-period amendment within 365 days; not confirmed restatement',
    'feature_schema_version':4,
    'sampling':'all positives; random year/form-matched 1:1 negatives within each cohort',
    'selection':'highest mean balanced PR-AUC on 2018, 2020, 2022 forward folds; fixed median CV tree count',
    'calibration':'2023Q1-Q3; F1 thresholds chosen separately for balanced and natural prevalence',
    'final_holdout':'originals filed 2024Q4-2025Q3 with 365-day outcome through 2026Q3',
    'holdout_status':'pristine only if this cohort has not previously been inspected for model choices',
    'selected':selected, 'trees_in_final_fit':tree_count, 'cv_leaderboard':leaderboard,
    'balanced_calibration':evaluate(cal[TARGET], p_cal, balanced_threshold),
    'natural_calibration':evaluate(cal_natural[TARGET], p_cal_natural, natural_threshold),
    'balanced_holdout':evaluate(holdout[TARGET], p_holdout, balanced_threshold),
    'natural_holdout_at_balanced_threshold':evaluate(holdout_natural[TARGET], p_holdout_natural, balanced_threshold),
    'natural_holdout_at_natural_threshold':evaluate(holdout_natural[TARGET], p_holdout_natural, natural_threshold),
    'natural_holdout_seen_issuers':subset_metrics(holdout_natural, p_holdout_natural,
                                                  seen_issuers, natural_threshold),
    'natural_holdout_unseen_issuers':subset_metrics(holdout_natural, p_holdout_natural,
                                                    ~seen_issuers, natural_threshold),
    'natural_holdout_10k':subset_metrics(holdout_natural, p_holdout_natural,
                                        holdout_natural.form.eq('10-K').to_numpy(), natural_threshold),
    'natural_holdout_10q':subset_metrics(holdout_natural, p_holdout_natural,
                                        holdout_natural.form.eq('10-Q').to_numpy(), natural_threshold),
    'balanced_always_negative_accuracy':float(1-holdout[TARGET].mean()),
    'natural_always_negative_accuracy':float(1-holdout_natural[TARGET].mean()),
}

OUT.mkdir(parents=True, exist_ok=True)
suffix = 'json' if selected['family'] == 'xgboost' else 'cbm'
model.save_model(str(OUT / f'holdout_evaluation_model.{suffix}'))
pd.DataFrame({'feature':model_columns, 'importance':model.feature_importances_}).sort_values(
    'importance', ascending=False).to_csv(OUT / 'feature_importance.csv', index=False)
with open(OUT / 'feature_schema.json', 'w') as handle:
    json.dump({'numeric':numeric, 'categorical':categorical, 'model_columns':model_columns,
               'family':selected['family'], 'fixed_trees':tree_count}, handle, indent=2)
with open(OUT / 'metrics.json', 'w') as handle:
    json.dump(report, handle, indent=2)
pd.DataFrame(leaderboard).to_csv(OUT / 'cv_leaderboard.csv', index=False)

# This later artifact uses every mature label. It has no independent test score.
all_data = balanced(df, 45)
all_model, all_columns = fit_fixed(selected, all_data, tree_count)
all_model.save_model(str(OUT / f'all_data_model.{suffix}'))
report['all_data_fit'] = {'rows':len(all_data), 'positives':int(all_data[TARGET].sum()),
                          'trees':tree_count, 'independent_evaluation':None}
with open(OUT / 'metrics.json', 'w') as handle:
    json.dump(report, handle, indent=2)
with open(OUT / 'all_data_feature_schema.json', 'w') as handle:
    json.dump({'model_columns':all_columns, 'family':selected['family']}, handle, indent=2)
print(json.dumps({k:report[k] for k in ('selected','trees_in_final_fit','balanced_holdout',
                                      'natural_holdout_at_natural_threshold')}, indent=2))
print('Saved GPU results to', OUT)
