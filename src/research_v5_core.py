"""Point-in-time features and fixed study design; no model fitting or networking."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

TARGET = 'y_amendment_proxy'
SEEDS = (42, 143, 244)
FOLDS = ((2018, 2020), (2020, 2022))
RATIOS = ('accruals_to_assets', 'net_margin', 'gross_margin', 'debt_to_assets',
          'cash_to_assets', 'working_capital_to_assets', 'receivables_to_revenue',
          'inventory_to_revenue', 'operating_cashflow_to_assets', 'revenue_to_assets',
          'liabilities_to_assets', 'retained_earnings_to_assets', 'filing_delay_days')
EXCLUDED = {'adsh', 'cik', 'period', 'period_date', 'filed', 'filed_date', 'filed_year',
            'sic', 'afs', 'wksi', 'days_to_amendment', TARGET, 'feature_schema_version',
            'label_observation_cutoff'}


def file_hash(path):
    result = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(2**20), b''):
            result.update(chunk)
    return result.hexdigest()


def load_data(path):
    df = pd.read_csv(path, dtype={'adsh':str, 'cik':str}, low_memory=False)
    required = EXCLUDED | {'form', 'fp', 'sic2', 'prior_amendment_count'}
    if required - set(df):
        raise ValueError(f'Missing columns: {sorted(required-set(df))}')
    if not df.feature_schema_version.eq(4).all():
        raise ValueError('Use the prepared schema-v4 CSV, not an earlier export.')
    df['filed_date'] = pd.to_datetime(df.filed_date, errors='raise')
    df['period_date'] = pd.to_datetime(df.period_date, errors='raise')
    df['cik'] = df.cik.astype(str).str.lstrip('0').replace('', '0')
    if not df.adsh.is_unique or df[TARGET].isna().any() or not set(df[TARGET]) <= {0,1}:
        raise ValueError('Invalid labels or duplicate accessions.')
    cutoff = pd.to_datetime(df.label_observation_cutoff)
    if cutoff.nunique() != 1 or cutoff.iloc[0] < pd.Timestamp('2026-09-30'):
        raise ValueError('Require label observation through 2026-09-30.')
    if not ((cutoff-df.filed_date).dt.days >= 365).all():
        raise ValueError('An outcome has less than 365 days of follow-up.')
    if not set(df.form) <= {'10-K','10-Q'}:
        raise ValueError('Predictors must come from original 10-K/10-Q filings.')
    df['sic2'] = df.sic2.fillna('unknown').astype(str)
    return df.reset_index(drop=True)


def balanced(part, seed=42):
    pos, neg = part[part[TARGET].eq(1)], part[part[TARGET].eq(0)]
    if pos.empty:
        raise ValueError('Cohort has zero positives.')
    controls = []
    for (year, form), group in pos.groupby(['filed_year','form']):
        pool = neg[neg.filed_year.eq(year) & neg.form.eq(form)]
        if len(pool) < len(group):
            raise ValueError(f'Insufficient controls: {year}, {form}')
        controls.append(pool.sample(len(group), replace=False, random_state=seed))
    return pd.concat([pos, *controls]).sample(frac=1, random_state=seed)


def date_part(df, start, end):
    return df[df.filed_date.between(pd.Timestamp(start),pd.Timestamp(end))]


def training_pool(df, fit_year, window=8):
    return date_part(df, f'{max(2009,fit_year-window+1)}-01-01', f'{fit_year}-12-31')


def add_mature_history(df):
    """A prior label is usable only AFTER its complete 365-day outcome window."""
    result = pd.DataFrame(index=df.index)
    for group_cols, prefix in [(['cik'],'issuer'), (['cik','form'],'issuer_form')]:
        count = np.zeros(len(df)); positive = np.zeros(len(df))
        for _, group in df.groupby(group_cols, sort=False):
            history = group.sort_values('filed_date')
            maturity = (history.filed_date + pd.Timedelta(days=365)).to_numpy('datetime64[D]')
            at = np.searchsorted(maturity, group.filed_date.to_numpy('datetime64[D]'), side='left')
            prefix_sum = np.r_[0, history[TARGET].to_numpy().cumsum()]
            count[group.index] = at
            positive[group.index] = prefix_sum[at]
        result[f'{prefix}_mature_count'] = count
        result[f'{prefix}_mature_positive_count'] = positive
        # Fixed smoothing constant; never estimated from later outcome labels.
        result[f'{prefix}_mature_rate'] = (positive + 0.4) / (count + 20)
    return pd.concat([df, result], axis=1)


def add_financial_history(df):
    columns = [c for c in RATIOS if c in df]
    # One aggregate per issuer/form/day prevents same-day information leakage.
    keys = ['cik','form','filed_date']
    days = df.groupby(keys, as_index=False)[columns].mean().sort_values(keys)
    groups = days.groupby(['cik','form'], sort=False)
    # Compute all ratios together; per-issuer Python callbacks are very slow
    # across hundreds of thousands of filings on a standard Colab runtime.
    past = groups[columns].shift()
    rolling = past.groupby([days.cik,days.form],sort=False).rolling(3,min_periods=1)
    means = rolling.mean().reset_index(level=[0,1],drop=True)
    stds = rolling.std().reset_index(level=[0,1],drop=True)
    derived = []
    for col in columns:
        days[f'past3_mean_{col}'] = means[col]
        days[f'past3_std_{col}'] = stds[col]
        derived += [f'past3_mean_{col}', f'past3_std_{col}']
    out = df.merge(days[keys+derived], on=keys, how='left', validate='many_to_one')
    extra = {}
    for col in columns:
        extra[f'deviation_past3_{col}'] = out[col]-out[f'past3_mean_{col}']
    out = pd.concat([out, pd.DataFrame(extra,index=out.index)], axis=1)
    # Calendar-year aggregates come solely from the PREVIOUS filing year.
    for col in columns:
        stats = df.groupby(['filed_year','form','sic2'])[col].agg(['median','count']).reset_index()
        stats['filed_year'] += 1
        stats.loc[stats['count'] < 20, 'median'] = np.nan
        name = f'prior_year_peer_median_{col}'
        stats = stats.rename(columns={'median':name})
        out = out.merge(stats[['filed_year','form','sic2',name]],
                        on=['filed_year','form','sic2'], how='left', validate='many_to_one')
        out[f'peer_deviation_{col}'] = out[col]-out[name]
    out['loss_with_positive_cashflow'] = ((out.net_income_to_assets < 0) &
                                           (out.operating_cashflow_to_assets > 0)).astype('int8')
    out['profit_with_negative_cashflow'] = ((out.net_income_to_assets > 0) &
                                            (out.operating_cashflow_to_assets < 0)).astype('int8')
    out['negative_equity'] = out.xbrl_StockholdersEquity.lt(0).astype('int8')
    facts = [c for c in df if c.startswith('xbrl_')]
    out['financial_fact_missing_fraction'] = out[facts].isna().mean(axis=1)
    return out


def add_event_history(df, events):
    """Public EDGAR index events, all strictly before the predictor date."""
    out = pd.DataFrame(index=df.index)
    events = events.copy()
    events['filed_date'] = pd.to_datetime(events.filed_date)
    events['cik'] = events.cik.astype(str).str.lstrip('0').replace('','0')
    categories = {'8k':['8-K','8-K/A'], 'late':['NT 10-K','NT 10-Q'],
                  '10ka':['10-K/A'], '10qa':['10-Q/A']}
    for name, forms in categories.items():
        event_groups = {cik:np.sort(g.filed_date.to_numpy('datetime64[D]'))
                        for cik,g in events[events.form.isin(forms)].groupby('cik')}
        counts = {n:np.zeros(len(df), dtype='int32') for n in (90,365,730)}
        gap = np.full(len(df), np.nan)
        for cik, rows in df.groupby('cik',sort=False):
            dates = event_groups.get(cik, np.array([],dtype='datetime64[D]'))
            query = rows.filed_date.to_numpy('datetime64[D]')
            end = np.searchsorted(dates,query,side='left')
            for span in counts:
                counts[span][rows.index] = end-np.searchsorted(dates,query-np.timedelta64(span,'D'),side='left')
            valid = end > 0
            gap[rows.index[valid]] = (query[valid]-dates[end[valid]-1]).astype('timedelta64[D]').astype(int)
        for span,values in counts.items():
            out[f'sec_{name}_count_{span}d'] = values
        out[f'sec_{name}_days_since'] = gap
    return pd.concat([df,out],axis=1)


def threshold_choice(y, probability):
    """Maximise the worst of precision/recall/F1/accuracy on CALIBRATION only."""
    y = np.asarray(y, dtype=int); p = np.asarray(probability, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2 or not np.isfinite(p).all():
        raise ValueError('Threshold selection requires finite scores and both classes.')
    order = np.argsort(-p, kind='stable'); ys = y[order]; ps = p[order]
    ends = np.r_[np.flatnonzero(ps[:-1] != ps[1:]),len(ps)-1]
    tp = ys.cumsum()[ends]; fp = ends+1-tp
    recall = tp/y.sum(); precision = tp/(ends+1)
    f1 = 2*precision*recall/np.maximum(precision+recall,1e-12)
    accuracy = (tp+(1-y).sum()-fp)/len(y)
    floor = np.minimum.reduce([precision,recall,f1,accuracy])
    i = int(np.argmax(floor))
    return {'threshold':float(ps[ends[i]]), 'calibration_min_metric':float(floor[i]),
            'calibration_all_metrics_ge_80':bool(np.max(floor) >= .80),
            'calibration_all_metrics_ge_85':bool(np.max(floor) >= .85)}


def text_manifest(df):
    parts = []
    for fit_year in (2018,2020,2021):
        pool = training_pool(df,fit_year)
        parts += [balanced(pool,seed) for seed in SEEDS]
    for start,end in [('2020-01-01','2020-12-31'),('2022-01-01','2022-12-31'),
                      ('2023-01-01','2023-09-30'),('2024-10-01','2025-09-30')]:
        parts.append(date_part(df,start,end))
    return pd.concat(parts).drop_duplicates('adsh')[['adsh','cik','form','filed_date']].sort_values('adsh')
