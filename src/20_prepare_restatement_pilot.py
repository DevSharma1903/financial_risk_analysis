"""Create a reproducible restatement review queue without assigning labels.

The pilot contains amendment candidates and matched no-amendment controls.
Every outcome remains pending until a person reviews primary SEC evidence.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def parse_filing_dates(values: pd.Series) -> pd.Series:
    """Parse SEC YYYYMMDD date keys and ISO dates without ns misinterpretation."""
    strings = values.astype(str).str.strip()
    compact = strings.str.fullmatch(r'\d{8}')
    result = pd.Series(pd.NaT, index=values.index, dtype='datetime64[ns]')
    result.loc[compact] = pd.to_datetime(strings.loc[compact], format='%Y%m%d', errors='raise')
    result.loc[~compact] = pd.to_datetime(strings.loc[~compact], errors='raise')
    return result


def proportional_sample(frame: pd.DataFrame, n: int, strata: list[str], seed: int) -> pd.DataFrame:
    """Sample up to n rows proportionally across strata, without replacement."""
    if n <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    n = min(n, len(frame))
    groups = list(frame.groupby(strata, dropna=False, sort=True, observed=True))
    sizes = np.asarray([len(group) for _, group in groups], dtype=float)
    ideal = n * sizes / sizes.sum()
    takes = np.minimum(np.floor(ideal).astype(int), sizes.astype(int))
    remaining = n - int(takes.sum())
    order = np.argsort(-(ideal - np.floor(ideal)), kind='stable')
    for idx in order:
        if remaining == 0:
            break
        if takes[idx] < sizes[idx]:
            takes[idx] += 1
            remaining -= 1
    pieces = [group.sample(int(take), random_state=seed)
              for (_, group), take in zip(groups, takes) if take]
    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def build_pilot(candidates: pd.DataFrame, proxy: pd.DataFrame, n_each: int = 1000,
                seed: int = 20261019, horizon_years: int | None = None) -> pd.DataFrame:
    required_candidates = {'adsh_original', 'adsh_amendment', 'form_original',
                           'filed_original', 'filed_amendment', 'days_after_original',
                           'amendment_url'}
    required_proxy = {'adsh', 'cik', 'form', 'filed_date', 'y_amendment_proxy'}
    if required_candidates - set(candidates) or required_proxy - set(proxy):
        raise ValueError('Candidate or proxy CSV lacks required columns.')
    if candidates.adsh_original.duplicated().any() or proxy.adsh.duplicated().any():
        raise ValueError('Original accession IDs must be unique in both inputs.')

    candidates = candidates.copy()
    if 'cik' in candidates:
        candidates['candidate_cik'] = candidates.cik.astype(str).str.lstrip('0').replace('', '0')
        candidates = candidates.drop(columns='cik')
    candidates['filed_original'] = parse_filing_dates(candidates.filed_original)
    candidates['filed_amendment'] = parse_filing_dates(candidates.filed_amendment)
    candidates['filed_year'] = candidates.filed_original.dt.year
    candidates['days_bin'] = pd.cut(candidates.days_after_original.astype(int),
                                    bins=[0, 30, 90, 180, 365, 1095, 1830],
                                    labels=['1-30', '31-90', '91-180', '181-365', '366-1095', '1096-1830'])
    proxy_fields = ['adsh', 'cik', 'form', 'filed_date', 'y_amendment_proxy'] + (['sic2'] if 'sic2' in proxy else [])
    candidates = candidates.merge(proxy[proxy_fields].rename(columns={
                                      'adsh':'adsh_original', 'cik':'proxy_cik',
                                      'form':'proxy_form', 'filed_date':'proxy_filed_date'}),
                                  on='adsh_original', how='inner', validate='one_to_one')
    if len(candidates) == 0:
        raise ValueError('No candidate originals matched the current prepared dataset.')
    within_proxy_window = candidates.days_after_original.astype(int).le(365)
    if not pd.to_numeric(candidates.loc[within_proxy_window, 'y_amendment_proxy'], errors='coerce').eq(1).all():
        raise ValueError('A candidate within 365 days does not align with the amendment-proxy index; check accession/date joins.')
    if 'candidate_cik' in candidates:
        if not candidates.candidate_cik.eq(
                candidates.proxy_cik.astype(str).str.lstrip('0').replace('', '0')).all():
            raise ValueError('Candidate issuer IDs do not match the prepared filings.')
    if not candidates.form_original.astype(str).eq(candidates.proxy_form.astype(str)).all():
        raise ValueError('Candidate original form does not match the prepared filing.')
    if not candidates.filed_original.eq(parse_filing_dates(candidates.proxy_filed_date)).all():
        raise ValueError('Candidate original date does not match the prepared filing.')
    mature_cutoff = None
    if horizon_years is not None:
        if horizon_years not in (1, 3, 5):
            raise ValueError('horizon_years must be 1, 3, or 5')
        if 'label_observation_cutoff' not in proxy:
            raise ValueError('The proxy table needs label_observation_cutoff to build a mature horizon cohort.')
        cutoffs = pd.to_datetime(proxy.label_observation_cutoff, errors='coerce').dropna().unique()
        if len(cutoffs) != 1:
            raise ValueError('Expected one common label_observation_cutoff in the proxy table.')
        mature_cutoff = pd.Timestamp(cutoffs[0]) - pd.DateOffset(years=horizon_years)
        candidates = candidates.loc[candidates.filed_original.le(mature_cutoff)].copy()
        if candidates.empty:
            raise ValueError(f'No amendment candidates are mature for a {horizon_years}-year review horizon.')
    cases = proportional_sample(candidates, n_each, ['filed_year', 'form_original', 'days_bin'], seed)

    controls = proxy.loc[
        pd.to_numeric(proxy.y_amendment_proxy, errors='coerce').eq(0) &
        ~proxy.adsh.astype(str).isin(candidates.adsh_original.astype(str))
    ].copy()
    controls['filed_date'] = parse_filing_dates(controls.filed_date)
    if mature_cutoff is not None:
        controls = controls.loc[controls.filed_date.le(mature_cutoff)].copy()
    controls['filed_year'] = controls.filed_date.dt.year
    # Match control allocation to the sampled cases by year and form. These
    # controls are not presumed free of a confirmed restatement; they require review.
    if 'sic2' not in cases:
        cases['sic2'] = 'unknown'
    if 'sic2' not in controls:
        controls['sic2'] = 'unknown'
    cases['sic2'] = cases.sic2.fillna('unknown').astype(str)
    controls['sic2'] = controls.sic2.fillna('unknown').astype(str)
    case_strata = cases.groupby(['filed_year', 'form_original', 'sic2'], dropna=False).size()
    control_strata = {
        key: group for key, group in controls.groupby(['filed_year', 'form', 'sic2'],
                                                     dropna=False, sort=False)
    }
    selected_controls = []
    for (year, form, industry), count in case_strata.items():
        pool = control_strata.get((year, form, industry), controls.iloc[0:0])
        if not pool.empty:
            selected_controls.append(pool.sample(min(int(count), len(pool)),
                                                 random_state=seed + int(year) + int(industry[:2])
                                                 if industry[:2].isdigit() else seed + int(year)))
    control_frame = (pd.concat(selected_controls, ignore_index=True) if selected_controls
                     else controls.iloc[0:0].copy())
    if len(control_frame) < len(cases):
        remaining = controls.loc[~controls.adsh.isin(control_frame.adsh)]
        year_form_counts = cases.groupby(['filed_year','form_original']).size()
        remaining_by_year_form = {
            key: group for key, group in remaining.groupby(['filed_year', 'form'],
                                                           dropna=False, sort=False)
        }
        extra_pieces = []
        for (year, form), count in year_form_counts.items():
            already = int(((control_frame.filed_year == year) & (control_frame.form == form)).sum())
            need = max(0, int(count) - already)
            pool = remaining_by_year_form.get((year, form), remaining.iloc[0:0])
            if need and not pool.empty:
                piece = pool.sample(min(need, len(pool)), random_state=seed + int(year) + 17)
                extra_pieces.append(piece)
                remaining = remaining.loc[~remaining.adsh.isin(piece.adsh)]
        if extra_pieces:
            selected_controls.extend(extra_pieces)
            control_frame = pd.concat(selected_controls, ignore_index=True)
        if len(control_frame) < len(cases):
            remaining = controls.loc[~controls.adsh.isin(control_frame.adsh)]
            extra = proportional_sample(remaining, len(cases)-len(control_frame), ['filed_year','form'], seed+1)
            control_frame = pd.concat([control_frame, extra], ignore_index=True)
    control_frame = control_frame.sample(min(len(cases), len(control_frame)),
                                         random_state=seed+2).reset_index(drop=True)

    case_rows = pd.DataFrame({
        'original_adsh': cases.adsh_original.astype(str),
        'cik': cases.proxy_cik.astype(str), 'original_form': cases.form_original.astype(str),
        'original_filed_date': cases.filed_original.dt.strftime('%Y-%m-%d'),
        'sampling_group': 'same_period_amendment_candidate',
        'candidate_event_adsh': cases.adsh_amendment.astype(str),
        'candidate_event_date': cases.filed_amendment.dt.strftime('%Y-%m-%d'),
        'candidate_event_url': cases.amendment_url.astype(str),
        'days_to_candidate_event': cases.days_after_original.astype(int),
    })
    control_rows = pd.DataFrame({
        'original_adsh': control_frame.adsh.astype(str), 'cik': control_frame.cik.astype(str),
        'original_form': control_frame.form.astype(str),
        'original_filed_date': control_frame.filed_date.dt.strftime('%Y-%m-%d'),
        'sampling_group': 'no_same_period_amendment_control',
        'candidate_event_adsh': '', 'candidate_event_date': '',
        'candidate_event_url': '', 'days_to_candidate_event': '',
    })
    out = pd.concat([case_rows, control_rows], ignore_index=True)
    if out.original_adsh.duplicated().any():
        raise ValueError('Pilot sampling produced duplicate originals.')
    # The adjudication fields intentionally remain empty. An absent amendment
    # candidate is not evidence for a confirmed-restatement negative.
    out['label'] = ''
    out['review_status'] = 'pending'
    out['event_type'] = ''
    out['event_date'] = ''
    out['evidence_url'] = ''
    out['evidence_excerpt'] = ''
    out['reviewer'] = ''
    out['review_notes'] = ''
    # The reviewer records how far the filing history was actually checked.
    # This is required to establish a mature negative for a chosen horizon.
    out['outcome_followup_through'] = ''
    out['review_queue_horizon_years'] = '' if horizon_years is None else int(horizon_years)
    return out.sort_values(['original_filed_date', 'sampling_group', 'original_adsh']).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--n-each', type=int, default=1000,
                        help='Maximum candidate and control filings; default 1,000 per group.')
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--horizon-years', type=int, choices=(1,3,5), default=3,
                        help='Keep filings mature through this horizon as of the prepared proxy cutoff (default: 3).')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    data = args.root / 'data'
    candidates_path = data / 'restatement_candidates.csv'
    proxy_path = data / 'colab_amendment_proxy.csv.gz'
    if not candidates_path.is_file() or not proxy_path.is_file():
        raise FileNotFoundError('Expected data/restatement_candidates.csv and data/colab_amendment_proxy.csv.gz')
    candidates = pd.read_csv(candidates_path, dtype={'adsh_original':str,'adsh_amendment':str})
    proxy = pd.read_csv(proxy_path, compression='gzip', usecols=lambda c: c in {
        'adsh','cik','form','filed_date','y_amendment_proxy','sic2','label_observation_cutoff'},
        dtype={'adsh':str,'cik':str,'form':str}, low_memory=False)
    pilot = build_pilot(candidates, proxy, args.n_each, args.seed, args.horizon_years)
    cutoff = pd.to_datetime(proxy.label_observation_cutoff, errors='raise').iloc[0]
    mature_cutoff = cutoff - pd.DateOffset(years=args.horizon_years)
    output = args.output or data / 'restatement_review_queue.csv'
    pilot.to_csv(output, index=False)
    print(f'Wrote {len(pilot):,} pending review rows to {output}')
    print(pilot.sampling_group.value_counts().to_string())
    print(f'Queue horizon: {args.horizon_years} years; eligible originals filed on or before {mature_cutoff:%Y-%m-%d}.')
    print('No labels were inferred. Review primary SEC evidence and record complete follow-up before assigning outcomes.')


if __name__ == '__main__':
    main()
