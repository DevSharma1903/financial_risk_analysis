"""Offline checks of the study boundaries. No model fitting or SEC requests."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
import research_v5_core as core
import sec_v5_evidence as evidence
spec=importlib.util.spec_from_file_location('training_v5',SRC/'14_train_v5_colab.py')
training=importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def filings(dates, labels=None):
    n=len(dates)
    frame=pd.DataFrame({'adsh':[f'0000000001-20-{i:06d}' for i in range(n)],
                        'cik':['1']*n,'form':['10-Q']*n,'fp':['Q1']*n,'sic2':['10']*n,
                        'filed_date':pd.to_datetime(dates),core.TARGET:labels or [0]*n,
                        'net_income_to_assets':np.arange(n,dtype=float),
                        'operating_cashflow_to_assets':np.arange(n,dtype=float),
                        'xbrl_StockholdersEquity':np.arange(n,dtype=float)})
    frame['filed_year']=frame.filed_date.dt.year
    return frame


class StudyBoundaries(unittest.TestCase):
    def test_mature_labels_are_strictly_historical(self):
        df=filings(['2020-01-01','2020-12-31','2021-01-01','2022-01-02'],[1,1,0,1])
        got=core.add_mature_history(df)
        self.assertEqual(got.issuer_mature_positive_count.tolist(),[0,0,1,2])
        changed=df.copy(); changed.loc[3,core.TARGET]=0
        pd.testing.assert_frame_equal(got.filter(regex='mature'),
                                      core.add_mature_history(changed).filter(regex='mature'))
        self.assertEqual(got.issuer_form_mature_count.tolist(),[0,0,1,3])

    def test_same_day_and_future_events_are_excluded(self):
        df=filings(['2020-01-01','2020-01-02'])
        events=pd.DataFrame({'cik':['1']*4,'form':['8-K']*4,
                              'filed_date':['2019-12-31','2020-01-01','2020-01-02','2030-01-01']})
        got=core.add_event_history(df,events)
        self.assertEqual(got.sec_8k_count_90d.tolist(),[1,2])
        self.assertEqual(got.sec_8k_days_since.tolist(),[1,1])
        self.assertTrue(got.sec_late_days_since.isna().all())

    def test_financial_history_has_no_same_day_or_future_rows(self):
        df=filings(['2020-01-01','2020-01-01','2020-04-01','2020-07-01'])
        df['net_margin']=[1,3,9,999]
        got=core.add_financial_history(df)
        self.assertTrue(got.past3_mean_net_margin.iloc[:2].isna().all())
        self.assertEqual(got.past3_mean_net_margin.iloc[2],2)
        changed=df.copy(); changed.loc[3,'net_margin']=-999
        np.testing.assert_allclose(got.past3_mean_net_margin,
                                   core.add_financial_history(changed).past3_mean_net_margin,
                                   equal_nan=True)

    def test_peer_values_come_from_prior_year(self):
        df=filings(['2019-06-01']*20+['2020-01-01','2020-12-01'])
        df['cik']=[str(i) for i in range(len(df))]
        df['net_margin']=[2.0]*20+[100,200]
        got=core.add_financial_history(df)
        self.assertTrue(got.prior_year_peer_median_net_margin.iloc[:20].isna().all())
        self.assertEqual(got.prior_year_peer_median_net_margin.iloc[20:].tolist(),[2,2])

    def test_vectorized_history_matches_independent_issuer_calculation(self):
        df=filings(pd.date_range('2020-01-01',periods=40,freq='7D'))
        df['cik']=[str(i%3) for i in range(len(df))]
        df['net_margin']=np.arange(len(df),dtype=float)
        df.loc[[3,8,20],'net_margin']=np.nan
        df=df.sample(frac=1,random_state=7).reset_index(drop=True)
        got=core.add_financial_history(df)
        for _,part in df.groupby('cik'):
            ordered=part.sort_values('filed_date')
            expected=ordered.net_margin.shift().rolling(3,min_periods=1)
            actual=got.set_index('adsh').loc[ordered.adsh]
            np.testing.assert_allclose(actual.past3_mean_net_margin,expected.mean(),equal_nan=True)
            np.testing.assert_allclose(actual.past3_std_net_margin,expected.std(),equal_nan=True)

    def test_balancing_matches_year_and_form_without_replacement(self):
        df=filings(['2020-01-01']*20+['2021-01-01']*20,[1,1]+[0]*18+[1]+[0]*19)
        sample=core.balanced(df,42)
        self.assertTrue(sample.adsh.is_unique)
        self.assertEqual(sample[core.TARGET].sum(),3)
        counts=sample.groupby(['filed_year','form',core.TARGET]).size().unstack()
        np.testing.assert_array_equal(counts[0],counts[1])

    def test_threshold_matches_exhaustive_search_including_ties(self):
        rng=np.random.default_rng(123)
        for _ in range(30):
            y=np.r_[0,1,rng.integers(0,2,98)]
            p=np.round(rng.random(100),1)
            metrics=[training.metric_report(y,p,t) for t in np.unique(p)]
            best=max(min(m[k] for k in ('accuracy','precision','recall','f1')) for m in metrics)
            chosen=core.threshold_choice(y,p)
            self.assertAlmostEqual(chosen['calibration_min_metric'],best)
            self.assertEqual(chosen['calibration_all_metrics_ge_85'],best>=.85)

    def test_outcome_metadata_never_enters_features(self):
        df=filings(['2020-01-01','2021-01-01'],[0,1])
        df['days_to_amendment']=[-1,2]; df['text_emb_0']=[1,2]
        df['issuer_mature_rate']=[.02,.02]
        numeric,categorical=training.feature_columns(df,list(df),{})
        self.assertFalse(set(numeric+categorical)&core.EXCLUDED)
        self.assertNotIn('text_emb_0',numeric)
        self.assertIn('issuer_mature_rate',numeric)

    def test_category_vocabulary_comes_only_from_training(self):
        train=filings(['2020-01-01'],[1]); future=filings(['2021-01-01'],[0])
        future['sic2']='99'
        x,p=training.matrices(train,future,['net_income_to_assets'],['sic2'])
        self.assertNotIn('sic2_99',p)
        self.assertEqual(p.sic2_10.iloc[0],0)
        self.assertEqual(x.columns.tolist(),p.columns.tolist())


class PublicEvidenceParsing(unittest.TestCase):
    def test_master_index_uses_only_valid_filing_paths(self):
        raw=(b'CIK|Company Name|Form Type|Date Filed|Filename\n'
             b'1|Example|10-Q|2020-01-02|edgar/data/1/0000000001-20-000001.txt\n'
             b'1|Example|8-K|2020-01-03|edgar/data/1/0000000001-20-000002.txt\n'
             b'1|Example|10-K/A|2020-01-04|edgar/data/99/0000000001-20-000003.txt\n')
        got=evidence.parse_master(raw)
        self.assertEqual(got.form.tolist(),['10-Q','8-K'])

    def test_original_text_excludes_amendments_and_exhibits(self):
        blob=(b'<SEC-HEADER>future-looking metadata</SEC-HEADER>'
              b'<DOCUMENT><TYPE>EX-99\n<TEXT>bad exhibit</TEXT></DOCUMENT>'
              b'<DOCUMENT><TYPE>10-K/A\n<TEXT>bad amendment</TEXT></DOCUMENT>'
              b'<DOCUMENT><TYPE>10-K\n<TEXT>original only</TEXT></DOCUMENT>')
        self.assertEqual(evidence.original_document(blob,'10-K'),'original only')
        self.assertIsNone(evidence.original_document(blob,'10-Q'))
        with self.assertRaises(ValueError):
            evidence.original_document(blob,'10-K/A')

    def test_access_denials_stop_and_cached_text_is_resumed(self):
        from urllib.error import HTTPError
        from unittest.mock import patch
        client=evidence.SecClient('Academic Research contact@example.org')
        with patch.object(evidence,'urlopen',side_effect=HTTPError('https://www.sec.gov/',403,'denied',{},None)) as opening:
            with self.assertRaisesRegex(RuntimeError,'403'):
                client.get('https://www.sec.gov/Archives/example.txt')
            self.assertEqual(opening.call_count,1)
        class FixtureClient:
            calls=0
            def get(self,*args,**kwargs):
                self.calls+=1
                return ('<DOCUMENT><TYPE>10-Q\n<TEXT><p>'+('material weakness disclosure '*100)+
                        '</p></TEXT></DOCUMENT>').encode()
        fixture=FixtureClient()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'text.sqlite'
            manifest=filings(['2020-01-01'])[['adsh','cik','form','filed_date']]
            evidence.collect_text(manifest,path,fixture)
            report=evidence.collect_text(manifest,path,fixture)
            self.assertEqual(fixture.calls,1)
            self.assertEqual(report['usable'],1)


if __name__=='__main__':
    unittest.main()
