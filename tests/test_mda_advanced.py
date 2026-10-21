"""Fast offline checks for full-section extraction, point-in-time text features,
and the human-review pilot. No SEC requests or model fitting are performed.
"""
import importlib.util
import gzip
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location('mda_advanced', SRC / '19_mda_advanced.py')
advanced = importlib.util.module_from_spec(spec)
spec.loader.exec_module(advanced)
spec = importlib.util.spec_from_file_location('mda_training', SRC / '17_train_mda_text_gpu.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)
spec = importlib.util.spec_from_file_location('review_pilot', SRC / '20_prepare_restatement_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def long_text(prefix):
    return ' '.join([prefix] * 240)


class FullMdaExtraction(unittest.TestCase):
    def test_10k_prefers_real_item_7_over_table_of_contents(self):
        html = '''<html><body>
        <p>Table of Contents</p><p>Item 7. Management discussion</p>
        <p>Item 7A. Market risk</p>
        <h2>Item 7. Management's Discussion and Analysis</h2>
        <p>''' + long_text('operations') + '''</p>
        <h2>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</h2>
        <p>''' + long_text('market') + '''</p><h2>Item 8. Financial Statements</h2>
        </body></html>'''
        text, status, words = advanced.extract_mda_section(html, '10-K')
        self.assertEqual(status, 'ok')
        self.assertGreaterEqual(words, 200)
        self.assertIn('operations', text)
        self.assertNotIn('market', text)

    def test_10q_item_2_stops_before_item_3(self):
        html = ('<p>Item 2. Management Discussion and Analysis</p><p>' +
                long_text('quarterly') +
                '</p><p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p>' +
                '<p>' + long_text('risk') + '</p><p>Item 4. Controls</p>')
        text, status, _ = advanced.extract_mda_section(html, '10-Q')
        self.assertEqual(status, 'ok')
        self.assertIn('quarterly', text)
        self.assertNotIn('risk', text)

    def test_missing_or_unsupported_sections_are_not_filled_from_whole_filing(self):
        text, status, words = advanced.extract_mda_section('<p>General filing text</p>', '10-K')
        self.assertEqual((text, status, words), ('', 'no_mda_section', 0))
        with self.assertRaises(ValueError):
            advanced.extract_mda_section('<p>text</p>', '10-K/A')

    def test_word_cap_is_reported(self):
        html = ('<p>Item 2. MD&amp;A</p><p>' + long_text('disclosure') * 40 +
                '</p><p>Item 3. Market risk</p>')
        text, status, words = advanced.extract_mda_section(html, '10-Q', max_words=300)
        self.assertEqual(status, 'ok_truncated')
        self.assertEqual(words, 300)
        self.assertEqual(len(text.split()), 300)

    def test_export_preserves_missing_text_rows_and_statuses(self):
        manifest = pd.DataFrame({
            'adsh':['a','b'], 'cik':['1','2'], 'form':['10-K','10-Q'],
            'filed_date':pd.to_datetime(['2020-01-01','2020-02-01']), 'y_proxy':[1,0],
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            db_path = root/'evidence_v5'/'mda_sections'/'sections.sqlite'
            db = advanced.open_section_db(db_path)
            db.executemany('INSERT INTO mda_sections VALUES (?,?,?,?,?,?,?,?)', [
                ('a','1','10-K','2020-01-01','ok',gzip.compress(b'example MD&A'),2,'hash-a'),
                ('b','2','10-Q','2020-02-01','no_mda_section',gzip.compress(b''),0,''),
            ])
            db.commit(); db.close()
            with patch.object(advanced,'build_manifest',return_value=manifest):
                output = advanced.export_dataset(root)
            result = pd.read_csv(output,compression='gzip',dtype={'adsh':str})
            self.assertEqual(result.adsh.tolist(),['a','b'])
            self.assertEqual(result.text_status.tolist(),['ok','no_mda_section'])
            self.assertEqual(result.mda_text.fillna('').tolist(),['example MD&A',''])


class PointInTimeTextChanges(unittest.TestCase):
    def test_legacy_label_columns_are_never_features(self):
        frame = pd.DataFrame({
            'form':['10-K','10-Q'], 'sic2':['10','20'],
            'text_emb_beginning_0':[0.1,0.2], 'mda_word_count':[10,20],
            'y':[0,1], 'legacy_y':[0,1], 'label':[0,1], 'y_proxy':[0,1],
            'y_amendment_proxy':[0,1],
        })
        numeric, categorical = training.columns_for(frame, with_tabular=False)
        self.assertEqual(categorical, ['form'])
        self.assertEqual(set(numeric), {'text_emb_beginning_0','mda_word_count'})

    def test_prior_text_link_is_strictly_earlier_and_same_issuer_form(self):
        frame = pd.DataFrame({
            'adsh':['a','b','c','d','e'], 'cik':['0001','1','1','1','2'],
            'form':['10-Q','10-Q','10-Q','10-K','10-Q'],
            'filed_date':['2020-01-01','2020-04-01','2020-04-01','2020-05-01','2020-06-01'],
        })
        links = training.previous_text_links(frame).set_index('adsh').prior_adsh
        self.assertTrue(pd.isna(links['a']))
        self.assertEqual(links['b'], 'a')
        self.assertEqual(links['c'], 'a')  # same-day filings do not become each other's history
        self.assertTrue(pd.isna(links['d'])) # another form
        self.assertTrue(pd.isna(links['e'])) # another issuer

    def test_issuer_holdout_assignment_is_stable_and_normalizes_cik(self):
        a = training.issuer_holdout_mask(['000123','987654','123'])
        b = training.issuer_holdout_mask(['123','987654','000123'])
        self.assertEqual(bool(a[0]), bool(b[2]))
        self.assertEqual(bool(a[2]), bool(b[0]))
        self.assertEqual(bool(a[0]), bool(a[2]))
        self.assertFalse(np.array_equal(training.issuer_holdout_mask(['1','2','3']),
                                         training.issuer_holdout_mask(['1','2','3'],salt='other')))

    def test_pairwise_groups_require_both_labels_per_year_form(self):
        frame = pd.DataFrame({'adsh':['a','b','c','d'], 'form':['10-K']*4,
            'filed_date':['2020-01-01','2020-02-01','2021-01-01','2021-02-01'],
            'y_amendment_proxy':[1,0,1,0]})
        order, groups = training._rank_groups(frame)
        self.assertEqual(groups, [2,2])
        self.assertEqual(frame.iloc[order].y_amendment_proxy.tolist(), [1,0,1,0])
        frame.loc[frame.filed_date.str.startswith('2021'),'y_amendment_proxy'] = 0
        with self.assertRaisesRegex(ValueError, 'both labels'):
            training._rank_groups(frame)

    def test_form_thresholds_use_2023_calibration_and_align_by_row(self):
        frame = pd.DataFrame({
            'form':['10-K']*40+['10-Q']*40,
            'y_amendment_proxy':[0,1]*40,
        })
        probabilities = np.tile([.2,.8],40)
        thresholds = training.calibrated_form_thresholds(frame, probabilities,.5,min_per_class=5)
        row_thresholds = training.thresholds_for_frame(frame,thresholds,.5)
        report = training.score_report(frame.y_amendment_proxy,probabilities,row_thresholds)
        self.assertEqual(set(thresholds), {'10-K','10-Q'})
        self.assertEqual(set(row_thresholds), {float(thresholds['10-K']['threshold']),
                                                float(thresholds['10-Q']['threshold'])})
        self.assertEqual(report['accuracy'],1.0)
        self.assertEqual(report['threshold'],'row_specific')


class ReviewPilot(unittest.TestCase):
    def test_candidate_and_control_rows_stay_pending_and_unlabeled(self):
        candidates = pd.DataFrame({
            'adsh_original':['pos1','pos2','pos3'],
            'adsh_amendment':['am1','am2','am3'],
            'form_original':['10-K','10-K','10-Q'],
            'filed_original':[20200101,20200201,20200301],
            'filed_amendment':[20200201,20200301,20200401],
            'days_after_original':[31,29,31],
            'amendment_url':['https://www.sec.gov/a','https://www.sec.gov/b','https://www.sec.gov/c'],
        })
        proxy = pd.DataFrame({
            'adsh':['pos1','pos2','pos3','neg1','neg2','neg3'],
            'cik':['1','2','3','4','5','6'], 'form':['10-K','10-K','10-Q','10-K','10-K','10-Q'],
            'filed_date':['2020-01-01','2020-02-01','2020-03-01','2020-01-10','2020-02-10','2020-03-10'],
            'y_amendment_proxy':[1,1,1,0,0,0], 'sic2':['10','10','20','10','10','20'],
        })
        result = pilot.build_pilot(candidates, proxy, n_each=2, seed=5)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.sampling_group.value_counts().to_dict(), {
            'same_period_amendment_candidate':2,
            'no_same_period_amendment_control':2,
        })
        self.assertTrue(result.label.eq('').all())
        self.assertTrue(result.review_status.eq('pending').all())
        self.assertTrue(result.original_adsh.is_unique)

    def test_late_amendment_candidate_is_not_rejected_by_365_day_proxy(self):
        candidates = pd.DataFrame({
            'adsh_original':['late'], 'adsh_amendment':['am-late'],
            'form_original':['10-K'], 'filed_original':[20200101],
            'filed_amendment':[20220501], 'days_after_original':[851],
            'amendment_url':['https://www.sec.gov/Archives/example.txt'],
        })
        proxy = pd.DataFrame({
            'adsh':['late','control'], 'cik':['1','2'], 'form':['10-K','10-K'],
            'filed_date':['2020-01-01','2020-01-05'], 'y_amendment_proxy':[0,0],
            'sic2':['10','10'],
        })
        result=pilot.build_pilot(candidates,proxy,n_each=1,seed=4)
        case=result[result.sampling_group.eq('same_period_amendment_candidate')]
        self.assertEqual(case.original_adsh.tolist(),['late'])
        self.assertEqual(case.days_to_candidate_event.tolist(),[851])
        self.assertTrue(result.label.eq('').all())
        self.assertTrue(result.evidence_url.eq('').all())

    def test_pilot_checks_proxy_consistency_only_inside_its_365_day_window(self):
        candidates = pd.DataFrame({
            'adsh_original':['neg'], 'adsh_amendment':['am'], 'form_original':['10-Q'],
            'filed_original':['2020-01-01'], 'filed_amendment':['2020-02-01'],
            'days_after_original':[31], 'amendment_url':['https://www.sec.gov/a'],
        })
        proxy = pd.DataFrame({'adsh':['neg','control'], 'cik':['1','2'], 'form':['10-Q','10-Q'],
            'filed_date':['2020-01-01','2020-01-02'], 'y_amendment_proxy':[0,0]})
        with self.assertRaisesRegex(ValueError, 'does not align'):
            pilot.build_pilot(candidates, proxy, n_each=1)

    def test_horizon_queue_excludes_immature_originals_before_balanced_sampling(self):
        candidates=pd.DataFrame({
            'adsh_original':['early','late'], 'adsh_amendment':['am-early','am-late'],
            'form_original':['10-K','10-K'], 'filed_original':[20200101,20250101],
            'filed_amendment':[20220501,20260102], 'days_after_original':[851,366],
            'amendment_url':['https://www.sec.gov/a','https://www.sec.gov/b'],
        })
        proxy=pd.DataFrame({
            'adsh':['early','late','control'], 'cik':['1','2','3'], 'form':['10-K']*3,
            'filed_date':['2020-01-01','2025-01-01','2020-02-01'],
            'y_amendment_proxy':[0,0,0], 'sic2':['10']*3,
            'label_observation_cutoff':['2026-09-30']*3,
        })
        result=pilot.build_pilot(candidates,proxy,n_each=1,seed=3,horizon_years=3)
        self.assertEqual(set(result.original_adsh),{'early','control'})
        self.assertTrue(result.review_queue_horizon_years.eq(3).all())


if __name__ == '__main__':
    unittest.main()
