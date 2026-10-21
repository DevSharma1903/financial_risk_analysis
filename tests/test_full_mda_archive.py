"""Tests for full-archive text preparation and leakage-safe text transforms."""
import gzip
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
sys.path.insert(0,str(SRC))

spec=importlib.util.spec_from_file_location("restatement_study",SRC/"22_restatement_study.py")
study=importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
prep_spec=importlib.util.spec_from_file_location("mda_archive_prep",SRC/"26_prepare_mda_archive.py")
prep=importlib.util.module_from_spec(prep_spec)
prep_spec.loader.exec_module(prep)


class FullArchivePreparation(unittest.TestCase):
    def test_preparation_drops_proxy_target_and_bounds_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); data=root/"data"; data.mkdir()
            rows=[]
            proxy=[]
            for i in range(4):
                adsh=f"a{i}"
                filed=f"2020010{i+1}"
                text="income cash controls "*(3 if i<3 else 1)
                rows.append({"adsh":adsh,"cik":"0001","form":"10-K","filed":filed,
                             "y":i%2,"mda_text":text})
                proxy.append({"adsh":adsh,"cik":"1","form":"10-K",
                              "filed_date":f"2020-01-0{i+1}"})
            pd.DataFrame(rows).to_csv(data/"mda_corpus.csv",index=False)
            pd.DataFrame(proxy).to_csv(data/"colab_amendment_proxy.csv.gz",index=False,
                                       compression="gzip")
            manifest=prep.prepare(root,min_words=3,max_words=5)
            result=pd.read_csv(data/"colab_mda_archive_unlabeled.csv.gz",compression="gzip")
        self.assertEqual(len(result),4)
        self.assertEqual(result.columns.tolist(),
                         ["adsh","cik","form","filed_date","mda_text","mda_word_count"])
        self.assertNotIn("y",result.columns)
        self.assertTrue(result.mda_word_count.le(5).all())
        self.assertFalse(manifest["legacy_proxy_label_read"])
        self.assertFalse(manifest["outcome_labels_in_output"])

    def test_archive_topics_fit_only_pre_holdout_and_ignore_old_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"archive.csv.gz"
            common="revenue operations cash accounting "*60
            rows=[]
            for i in range(24):
                rows.append({"adsh":f"train{i}","filed_date":f"2019-01-{(i%28)+1:02d}",
                             "mda_text":common+(" liquidity "*i),"y_proxy":i%2})
            rows.append({"adsh":"future","filed_date":"2022-01-01",
                         "mda_text":common+" secretfutureterm "*60,"y_proxy":1})
            pd.DataFrame(rows).to_csv(path,index=False,compression="gzip")
            frame=pd.DataFrame({"mda_text":[common],"prior_mda_text":[common+" cash "*10]})
            result,meta,vectorizer,_=study._archive_topic_features(
                frame,path,2021,n_components=3,min_df=1,max_features=100)
        self.assertEqual(meta["archive_rows"],25)
        self.assertEqual(meta["archive_fit_rows"],24)
        self.assertEqual(meta["archive_rows_excluded_as_holdout_or_later"],1)
        self.assertFalse(meta["legacy_proxy_labels_read"])
        self.assertNotIn("secretfutureterm",vectorizer.vocabulary_)
        self.assertEqual(result["archive_topics"].shape,(1,9))
        self.assertTrue(np.isfinite(result["archive_topics"]).all())

    def test_full_archive_view_aligns_feature_names_and_matrix(self):
        frame=pd.DataFrame({"adsh":["a","b","c"],"cik":["1","2","3"],
            "form":["10-K","10-Q","10-K"],"fp":["FY","Q1","FY"],
            "sic2":["10","20","10"],"filed_date":["2020-01-01"]*3,
            "filed_year":[2020]*3,"target_restatement":[1,0,1],"xbrl_Assets":[1.,2.,3.]})
        embeddings={"novelty":np.ones((3,2),dtype=np.float32),
                    "novelty_names":["mda_change__0","mda_current__0"],
                    "archive_topics":np.ones((3,1),dtype=np.float32),
                    "archive_topic_names":["archive_current_topic__000"]}
        x_train,x_valid,names=study._features(frame.iloc[:2],frame.iloc[2:],"full_archive",
            embeddings,np.array([0,1]),np.array([2]),return_names=True)
        self.assertEqual(x_train.shape[1],len(names))
        self.assertEqual(x_valid.shape[1],len(names))
        self.assertIn("archive_current_topic__000",names)


if __name__=="__main__":
    unittest.main()
