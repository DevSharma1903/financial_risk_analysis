"""Offline tests for transparent amendment/event proxy labels and leakage guards."""
from pathlib import Path
import gzip
import importlib.util
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"
sys.path.insert(0,str(SRC))
spec=importlib.util.spec_from_file_location("filing_event_proxy",SRC/"28_filing_event_proxy.py")
event=importlib.util.module_from_spec(spec)
spec.loader.exec_module(event)
study_spec=importlib.util.spec_from_file_location("restatement_study",SRC/"22_restatement_study.py")
study=importlib.util.module_from_spec(study_spec)
study_spec.loader.exec_module(study)


def row(adsh, group, status, event_type="", event_date="", followup="", candidate="", candidate_date=""):
    return {
        "original_adsh":adsh,"original_filed_date":"2020-01-01",
        "sampling_group":group,"candidate_event_adsh":candidate,
        "candidate_event_date":candidate_date,"review_status":status,
        "event_type":event_type,"event_date":event_date,
        "outcome_followup_through":followup,"review_queue_horizon_years":3,
    }


class FilingEventProxy(unittest.TestCase):
    def test_proxy_and_composite_targets_keep_unresolved_controls_missing(self):
        ledger=pd.DataFrame([
            row("candidate",event.CANDIDATE_GROUP,"pending",candidate="amendment",candidate_date="2021-02-01"),
            row("restatement",event.CONTROL_GROUP,"reviewed","restatement","2021-02-01","2021-02-01"),
            row("negative",event.CONTROL_GROUP,"reviewed","no_confirmed_restatement","","2023-01-01"),
            row("pending",event.CONTROL_GROUP,"pending"),
        ])
        result=event.derive_targets(ledger,3).set_index("original_adsh")
        self.assertEqual(result.target_same_period_amendment_proxy.to_dict(),
                         {"candidate":1,"restatement":0,"negative":0,"pending":0})
        self.assertEqual(int(result.loc["candidate","target_amendment_or_screened_event"]),1)
        self.assertEqual(int(result.loc["restatement","target_amendment_or_screened_event"]),1)
        self.assertEqual(int(result.loc["negative","target_amendment_or_screened_event"]),0)
        self.assertTrue(pd.isna(result.loc["pending","target_amendment_or_screened_event"]))

    def test_out_of_horizon_candidate_and_wrong_horizon_are_rejected(self):
        late=pd.DataFrame([row("late",event.CANDIDATE_GROUP,"pending",candidate="amendment",
                               candidate_date="2024-01-02")])
        with self.assertRaisesRegex(ValueError,"outside the selected outcome horizon"):
            event.derive_targets(late,3)
        wrong=late.copy(); wrong["review_queue_horizon_years"]=1
        with self.assertRaisesRegex(ValueError,"does not match"):
            event.derive_targets(wrong,3)

    def test_model_table_drops_outcome_and_legacy_proxy_predictors(self):
        ledger=pd.DataFrame([
            row("a",event.CANDIDATE_GROUP,"pending",candidate="amendment",candidate_date="2021-01-01"),
            row("b",event.CONTROL_GROUP,"reviewed","no_confirmed_restatement","","2023-01-01"),
            row("c",event.CONTROL_GROUP,"pending"),
        ])
        cohort=event.derive_targets(ledger,3)
        proxy=pd.DataFrame({
            "adsh":["a","b","c"],"cik":["1","2","3"],"form":["10-K"]*3,
            "fp":["FY"]*3,"sic2":["10","20","30"],
            "filed_date":["2020-01-01"]*3,
            "period_date":["2019-12-31"]*3,
            "xbrl_Assets":[10.0,20.0,30.0],"y_amendment_proxy":[1,0,0],
            "days_to_amendment":[100,0,0],"feature_schema_version":[4]*3,
            "label_observation_cutoff":["2026-09-30"]*3,
        })
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder); db=event.sqlite3.connect(folder/"sections.sqlite")
            db.execute("""CREATE TABLE restatement_mda (
                adsh TEXT PRIMARY KEY, cik TEXT NOT NULL, form TEXT NOT NULL,
                filed_date TEXT NOT NULL, status TEXT NOT NULL, body BLOB NOT NULL,
                word_count INTEGER NOT NULL, sha256 TEXT NOT NULL)""")
            for adsh,cik in [("a","1"),("b","2"),("c","3")]:
                db.execute("INSERT INTO restatement_mda VALUES (?,?,?,?,?,?,?,?)",
                    (adsh,cik,"10-K","2020-01-01","ok",gzip.compress(b"sample MD&A"),2,"hash"))
            db.commit(); db.close()
            output=folder/"filing_event.csv.gz"
            result=event.prepare_model_data(proxy,cohort,folder/"sections.sqlite",output)
            prepared=pd.read_csv(output,compression="gzip")
        self.assertEqual(result["rows"],3)
        self.assertIn("xbrl_Assets",prepared)
        self.assertIn("target_same_period_amendment_proxy",prepared)
        self.assertIn("target_amendment_or_screened_event",prepared)
        for forbidden in ("y_amendment_proxy","days_to_amendment","sampling_group",
                          "candidate_event_adsh","event_date","review_status"):
            self.assertNotIn(forbidden,prepared.columns)
        self.assertIn("label_provenance",prepared.columns)
        self.assertFalse(prepared.target_amendment_or_screened_event.notna().all())

    def test_model_features_ignore_all_proxy_target_and_provenance_columns(self):
        frame=pd.DataFrame({
            "adsh":["a","b","c"],"cik":["1","2","3"],"form":["10-K"]*3,
            "fp":["FY"]*3,"sic2":["10"]*3,"filed_date":["2020-01-01"]*3,
            "filed_year":[2020]*3,"target_same_period_amendment_proxy":[1,0,1],
            "target_amendment_or_screened_event":[1,0,1],"label_provenance_code":[7,8,9],
            "xbrl_Assets":[10.0,20.0,30.0],
        })
        embeds={"text":np.ones((3,2),dtype="float32"),
                "text_names":["text__0","text__1"]}
        x_train,x_valid,names=study._features(frame.iloc[:2],frame.iloc[2:],"financial",
            embeds,np.array([0,1]),np.array([2]),return_names=True)
        self.assertTrue(all("target_" not in name and "label_" not in name for name in names))
        self.assertEqual(x_train.shape[1],x_valid.shape[1])


if __name__=="__main__":
    unittest.main()
