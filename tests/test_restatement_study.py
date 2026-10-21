"""Offline correctness checks for reviewed-outcome labels and data preparation."""
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
validator_spec=importlib.util.spec_from_file_location("restatement_validator",SRC/"11_validate_restatement_labels.py")
validator=importlib.util.module_from_spec(validator_spec)
validator_spec.loader.exec_module(validator)


def reviewed_row(adsh, filed, label, event_type, event_date, followup):
    return {
        "original_adsh":adsh,"original_filed_date":filed,"label":label,
        "review_status":"reviewed","event_type":event_type,"event_date":event_date,
        "outcome_followup_through":followup,"evidence_url":"https://www.sec.gov/Archives/example.txt",
        "evidence_excerpt":"Reviewed source evidence excerpt.","reviewer":"reviewer-1",
        "review_notes":"Manually reviewed.",
    }


class RestatementLabels(unittest.TestCase):
    def test_financial_trajectory_uses_only_strictly_earlier_same_issuer_form_filings(self):
        proxy=pd.DataFrame({
            "adsh":["p1","p2","same_day","future","target","other"],
            "cik":["0001","1","1","1","1","1"],
            "form":["10-Q","10-Q","10-Q","10-Q","10-Q","10-K"],
            "filed_date":["2018-01-01","2019-01-01","2020-01-01","2021-01-01",
                          "2020-01-01","2019-01-01"],
            "gross_margin":[.1,.2,9.,100.,.4,50.],
        })
        targets=proxy.loc[proxy.adsh.eq("target")].copy()
        result=study._financial_trajectory_features(proxy,targets).iloc[0]
        self.assertEqual(result.trajectory_prior_filing_count,2)
        self.assertAlmostEqual(result["trajectory__gross_margin__slope_per_filing"],.15)
        self.assertAlmostEqual(result["trajectory__gross_margin__std"],np.std([.1,.2,.4]))
        self.assertAlmostEqual(result["trajectory__gross_margin__range"],.3)
        self.assertAlmostEqual(result["trajectory__gross_margin__latest_step"],.2)

    def test_embedding_chunking_covers_the_entire_section(self):
        words=[f"w{i}" for i in range(1301)]
        chunks=study._document_chunks(" ".join(words),chunk_words=140)
        self.assertEqual(len(chunks),10)
        self.assertEqual(chunks[0].split()[0],"w0")
        self.assertEqual(chunks[-1].split()[-1],"w1300")
        self.assertEqual(sum(len(chunk.split()) for chunk in chunks),len(words))

    def test_horizons_keep_event_dates_and_exclude_immature_negatives(self):
        ledger=pd.DataFrame([
            reviewed_row("a","2020-01-01",1,"restatement","2020-07-01","2020-07-01"),
            reviewed_row("b","2020-01-01",1,"restatement","2022-02-01","2022-02-01"),
            reviewed_row("c","2020-01-01",0,"no_confirmed_restatement","","2026-01-01"),
            reviewed_row("d","2020-01-01",0,"no_confirmed_restatement","","2021-02-01"),
            {**reviewed_row("pending","2020-01-01","","","",""),"review_status":"pending",
             "evidence_url":"","evidence_excerpt":"","reviewer":""},
        ])
        one,immature_one=study.derive_horizon_labels(ledger,1)
        three,immature_three=study.derive_horizon_labels(ledger,3)
        self.assertEqual(dict(zip(one.original_adsh,one.target_restatement)),{"a":1,"b":0,"c":0,"d":0})
        self.assertEqual(dict(zip(three.original_adsh,three.target_restatement)),{"a":1,"b":1,"c":0})
        self.assertEqual((immature_one,immature_three),(0,1))

    def test_only_sec_primary_urls_and_reviewed_rows_can_enter_model_cohort(self):
        ledger=pd.DataFrame([reviewed_row("a","2020-01-01",0,"no_confirmed_restatement","","2022-01-01")])
        ledger.loc[0,"evidence_url"]="https://example.org/not-sec"
        self.assertTrue(any("SEC.gov" in e for e in study.validate_ledger(ledger)))

    def test_cli_validator_accepts_early_positive_and_rejects_immature_negative(self):
        positive=pd.DataFrame([reviewed_row("a","2020-01-01",1,"restatement","2020-06-01","2020-06-01")])
        self.assertEqual(validator.validate_ledger(positive,min_followup_years=3),[])
        negative=pd.DataFrame([reviewed_row("b","2020-01-01",0,"no_confirmed_restatement","","2022-12-31")])
        errors=validator.validate_ledger(negative,min_followup_years=3)
        self.assertTrue(any("3 years" in e for e in errors))

    def test_prior_filing_link_is_strictly_earlier_same_issuer_and_form(self):
        proxy=pd.DataFrame({
            "adsh":["prior","same-day","other-form","target"],
            "cik":["0001","1","1","1"],"form":["10-Q","10-Q","10-K","10-Q"],
            "filed_date":["2020-01-01","2020-04-01","2020-03-01","2020-04-01"],
        })
        targets=pd.DataFrame({"original_adsh":["target"],"cik":["1"],"original_form":["10-Q"],
                              "original_filed_date":[pd.Timestamp("2020-04-01")]})
        links=study._prior_filing_links(proxy,targets)
        self.assertEqual(links.prior_adsh.tolist(),["prior"])

    def test_holdout_uses_only_latest_filing_years(self):
        frame=pd.DataFrame({"filed_date":[f"{year}-01-01" for year in range(2015,2020)]})
        mask,start=study.temporal_holdout_mask(frame,.20)
        self.assertEqual(start,2019)
        self.assertEqual(mask.tolist(),[False,False,False,False,True])

    def test_fusion_features_keep_same_schema_and_exclude_proxy_outcomes(self):
        frame=pd.DataFrame({
            "adsh":["a","b","c"],"cik":["1","2","3"],"form":["10-K","10-Q","10-K"],
            "fp":["FY","Q1","FY"],"sic2":["10","20","10"],"filed_date":["2020-01-01"]*3,
            "filed_year":[2020]*3,"target_restatement":[1,0,1],"y_amendment_proxy":[0,1,1],
            "days_to_amendment":[0,20,30],"xbrl_Assets":[1.,2.,3.],
        })
        embeds={"text":np.ones((3,2),dtype=np.float32),"novelty":np.ones((3,4),dtype=np.float32),
                "novelty_names":["mda_current_embedding__000","mda_prior_embedding__000",
                    "mda_change_embedding__000","mda_abs_change_embedding__000"]}
        x_train,x_valid,feature_names=study._features(frame.iloc[:2],frame.iloc[2:],"fusion",embeds,
                                         np.array([0,1]),np.array([2]),return_names=True)
        changed=frame.iloc[2:].copy()
        changed["target_restatement"]=0; changed["y_amendment_proxy"]=0; changed["days_to_amendment"]=-999
        _,x_valid_changed=study._features(frame.iloc[:2],changed,"fusion",embeds,
                                           np.array([0,1]),np.array([2]))
        self.assertEqual(x_train.shape[1],x_valid.shape[1])
        self.assertEqual(x_valid.shape[1],len(feature_names))
        np.testing.assert_array_equal(x_valid,x_valid_changed)

    def test_interpretable_view_names_align_and_keep_lexical_change_features(self):
        frame=pd.DataFrame({
            "adsh":["a","b","c"],"cik":["1","2","3"],"form":["10-K","10-Q","10-K"],
            "fp":["FY","Q1","FY"],"sic2":["10","20","10"],
            "filed_date":["2020-01-01"]*3,"filed_year":[2020]*3,
            "target_restatement":[1,0,1],"xbrl_Assets":[1.,2.,3.],
        })
        embeds={"interpretable":np.ones((3,4),dtype=np.float32),
                "interpretable_names":["lex_current__lex_error_per_1k",
                    "lex_prior__lex_error_per_1k","lex_change__lex_error_per_1k",
                    "availability__current_mda"]}
        x_train,x_valid,names=study._features(frame.iloc[:2],frame.iloc[2:],"interpretable",
            embeds,np.array([0,1]),np.array([2]),return_names=True)
        self.assertEqual(x_train.shape[1],len(names))
        self.assertEqual(x_valid.shape[1],len(names))
        self.assertIn("lex_change__lex_error_per_1k",names)

    def test_class_weights_retain_rows_and_equalize_total_class_weight(self):
        frame=pd.DataFrame({"target_restatement":[1,0,0,0,0]})
        weights=study._class_weights(frame)
        self.assertEqual(len(weights),len(frame))
        self.assertAlmostEqual(float(weights[frame.target_restatement.eq(1)].sum()),
                               float(weights[frame.target_restatement.eq(0)].sum()))

    def test_paired_auc_bootstrap_compares_same_cases_and_controls(self):
        y=np.array([0,0,0,0,1,1,1,1])
        better=np.array([.1,.2,.3,.4,.6,.7,.8,.9])
        worse=1-better
        result=study._paired_auc_difference(y,better,worse,seed=7,replicates=100)
        self.assertGreater(result["pr_auc_difference"]["estimate"],0)
        self.assertLessEqual(result["pr_auc_difference"]["lower_95"],
                             result["pr_auc_difference"]["upper_95"])

    def test_model_data_joins_only_reviewed_labels_and_excludes_proxy_target(self):
        ledger=pd.DataFrame([
            reviewed_row("target","2020-01-01",1,"restatement","2020-06-01","2020-06-01"),
            reviewed_row("negative","2020-02-01",0,"no_confirmed_restatement","","2022-03-01"),
        ])
        proxy=pd.DataFrame({
            "adsh":["prior","target","negative"],"cik":["1","1","2"],
            "form":["10-Q","10-Q","10-Q"],"fp":["Q4","Q1","Q1"],
            "sic2":["10","10","20"],
            "filed_date":["2019-10-01","2020-01-01","2020-02-01"],
            "period_date":["2019-09-30","2019-12-31","2019-12-31"],
            "feature_schema_version":[4,4,4],"label_observation_cutoff":["2026-09-30"]*3,
            "y_amendment_proxy":[0,1,0],"days_to_amendment":[0,100,0],
            "xbrl_Assets":[10.,11.,20.],"delta_xbrl_Assets":[0.,1.,0.],
            "prior_amendments_365d":[0,0,0],
        })
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            proxy_path=td/"proxy.csv.gz"; proxy.to_csv(proxy_path,index=False,compression="gzip")
            ledger_path=td/"ledger.csv"; ledger.to_csv(ledger_path,index=False)
            db=study.section_db(td/"sections.sqlite")
            for adsh,form,date in [("prior","10-Q","2019-10-01"),("target","10-Q","2020-01-01")]:
                db.execute("INSERT INTO restatement_mda VALUES (?,?,?,?,?,?,?,?)",
                           (adsh,"1",form,date,"ok",gzip.compress(b"sample MD&A text"),3,"hash"))
            db.commit(); db.close()
            output=td/"model.csv.gz"
            report=study.prepare_model_data(proxy_path,ledger_path,td/"sections.sqlite",1,output)
            result=pd.read_csv(output,compression="gzip",dtype={"adsh":str})
        self.assertEqual(report["rows"],2)
        self.assertEqual(result.target_restatement.tolist(),[1,0])
        self.assertEqual(result.loc[result.adsh.eq("target"),"prior_adsh"].iloc[0],"prior")
        self.assertNotIn("y_amendment_proxy",result)
        self.assertNotIn("days_to_amendment",result)
        self.assertNotIn("evidence_excerpt",result)

    def test_targeted_text_collection_caches_only_reviewed_filings_and_prior(self):
        ledger=pd.DataFrame([reviewed_row("target","2020-01-01",1,"restatement","2020-06-01","2020-06-01")])
        proxy=pd.DataFrame({
            "adsh":["prior","target"],"cik":["1","1"],"form":["10-Q","10-Q"],
            "filed_date":["2019-10-01","2020-01-01"],
            "feature_schema_version":[4,4],"label_observation_cutoff":["2026-09-30"]*2,
        })
        body=("<DOCUMENT><TYPE>10-Q\n<TEXT><html><p>Item 2. Management Discussion and Analysis</p><p>"+
              " ".join(["quarterly operations discussion"]*240)+
              "</p><p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p></html></TEXT></DOCUMENT>").encode()
        class FakeClient:
            calls=0
            def get(self,url,**kwargs):
                self.calls+=1
                return body
        client=FakeClient()
        with tempfile.TemporaryDirectory() as td:
            db_path=Path(td)/"sections.sqlite"
            first=study.collect_reviewed_mda(proxy,ledger,1,db_path,"A Researcher contact@example.org",max_new=0,client=client)
            second=study.collect_reviewed_mda(proxy,ledger,1,db_path,"A Researcher contact@example.org",max_new=0,client=client)
        self.assertEqual(first["unique_filings"],2)
        self.assertEqual(first["pending"],0)
        self.assertEqual(client.calls,2)
        self.assertEqual(second["attempted_this_session"],0)


if __name__=="__main__":
    unittest.main()
