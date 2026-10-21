"""Generate the GPU Colab notebook for the filing-event proxy experiments."""
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def markdown(value):
    return {"cell_type":"markdown","metadata":{},"source":value.strip().splitlines(keepends=True)}


def code(value):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
            "source":value.strip().splitlines(keepends=True)}


def build():
    names=["research_v5_core.py", "sec_v5_evidence.py", "19_mda_advanced.py",
           "22_restatement_study.py", "28_filing_event_proxy.py"]
    sources={name:(ROOT/"src"/name).read_text(encoding="utf-8") for name in names}
    cells=[
        markdown("""# Filing-event proxy experiments: MD&A + GPU

This Colab notebook runs two separately named outcomes on a 1:1 sampled SEC filing cohort. **Outcome A** is an observed same-period 10-K/A or 10-Q/A within three years. It is an amendment proxy and can include routine amendments; it is not a confirmed restatement label. **Outcome B** is a broader composite: the same-period amendment event or a reviewed issuer-level SEC restatement/revision event within three years. Pending control outcomes are excluded from Outcome B. Some review outcomes were AI-assisted, so treat both results as exploratory.

The model receives only the original filing's point-in-time financial features and original/prior MD&A text. Candidate accession fields, sampling stratum, review status, evidence text, future dates, and legacy proxy label columns are excluded from predictors. Full-archive MD&A text supplies unsupervised language topics, fit only on documents filed before the future holdout. The comparison includes financial-only, text-only, interpretable language, fused, and full-archive feature views using GPU XGBoost and CatBoost. Issuers are grouped in development folds; the latest filing years are held out. Threshold selection uses development predictions, then each target's holdout is evaluated once.

The 1:1 results describe this sampled cohort only. They do not estimate natural-prevalence precision and must not be described as confirmed-restatement prediction. The notebook requires a Colab GPU and will stop if CUDA is unavailable."""),
        code('%pip -q uninstall -y xgboost xgboost-cu12\n%pip -q install "pandas>=2.2,<3" "numpy>=1.26,<3" "scikit-learn>=1.5,<2" "xgboost-cu12>=3.2,<4" "catboost>=1.2,<2" "sentence-transformers>=3,<6" "beautifulsoup4>=4.12,<5"'),
        markdown("If Colab requests a runtime restart after installing packages, restart, reselect the GPU runtime, and run cells from the top."),
        code('import torch, subprocess, xgboost as xgb\nassert torch.cuda.is_available(), "Select Runtime > Change runtime type > GPU, then rerun."\nassert xgb.build_info().get("USE_CUDA",False), "The installed XGBoost does not support CUDA; reinstall in a GPU runtime."\nprint("CUDA:",torch.version.cuda,"GPU:",torch.cuda.get_device_name(0),"XGBoost CUDA:",xgb.build_info().get("CUDA_VERSION"))\nsubprocess.run(["nvidia-smi"],check=True)'),
        code('from google.colab import drive\ndrive.mount("/content/drive")\nfrom pathlib import Path\nROOT=Path("/content/drive/MyDrive/financial-risk-detector")\nDATA=ROOT/"data"\nPROXY=DATA/"colab_amendment_proxy.csv.gz"\nLEDGER=DATA/"restatement_review_queue_ai_refined.csv"\nif not LEDGER.is_file(): LEDGER=DATA/"restatement_review_queue.csv"\nARCHIVE=DATA/"colab_mda_archive_unlabeled.csv.gz"\nfor path in (PROXY,LEDGER,ARCHIVE): assert path.is_file(), f"Missing upload: {path}"\nHORIZON_YEARS=3\nCACHE=ROOT/f"evidence_restatement/{HORIZON_YEARS}y/sections.sqlite"\nDATASET=DATA/f"filing_event_proxy_h{HORIZON_YEARS}y_fulltext_trajectory_v2_model.csv.gz"\nprint("Proxy:",PROXY,"\\nReview queue:",LEDGER,"\\nUnlabeled MD&A archive:",ARCHIVE,"\\nSEC cache:",CACHE)'),
        code('import sys, importlib, importlib.util\nMODULES=Path("/content/filing_event_modules"); MODULES.mkdir(parents=True,exist_ok=True)\nSOURCES='+repr(sources)+'\nfor name,source in SOURCES.items(): (MODULES/name).write_text(source,encoding="utf-8")\nif str(MODULES) not in sys.path: sys.path.insert(0,str(MODULES))\nimport research_v5_core as core; import sec_v5_evidence as evidence\nimportlib.reload(core); importlib.reload(evidence)\nspec=importlib.util.spec_from_file_location("restatement_study",MODULES/"22_restatement_study.py")\nstudy=importlib.util.module_from_spec(spec); spec.loader.exec_module(study)\nspec=importlib.util.spec_from_file_location("filing_event_proxy",MODULES/"28_filing_event_proxy.py")\nevent=importlib.util.module_from_spec(spec); spec.loader.exec_module(event)\nimport pandas as pd\nprint("Embedded study and target modules loaded.")'),
        code('ledger=pd.read_csv(LEDGER,dtype={"original_adsh":str},low_memory=False)\nproxy=pd.read_csv(PROXY,compression="gzip",dtype={"adsh":str,"cik":str},low_memory=False)\nevent.validate_maturity(proxy,ledger,HORIZON_YEARS)\ncohort=event.derive_targets(ledger,HORIZON_YEARS)\nprint(event.summarize(cohort))\nprint("Target A: all observed candidate/control rows. Target B: unresolved controls remain missing and are excluded at fit time.")'),
        markdown("The next cell resumes the Drive SQLite cache. It may take multiple Colab sessions. If `pending` is above zero, rerun setup and collection later; do not launch two sessions against the same cache at once. SEC requests are sequential, limited to two per second, and stop on access denial."),
        code('import getpass\nUSER_AGENT=getpass.getpass("Real SEC User-Agent (your name/organization and contact email): ").strip()\nMAX_NEW_PER_SESSION=500\nadvanced=study._load_module("event_mda_extractor",MODULES/"19_mda_advanced.py")\ncollection=event.collect_mda(proxy,cohort,CACHE,USER_AGENT,max_new=MAX_NEW_PER_SESSION,study=study,evidence=evidence,advanced=advanced)\nprint(collection)\nif collection["pending"]: raise RuntimeError("Text collection is incomplete. Rerun setup and collection in a later session until pending is 0.")'),
        code('prepared=event.prepare_model_data(proxy,cohort,CACHE,DATASET)\nprint(prepared)\nprint("Predictor file:",DATASET)'),
        code('import hashlib\ndef sha256_file(path):\n    digest=hashlib.sha256()\n    with Path(path).open("rb") as stream:\n        for block in iter(lambda:stream.read(1024*1024),b""):\n            digest.update(block)\n    return digest.hexdigest()\nOUT_A=ROOT/f"colab_outputs/filing_event_proxy_h{HORIZON_YEARS}y_fulltext_trajectory_v2/same_period_amendment"\nOUT_B=ROOT/f"colab_outputs/filing_event_proxy_h{HORIZON_YEARS}y_fulltext_trajectory_v2/amendment_or_screened_event"\naudit={"horizon_years":HORIZON_YEARS,"cohort":event.summarize(cohort),\n       "input_sha256":{"proxy":sha256_file(PROXY),"review_queue":sha256_file(LEDGER),\n                        "unlabeled_archive":sha256_file(ARCHIVE),"prepared_dataset":sha256_file(DATASET)},\n       "target_rules_sha256":sha256_file(MODULES/"28_filing_event_proxy.py"),\n       "prediction_rule":"Only original-filing features; target/provenance/candidate/review fields are excluded."}\nfor folder in (OUT_A,OUT_B):\n    folder.mkdir(parents=True,exist_ok=True)\n    (folder/"label_provenance.json").write_text(json.dumps(audit,indent=2,default=str),encoding="utf-8")\nprint("Saved cohort and input hashes to each output folder.")'),
        markdown("## GPU model comparisons\n\nBoth target runs use identical original/current/prior MD&A predictors and the same full archive. The second target omits unresolved controls; its counts are printed before fitting. The run selects a model and threshold using development folds, then evaluates the latest filing years."),
        code('import json\nTARGET_A="target_same_period_amendment_proxy"\nDEFINITION_A=("1 when a same-period 10-K/A or 10-Q/A candidate was observed within three years; "\n              "0 for the sampled no-amendment control. This proxy may include routine amendments.")\nmetrics_a=study.train_gpu(DATASET,OUT_A,archive_path=ARCHIVE,target_column=TARGET_A,target_definition=DEFINITION_A)\nprint(json.dumps(metrics_a,indent=2))'),
        code('TARGET_B="target_amendment_or_screened_event"\nDEFINITION_B=("1 for a same-period 10-K/A or 10-Q/A candidate, or a reviewed issuer-level restatement/revision event, "\n              "within three years; 0 only for a reviewed control with complete follow-up and no qualifying event. "\n              "Pending control outcomes are excluded. Some review screening was AI-assisted.")\nmetrics_b=study.train_gpu(DATASET,OUT_B,archive_path=ARCHIVE,target_column=TARGET_B,target_definition=DEFINITION_B)\nprint(json.dumps(metrics_b,indent=2))\nprint("\\nOutput A:",OUT_A,"\\nOutput B:",OUT_B)'),
        markdown("""## Report the results

Use each output folder's `metrics.json`, `study_report.md`, grouped CV leaderboard, future-holdout predictions, and SHAP summaries. Report the targets by their exact names. Outcome A measures prediction of a sampled same-period amendment proxy. Outcome B is a mixed-evidence composite with unresolved controls excluded. Do not merge their scores or present either one as a confirmed-restatement estimate. The latest-year 1:1 metrics are case-control sample metrics, not estimates of real-world precision. The `full_archive_text_features` field records the archive cutoff and confirms the legacy archive label was not read.""")]
    notebook={"nbformat":4,"nbformat_minor":4,
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                    "language_info":{"name":"python"},
                    "colab":{"name":"filing_event_proxy_gpu_colab.ipynb","provenance":[]},
                    "accelerator":"GPU"},"cells":cells}
    path=ROOT/"notebooks"/"filing_event_proxy_gpu_colab.ipynb"
    path.write_text(json.dumps(notebook,indent=1),encoding="utf-8")
    print(f"Wrote {path}")


if __name__=="__main__":
    build()
