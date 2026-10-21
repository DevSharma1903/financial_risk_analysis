"""Build a self-contained Colab notebook for the confirmed-restatement study."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def markdown(value):
    return {"cell_type":"markdown","metadata":{},"source":value.strip().splitlines(keepends=True)}


def code(value):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
            "source":value.strip().splitlines(keepends=True)}


def build():
    names=["research_v5_core.py","sec_v5_evidence.py","19_mda_advanced.py",
           "11_validate_restatement_labels.py","22_restatement_study.py"]
    sources={name:(ROOT/"src"/name).read_text(encoding="utf-8") for name in names}
    cells=[
        markdown("""# Confirmed-restatement MD&A study — Colab GPU

This notebook tests whether MD&A predicts a later **issuer-level financial-statement restatement** after an original 10-K/10-Q. It does not use the amendment proxy as the target, and the event need not restate the exact period of the original filing. It evaluates a fixed 1:1 case-control cohort and reports development and forward-year holdout metrics. No score is promised in advance.

The ledger may contain human-adjudicated labels or an AI-assisted SEC evidence screen. AI-assisted labels are exploratory; the validator checks structure and source links, not accounting correctness. Do not report confirmatory or human-validated performance until outcomes have been human-audited. Pending rows are never used.

Choose one outcome horizon before the run: 1, 3, or 5 years. A negative needs documented SEC filing review through the end of that horizon. A confirmed event date is retained, so targets at other horizons can be derived later. SEC text is downloaded only for reviewed originals and their immediately prior same-issuer/form filing. Downloads are sequential, limited to two requests per second, and stop on access denial. Use a real identifying SEC User-Agent.

The next version adds strictly point-in-time financial trajectories from up to three earlier same-issuer/same-form filings, and sentence embeddings now cover the full available MD&A section (up to 6,000 words) instead of only its opening portion. It compares structured, interpretable-text, semantic-text, and fused features using GPU XGBoost/CatBoost. Training retains every development row using class-weighted loss; validation and the primary holdout remain unique, year/form-matched 1:1 samples. Model selection uses issuer-grouped development PR-AUC, and the threshold is set from development predictions only. The latest filing years stay untouched until final evaluation. The returned 1:1 metrics apply only to this reviewed case-control sample; they do not establish performance at natural prevalence."""),
        code('%pip -q uninstall -y xgboost xgboost-cu12\n%pip -q install "pandas>=2.2,<3" "numpy>=1.26,<3" "scikit-learn>=1.5,<2" "xgboost-cu12>=3.2,<4" "catboost>=1.2,<2" "sentence-transformers>=3,<6" "beautifulsoup4>=4.12,<5"'),
        markdown("If Colab asks for a runtime restart after installation, restart, reselect GPU, and run all cells from the top."),
        code('import torch, subprocess, xgboost as xgb\nassert torch.cuda.is_available(), "Select Runtime > Change runtime type > T4 GPU, then rerun."\nassert xgb.build_info().get("USE_CUDA",False), "Installed XGBoost lacks CUDA; reinstall dependencies and restart the GPU runtime."\nprint("CUDA:",torch.version.cuda,"GPU:",torch.cuda.get_device_name(0),"XGBoost CUDA:",xgb.build_info().get("CUDA_VERSION"))\nsubprocess.run(["nvidia-smi"],check=True)'),
        code('from google.colab import drive\ndrive.mount("/content/drive")\nfrom pathlib import Path\nROOT=Path("/content/drive/MyDrive/financial-risk-detector")\nPROXY=ROOT/"data/colab_amendment_proxy.csv.gz"\nLEDGER=ROOT/"data/restatement_review_queue.csv"\nassert PROXY.is_file(), f"Upload the prepared feature CSV: {PROXY}"\nassert LEDGER.is_file(), f"Upload the selected review ledger as restatement_review_queue.csv: {LEDGER}"\nHORIZON_YEARS=3  # Choose exactly one: 1, 3, or 5\nassert HORIZON_YEARS in (1,3,5)\nOUTPUT=ROOT/f"colab_outputs/confirmed_restatement_{HORIZON_YEARS}y_fulltext_trajectory_v2"\nCACHE=ROOT/f"evidence_restatement/{HORIZON_YEARS}y/sections.sqlite"\nprint("Proxy:",PROXY,"\\nLedger:",LEDGER,"\\nHorizon:",HORIZON_YEARS,"years")'),
        code('from pathlib import Path\nimport sys, importlib, importlib.util\nMODULES=Path("/content/restatement_modules"); MODULES.mkdir(parents=True,exist_ok=True)\nSOURCES='+repr(sources)+'\nfor name,source in SOURCES.items(): (MODULES/name).write_text(source,encoding="utf-8")\nif str(MODULES) not in sys.path: sys.path.insert(0,str(MODULES))\nimport research_v5_core as core; import sec_v5_evidence as evidence\nimportlib.reload(core); importlib.reload(evidence)\nspec=importlib.util.spec_from_file_location("restatement_study",MODULES/"22_restatement_study.py")\nstudy=importlib.util.module_from_spec(spec); spec.loader.exec_module(study)\nprint("Embedded research modules loaded.")'),
        code('import pandas as pd, getpass\nledger=pd.read_csv(LEDGER,dtype={"original_adsh":str},low_memory=False)\nerrors=study.validate_ledger(ledger)\nif errors: raise ValueError("Ledger errors:\\n- "+"\\n- ".join(errors))\nreviewed=ledger.review_status.fillna("").astype(str).str.lower().eq("reviewed").sum()\nprint(f"Queue rows: {len(ledger):,}; reviewed: {reviewed:,}; pending: {len(ledger)-reviewed:,}")\nif not reviewed: raise ValueError("No outcomes are marked reviewed yet. Prepare an evidence-reviewed ledger first; proxy labels cannot substitute.")\nproxy=pd.read_csv(PROXY,dtype={"adsh":str,"cik":str},low_memory=False)\nUSER_AGENT=getpass.getpass("Real SEC User-Agent (name/organization and contact email): ").strip()\nMAX_NEW_PER_SESSION=500  # Resume this cell in later sessions until pending reaches 0; use 0 for all remaining.\nreport=study.collect_reviewed_mda(proxy,ledger,HORIZON_YEARS,CACHE,USER_AGENT,max_new=MAX_NEW_PER_SESSION)\nprint(report)'),
        markdown("""## Prepare data and train

If the preceding cell reports `pending` above zero, rerun it in a later session until all current and prior MD&A sections are cached. Then run the next cell. It retains every horizon-eligible reviewed row, records text coverage, and never adds labels to pending rows."""),
        code('if report["pending"]: raise RuntimeError("SEC text collection is incomplete. Rerun the collection cell until pending is 0.")\nDATASET=ROOT/f"data/confirmed_restatement_h{HORIZON_YEARS}y_fulltext_trajectory_v2.csv.gz"\nprepared=study.prepare_model_data(PROXY,LEDGER,CACHE,HORIZON_YEARS,DATASET)\nprint(prepared)\ncounts=pd.read_csv(DATASET,compression="gzip",usecols=["target_restatement"]).target_restatement.value_counts().to_dict()\nprint("Usable target counts:",counts)'),
        code('metrics=study.train_gpu(DATASET,OUTPUT)\nprint(json.dumps(metrics,indent=2))\nprint("\\nOutput folder:",OUTPUT)\nprint("The run saves six model/view evaluations, their model files, study_report.md, and SHAP summaries when available.")'),
        markdown("""## Research question and reporting

**Suggested title:** *Does MD&A change add predictive value for a later SEC-reported restatement? A three-year matched temporal-holdout study.*

**Primary question:** Do current MD&A language and its change from the prior comparable filing improve prediction beyond structured financial features? The three prespecified views are evaluated on the same forward holdout, with each model and threshold selected using development data only. The main incremental comparison is the paired holdout PR-AUC difference between the financial-plus-text view and the financial baseline.

Report the financial-only, interpretable-language, and semantic-change comparisons. Use `grouped_cv_leaderboard.csv` for development model selection; the frozen forward holdout is reported once. `feature_importance_shap.csv` lists feature attributions, and `feature_group_importance_shap.csv` summarizes them by financial, lexical, embedding, and availability groups. These explain the fitted model; they do not establish causation.

## Reading the result

`balanced_holdout_1_to_1` is the primary matched-sample metric set. It is evaluated on the latest filing years, which were excluded from training and threshold selection, with year/form-matched 1:1 sampling. Its `*_ci95` fields are stratified bootstrap intervals for that sample. Companies can appear in earlier development years; this reflects prediction for continuing issuers, while development folds group issuers. `holdout_adequate_for_claim` requires at least 30 cases per class and 10 distinct issuers per class. If false, treat the metrics as preliminary. `matched_sample_sensitivity_50` shows how the result changes across 50 fixed-threshold 1:1 matches; it is a matching sensitivity analysis, not a confidence interval. `development_cv_1_to_1` selects the model and threshold; do not present it as a final estimate. The curated all-row holdout is not a natural-prevalence estimate because the review queue was sampled. `study_report.md` assembles counts, results, and limitations for the report draft.

If the GPU cell says there are too few positive/negative rows or issuer groups, expand the reviewed ledger first. Do not balance by duplicating rows, include proxy labels as target, or tune on the held-out results. After selecting a horizon and model using development evidence, collect new mature filings for a future external test.""")]
    notebook={"nbformat":4,"nbformat_minor":4,
              "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                          "language_info":{"name":"python"},"colab":{"name":"confirmed_restatement_gpu_colab.ipynb","provenance":[]},"accelerator":"GPU"},
              "cells":cells}
    path=ROOT/"notebooks"/"confirmed_restatement_gpu_colab.ipynb"
    path.write_text(json.dumps(notebook,indent=1),encoding="utf-8")
    print(f"Wrote {path}")


if __name__=="__main__":
    build()
