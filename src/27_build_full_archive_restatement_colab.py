"""Build a self-contained Colab GPU notebook using the full MD&A text archive."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


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
        markdown("""# Full-archive MD&A/restatement study — Colab GPU

This version adds the compact historical MD&A text archive to the confirmed-restatement study. It uses every eligible archive document filed before the forward holdout to fit an unlabeled TF-IDF/SVD text representation. It never reads the archive's legacy amendment-proxy labels. Restatement labels still come only from the reviewed outcome ledger.

The study target is an issuer-level SEC-reported financial-statement restatement within three years after an original 10-K/10-Q. It tests whether MD&A language adds predictive information beyond structured financial features. Reviewed examples from earlier years form the development/training data; later years remain untouched for evaluation. The 1:1 metric set and all-row holdout are both reported. No metric level is guaranteed.

The full text archive was selected for an earlier amendment-proxy experiment, so it is not a natural-population sample. It is used only to learn unlabeled text vocabulary/topics; its proxy labels are excluded. The notebook also retrieves missing reviewed original/prior MD&A sections from SEC and resumes from the Drive cache. AI-assisted restatement labels remain exploratory pending human audit."""),
        code('%pip -q uninstall -y xgboost xgboost-cu12\n%pip -q install "pandas>=2.2,<3" "numpy>=1.26,<3" "scikit-learn>=1.5,<2" "xgboost-cu12>=3.2,<4" "catboost>=1.2,<2" "sentence-transformers>=3,<6" "beautifulsoup4>=4.12,<5"'),
        markdown("If Colab requests a runtime restart, restart it, reselect GPU, then run all cells from the top."),
        code('import torch, subprocess, xgboost as xgb\nassert torch.cuda.is_available(), "Select Runtime > Change runtime type > GPU, then rerun."\nassert xgb.build_info().get("USE_CUDA",False), "Installed XGBoost lacks CUDA; reinstall dependencies and restart the GPU runtime."\nprint("CUDA:",torch.version.cuda,"GPU:",torch.cuda.get_device_name(0),"XGBoost CUDA:",xgb.build_info().get("CUDA_VERSION"))\nsubprocess.run(["nvidia-smi"],check=True)'),
        code('from google.colab import drive\ndrive.mount("/content/drive")\nfrom pathlib import Path\nROOT=Path("/content/drive/MyDrive/financial-risk-detector")\nPROXY=ROOT/"data/colab_amendment_proxy.csv.gz"\nLEDGER=ROOT/"data/restatement_review_queue.csv"\nARCHIVE=ROOT/"data/colab_mda_archive_unlabeled.csv.gz"\nfor path in (PROXY,LEDGER,ARCHIVE): assert path.is_file(), f"Upload required input: {path}"\nHORIZON_YEARS=3\nOUTPUT=ROOT/f"colab_outputs/confirmed_restatement_full_archive_{HORIZON_YEARS}y_fulltext_trajectory_v2"\nCACHE=ROOT/f"evidence_restatement/{HORIZON_YEARS}y/sections.sqlite"\nprint("Proxy:",PROXY,"\\nLedger:",LEDGER,"\\nUnlabeled MD&A archive:",ARCHIVE,"\\nOutput:",OUTPUT)'),
        code('from pathlib import Path\nimport sys, importlib, importlib.util\nMODULES=Path("/content/restatement_modules_full_archive"); MODULES.mkdir(parents=True,exist_ok=True)\nSOURCES='+repr(sources)+'\nfor name,source in SOURCES.items(): (MODULES/name).write_text(source,encoding="utf-8")\nif str(MODULES) not in sys.path: sys.path.insert(0,str(MODULES))\nimport research_v5_core as core; import sec_v5_evidence as evidence\nimportlib.reload(core); importlib.reload(evidence)\nspec=importlib.util.spec_from_file_location("restatement_study_full_archive",MODULES/"22_restatement_study.py")\nstudy=importlib.util.module_from_spec(spec); spec.loader.exec_module(study)\nprint("Embedded full-archive study modules loaded.")'),
        code('import pandas as pd, getpass\nledger=pd.read_csv(LEDGER,dtype={"original_adsh":str},low_memory=False)\nerrors=study.validate_ledger(ledger)\nif errors: raise ValueError("Ledger errors:\\n- "+"\\n- ".join(errors))\nreviewed=ledger.review_status.fillna("").astype(str).str.lower().eq("reviewed").sum()\nprint(f"Queue rows: {len(ledger):,}; reviewed: {reviewed:,}; pending outcome reviews: {len(ledger)-reviewed:,}")\nif not reviewed: raise ValueError("No reviewed outcome rows are available.")\narchive=pd.read_csv(ARCHIVE,compression="gzip",usecols=["adsh","filed_date","mda_word_count"],low_memory=False)\nprint(f"Unlabeled archive: {len(archive):,} filings; years {pd.to_datetime(archive.filed_date).dt.year.min()}–{pd.to_datetime(archive.filed_date).dt.year.max()}; usable documents: {(archive.mda_word_count>=200).sum():,}")\nproxy=pd.read_csv(PROXY,compression="gzip",dtype={"adsh":str,"cik":str},low_memory=False)\nUSER_AGENT=getpass.getpass("Real SEC User-Agent (name/organization and contact email): ").strip()\nMAX_NEW_PER_SESSION=500\nreport=study.collect_reviewed_mda(proxy,ledger,HORIZON_YEARS,CACHE,USER_AGENT,max_new=MAX_NEW_PER_SESSION)\nprint(report)'),
        markdown("If text collection reports pending documents, rerun the setup and collection cells in a later session until pending is 0. The Drive cache resumes the work. Stop on SEC 403/429; do not retry rapidly or bypass access controls."),
        code('if report["pending"]: raise RuntimeError("SEC text collection is incomplete; resume the collection cell until pending is 0.")\nDATASET=ROOT/f"data/confirmed_restatement_h{HORIZON_YEARS}y_fulltext_trajectory_v2.csv.gz"\nprepared=study.prepare_model_data(PROXY,LEDGER,CACHE,HORIZON_YEARS,DATASET)\nprint(prepared)\ncounts=pd.read_csv(DATASET,compression="gzip",usecols=["target_restatement"]).target_restatement.value_counts().to_dict()\nprint("Usable reviewed target counts:",counts)'),
        code('import json\nmetrics=study.train_gpu(DATASET,OUTPUT,archive_path=ARCHIVE)\nprint(json.dumps(metrics,indent=2))\nprint("\\nOutputs:",OUTPUT)\nprint("The run saves up to eight GPU model/view evaluations, holdout predictions, study report, model files, and SHAP summaries.")'),
        markdown("""## Reading the results

`full_archive_text_features` in `metrics.json` records archive rows, rows used before the holdout, cutoff year, vocabulary size, and confirms that legacy proxy labels were not read. `full_archive` is the financial + current/prior MiniLM + lexical-change + archive-topic feature view. Compare it with `financial`, `interpretable`, and `fusion` on the same future holdout. The primary question is whether text improves PR-AUC over the financial baseline.

`grouped_cv_leaderboard.csv` is for development model selection; the temporal holdout is evaluated once. `future_holdout_predictions.csv`, `study_report.md`, `feature_importance_shap.csv`, and `feature_group_importance_shap.csv` are also saved. SHAP describes fitted model associations; it does not establish causation.

The archive supplies unlabeled text representation learning, not extra restatement labels. Only reviewed outcomes enter the supervised task. The curated cohort and AI-assisted labels limit claims; report the matched 1:1 and full holdout metrics with their intervals, and do not describe them as population-level or human-validated.""")]
    notebook={"nbformat":4,"nbformat_minor":4,
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                    "language_info":{"name":"python"},
                    "colab":{"name":"confirmed_restatement_full_archive_gpu_colab.ipynb","provenance":[]},
                    "accelerator":"GPU"},"cells":cells}
    path=ROOT/"notebooks"/"confirmed_restatement_full_archive_gpu_colab.ipynb"
    path.write_text(json.dumps(notebook,indent=1),encoding="utf-8")
    print(f"Wrote {path}")


if __name__=="__main__":
    build()
