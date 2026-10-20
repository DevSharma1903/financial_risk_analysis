"""Build self-contained notebooks from the reviewed source modules. No fitting."""
from pathlib import Path
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def markdown(text):
    return {'cell_type':'markdown','metadata':{},'source':text.strip().splitlines(keepends=True)}


def code(text):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
            'source':text.strip().splitlines(keepends=True)}


def embedded(names):
    sources={name:(ROOT/'src'/name).read_text(encoding='utf-8') for name in names}
    return code('from pathlib import Path\nimport sys, importlib\n'
                'MODULES = Path("/content/research_v5_modules")\n'
                'MODULES.mkdir(parents=True, exist_ok=True)\n'
                f'SOURCES = {sources!r}\n'
                'for name, source in SOURCES.items():\n'
                '    (MODULES/name).write_text(source, encoding="utf-8")\n'
                'if str(MODULES) not in sys.path:\n'
                '    sys.path.insert(0, str(MODULES))\n'
                'import research_v5_core as core\nimport sec_v5_evidence as evidence\n'
                'importlib.reload(core)\nimportlib.reload(evidence)\n'
                'print("Embedded source modules ready.")')


def write_notebook(name,cells,gpu=False):
    metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
              'language_info':{'name':'python'},'colab':{'name':name,'provenance':[]}}
    if gpu:
        metadata['accelerator']='GPU'
    notebook={'nbformat':4,'nbformat_minor':4,'metadata':metadata,'cells':cells}
    path=ROOT/'notebooks'/name
    path.write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    return path


def build():
    mount=code('from google.colab import drive\n'
               'drive.mount("/content/drive")\n'
               'from pathlib import Path\n'
               'ROOT = Path("/content/drive/MyDrive/financial-risk-detector")\n'
               'DATASET = ROOT/"data/colab_amendment_proxy.csv.gz"\n'
               'if not DATASET.exists():\n'
               '    DATASET = ROOT/"colab_amendment_proxy.csv.gz"\n'
               'assert DATASET.is_file(), f"Prepared schema-v4 CSV missing: {DATASET}"\n'
               'print("Reusing:", DATASET)')
    evidence_cells=[
        markdown('''# 1. Research v5 — prepare public SEC evidence

Run this notebook first. Reuse the prepared CSV already in Drive; **do not upload raw data**.
This notebook fits no models, so a CPU runtime is sufficient. Model training happens in notebook 2 on GPU.

The required step caches 67 complete quarterly SEC master indexes (2009Q1–2025Q3), keeping only relevant filing records. Original documents are optional and take much longer. Downloads are sequential, identified, limited to two requests/second, resumable, and stop on SEC access denial. Use a real name/organization and contact email at the prompt. No identity is saved in outputs.

These features enrich the existing amendment proxy; they do not turn it into a confirmed-restatement label. Academic use only. [SEC access guidance](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data).'''),
        code('%pip -q install "pandas>=2.2,<3" "numpy>=1.26,<3" "pyarrow>=16,<25" "beautifulsoup4>=4.12,<5"'),
        mount,
        embedded(['research_v5_core.py','sec_v5_evidence.py']),
        markdown('''## Required: filing-history indexes

Run the next cell once; reruns skip cached quarters. If the SEC returns 403, stop and retry later on an allowed connection. No bypass is attempted. The same downloader can run locally with `python src/sec_v5_evidence.py --root .`; then copy only `evidence_v5/indexes/` to the project folder in Drive.'''),
        code('import getpass, os\n'
             'client = evidence.SecClient(os.environ.get("SEC_USER_AGENT") or\n'
             '                            getpass.getpass("Real SEC User-Agent (name/organization and email): "))\n'
             'events = evidence.load_sec_history(ROOT/"evidence_v5/indexes", client)\n'
             'print(f"READY: {len(events):,} relevant indexed filings.")\n'
             'print("Next: open financial_risk_detector_v5_gpu_colab.ipynb with GPU enabled.")\n'
             'del events'),
        markdown('''## Optional: original filing text

For the first experiment, keep `DOWNLOAD_TEXT=False` and go to notebook 2. The historical features and seven model strategies work without text.

For the additional two disclosure models, set `DOWNLOAD_TEXT=True`. This needs many original documents; the cell prints the exact manifest size. `MAX_NEW_TEXT=3000` limits each session; rerun this cell to continue. Set it to `0` to process all pending requests in one session. You can interrupt and resume; progress is committed every 25 records. Cached excerpts are capped at 10,000 characters per filing, not whole raw submissions. Allow additional Drive space for that cache.

Enable `USE_TEXT=True` in notebook 2 only after `pending` reaches zero and at least 90% of documents are usable. Failures remain in evaluation with missing-text flags. Embeddings are computed on GPU in notebook 2. The encoder is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache-2.0, with its revision recorded.'''),
        code('DOWNLOAD_TEXT = False\nMAX_NEW_TEXT = 3000\n'
             'if DOWNLOAD_TEXT:\n'
             '    df = core.load_data(DATASET)\n'
             '    manifest = core.text_manifest(df)\n'
             '    del df\n'
             '    manifest.to_csv(ROOT/"evidence_v5/text_manifest.csv.gz", index=False, compression="gzip")\n'
             '    coverage = evidence.collect_text(manifest, ROOT/"evidence_v5/original_text.sqlite",\n'
             '                                     client, max_new=MAX_NEW_TEXT)\n'
             'else:\n'
             '    print("Text skipped. Use USE_TEXT=False in the GPU notebook.")')]
    gpu_cells=[
        markdown('''# 2. Research v5 — GPU model comparison

First complete the required index step in notebook 1. Reuse your existing schema-v4 CSV in Drive. No raw-data upload or new CSV preparation is needed.

1. Select **Runtime → Change runtime type → T4 GPU** (or another available NVIDIA GPU), then Save.
2. Run cells in order, approve Drive mounting, and leave `USE_TEXT=False` for the first experiment.
3. Keep the runtime connected. Completed model members are checkpointed to Drive. After a disconnect, choose GPU again and rerun all cells; unchanged completed members resume.

Seven strategies compare multiple balanced samples, recent history, XGBoost, CatBoost, a neural network, separate 10-K/10-Q models, and all controls with equal total class weight. An optional text experiment adds two strategies. **No CPU training fallback.** PyTorch uses Colab's installed CUDA build.

This is an attempt to improve performance, not a promise of 80–85%. The 2024Q4–2025Q3 cohort was already inspected, so these are development comparisons. Thresholds use 2023 calibration labels only; both balanced and full-population results are saved.'''),
        code('import torch\nimport subprocess\n'
             'assert torch.cuda.is_available(), "Select Runtime > Change runtime type > GPU, then rerun."\n'
             'print("CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))\n'
             'subprocess.run(["nvidia-smi"], check=True)'),
        code('%pip -q install "xgboost>=2.1,<4" "catboost>=1.2.7,<2" "scikit-learn>=1.5,<2" "pandas>=2.2,<3" "numpy>=1.26,<3" "pyarrow>=16,<25" "beautifulsoup4>=4.12,<5"'),
        markdown('If Colab requests a runtime restart after installation, restart, then run cells again with GPU enabled.'),
        mount,
        code('USE_TEXT = False  # True only after optional evidence collection is complete.\n'
             'if USE_TEXT:\n'
             '    import sys, subprocess\n'
             '    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",\n'
             '                           "sentence-transformers>=3,<6"])'),
        embedded(['research_v5_core.py','sec_v5_evidence.py','14_train_v5_colab.py']),
        markdown('''## Train and evaluate

All model fits explicitly use CUDA. CPU activity during CSV reading, feature preparation, thresholds, and reporting is expected. Evaluation models use iteration counts selected by chronological validation; early stopping never sees the development cohort.

Outputs go to `MyDrive/financial-risk-detector/colab_outputs/research_v5_<fingerprint>/`. Fingerprints include data, code, dependencies, evidence, and GPU type. Changing those starts a new run. A checkpoint resumes complete model members; an interrupted member is retrained. Do not change settings based on development scores and then call that cohort an untouched test.'''),
        code('import importlib.util\n'
             'spec = importlib.util.spec_from_file_location("training_v5", MODULES/"14_train_v5_colab.py")\n'
             'training = importlib.util.module_from_spec(spec)\n'
             'spec.loader.exec_module(training)\n'
             'OUTPUT = training.run(ROOT, use_text=USE_TEXT)'),
        code('import json\nimport pandas as pd\n'
             'metrics = json.loads((OUTPUT/"metrics.json").read_text())\n'
             'summary = pd.DataFrame({name: metrics[name]["development"]\n'
             '                        for name in ("balanced", "natural")}).T\n'
             'display(summary[["n", "prevalence", "accuracy", "precision", "recall", "f1",\n'
             '                 "pr_auc", "roc_auc", "all_four_metrics_ge_80", "all_four_metrics_ge_85"]])\n'
             'print("Saved to:", OUTPUT)\n'
             'print("Return metrics.json, cv_leaderboard.csv, provenance.json, and development_predictions.csv.gz.")'),
        markdown('''## Read the results

`balanced` uses one control per positive. `natural` includes every eligible filing. Read precision, recall, F1, and accuracy together with prevalence and the confusion matrix; high natural accuracy alone is not success. `issuer_bootstrap_95pct` reports uncertainty. No difficult cases are removed to inflate results.

The target remains an FSDS-observed same-period amendment within 365 days. These are academic screening experiments, not confirmed-restatement or fraud determinations. Public SEC access follows the declared-identity and throttling guidance; do not bypass denied access.''')]
    notebooks=[write_notebook('financial_risk_detector_v5_evidence_colab.ipynb',evidence_cells),
               write_notebook('financial_risk_detector_v5_gpu_colab.ipynb',gpu_cells,gpu=True)]
    archive=ROOT/'colab_v5_run_files.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as output:
        for path in notebooks+[ROOT/'src'/n for n in ('research_v5_core.py','sec_v5_evidence.py',
                                                    '14_train_v5_colab.py','15_build_v5_notebooks.py')]+[ROOT/'README.md']:
            output.write(path,path.relative_to(ROOT).as_posix())
    print(f'Built two self-contained notebooks and {archive.name} ({archive.stat().st_size:,} bytes).')


if __name__=='__main__':
    build()
