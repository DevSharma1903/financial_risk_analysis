"""Build self-contained preparation and GPU-training notebooks for expanded MD&A."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def markdown(value):
    return {'cell_type':'markdown','metadata':{},'source':value.strip().splitlines(keepends=True)}


def code(value):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
            'source':value.strip().splitlines(keepends=True)}


def embedded(names, directory, require_sec_evidence=True):
    sources={name:(ROOT/'src'/name).read_text(encoding='utf-8') for name in names}
    imports=('import research_v5_core as core\n'
             'importlib.reload(core)\n')
    if require_sec_evidence:
        imports += ('import sec_v5_evidence as evidence\n'
                    'importlib.reload(evidence)\n')
    return code('from pathlib import Path\nimport sys, importlib, importlib.util\n'
        f'MODULES=Path("/content/{directory}"); MODULES.mkdir(parents=True,exist_ok=True)\n'
        f'SOURCES={sources!r}\n'
        'for name,source in SOURCES.items(): (MODULES/name).write_text(source,encoding="utf-8")\n'
        'if str(MODULES) not in sys.path: sys.path.insert(0,str(MODULES))\n'
        f'{imports}'
        'print("Embedded research code loaded.")')


def write(name, cells, gpu=False):
    metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
              'language_info':{'name':'python'},'colab':{'name':name,'provenance':[]}}
    if gpu:
        metadata['accelerator']='GPU'
    path=ROOT/'notebooks'/name
    path.write_text(json.dumps({'nbformat':4,'nbformat_minor':4,
        'metadata':metadata,'cells':cells},indent=1),encoding='utf-8')
    return path


def build():
    prepare_cells=[
        markdown('''# Prepare full SEC MD&A sections (data collection only)

This notebook fetches only the original filings in the project’s fixed text manifest, using the public SEC archive. It extracts 10-K Item 7 and 10-Q Item 2. It fits no models and does not change proxy labels. The cache resumes across Colab sessions. Use a normal CPU runtime for this step; model fitting is in the companion GPU notebook.

SEC requests are sequential and limited to two per second. Enter your real identifying User-Agent when prompted. If the SEC returns 403, the request code stops; do not bypass that denial. The default is 3,000 filings per session so each run can checkpoint to Drive.'''),
        code('%pip -q install "pandas>=2.2,<3" "numpy>=1.26,<3" "pyarrow>=16,<25" "beautifulsoup4>=4.12,<5"'),
        markdown('If Colab requests a runtime restart after installation, restart and run all setup cells again.'),
        code('from google.colab import drive\ndrive.mount("/content/drive")\n'
             'from pathlib import Path\nROOT=Path("/content/drive/MyDrive/financial-risk-detector")\n'
             'assert (ROOT/"data/colab_amendment_proxy.csv.gz").is_file(), "Upload the prepared proxy CSV to Drive/data first."\n'
             'print("Drive root:",ROOT)'),
        embedded(['research_v5_core.py','sec_v5_evidence.py','19_mda_advanced.py'],'mda_section_modules'),
        code('spec=importlib.util.spec_from_file_location("mda_advanced",MODULES/"19_mda_advanced.py")\n'
             'prepare=importlib.util.module_from_spec(spec); spec.loader.exec_module(prepare)\n'
             'import getpass, json\n'
             'USER_AGENT=getpass.getpass("SEC User-Agent (real name/organization and contact email): ").strip()\n'
             'MAX_NEW_PER_SESSION=3000  # Set to 0 only if you intend to fetch every remaining filing this session.\n'
             'report=prepare.collect_sections(ROOT,USER_AGENT,max_new=MAX_NEW_PER_SESSION)\n'
             'if report["pending"]==0:\n'
             '    dataset=prepare.export_dataset(ROOT)\n'
             '    report["dataset"]=str(dataset); report["dataset_bytes"]=dataset.stat().st_size\n'
             'print(json.dumps(report,indent=2))\n'
             'if report["pending"]:\n'
             '    print("Progress is saved in Drive. Rerun this cell in a later session until pending is 0.")'),
        markdown('''## Resume and finish

When `pending` is nonzero, rerun the last cell in a later Colab session. Completed rows and manifest fingerprints are cached under `evidence_v5/mda_sections/`. Once the manifest is complete, the notebook writes `data/colab_mda_text_expanded.csv.gz`. Missing or unparseable sections remain as rows with an explicit status and empty text; they are not relabeled or silently removed.''')]
    prepare_path=write('financial_risk_detector_mda_sections_prepare_colab.ipynb',prepare_cells)

    train_cells=[
        markdown('''# Expanded MD&A experiments — Colab GPU only

This run expands the supplied MD&A sample toward the 111,189-row fixed text manifest. It adds strict Item 7/Item 2 text, document-max embedding pooling, prior same-issuer/same-form text similarity and embedding changes, word/character TF-IDF, and an XGBoost pairwise ranker grouped by filing year and form. All model fitting and sentence embedding require CUDA; there is no CPU fitting fallback.

The headline results remain year/form matched 1:1. The 2024Q4–2025Q3 period has already been inspected, so it remains development data. A stable 20% CIK holdout is excluded from model selection and fitting and reported separately as an exploratory issuer-disjoint check; because the dates have already been examined in aggregate, it is not a pristine test. Scores are academic amendment-proxy results, not confirmed-restatement or natural-population estimates.'''),
        code('import torch, subprocess\nassert torch.cuda.is_available(), "Select Runtime > Change runtime type > GPU and rerun."\n'
             'print("CUDA:",torch.version.cuda,"GPU:",torch.cuda.get_device_name(0))\n'
             'subprocess.run(["nvidia-smi"],check=True)'),
        code('%pip -q uninstall -y xgboost xgboost-cu12\n'
             '%pip -q install "xgboost-cu12>=3.2,<4" "scikit-learn>=1.5,<2" "sentence-transformers>=3,<6" "pandas>=2.2,<3" "numpy>=1.26,<3" "pyarrow>=16,<25" "scipy>=1.12,<2" "joblib>=1.3,<2"'),
        markdown('If Colab requests a runtime restart after installing packages, restart, select GPU again, and run all cells from the top.'),
        code('from google.colab import drive\ndrive.mount("/content/drive")\n'
             'from pathlib import Path\nROOT=Path("/content/drive/MyDrive/financial-risk-detector")\n'
             'EXPANDED_TEXT_PATH=ROOT/"data/colab_mda_text_expanded.csv.gz"\n'
             'SAMPLED_TEXT_PATH=ROOT/"data/colab_mda_text_case_control.csv.gz"\n'
             'TEXT_PATH=EXPANDED_TEXT_PATH if EXPANDED_TEXT_PATH.is_file() else SAMPLED_TEXT_PATH\n'
             'PROXY_PATH=ROOT/"data/colab_amendment_proxy.csv.gz"\n'
             'assert TEXT_PATH.is_file(), "Upload the existing case-control text CSV, or finish the companion SEC section-preparation notebook."\n'
             'assert PROXY_PATH.is_file(), f"Missing current proxy labels: {PROXY_PATH}"\n'
             'print("Text input:",TEXT_PATH,"rows are read by the training pipeline; expanded sections are preferred when present.")'),
        embedded(['research_v5_core.py','17_train_mda_text_gpu.py'],
                 'mda_advanced_gpu_modules',require_sec_evidence=False),
        code('spec=importlib.util.spec_from_file_location("mda_training",MODULES/"17_train_mda_text_gpu.py")\n'
             'training=importlib.util.module_from_spec(spec); spec.loader.exec_module(training)\n'
             'OUTPUT=training.run(ROOT,text_path=TEXT_PATH,experiment_name="mda_text_expanded")\n'
             'print("Output folder:",OUTPUT)'),
        code('import json,pandas as pd\n'
             'report=json.loads((OUTPUT/"metrics.json").read_text())\n'
             'display(pd.DataFrame({"development":report["development"],\n'
             '                     "issuer-disjoint development":report["issuer_disjoint_development"]}).T)\n'
             'display(pd.DataFrame(report["development_control_sampling_sensitivity"]["summary"]).T)\n'
             'print("Return metrics.json, cv_leaderboard.csv, development_predictions_case_control.csv.gz, and issuer_disjoint_predictions_case_control.csv.gz from",OUTPUT)')]
    train_path=write('financial_risk_detector_mda_advanced_gpu_colab.ipynb',train_cells,gpu=True)
    print(f'Wrote {prepare_path}\nWrote {train_path}')


if __name__=='__main__':
    build()
