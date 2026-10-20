"""Build the standalone Colab notebook for the corrected MD&A text experiment."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def md(value):
    return {'cell_type':'markdown','metadata':{},'source':value.strip().splitlines(keepends=True)}


def code(value):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
            'source':value.strip().splitlines(keepends=True)}


def build():
    sources={name:(ROOT/'src'/name).read_text(encoding='utf-8') for name in
             ('research_v5_core.py','17_train_mda_text_gpu.py')}
    cells=[
        md('''# GPU MD&A text experiment - corrected amendment proxy

This experiment uses the existing, selected MD&A corpus; it does not fetch another 111,000 SEC documents. Its labels are rejoined from the current schema-v4 file by accession. The corpus's old balanced `y` label is excluded because it disagreed with the current target for 728 filings.

**Scope:** these outputs measure a year/form matched case-control text sample. The source corpus was selected and approximately balanced, and it has text for only a small part of the full filing population. This notebook therefore does not estimate natural-prevalence precision, recall, or accuracy. Its 2024Q4 to 2025Q3 cohort was already inspected in v5, so this is further development, not a fresh final test. The three-fold leaderboard guides model selection and is not an untouched estimate after comparing candidates.

Use only as an academic amendment-proxy experiment. All model fitting and text embedding run on the Colab GPU. It compares text-only and text-plus-financial XGBoost models with GPU neural models, and adds changes from the previous same-form filing. Text uses pooled embeddings from the beginning, middle, and ending of each supplied MD&A, plus transparent language and document-style counts. Training controls are matched to cases within filing year and form. Early stopping uses an inner year; reported CV scores come from a later outer year.'''),
        code('import torch, subprocess\n'
             'assert torch.cuda.is_available(), "Choose Runtime > Change runtime type > T4 GPU, then rerun."\n'
             'print("CUDA:",torch.version.cuda,"GPU:",torch.cuda.get_device_name(0))\n'
             'subprocess.run(["nvidia-smi"],check=True)'),
        code('%pip -q install "xgboost>=2.1,<4" "scikit-learn>=1.5,<2" "sentence-transformers>=3,<6" "pandas>=2.2,<3" "numpy>=1.26,<3" "pyarrow>=16,<25"'),
        md('If Colab requests a runtime restart after installation, restart, select GPU again, and rerun the cells from the top.'),
        code('from google.colab import drive\ndrive.mount("/content/drive")\n'
             'from pathlib import Path\n'
             'ROOT=Path("/content/drive/MyDrive/financial-risk-detector")\n'
             'for path in (ROOT/"data/colab_amendment_proxy.csv.gz",ROOT/"data/colab_mda_text_case_control.csv.gz"):\n'
             '    assert path.is_file(),f"Missing input file: {path}"\n'
             'print("Both prepared CSV files are ready.")'),
        code('import sys,importlib,importlib.util\nfrom pathlib import Path\n'
             'MODULES=Path("/content/mda_text_modules"); MODULES.mkdir(parents=True,exist_ok=True)\n'
             f'SOURCES={sources!r}\n'
             'for name,source in SOURCES.items(): (MODULES/name).write_text(source,encoding="utf-8")\n'
             'if str(MODULES) not in sys.path: sys.path.insert(0,str(MODULES))\n'
             'import research_v5_core as core\nimportlib.reload(core)\n'
             'spec=importlib.util.spec_from_file_location("mda_text_training",MODULES/"17_train_mda_text_gpu.py")\n'
             'training=importlib.util.module_from_spec(spec); spec.loader.exec_module(training)\n'
             'print("Self-contained training code loaded.")'),
        md('''## Run the GPU experiment

The first run embeds the filing text with a frozen MiniLM encoder and caches the vectors in Drive. Later runs reuse that cache. Completed CV folds and final members are checkpointed with source/data fingerprints, so rerunning after an interruption resumes compatible work. XGBoost and the neural models fit on CUDA; the code stops if a GPU is unavailable or a model falls back to CPU. Forward model selection uses three outer years (2018, 2020, 2022) and earlier inner stop years (2016, 2018, 2020).

The threshold is selected using matched 2023 cases and controls. Development metrics use matched cases and controls by year/form; a 20-draw sensitivity summary keeps the calibrated threshold fixed and varies the matched sample. The notebook also compares the text models with prior v5 scores on the same selected accessions when present. These results do not describe all filings.'''),
        code('OUTPUT=training.run(ROOT)\nprint("Output folder:",OUTPUT)'),
        code('import json,pandas as pd\n'
             'report=json.loads((OUTPUT/"metrics.json").read_text())\n'
             'display(pd.DataFrame({"text experiment":report["development"]}).T)\n'
             'display(pd.DataFrame(report["development_control_sampling_sensitivity"]["summary"]).T)\n'
             'if "v5_tabular_on_same_matched_development" in report:\n'
             '    display(pd.DataFrame({"v5 tabular baseline":report["v5_tabular_on_same_matched_development"]}).T)\n'
             'print("Return metrics.json, cv_leaderboard.csv, and development_predictions_case_control.csv.gz from",OUTPUT)')]
    notebook={'nbformat':4,'nbformat_minor':4,'metadata':{
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'accelerator':'GPU',
        'colab':{'name':'financial_risk_detector_mda_text_gpu_colab.ipynb','provenance':[]}},
        'cells':cells}
    path=ROOT/'notebooks/financial_risk_detector_mda_text_gpu_colab.ipynb'
    path.write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    print(f'Wrote {path}')


if __name__=='__main__':
    build()
