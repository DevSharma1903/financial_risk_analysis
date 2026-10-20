# Financial Risk Detector: academic SEC amendment study

Large SEC datasets, cached filings, compressed exports, trained model artifacts, and local credentials are intentionally excluded from Git. Follow the data paths below when preparing a local or Colab run; the source code and clean notebooks are included in the repository.

## What the model predicts

The target is whether an original SEC 10-K or 10-Q receives a **same-period 10-K/A or 10-Q/A within 365 days**. This is an amendment proxy, not a confirmed restatement, fraud label, or investment signal. An amendment can have many causes. Use the outputs only for academic research, not investment, lending, audit, or operational decisions.

The study uses every positive filing and one randomly selected negative from the same filing year and form in each balanced cohort. It also reports performance on **all** filings in the later cohort, where amendments are rare. A high balanced accuracy or precision does not imply high natural-prevalence precision. There is no honest guarantee that precision, recall, F1, or accuracy will exceed 85%.

## Current experiment: research v5 (reuse the CSV already on Drive)

The uploaded v4 run has 72.57% balanced accuracy, 70.06% precision, 78.82% recall, and 74.18% F1. At natural prevalence its selected threshold yields 85.65% accuracy, 4.73% precision, and 53.13% recall. These are the baseline results, **not new v5 scores**. V5 has to be trained on Colab before its improvement can be measured.

1. Use the v5 notebooks included in this repository. Leave `MyDrive/financial-risk-detector/data/colab_amendment_proxy.csv.gz` where it is. **Do not upload the 27 GB raw folder or prepare the CSV again.** The notebooks embed their source code; separate script uploads are unnecessary.
2. In Colab choose **File > Upload notebook**, and open [notebooks/financial_risk_detector_v5_evidence_colab.ipynb](notebooks/financial_risk_detector_v5_evidence_colab.ipynb). Run the cells in order and mount Drive. A CPU runtime is sufficient for evidence collection.
3. Enter a real SEC User-Agent (name/organization and contact email) when prompted. The required step caches 67 quarterly master indexes, retaining relevant filing records. Keep `DOWNLOAD_TEXT=False` for the first run. It prints `READY` when the required evidence is complete. An interrupted download resumes cached quarters.
4. Open [notebooks/financial_risk_detector_v5_gpu_colab.ipynb](notebooks/financial_risk_detector_v5_gpu_colab.ipynb). Select **Runtime > Change runtime type > T4 GPU** (or another NVIDIA GPU). Run cells in order. Keep `USE_TEXT=False`. The first cell confirms CUDA and prints `nvidia-smi`; every training backend explicitly uses GPU and refuses CPU fallback.
5. If Colab requests a restart after installing dependencies, restart and run the cells again. Completed model members resume from Drive when code, data, dependencies and GPU type match. The pipeline leaves v4 outputs intact and writes to `colab_outputs/research_v5_<fingerprint>/`.
6. Return `metrics.json`, `cv_leaderboard.csv`, `provenance.json`, and `development_predictions.csv.gz` from the printed output directory. In `metrics.json`, inspect `balanced.development` and `natural.development`, together with confidence intervals and `all_four_metrics_ge_80`/`all_four_metrics_ge_85`.

If SEC access is denied, the collector stops. Retry later on an allowed connection; do not bypass the denial. The evidence collector also works locally without model fitting: `python src/sec_v5_evidence.py --root .`. It uses `SEC_USER_AGENT` if set, otherwise prompts. Copy only the resulting `evidence_v5/indexes/` directory into `MyDrive/financial-risk-detector/evidence_v5/indexes/`. Local prerequisites are pandas, numpy, pyarrow and beautifulsoup4.

### What v5 changes

- Complete public filing-index history adds strictly earlier 8-K, late-filing notice, and amendment counts across several time windows. The existing FSDS proxy labels are retained.
- Prior issuer outcomes enter features only after their entire 365-day observation window has closed. Financial features add three-filing trends, deviations, missingness, and comparisons to the prior year's industry peers.
- Seven strategies compare a v4-feature reference, three balanced negative samples, a recent-years model, CatBoost, separate 10-K/10-Q specialists, a GPU neural network, and an all-controls model with equal total positive/negative weight. The reference uses v4 features; it is not a byte-for-byte reproduction of the prior training run.
- Forward validation selects a single model or an average of the top three. Balanced and natural-prevalence objectives select their models separately. Calibration chooses the threshold maximizing the worst of accuracy, precision, recall and F1.
- Completed model members, feature schemas, GPU confirmation, dependency versions, source/data fingerprints and prediction files are saved. Issuer-level bootstrap intervals show uncertainty. No model is fitted locally.

Optional text experiment: in the evidence notebook set `DOWNLOAD_TEXT=True`. Rerun its last cell until `pending` is zero (`MAX_NEW_TEXT=3000` per session; `0` means all remaining). It fetches only original submissions and extracts compact disclosure windows; it excludes amendments, exhibits and SEC headers. The current CSV produces a manifest of **111,189 filings**: at two requests/second, downloads alone require at least 15.4 hours before network delays, plus additional cache space. This is unnecessary for the first experiment. At least 90% usable text is required, and missing documents remain in evaluation. Then set `USE_TEXT=True` in the GPU notebook. A frozen, revision-recorded [Apache-2.0 MiniLM encoder](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) creates GPU embeddings, and two additional disclosure models are compared. These are keyword-centered excerpts, not verified SEC item sections.

### Standalone MD&A text experiment (prepared corpus already local)

Use [notebooks/financial_risk_detector_mda_text_gpu_colab.ipynb](notebooks/financial_risk_detector_mda_text_gpu_colab.ipynb) for the existing `data/colab_mda_text_case_control.csv.gz`; upload it and `data/colab_amendment_proxy.csv.gz` to Drive under `data/`. The experiment uses the current accession-matched proxy labels and ignores the corpus' legacy `y`. It compares GPU XGBoost and neural models with text-only and text-plus-financial features, including point-in-time changes from the previous same-form filing. Training controls are matched by filing year and form. Early stopping occurs on an inner year, while model selection uses later outer years; those folds select models and are not a pristine score estimate after trying candidates. The report includes a 20-draw matched-sample sensitivity check with a fixed calibrated threshold. Completed GPU folds and final model members are checkpointed by source/data fingerprint for compatible resume after interruption. Outputs remain selected case-control development metrics, not natural-population estimates or a fresh final test.

### Expanded MD&A and confirmed-label pilot

The expanded experiment adds bounded Item 7 (10-K) / Item 2 (10-Q) extraction, text novelty and embedding changes against a strictly earlier same-issuer/form filing, streamed full-document embeddings, sparse word/character TF-IDF, a GPU pairwise ranker, and separate 10-K/10-Q model specialists. It calibrates optional form-specific thresholds on 2023 only, with a minimum group-size fallback. It also permanently excludes legacy target aliases from predictors. Results are still measured against the existing amendment proxy; this cannot turn proxy labels into confirmed restatements or guarantee higher scores.

1. The notebooks are already generated at [notebooks/financial_risk_detector_mda_sections_prepare_colab.ipynb](notebooks/financial_risk_detector_mda_sections_prepare_colab.ipynb) and [notebooks/financial_risk_detector_mda_advanced_gpu_colab.ipynb](notebooks/financial_risk_detector_mda_advanced_gpu_colab.ipynb). After changing their source scripts, regenerate them with `python src/21_build_mda_advanced_colab.py`.
2. Keep `data/colab_amendment_proxy.csv.gz` on Drive at `MyDrive/financial-risk-detector/data/`. No 27 GB raw-data upload is needed. For a run right away, also upload the existing `data/colab_mda_text_case_control.csv.gz`; the GPU notebook uses expanded text if it exists, otherwise it uses this existing sample.
3. To expand the text, open the preparation notebook with a CPU runtime, mount Drive, enter a real SEC User-Agent when prompted, and run the cells. It fetches only the fixed manifest's original public filings, at no more than two requests/second, and stops on HTTP 403. The manifest has about 111,189 filings: the request floor is about 15.4 hours, with no guarantee about network or filing response time. The notebook checkpoints into `evidence_v5/mda_sections/` in Drive; rerun its collection cell across sessions until `pending` is zero. It then writes `data/colab_mda_text_expanded.csv.gz`. Monitor available Drive storage while the compressed filing cache accumulates; expansion is optional.
4. Open the advanced GPU notebook, select **Runtime > Change runtime type > T4 GPU**, and run the cells in order. It checks CUDA before training and each model trainer refuses a CPU fallback. If Colab asks for a runtime restart after package installation, select GPU again and rerun from the top. The output path is printed; return `metrics.json`, `cv_leaderboard.csv`, `development_predictions_case_control.csv.gz`, and (when available) `issuer_disjoint_predictions_case_control.csv.gz`.
5. The new issuer-disjoint score is exploratory because the date range was previously inspected. Do not present it or the matched 1:1 results as a fresh independent estimate. Do not tune against these metrics and then call them untouched test performance.

For a distinct confirmed-restatement target, `python src/20_prepare_restatement_pilot.py --n-each 150` creates `data/restatement_review_pilot.csv` with 150 amendment candidates and 150 proxy-negative controls. The supplied local pilot is already generated. Every label is intentionally pending: a reviewer must check primary SEC evidence and document the event type/date/link before using these rows. This pilot is separate from proxy training and should not be used to select a model until adjudication is complete.

### Evaluation boundaries

V5 validates on 2020 and 2022, with training ending in 2018 and 2020 respectively. Final evaluation models train through 2021, calibrate thresholds on 2023Q1–Q3, and evaluate on 2024Q4–2025Q3. The gaps allow 365-day labels to mature. The latter cohort **has now been inspected in v4**, so v5 calls it development data; it cannot establish untouched future performance. No 80–85% result is promised or fabricated. A new independent evaluation will be necessary before making a generalization claim.

Current SEC index snapshots can reflect subsequent corrections or removals, and history begins in 2009. Original text availability is reported by year and label. Public data access is sequential, at most two requests/second, with a declared identity, backoff, and a stop on HTTP 403. See [SEC access and index guidance](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data). These controls support responsible academic use; they are not a blanket legal certification.

## Previous v4 workflow: prepare from local raw data

The existing `data/raw/` directory is about 27 GB. Preparing the CSV locally avoids uploading that whole tree to Drive; **all model fitting still happens only on the Colab GPU**. From PowerShell in the project directory:

```powershell
$env:SEC_USER_AGENT = "Your Real Name your_email@example.com"
python src/12_update_sec_quarters.py
python src/09_prepare_colab_dataset.py
```

Upload `data/colab_amendment_proxy.csv.gz` to `MyDrive/financial-risk-detector/data/colab_amendment_proxy.csv.gz`. Then open [notebooks/financial_risk_detector_gpu_colab.ipynb](notebooks/financial_risk_detector_gpu_colab.ipynb) in Colab, select **Runtime > Change runtime type > GPU**, and run all cells. It checks CUDA before fitting. Read `MyDrive/financial-risk-detector/colab_outputs/schema_v4_final/metrics.json` afterward.

## Alternative: prepare and train in Colab

1. In Google Drive, create `MyDrive/financial-risk-detector/`. Upload the existing `data/raw/` tree to `MyDrive/financial-risk-detector/data/raw/`. It should contain all 67 quarters from `2009Q1` through `2025Q3`, each with `sub.txt` and `num.txt`.
2. Upload [src/09_prepare_colab_dataset.py](src/09_prepare_colab_dataset.py) and [src/12_update_sec_quarters.py](src/12_update_sec_quarters.py) to `MyDrive/financial-risk-detector/src/`.
3. Open [notebooks/prepare_sec_data_colab.ipynb](notebooks/prepare_sec_data_colab.ipynb) in Colab and run all cells. Enter a real identifying SEC User-Agent with your name or organization and contact email when prompted. The notebook downloads the missing public SEC quarters (`2025Q4`, `2026Q1`, `2026Q2`, `2026Q3`) sequentially, skips complete existing quarters, and prepares `MyDrive/financial-risk-detector/data/colab_amendment_proxy.csv.gz`. Do not replace a complete quarter with a partial archive. Preparation may take time; it fits no model.
4. Open [notebooks/financial_risk_detector_gpu_colab.ipynb](notebooks/financial_risk_detector_gpu_colab.ipynb), choose **Runtime > Change runtime type > GPU**, then run all cells. The notebook checks CUDA before fitting and uses GPU XGBoost and CatBoost. Its training code is embedded, so no separate script upload is needed. If Colab disconnects during model comparison, run it again; outputs are written only after the full comparison and evaluation finish.
5. Read `MyDrive/financial-risk-detector/colab_outputs/schema_v4_final/metrics.json`. Compare `balanced_holdout`, `natural_holdout_at_natural_threshold`, and the always-negative accuracy baselines. The final holdout model, separate all-data model, model feature columns, and chronological CV leaderboard are saved in the same directory.

The data preparation step requires the four newer SEC quarters because an original filed in 2025Q3 must have a complete year of label follow-up. The preparation script refuses missing quarters. The training notebook refuses older feature schemas and a label cutoff before 2026-09-30.

## What changed from the earlier run

The older all-data XGBoost artifact used **one tree**: early stopping on 2021 selected iteration zero, and the all-data refit copied that count. V4 selected settings on balanced forward folds validated in 2018, 2020, and 2022; it fixed the final tree count to the median selected by those folds and refused fewer than 25 trees. It fitted through 2021, chose thresholds on 2023Q1-Q3, and evaluated on originals filed 2024Q4-2025Q3. The uploaded results have now been inspected, so that later cohort is development evidence for subsequent changes, including v5.

Schema v4 also adds issuer history known **strictly before** each original filing: prior amendment count, amendments in the preceding year, time since an earlier amendment, earlier original filing count, time since an earlier original, and past mean reporting delay. Numeric financial facts and ratios remain as-filed. Filing IDs, issuer IDs, dates, label timing, and the label observation cutoff are excluded from predictors. The SEC financial statement data sets do not contain complete 8-K history, so this experiment does not present an incomplete 8-K count as a reliable feature.

## Limits and source use

SEC Financial Statement Data Sets can contain filing or extraction errors and do not include every disclosure. This model should not be described as detecting confirmed restatements. Confirmed-restatement research needs a separately reviewed event ledger with primary filing evidence; [src/11_validate_restatement_labels.py](src/11_validate_restatement_labels.py) checks the ledger structure, but cannot verify human judgments.

The quarterly downloader uses only public SEC archives, an identifying User-Agent, sequential requests, retry backoff, and atomic extraction. Do not use fabricated contact details or bypass access restrictions. See the [SEC data set scope](https://www.sec.gov/dera/data/financial-statement-data-sets.html) and [SEC developer access guidance](https://www.sec.gov/about/developer-resources).
