# Financial Risk Detector: confirmed restatement study

## Research question

Can the MD&A in an original SEC 10-K or 10-Q help predict whether the issuer later reports a confirmed financial-statement restatement within a selected 1-, 3-, or 5-year horizon?

The current target is **issuer-level**: a qualifying event need not restate the exact period covered by the original MD&A filing. Do not describe it as period-specific prediction. It is not the old same-period amendment proxy. A 10-K/A or 10-Q/A is only a review candidate: routine amendments, immaterial revisions, corporate charter changes, stock splits, and new accounting-standard adoption are not automatically restatements. Outcomes need primary SEC evidence. The study is academic and is not a fraud detector or an investment, lending, audit, or operational tool.

The primary requested score is performance on a reviewed 1:1 case-control cohort. The queue intentionally samples amendment candidates and controls, so this does not estimate real-world prevalence or precision. No code can guarantee 85% or higher metrics. The notebook reports development model-selection metrics separately from a latest-years holdout; the holdout is only useful when it contains enough reviewed outcomes and issuers.

## What changed

- Same-period amendment candidates now cover up to five years, so events after day 365 can be reviewed. The retrieval window is not the outcome rule.
- A new ledger validator requires event dates, primary SEC evidence, reviewer identity, and an explicit outcome follow-up date. Pending rows never become labels.
- An AI-assisted SEC full-text screen searches each issuer's horizon, retrieves matching SEC documents, and leaves ambiguous hits pending. It is an exploratory label pass, not expert or human adjudication. Its query cache and evidence manifest make the screen auditable and resumable.
- Horizon labels are derived from the documented event date. For a negative at horizon H, the review must cover the full H years. Positive events inside H can be used as soon as the event is documented.
- The GPU notebook downloads only reviewed original filings and their immediately prior same-issuer/form filings. It adds point-in-time ratio trajectories from up to three earlier same-issuer/form filings and embeds the full available MD&A section (up to 6,000 words), rather than embedding only the opening 1,120 words. It compares structured financial features, interpretable MD&A cues and changes, and financial plus MiniLM semantic changes across GPU XGBoost and CatBoost. These are new hypotheses; their effect must be measured on the untouched future holdout.
- Model and threshold selection use issuer-grouped development folds. The latest filing years are held aside, and the main report includes matched 1:1 metrics.
- SEC downloads use public filing archives, an identifying User-Agent, sequential requests at no more than two per second, resumable storage, and stop on HTTP 403.

## Full MD&A archive extension

The first confirmed-restatement notebook uses reviewed outcome rows and fetches their current/prior MD&A from SEC. The full-archive version also uses the locally collected MD&A corpus (about 16,400 sections, 2009–2025) to learn an unlabeled TF-IDF/SVD language representation. Only archive text filed before the future holdout fits that representation. The corpus's legacy amendment-proxy target is not loaded. Supervised restatement labels still come only from the reviewed outcome ledger, so this addition does not turn the proxy into a restatement label or train on the holdout.

When the current Colab run finishes, prepare and build the next notebook locally:

```powershell
python src/26_prepare_mda_archive.py
python src/27_build_full_archive_restatement_colab.py
```

The first command reads `data/mda_corpus.csv`, validates accession/form/date against the prepared proxy, truncates very long sections to 20,000 words, and writes the label-free `data/colab_mda_archive_unlabeled.csv.gz` plus a manifest. It does not fit a model. The 27 GB raw filing tree is not uploaded.

Create `MyDrive/financial-risk-detector/data/` and upload:

1. `data/colab_amendment_proxy.csv.gz`
2. `data/restatement_review_queue_ai_refined.csv`, renamed in Drive to `restatement_review_queue.csv` (exploratory labels)
3. `data/colab_mda_archive_unlabeled.csv.gz`

Open `notebooks/confirmed_restatement_full_archive_gpu_colab.ipynb` in Colab, select a GPU runtime, and run cells in order. It reuses the existing three-year SEC text cache in Drive when available; otherwise it resumes text collection. Results go to `colab_outputs/confirmed_restatement_full_archive_3y/`. The archive is selected from an earlier amendment-proxy case-control corpus, so it supports language feature learning but is not representative of natural filing prevalence. The study's restatement labels remain AI-assisted and exploratory until human-audited.

## Higher-output filing-event proxy experiment

This is a separate, broader task intended to use the full original MD&A archive and a larger, balanced filing sample. It does **not** measure confirmed-restatement prediction:

- **Outcome A — same-period amendment proxy:** 1 means a later same-period 10-K/A or 10-Q/A candidate was observed within three years; 0 means the sampled control had no such candidate. Routine amendments may be included.
- **Outcome B — amendment or screened event:** Outcome A positives, plus reviewed issuer-level restatement/revision events within three years. A control is negative only with reviewed follow-up through the full horizon. Pending controls stay unlabeled and are excluded.

The local AI-refined queue currently produces 2,000 Outcome A rows (1,000 per class) and 1,732 Outcome B rows (1,042 positive, 690 negative); 268 unresolved controls are excluded from Outcome B. These labels come from the sample design and an AI-assisted screen, so report the exact target and describe the estimates as exploratory case-control metrics. The model never receives the queue's candidate-event fields, review fields, evidence, or legacy proxy label as predictors. This may produce stronger metrics, but no percentage is guaranteed.

After the current Colab run finishes, generate the new notebook locally:

```powershell
python src/29_build_filing_event_gpu_colab.py
```

Upload these files to `MyDrive/financial-risk-detector/data/`:

1. `colab_amendment_proxy.csv.gz`
2. `restatement_review_queue_ai_refined.csv` (the notebook also accepts `restatement_review_queue.csv`)
3. `colab_mda_archive_unlabeled.csv.gz`

Open `notebooks/filing_event_proxy_gpu_colab.ipynb`, choose a **GPU runtime**, and run cells in order. The notebook checks CUDA and stops if it is unavailable. It reuses `evidence_restatement/3y/sections.sqlite` when that cache is already on Drive; do not run it at the same time as another notebook writing that cache. SEC text collection is resumable and requests at most 500 new filings per session; rerun the setup and collection cells in a later session until pending reaches zero. Use a real identifying SEC User-Agent and stop if SEC denies access.

Results are written to `colab_outputs/filing_event_proxy_h3y/same_period_amendment/` and `colab_outputs/filing_event_proxy_h3y/amendment_or_screened_event/`. Each folder includes a matched future-year holdout, development CV leaderboard, predictions, model files, and a report naming its target. Precision from either 1:1 sample is not population precision. Do not present these proxy scores as confirmed-restatement results.

## Exact steps

### 1. Build the review queue locally

Run PowerShell from the repository folder. This reads the existing local `data/raw/` SEC quarterly files and the prepared feature CSV; it does not fit a model or make SEC requests.

```powershell
python src/08_build_restatement_candidates.py --max-years 5
python src/20_prepare_restatement_pilot.py --n-each 1000 --horizon-years 3
```

The first command rebuilds `data/restatement_candidates.csv` from same-period 10-K/A and 10-Q/A filings within five calendar years. The second creates `data/restatement_review_queue.csv` with up to 1,000 candidate originals and 1,000 comparison controls that have a mature three-year follow-up window. Candidate and control status is for sampling only; neither one assigns an outcome. Use this new queue; the earlier `restatement_review_pilot.csv` lacks the follow-up field required by this pipeline. Use `--horizon-years 1` or `5` only if you will also set that same horizon in the validator and Colab notebook.

### 2. Run the AI-assisted exploratory screen

To prepare a provisional ledger without hand-labeling every row, search the SEC full-text index and retrieve the matching primary documents. Supply a real identifying name and email in the User-Agent:

```powershell
python src/24_review_restatement_queue.py --user-agent "Your Name your.email@example.com"
python src/25_refine_restatement_queue.py
python src/11_validate_restatement_labels.py data/restatement_review_queue_ai_refined.csv --horizon-years 3
```

This writes `data/restatement_review_queue_ai_refined.csv`, `data/restatement_review_evidence_manifest.csv`, and a resumable search cache. Clearly supported cases are labeled; uncertain evidence remains pending. A negative means the SEC full-text screen found no qualifying event and is provisional. The structural validator checks fields and SEC links, not accounting correctness. Use this ledger only for exploratory model fitting; human adjudication is required before reporting confirmatory metrics.

The SEC client requests sequentially at no more than two per second. It stops on HTTP 403 or 429 and backs off on temporary 5xx responses. Do not bypass access denial.

### 3. Review the outcomes for confirmatory research

For each original filing, review the issuer's SEC filing history and primary filings through the longest horizon you plan to study. Complete or adjudicate `data/restatement_review_queue.csv`:

- `label=1`, `event_type=restatement`: the issuer identifies a material error in previously issued financial statements, a restatement, or non-reliance due to an accounting error. Record the earliest supported event date.
- `label=0`, `event_type=revision_not_restatement`: a documented immaterial revision or prior-period adjustment that is not a restatement.
- `label=0`, `event_type=other_amendment`: an amendment for another reason, with no confirmed restatement through the required follow-up.
- `label=0`, `event_type=no_confirmed_restatement`: no qualifying restatement found after reviewing the issuer's SEC filings through the entered follow-up date.

For every reviewed row, fill `outcome_followup_through`, `evidence_url` (a primary `https://www.sec.gov/` filing), `evidence_excerpt`, `reviewer`, and `review_notes`. For negative labels, the follow-up date must be at least the original filing date plus the selected horizon. Leave uncertain cases `review_status=pending`; do not force them into either class. An amendment alone is not evidence of a restatement.

Validate the completed queue for the 3-year horizon (change `3` to `1` or `5` if that is your preselected target):

```powershell
python src/11_validate_restatement_labels.py data/restatement_review_queue.csv --horizon-years 3
```

### 4. Run the GPU notebook

Generate the self-contained notebook:

```powershell
python src/23_build_restatement_colab.py
```

In Google Drive, use `MyDrive/financial-risk-detector/` and upload:

1. `data/colab_amendment_proxy.csv.gz`
2. `data/restatement_review_queue_ai_refined.csv`, renamed on Drive to `restatement_review_queue.csv` for an exploratory run, or the human-adjudicated `data/restatement_review_queue.csv` for a confirmatory run
3. Optionally, `data/restatement_review_evidence_manifest.csv` for audit; the notebook does not read it

The 27 GB raw folder and the old sampled text CSV are not needed. Open `notebooks/confirmed_restatement_gpu_colab.ipynb` in Colab, choose **Runtime > Change runtime type > T4 GPU**, and run all cells in order. This version writes to `colab_outputs/confirmed_restatement_3y_fulltext_trajectory_v2/` and uses a separate prepared dataset, preserving the previous run for comparison. Set `HORIZON_YEARS` to the same value used for review. Enter a real name/organization and contact email for the SEC User-Agent. The text collector checkpoints to Drive; if it reports pending filings, rerun that cell in a later session until pending is zero. If SEC access is denied with 403, stop and retry later on an allowed connection.

The notebook saves `metrics.json`, `grouped_cv_leaderboard.csv`, `future_holdout_predictions.csv`, `study_report.md`, SHAP feature and feature-group attribution CSVs when supported, and model files for the compared GPU configurations in `colab_outputs/confirmed_restatement_<H>y/`.

## Reading the metrics

The compact model comparison crosses three feature views (financial-only, interpretable MD&A cue rates and changes, and financial plus semantic MD&A changes) with two GPU model families (XGBoost and CatBoost). Training uses every development row with class-weighted loss rather than discarding most controls. Validation and test metrics remain year/form-matched 1:1 samples with no replacement. Model and threshold choices use issuer-grouped development predictions; the latest years remain untouched until final evaluation. The same holdout is reported for all feature/model combinations, and paired PR-AUC/ROC-AUC differences compare each text view with its financial-only baseline within the same model family. SHAP feature attributions describe the selected model; they are not causal effects.

Use `holdout_results_by_view` to compare all six predeclared model/view combinations on the same future holdout. `paired_holdout_auc_differences` gives paired bootstrap differences from each model family’s financial-only baseline. `balanced_holdout_1_to_1` gives the selected development model’s matched-sample result, measured on the latest filing years withheld from fitting and threshold selection; its `balanced_holdout_1_to_1_ci95` fields show stratified bootstrap intervals. Issuers may also appear in earlier development years, matching use on continuing companies; development folds themselves group issuers. `matched_sample_sensitivity_50` shows how the fixed-threshold metrics change across 50 1:1 matches; it is a matching sensitivity analysis, not a confidence interval. `development_cv_1_to_1` is for choosing the feature set and threshold; it is not a final score. `holdout_adequate_for_claim` requires at least 30 held-out examples per class and 10 distinct issuers per class. If it is false, review more outcomes from more issuers before making a strong claim.

Suggested report title: **Does MD&A change add predictive value for a later SEC-reported restatement? A three-year matched temporal-holdout study.** The main question is whether current MD&A language and its change from the prior comparable filing improve prediction beyond structured financial features. `study_report.md` contains the run-specific counts, results, and limitations for a draft. The AI-assisted labels remain exploratory; this small sampled cohort cannot support a claim of population-level predictive performance.

Because the review queue is deliberately enriched with amendment candidates and balanced controls, even the holdout is not a natural-prevalence estimate. Do not report its precision as population precision. The threshold is selected on development folds to maximize the weakest of accuracy, precision, recall, and F1, but optimizing a threshold cannot make weak separation disappear or guarantee a score. For a publishable estimate, freeze the target/model after development and evaluate on a later, untouched, mature cohort.

## Legacy amendment-proxy experiments

The v4/v5 and MDA notebooks remain available for reproducing the earlier **same-period amendment proxy** work. That proxy asks whether a same-period 10-K/A or 10-Q/A appears, not whether a restatement is confirmed. Its results are not results for this research question. In particular, do not reuse proxy labels, the legacy text dataset's `y`, or the proxy prediction as the confirmed-restatement target.

## Source access and use

The SEC materials used here are public filing records. Provide a real identifying User-Agent, request sequentially, respect SEC rate/access responses, and stop rather than bypassing a denial. See the [SEC access guidance](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data) and [SEC Financial Statement Data Sets description](https://www.sec.gov/dera/data/financial-statement-data-sets.html). These project controls support responsible academic research; they are not a legal certification.
