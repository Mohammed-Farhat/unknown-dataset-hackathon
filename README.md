# Unknown Dataset Hackathon: one program for any table

`main.py` takes an unfamiliar table and solves the machine-learning task that comes with it. The task can be **classification**, **regression** or **clustering**. The program needs no human input:

1. It reads the dataset's metadata (task, feature types, number of groups).
2. It inspects the data and writes a structured **data profile**.
3. It chooses its preparation and modelling steps.
4. It writes **predictions**.

A typical dataset takes 5–20 seconds on a laptop CPU, about 10 s on average. The first run after installation is a little slower while Python caches the libraries. Three safeguards stop any run from reaching the 60-second limit: an internal deadline, an early fallback output, and a watchdog.

> Course: Introduction to Machine Learning. Unknown Dataset Hackathon, six hours, three students per team.

## Team

| Name | Student ID |
|---|---|
| Mohammed Farhat | 202772669 |
| Ismail Abou Zeid | 202674504 |
| Rena Tarhini | 202771173 |

---

## How to run (for the assessment agent)

**1. Install** (Python 3.10–3.14, CPU only, no internet needed at run time):

```bash
pip install -r requirements.txt
```

**2. Run one dataset.** Pick whichever form matches how the data is given:

```bash
# a) a folder holding the data + a description (train.csv, test.csv, metadata.json / .yaml / README.md ...)
python main.py path/to/dataset_folder path/to/output_folder

# b) explicit files, in any order (train table first, then test); inputs are never overwritten
python main.py train.csv test.csv metadata.json out/
python main.py --train train.csv --test test.csv --metadata metadata.json --output-dir out/

# c) supervised task from bare CSVs (no description file)
python main.py --train train.csv --test test.csv --task classification --target Churn --output-dir out/

# d) clustering a single table into k groups
python main.py --train customers.csv --task clustering --n-clusters 5 --output-dir out/

# e) many datasets at once (one sub-folder per dataset); each runs in its own process with its own 60 s clock
python main.py path/to/all_datasets path/to/output_root

# f) check how a dataset is understood, without training (about 1 s)
python main.py path/to/dataset_folder --dry-run
```

If the description is not detected, pass `--task`, `--target`, `--n-clusters` or `--id-column`; flags always win. A free-text `--description "..."` or inline JSON (`--metadata '{"task": "clustering", "n_clusters": 4}'`) also works.

**3. Read the results** in the output folder (default `outputs/<dataset_name>/`):

| File | Content |
|---|---|
| `predictions.csv` | `<id column>, prediction`. One row per test row, in input order (every row for single-table clustering). Classes keep their original labels; clusters are `0..k-1`. |
| `profile.json` | The structured data profile, see [Data profile](#data-profile). An identical copy is written as `data_profile.json`. |

`stdout` gets exactly one JSON line. It contains:
- `status`
- `resolved`: the files, task, target, id column and k that were used
- absolute `predictions_path` and `profile_path`
- `profile_summary`: rows, features, missing values, constant / duplicate / id / irrelevant columns, imbalance ratio, data issues
- `runtime_seconds`

Logs go to `stderr`. The exit code is 0 whenever predictions were written. The same instructions, in short, are in [AGENTS.md](AGENTS.md).

**What the program works out by itself:**
- **Task, target and number of groups:** from the metadata (JSON/YAML, about 75 key synonyms plus fuzzy matching such as `task_family` or `required_number_of_groups`, and column roles like `{"name": "Churn", "role": "target"}`), from a free-text README.md / task.txt, or from flags. Without a description, the target is the train column missing from test, and the task follows from its type or from the folder name (`clf_…`, `reg_…`, `cluster_…`).
- **Feature types and data issues** (missing values, scales, categoricals, constant / duplicate / ID / irrelevant columns, imbalance): detected from the data.
- **Files:** recognised by name (`train`, `test`, `data`, `X_train` + `y_train`, `sample_submission`, …), or taken from paths written inside the metadata (`"train_path": …`). Formats: CSV/TSV (any separator, decimal commas, UTF-8/cp1252/latin-1), gzip/bz2/xz/zip-compressed CSV, JSON lines, Parquet (when `pyarrow` is installed).

**Other ways to run it:**
- `python main.py --help` lists every option.
- `python -m automl` and the aliases `run.py`, `solution.py` and `predict.py` behave exactly like `main.py`.
- From Python:

```python
from automl import run, solve
result = run("dataset_folder", "output_folder")      # {"predictions": DataFrame, "profile": dict, ...}
preds, profile = solve(train_df, test_df, {"task": "regression", "target": "SalePrice"})
```

**Try it on the bundled examples:**

```bash
python main.py examples/classification_demo outputs/classification_demo
python main.py examples/regression_demo     outputs/regression_demo
python main.py examples/clustering_demo     outputs/clustering_demo
```

## Input / output contract

We received only the written announcement. It does not specify the input/output format, and the practice datasets, baseline, checker and scoring rules it mentions were not available to us. The program therefore implements the most common hackathon layout and accepts many variants of it, listed under [Assumptions](#assumptions). Every contract-specific detail lives in one file, [`automl/io_contract.py`](automl/io_contract.py).

### Standard entry point

```bash
python main.py <dataset_dir> [<output_dir>]
python main.py --data-dir <dataset_dir> --output-dir <output_dir>
python main.py --train train.csv --test test.csv --metadata metadata.json --output-dir out/
python main.py <folder_with_many_datasets> <output_root>      # batch: one process per dataset
python -m automl <dataset_dir> <output_dir>                   # same thing, as a module
```

`run.py`, `solution.py` and `predict.py` are aliases of `main.py` with the same command line, for checkers that expect one of those names.

- **Path flags:** `--input/--dataset/--data`, `--output/--out/-o`, and environment variables `DATA_DIR` and `OUTPUT_DIR` are all accepted. Unknown flags are ignored.
- **Metadata overrides:** `--task`, `--target`, `--n-clusters`, `--id-column` and `--metric` override the metadata.
- **Output as a file:** if the output path ends in `.csv`, it is used as the predictions file and the profile is written next to it.

Python API:

```python
from automl import run, solve
result = run("path/to/dataset_dir", "path/to/output_dir")   # {"predictions": DataFrame, "profile": dict, ...}
preds, profile = solve(train_df, test_df, {"task": "regression", "target": "SalePrice"})
```

### Input: one folder per dataset

| File | Content |
|---|---|
| `train.csv` | Feature columns plus the target column (supervised). For clustering: feature columns only, no labels. |
| `test.csv` | Feature columns only. Optional for clustering. |
| `metadata.json` | `task`, `target`, `id_column`, `feature_types` (`{column: "numeric" \| "categorical" \| ...}`), `n_clusters` (clustering), optional `metric` |

Other layouts are detected automatically:

- **Other file names:** `data.csv`, `X_train.csv` + `y_train.csv`, `train_labels.csv`, `validation.csv`, or a single table with a `split` column.
- **Other file formats:** TSV, `;`-separated with decimal commas, UTF-8/cp1252/latin-1 encodings, JSON lines, Parquet (when `pyarrow` is installed), and YAML metadata.
- **Own outputs:** files written by the program (`predictions.csv`, `profile.json`) are never read back as inputs, even when the output folder is the input folder.
- **Sample submission:** if a `sample_submission.csv` is present, its column names are copied.
- **Metadata keys:** about 60 synonyms are recognised, for example `task_type`, `problem_type`, `label`, `target_column`, `num_clusters`, `k`, or `features: [{name, type}]`.
- **Missing metadata entries:** if the metadata lacks something, it is inferred. The target is the train column missing from test; the task comes from the target's type.

### Output (written to `<output_dir>`, default `outputs/<dataset_name>/`)

| File | Content |
|---|---|
| `predictions.csv` | Columns `<id_column>, prediction`, one row per test row, in input order. Without an id column, the file has only `prediction`. |
| `profile.json` | The structured data profile, described [below](#data-profile). A copy is written as `data_profile.json`. |

Prediction values by task:

- **Classification:** labels in their original form (`Yes`/`No`, `1`/`0`, …).
- **Regression:** numbers.
- **Clustering:** integers `0..k-1`, with exactly `k` non-empty groups. If `test.csv` exists, train and test rows are clustered together and only the test rows are written.


### Adapting to a different contract

If the assessment environment uses a different format, only [`automl/io_contract.py`](automl/io_contract.py) needs to change:

| What differs | Where to change it |
|---|---|
| Dataset file names | `find_files` |
| Metadata key names | `KEYS` |
| Output file names | `resolve_output_paths` |
| Prediction column names | `prediction_frame` |
| Profile file name | `write_profile` |

The modelling code never touches file names.

---

## Approach

```
load + detect contract -> profile -> clean & type columns -> instant fallback output
      -> task-specific modelling (time-boxed) -> final predictions + completed profile
```

### How each announced data issue is handled

| Issue (from the announcement) | Detection | Handling |
|---|---|---|
| Missing values | `isna()` per column, in train and test | Gradient boosting uses native NaN routing. Linear, ExtraTrees and clustering models use median imputation plus missing-indicator columns (linear and ExtraTrees only), or the most frequent value for categoricals. |
| Differently scaled features | Ratio of the largest to the smallest feature standard deviation (flagged when > 10) | Tree models are scale-free. Linear models standardise, log-transform skewed non-negative columns and clip outliers to ±5 σ. Clustering standardises. |
| Categorical columns | Declared types; otherwise object, bool and category dtypes | Native categorical splits in gradient boosting (top 60 levels plus `__other__`). Capped one-hot encoding for linear models and clustering. Frequency encoding for high-cardinality columns. |
| Numbers stored as text, dates | ≥ 95% of values parse as numbers / ≥ 90% parse as dates | Text numbers are converted (e.g. Telco `TotalCharges`). Timestamps are expanded into year, month, day, weekday, hour, weekend flag, day of year and elapsed days. |
| Constant columns | `nunique(dropna=True) <= 1` | Dropped |
| Duplicate columns | Column identical to another (NaN equal to NaN, dtype ignored) | The copy is dropped |
| ID-like columns | Declared id; id-like name with ≥ 95% unique values; `0..n-1` counters; unique short codes | Dropped (an ID would otherwise ruin clustering) |
| Train-only columns | Present in train but absent from test (target leakage such as Bike `casual`/`registered`) | Dropped |
| Irrelevant features | Supervised: mutual information no higher than a permuted-target null **and** negligible held-out permutation importance (or zero importance with tiny mutual information). Precision 0.93 / recall 0.91 on our synthetic tables. Clustering: saliency selection (below). | Reported in the profile. Supervised models rely on regularised trees; clustering removes them, which matters most there. |
| Imbalanced classes | `max/min` class count ≥ 3 | Macro-F1-oriented decision rule tuned on out-of-fold probabilities (binary threshold or per-class weights); `class_weight="balanced"` variant when the minority is under 10% |
| Nonlinear relationships | Tree-model cross-validation score minus linear score > 0.03 | Gradient boosting and ExtraTrees capture them. The ensemble weights decide between tree and linear models. |

### Classification and regression

Implemented in [`automl/supervised.py`](automl/supervised.py).

**Candidates, all scikit-learn, in order:**
- `HistGradientBoosting` (native categoricals and NaN, early stopping).
- A linear model on one-hot plus scaled features (`LogisticRegression` / `Ridge` with automatic alpha).
- `RandomForest` (classification).
- A regularised gradient-boosting variant.
- `ExtraTrees`.
- For skewed non-negative regression targets (prices, counts): gradient-boosting and Ridge models fitted on `log1p(y)`, with a smearing correction.

Forests grow fully on small tables, with bounded tree size on large ones.

The candidate set was chosen by an offline experiment on 18 tables. k-NN, a faster gradient-boosting setting and a deeper gradient-boosting setting were also tried and dropped: they never helped, or their gain did not pay for their runtime.

**Cross-validation:** 5-fold out-of-fold (OOF) predictions, stratified for classification.
- Below 600 rows, the 5 folds are repeated twice.
- Above 30k rows, model selection runs 3 folds on a 30k-row sample, followed by a full refit.
- Test predictions are the average of the fold models, so no extra refit is needed.

**Blending:** greedy ensemble selection (Caruana) on the OOF predictions, optimising the target metric. Classification uses log-loss only as a tie-breaker; on its own it would reject accurate but over-confident forests.

**Decision rule (classification):** the default metric is macro-F1, chosen because of the emphasis on imbalance. The rule is tuned on the blended OOF probabilities and kept only if it improves OOF macro-F1 by more than 0.002. Accuracy is essentially unchanged.

**Regression output:** predictions are clipped to the training range ± 10%, and never below 0 for non-negative targets.

**Metric:** if the metadata names one (`accuracy`, `balanced_accuracy`, `f1`, `roc_auc`, `rmse`, `mae`, `rmsle`, `r2`, …), models are selected and tuned for it.

### Clustering with a given number of groups

Implemented in [`automl/clustering.py`](automl/clustering.py). The recipe was selected with [`tools/cluster_bench.py`](tools/cluster_bench.py) on 44 labelled problems. Mean ARI went from 0.81 for plain KMeans to **0.94**.

1. **Matrix:** numeric columns get median imputation and standardisation; categorical columns are one-hot encoded unscaled (each block has variance ≤ 1, like one standard column).
2. **Removing irrelevant features without labels:**
   - **Saliency:** iterative η² saliency, i.e. the share of each feature's variance explained by the current KMeans clusters.
   - **Leave-one-out check:** a noise feature cannot be predicted by clusters found without it.
   - **Dependence protection:** a column whose mutual information with another column is ≥ 0.12 is never dropped. Cluster structure makes informative columns depend on each other, even for rings and Mall-style income × spending groups; injected noise is independent of everything.
3. **Model:**
   - All-numeric data: a Gaussian mixture, full vs diagonal covariance chosen by BIC.
   - Categorical or low-cardinality features present: KMeans (`n_init=10`).
   - Spectral clustering on the k-NN graph replaces the mixture only when the mixture cuts through dense neighbourhoods (k-NN cut fraction ≥ 0.01) and spectral is at least 4 times cleaner. This handles moons and rings.
4. Labels are relabelled `0..k-1` deterministically (`random_state=0` everywhere).

### Time budget (60 s per dataset)

- The clock starts at the first line of `main.py`, before the library imports.
- A valid **fallback output** (majority class / mean / round-robin groups) is written as soon as the data is loaded and profiled: about 3 s on typical tables, more on very large files. The first fitted model then overwrites it, and the final blend overwrites that. Every write is atomic.
- **Model search** stops 20 s after start, or 12 s after modelling begins when loading a big table took long.
  - The first model may use the whole internal budget (40 s), so a strong model always finishes.
  - Each later candidate's cost is projected after its first fold; candidates that would overrun are cut short (partial folds) or skipped.
  - A failing candidate is isolated and never stops the others.
- A **watchdog** exits cleanly at 52 s. The latest outputs are already on disk, and a JSON status line is still printed.
- In batch mode every dataset runs in its own process with its own clock.
- **Threads:** `min(cpu_count, affinity, cgroup quota, 8)`, set through `threadpoolctl`. Folds run sequentially, so threads are never oversubscribed.

### Data profile

`profile.json` uses plain pandas definitions on the tables exactly as loaded, so the organisers' own pandas computations should agree with it:

| Key | Definition |
|---|---|
| `task`, `target`, `id_column`, `n_clusters`, `metric` | As resolved from the metadata or inferred |
| `n_rows`, `n_test_rows`, `n_columns`, `n_features` | Train shape; features exclude the target and id |
| `feature_types` | Declared types copied exactly; otherwise `numeric` / `categorical` / `datetime` / `text` |
| `numeric_features`, `categorical_features`, `datetime_features` | Lists |
| `missing_values` | `df[c].isna().sum()` for **every** column, zeros included |
| `columns_with_missing`, `missing_fraction`, `test_missing_values` | Related missing-value statistics |
| `constant_columns` | `nunique(dropna=True) <= 1`, including all-NaN columns |
| `duplicate_columns`, `duplicate_column_pairs`, `duplicate_columns_pandas` | Exact copies of another column (NaN equal to NaN, dtype ignored). `duplicate_columns` names the copy: `x_copy`, `x_dup` or `x.1` when the names say so, otherwise the later column. `duplicate_columns_pandas` is exactly `df.T.duplicated()`, which always names the later column. |
| `id_like_columns`, `near_constant_columns`, `n_duplicate_rows` | |
| `scaling` | Standard-deviation ratio; `differently_scaled` when > 10 |
| `class_balance` (classification) | Class counts and distribution, majority/minority class, `imbalance_ratio = max/min`, `is_imbalanced` (ratio ≥ 3) |
| `target_stats` (regression) | `describe()` statistics plus skew |
| `irrelevant_features` | See the table above |
| `nonlinearity_detected` | See the table above |
| `data_issues` | One boolean per announced issue: missing values, different scales, categorical, constant, duplicate, irrelevant, imbalance, nonlinear. `null` where an issue does not apply, e.g. class imbalance in clustering. |
| `columns` | Per-column details: dtype, declared and inferred type, missing count, unique count, constant / duplicate / id flags, statistics |
| `preprocessing` | Dropped columns with the reason for each, and the columns used |
| `model` | Candidates with their cross-validation scores, ensemble weights, decision rule, timings. For clustering: algorithm, selected features, cluster sizes. |
| `runtime_seconds`, `status` | |

---

## Results on our local practice suites

The local checker [`tools/benchmark.py`](tools/benchmark.py) runs `main.py` in a fresh process per dataset, the way an assessment agent would. It then validates the output format, scores the predictions against the held-back labels, and checks the profile against the issues we injected. We used two suites:

- **Practice suite.** Real versions of the announcement's examples (OpenML copies of Telco churn, AI4I 2020, House Prices, plus Mall-like and penguin tables), with extra issues injected.
- **Benchmark suite.** Hidden-like synthetic and real tables.

Measured on a 16-thread Windows laptop. Timings include Python start-up and imports.

| Suite | Classification macro-F1 / accuracy | Regression R² | Clustering ARI / NMI | Profile checks passed | Runtime mean / max (s) | Format problems |
|---|---|---|---|---|---|---|
| practice (9) | 0.776 / 0.848 | 0.920 | 0.770 / 0.828 | 0.96 | 11.7 / 22.5 | 0 |
| benchmark (18) | 0.747 / 0.822 | 0.850 | 0.887 / 0.886 | 0.96 | 10.8 / 20.5 | 0 |
| examples (3) | 0.780 / 0.855 | 0.796 | 0.991 / 0.987 | 0.90 | 7.3 / 9.4 | 0 |

- **Profile checks:** row count, every missing-value count, constant columns, duplicate columns, declared feature types, and recall/precision of the irrelevant-feature list. Real tables lower the precision part, because their naturally useless columns (e.g. `gender` in Telco) are flagged but were not injected by us.
- **Linux:** the same code was verified on Ubuntu with Python 3.14 and the pinned wheels. All tests pass; the example runs take 1–8 s.
- **Stress tests:** all finished in at most 28 s with valid outputs.
  - **Large and wide tables:** a 120k-row table with 5,000-level categoricals and timestamps; a 45k × 200 regression (test R² 0.94); a 300-column table.
  - **Unusual shapes:** a 20-class problem, a 40-row table, a class with only two training rows.
  - **Clustering at scale:** a 60k-row table with 30 noise columns (ARI 1.0).
  - **Input quirks:** latin-1 `;`-separated files with decimal commas, and output folder = input folder.
- **Batch mode:** all 18 benchmark datasets in a single invocation, each in its own process. Every one reached its final model.
- **Unusual dataset layouts:** more than 20 are covered by `tests/`.

<details><summary>Per-dataset results</summary>

| Dataset | Task | Score on hidden labels | Runtime (s) |
|---|---|---|---|
| `clf_machine_failure` | classification | macro-F1 0.879 (accuracy 0.984) | 11.1 |
| `clf_synthetic_multiclass` | classification | macro-F1 0.691 (accuracy 0.746) | 18.8 |
| `clf_telco_churn` | classification | macro-F1 0.758 (accuracy 0.813) | 11.6 |
| `clu_mall_customers` | clustering | ARI 1.000 (NMI 1.000) | 3.6 |
| `clu_penguins` | clustering | ARI 0.661 (NMI 0.690) | 3.7 |
| `clu_synthetic_blobs` | clustering | ARI 0.649 (NMI 0.795) | 4.0 |
| `reg_bike_demand` | regression | R² 0.984 | 17.1 |
| `reg_house_prices` | regression | R² 0.858 | 22.5 |
| `reg_synthetic_friedman` | regression | R² 0.919 | 12.6 |
| `b_clf_adult` | classification | macro-F1 0.800 (accuracy 0.854) | 16.6 |
| `b_clf_credit_g` | classification | macro-F1 0.714 (accuracy 0.760) | 8.5 |
| `b_clf_synth_0` | classification | macro-F1 0.648 (accuracy 0.832) | 11.7 |
| `b_clf_synth_1` | classification | macro-F1 0.810 (accuracy 0.810) | 20.5 |
| `b_clf_synth_2` | classification | macro-F1 0.667 (accuracy 0.688) | 15.1 |
| `b_clf_synth_3` | classification | macro-F1 0.844 (accuracy 0.988) | 15.5 |
| `b_clu_blobs_0` | clustering | ARI 0.638 (NMI 0.658) | 3.7 |
| `b_clu_blobs_1` | clustering | ARI 0.969 (NMI 0.969) | 4.2 |
| `b_clu_blobs_2` | clustering | ARI 0.951 (NMI 0.941) | 3.7 |
| `b_clu_blobs_3` | clustering | ARI 0.961 (NMI 0.971) | 4.4 |
| `b_clu_iris` | clustering | ARI 0.904 (NMI 0.900) | 3.5 |
| `b_clu_wine` | clustering | ARI 0.898 (NMI 0.876) | 3.9 |
| `b_reg_bike_hourly` | regression | R² 0.947 | 20.5 |
| `b_reg_synth_0` | regression | R² 0.883 | 9.4 |
| `b_reg_synth_1` | regression | R² 0.979 | 17.0 |
| `b_reg_synth_2` | regression | R² 0.863 | 11.0 |
| `b_reg_synth_3` | regression | R² 0.990 | 18.0 |
| `b_reg_wine_quality` | regression | R² 0.435 | 7.1 |

The synthetic tables are noisy by construction (overlapping classes and clusters, flipped labels), so their scores are not near 1. Their purpose is to compare strategies.
</details>

## Reproducing everything

```bash
pip install -r requirements-dev.txt

# 1. practice data: 9 datasets mirroring the announcement's examples
#    (Telco churn, AI4I machine failure, House Prices, bike demand, Mall customers,
#    plus synthetic tables with every announced data issue), and 18 extra
#    "hidden-like" benchmark datasets. Real tables come from OpenML; without
#    internet, add --offline to use synthetic stand-ins.
python tools/make_practice_data.py --suite practice
python tools/make_practice_data.py --suite bench

# 2. local checker + scorer: runs main.py as a subprocess per dataset,
#    validates the outputs and scores them against the hidden labels
python tools/benchmark.py --data data/practice --labels data/private_practice
python tools/benchmark.py --data data/bench    --labels data/private_bench

# 3. clustering strategy comparison (how the clustering defaults were chosen)
python tools/cluster_bench.py

# 4. tests (contract variants, profile definitions, end-to-end examples)
python -m pytest -q

# 5. final checklist before submission
python tools/presubmit_check.py
```

All randomness is seeded (`random_state=0`), so reruns on the same machine give byte-identical predictions; we checked this. On a much slower CPU the time budget may evaluate fewer candidate models.

## Assumptions

We worked only from the written announcement. The practice datasets, baseline, input/output contract, local checker and scoring rules it mentions were not available to us, and questions to the organisers were not allowed. These are the assumptions we made and why. Every one of them can be changed in a single place.

1. **Dataset layout.** Each dataset is a folder with `train.csv`, `test.csv` (supervised; optional for clustering) and a metadata file (`metadata.json`, or any `*.json` / `*.yaml` with task information). The variants listed in the contract section are also detected. If several datasets are given in one folder, each runs in its own process with its own 60-second clock.
2. **Metadata.** It states the task, the target column, the feature types and, for clustering, the number of groups, as the announcement says. Key names are matched against about 60 synonyms, nested structures included. If something is missing, it is inferred:
   - The target is the column present in train but absent from test.
   - The task comes from the target's type.
   - Without a target, the task is clustering. If `k` is not given either, it is chosen by silhouette over 2..10.
3. **Outputs.**
   - `predictions.csv` holds `<id_column>, prediction`, one row per test row, in input order. Without an id column it holds only `prediction`.
   - `profile.json` (copy: `data_profile.json`) goes in the output folder, by default `outputs/<dataset_name>/`.
   - If the metadata or a `sample_submission.csv` names other columns or file names, those are used instead.
4. **Clustering rows.** If the clustering dataset has a `test.csv`, train and test rows are clustered together and labels are written for the test rows. Otherwise every row of the table gets a label. Labels are integers `0..k-1` with exactly `k` groups.
5. **Metrics.** Unless the metadata names a metric, we assume:
   - **Classification:** macro-F1, because the announcement stresses imbalanced classes. The tuned decision rule leaves accuracy essentially unchanged.
   - **Regression:** R² / RMSE, so we minimise squared error.
   - **Clustering:** adjusted Rand index (or NMI). Both reward recovering the true groups, and our choices rank the same under both.
6. **Speed credit.** We assume it grows as runtime falls, and is only awarded when predictions beat a trivial baseline. We therefore aim for about 10 s per dataset and never go above 40 s. Speed is never traded for a measurable loss in quality.
7. **Profile correctness.** We assume the organisers compute the reference profile with plain pandas on the files as given. Our definitions follow pandas defaults (see the profile table). The two thresholds that pandas does not define are documented in the profile itself:
   - Imbalanced means a `max/min` class-count ratio ≥ 3.
   - Differently scaled means a feature standard-deviation ratio > 10.
8. **Feature types.** Declared types are trusted, with two exceptions:
   - A declared-categorical column that holds timestamps is expanded into date parts.
   - A declared-numeric column holding text is coerced to numbers.
   Columns that are ID-like, constant, exact duplicates, or present only in train are not used as features. Neither is a column the metadata leaves out when it clearly lists every feature (for example a stray `label` column next to clustering features). All of these are still reported in the profile.
9. **Rows with a missing target** are dropped from training. A dataset whose training labels contain a single class predicts that class.
10. **Size limits.** We assume the published limits are tens of thousands of rows and up to a few hundred columns. Larger tables still work: model selection runs on a 30k-row sample, then the selected models are refitted on all rows.
11. **Assessment environment.** We assume CPU only, Linux or Windows, Python 3.10–3.14, a fresh `pip install -r requirements.txt`, and no internet access at run time. Only the scikit-learn stack is used, to avoid installation failures. Thread usage follows the CPU quota of the container.
12. **Determinism.** All seeds are fixed (`random_state=0`). Reruns on the same machine give byte-identical predictions. Only a much slower CPU could change the result, by fitting fewer candidate models in the time budget.

## Repository structure

```
main.py                  standard entry point (CLI + importable run/solve)
AGENTS.md                short run instructions for automated agents
automl/
  cli.py                 argument parsing, batch mode, JSON summary
  io_contract.py         dataset discovery, metadata normalisation, output writing
  budget.py              time budget, CPU detection, watchdog
  features.py            column typing, constant/duplicate/id detection, feature building
  profiling.py           structured data profile + irrelevant-feature test
  supervised.py          classification / regression: candidates, CV, blending, decision rule
  clustering.py          clustering: feature saliency, GMM/KMeans/spectral selection
  pipeline.py            orchestration, fallback outputs, public API
examples/                three small datasets (one per task) + their hidden labels in _labels/
tests/                   pytest suite
tools/
  make_practice_data.py  practice / benchmark dataset generator
  benchmark.py           local checker + scorer + contest-score estimate
  cluster_bench.py       clustering strategy benchmark
  presubmit_check.py     submission checklist
requirements.txt         pinned runtime dependencies
requirements-dev.txt     + pytest
```

## Development notes

- As the announcement allows, AI assistants were used during development. Several independent assistant "advisors" proposed designs for the contract, the supervised pipeline and clustering, and the team kept the options that won on the benchmarks above.
- The final program is self-contained. It makes no network calls and no AI or API calls at run time, and it writes nothing outside the output folder.
