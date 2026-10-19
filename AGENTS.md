# Running this submission

Unknown-dataset AutoML for tabular classification, regression and clustering. Full details are in [README.md](README.md).

Install (Python 3.10–3.14, CPU only, no network needed at run time):

```bash
pip install -r requirements.txt
```

Run one dataset. Each run finishes in well under 60 s, typically 5–20 s.

```bash
python main.py <dataset_folder> <output_folder>                     # folder with data + description
python main.py train.csv test.csv metadata.json <output_folder>    # explicit files (any order)
python main.py --train data.csv --task clustering --n-clusters 5 -o <output_folder>
python main.py --train train.csv --test test.csv --task classification --target <column> -o <output_folder>
python main.py <folder_of_dataset_folders> <output_root>          # batch: one process per dataset
python main.py <dataset_folder> --dry-run                           # show how the input is understood, no training
```

How the input is understood:
- **Description files:** JSON or YAML metadata (task, target, id column, feature types, number of clusters; key names are matched flexibly), a README.md or task.txt, or `--description "..."`.
- **Flags:** `--task`, `--target`, `--n-clusters`, `--id-column` and `--metric` always win over anything read from files.
- **Missing information is inferred:** the target is the train column absent from test; the task comes from the target type or the folder name.

Outputs, in `<output_folder>` (default `outputs/<dataset_name>/`):
- `predictions.csv`: `<id column>, prediction`, one row per test row, in input order (every row for single-table clustering). Classes keep their original labels; clusters are `0..k-1`.
- `profile.json`: the structured data profile (copy: `data_profile.json`).
- stdout: exactly one JSON line with `status`, `resolved` (files, task, target, id column, k), absolute output paths, `profile_summary` and `runtime_seconds`. Logs go to stderr.
- Exit code 0 when predictions were written.

Python API: `from automl import run, solve`
- `run(dataset_folder, output_folder)`
- `solve(train_df, test_df, {"task": ..., "target": ...})` returns `(predictions, profile)`

Tests: `pip install -r requirements-dev.txt` (adds pytest), then `python -m pytest -q`.
