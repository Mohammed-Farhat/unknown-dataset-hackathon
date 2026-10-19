"""Contract robustness and profiling tests (fast, synthetic, offline).

Run with:  python -m pytest -q
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from automl.features import constant_columns, duplicate_columns, plan_columns  # noqa: E402
from automl.io_contract import Metadata  # noqa: E402
from automl.pipeline import run, solve  # noqa: E402


def _toy_supervised(n=300, task="classification", seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n) * 1000,
                      "c": rng.choice(["x", "y", "z"], n), "const": 1.0})
    X["a_copy"] = X["a"]
    X.loc[rng.random(n) < 0.1, "b"] = np.nan
    signal = X["a"] + (X["c"] == "x") * 1.5
    y = (signal > 0.3).map({True: "yes", False: "no"}) if task == "classification" \
        else signal * 10 + rng.normal(size=n)
    X.insert(0, "id", np.arange(n))
    return X, y


def _write(tmp: Path, train, test, meta, meta_name="metadata.json"):
    tmp.mkdir(parents=True, exist_ok=True)
    train.to_csv(tmp / "train.csv", index=False)
    if test is not None:
        test.to_csv(tmp / "test.csv", index=False)
    if meta is not None:
        if meta_name.endswith(".json"):
            (tmp / meta_name).write_text(json.dumps(meta))
        else:
            (tmp / meta_name).write_text("\n".join(f"{k}: {v}" for k, v in meta.items()))
    return tmp


# --------------------------------------------------------------------------- profiling
def test_constant_and_duplicate_definitions():
    df = pd.DataFrame({"a": [1, 2, 3, np.nan], "b": [5, 5, 5, np.nan], "c": [np.nan] * 4,
                       "d": [1.0, 2.0, 3.0, np.nan], "e": ["u", "v", "u", "v"],
                       "e_dup": ["u", "v", "u", "v"]})
    assert constant_columns(df) == ["b", "c"]          # nunique(dropna=True) <= 1
    pairs = duplicate_columns(df)
    assert ("a", "d") in pairs                         # int vs float with NaN: duplicates
    assert ("e", "e_dup") in pairs


def test_copy_name_is_reported_as_duplicate_even_if_first():
    df = pd.DataFrame({"x_copy": [1, 2, 3], "x": [1, 2, 3]})
    assert duplicate_columns(df) == [("x", "x_copy")]


def test_profile_counts_match_pandas(tmp_path):
    X, y = _toy_supervised()
    tr = X.iloc[:200].assign(target=y[:200].values)
    te = X.iloc[200:]
    d = _write(tmp_path / "ds", tr, te, {"task": "classification", "target": "target",
                                          "id_column": "id"})
    res = run(d, tmp_path / "out")
    prof = res["profile"]
    assert prof["status"] == "final" and not prof.get("errors")
    assert prof["n_rows"] == 200
    assert prof["missing_values"] == {c: int(v) for c, v in tr.isna().sum().items()}
    assert prof["constant_columns"] == ["const"]
    assert prof["duplicate_columns"] == ["a_copy"]
    assert prof["data_issues"]["categorical_columns"] is True
    assert prof["data_issues"]["differently_scaled_features"] is True
    assert json.loads((tmp_path / "out" / "profile.json").read_text())["task"] == "classification"


# --------------------------------------------------------------------------- contract
@pytest.mark.parametrize("meta", [
    {"task": "classification", "target": "target", "id_column": "id"},
    {"task_type": "binary_classification", "label": "target", "id_col": "id"},
    {"problem": {"type": "classification", "target_column": "target"}},
    None,  # no metadata: target = column missing from test, task inferred
])
def test_metadata_variants(tmp_path, meta):
    X, y = _toy_supervised()
    tr = X.iloc[:200].assign(target=y[:200].values)
    te = X.iloc[200:]
    d = _write(tmp_path / "ds", tr, te, meta)
    res = run(d, tmp_path / "out")
    assert res["profile"]["status"] == "final", res["profile"].get("errors")
    pred = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert len(pred) == 100
    assert set(pred["prediction"].astype(str)) <= {"yes", "no"}
    assert res["profile"]["task"] == "classification"


def test_yaml_metadata_and_regression(tmp_path):
    X, y = _toy_supervised(task="regression")
    tr = X.iloc[:200].assign(price=y[:200].values)
    d = _write(tmp_path / "ds", tr, X.iloc[200:], {"task": "regression", "target": "price"},
               meta_name="task.yaml")
    run(d, tmp_path / "out")
    pred = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert list(pred.columns) == ["id", "prediction"] and len(pred) == 100
    assert np.isfinite(pred["prediction"]).all()


def test_clustering_single_table(tmp_path):
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [8, 8], [0, 8]])
    lab = rng.integers(0, 3, 300)
    df = pd.DataFrame(centers[lab] + rng.normal(size=(300, 2)), columns=["u", "v"])
    df["noise"] = rng.normal(size=300)
    d = tmp_path / "clu"
    d.mkdir()
    df.to_csv(d / "data.csv", index=False)
    (d / "meta.json").write_text(json.dumps({"task": "clustering", "n_clusters": 3}))
    run(d, tmp_path / "out")
    pred = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert len(pred) == 300 and sorted(pred["prediction"].unique()) == [0, 1, 2]
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(lab, pred["prediction"]) > 0.9


def test_in_memory_api():
    X, y = _toy_supervised(task="regression")
    preds, prof = solve(X.iloc[:200].assign(y=y[:200].values), X.iloc[200:],
                        {"task": "regression", "target": "y"})
    assert len(preds) == 100 and prof["task"] == "regression"
    assert prof["status"] == "final", prof.get("errors")


def test_metadata_synonyms():
    m = Metadata.from_dict({"Task": "Clustering", "num_clusters": "4",
                            "features": [{"name": "a", "type": "numerical"},
                                         {"name": "b", "type": "nominal"}]})
    assert m.task == "clustering" and m.n_clusters == 4
    assert m.feature_types == {"a": "numeric", "b": "categorical"}


def test_feature_named_like_metadata_key_is_not_confused():
    m = Metadata.from_dict({"task": "regression", "target": "y",
                            "feature_types": {"type": "categorical", "id": "numeric"}})
    assert m.task == "regression" and m.id_column is None


# --------------------------------------------------------------------------- CLI
def test_cli_flags_and_summary(tmp_path):
    X, y = _toy_supervised()
    tr = X.iloc[:200].assign(target=y[:200].values)
    d = _write(tmp_path / "ds", tr, X.iloc[200:], {"task": "classification", "target": "target"})
    out = tmp_path / "o"
    cp = subprocess.run([sys.executable, str(ROOT / "main.py"), "--data-dir", str(d),
                         "--output-dir", str(out), "--quiet", "--some-unknown-flag"],
                        capture_output=True, text=True, timeout=120)
    assert cp.returncode == 0, cp.stderr
    summary = json.loads(cp.stdout.strip().splitlines()[-1])
    assert summary["status"] == "ok" and summary["runtime_seconds"] < 60
    assert (out / "predictions.csv").exists() and (out / "profile.json").exists()


def test_plan_drops_leaky_train_only_columns():
    tr = pd.DataFrame({"a": [1, 2, 3, 4], "leak": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    te = pd.DataFrame({"a": [5, 6]})
    plan = plan_columns(tr, te, "y", None, {})
    assert plan.dropped.get("leak") == "absent_from_test"


def test_batch_mode_subfolders_and_flat_files(tmp_path):
    X, y = _toy_supervised()
    tr = X.iloc[:200].assign(target=y[:200].values)
    root = tmp_path / "many"
    _write(root / "ds_a", tr, X.iloc[200:], {"task": "classification", "target": "target"})
    _write(root / "ds_b", tr, X.iloc[200:], {"task": "classification", "target": "target"})
    out = tmp_path / "out"
    cp = subprocess.run([sys.executable, str(ROOT / "main.py"), str(root), str(out), "--quiet"],
                        capture_output=True, text=True, timeout=300)
    assert cp.returncode == 0, cp.stderr
    for name in ("ds_a", "ds_b"):  # each child process must have its own clock
        prof = json.loads((out / name / "profile.json").read_text())
        assert prof["status"] == "final", prof.get("errors")
    flat = tmp_path / "flat"
    flat.mkdir()
    for name in ("clf_01", "clf_02"):
        tr.to_csv(flat / f"{name}_train.csv", index=False)
        X.iloc[200:].to_csv(flat / f"{name}_test.csv", index=False)
        (flat / f"{name}_metadata.json").write_text(json.dumps({"task": "classification",
                                                                "target": "target"}))
    cp = subprocess.run([sys.executable, str(ROOT / "main.py"), str(flat), str(tmp_path / "o2"),
                         "--quiet"], capture_output=True, text=True, timeout=300)
    assert cp.returncode == 0, cp.stderr
    assert len(pd.read_csv(tmp_path / "o2" / "clf_02" / "predictions.csv")) == 100


@pytest.mark.parametrize("script", ["run.py", "solution.py", "predict.py"])
def test_alias_entry_points(tmp_path, script):
    cp = subprocess.run([sys.executable, str(ROOT / script), str(ROOT / "examples" / "clustering_demo"),
                         str(tmp_path), "--quiet"], capture_output=True, text=True, timeout=120)
    assert cp.returncode == 0, cp.stderr
    assert (tmp_path / "predictions.csv").exists()


def test_rare_class_with_two_rows(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = np.where(X.a > 0, "p", "q").astype(object)
    y[:2] = "rare"
    d = _write(tmp_path / "ds", X.iloc[:300].assign(t=y[:300]), X.iloc[300:],
               {"task": "classification", "target": "t"})
    res = run(d, tmp_path / "out")
    assert res["profile"]["status"] == "final"
    assert "hgb" in res["profile"]["model"]["candidates"]


def test_latin1_semicolon_decimal_comma(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"caf\u00e9": rng.choice(["\u00e9", "\u00fc"], 300), "x": rng.normal(size=300).round(3)})
    df["y"] = df.x * 3
    d = tmp_path / "ds"
    d.mkdir()
    df.iloc[:240].to_csv(d / "train.csv", index=False, sep=";", decimal=",", encoding="latin-1")
    df.drop(columns="y").iloc[240:].to_csv(d / "test.csv", index=False, sep=";", decimal=",",
                                          encoding="latin-1")
    (d / "metadata.json").write_text(json.dumps({"task": "regression", "target": "y"}))
    run(d, tmp_path / "out")
    pred = pd.read_csv(tmp_path / "out" / "predictions.csv")
    assert np.corrcoef(pred["prediction"], df.y.iloc[240:])[0, 1] > 0.95


def test_output_dir_equal_to_input_dir_rerun(tmp_path):
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(300, centers=4, random_state=1)
    d = tmp_path / "ds"
    d.mkdir()
    pd.DataFrame(X, columns=["u", "v"]).to_csv(d / "data.csv", index=False)
    (d / "meta.json").write_text(json.dumps({"task": "clustering", "n_clusters": 4}))
    for _ in range(2):  # the second run must not read the first run's predictions.csv
        run(d, d)
    pred = pd.read_csv(d / "predictions.csv")
    assert len(pred) == 300 and pred["prediction"].nunique() == 4


def test_metadata_with_file_paths(tmp_path):
    X, y = _toy_supervised(task="regression")
    data = tmp_path / "files"
    data.mkdir()
    X.iloc[:200].assign(price=y[:200].values).to_csv(data / "a.csv", index=False)
    X.iloc[200:].to_csv(data / "b.csv", index=False)
    spec = tmp_path / "spec"
    spec.mkdir()
    (spec / "task.json").write_text(json.dumps({"task": "regression", "target": "price",
                                                "train_path": "../files/a.csv",
                                                "test_path": "../files/b.csv"}))
    res = run(spec, tmp_path / "out")
    assert res["profile"]["status"] == "final" and len(res["predictions"]) == 100


def test_task_from_text_description(tmp_path):
    from sklearn.datasets import make_blobs
    X, lab = make_blobs(300, centers=5, random_state=2)
    d = tmp_path / "mall"
    d.mkdir()
    pd.DataFrame(X, columns=["income", "spending"]).to_csv(d / "customers.csv", index=False)
    (d / "README.md").write_text("Customer segmentation: group the customers into 5 clusters.")
    res = run(d, tmp_path / "out")
    assert res["profile"]["task"] == "clustering"
    assert res["predictions"]["prediction"].nunique() == 5


def test_task_from_folder_name(tmp_path):
    X, y = _toy_supervised()
    d = tmp_path / "clf_07"
    _write(d, X.iloc[:200].assign(outcome=y[:200].values), X.iloc[200:], None)
    res = run(d, tmp_path / "out")
    assert res["profile"]["task"] == "classification"
    assert set(res["predictions"]["prediction"].astype(str)) <= {"yes", "no"}


def _cli(*args, timeout=200):
    cp = subprocess.run([sys.executable, str(ROOT / "main.py")] + [str(a) for a in args] + ["--quiet"],
                        capture_output=True, text=True, timeout=timeout)
    assert cp.returncode == 0, cp.stderr
    return json.loads(cp.stdout.strip().splitlines()[-1])


def test_positional_files_never_overwrite_inputs(tmp_path):
    X, y = _toy_supervised()
    d = _write(tmp_path / "ds", X.iloc[:200].assign(target=y[:200].values), X.iloc[200:],
               {"task": "classification", "target": "target", "id_column": "id"})
    before = {f: (d / f).read_bytes() for f in ("train.csv", "test.csv", "metadata.json")}
    out = tmp_path / "out"
    summary = _cli(d / "train.csv", d / "test.csv", d / "metadata.json", out)
    assert summary["status"] == "ok" and summary["resolved"]["target"] == "target"
    assert (out / "predictions.csv").exists()
    assert all((d / f).read_bytes() == b for f, b in before.items())


def test_fuzzy_keys_roles_inline_and_dry_run(tmp_path):
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(300, centers=3, random_state=1)
    d = tmp_path / "fuzzy"
    d.mkdir()
    pd.DataFrame(X, columns=["u", "v"]).to_csv(d / "data.csv", index=False)
    (d / "info.json").write_text(json.dumps({"task_family": "Clustering",
                                             "required_number_of_groups": 3}))
    s = _cli(d, "--dry-run")
    assert s["status"] == "dry_run" and s["resolved"]["task"] == "clustering"
    assert s["resolved"]["n_clusters"] == 3
    s = _cli("--train", d / "data.csv", "--metadata", '{"task": "clustering", "n_clusters": 4}',
             "-o", tmp_path / "o2")
    assert s["resolved"]["n_clusters"] == 4
    Xs, y = _toy_supervised()
    r = tmp_path / "roles"
    _write(r, Xs.iloc[:200].assign(Churn=y[:200].values), Xs.iloc[200:], None)
    (r / "schema.json").write_text(json.dumps({"columns": [
        {"name": "id", "role": "id"}, {"name": "Churn", "type": "string", "role": "target"}]}))
    s = _cli(r, "--dry-run")
    assert s["resolved"]["target"] == "Churn" and s["resolved"]["id_column"] == "id"


def test_gzip_tables(tmp_path):
    X, y = _toy_supervised()
    d = tmp_path / "gz"
    d.mkdir()
    X.iloc[:200].assign(target=y[:200].values).to_csv(d / "train.csv.gz", index=False)
    X.iloc[200:].to_csv(d / "test.csv.gz", index=False)
    (d / "metadata.json").write_text(json.dumps({"task": "classification", "target": "target"}))
    s = _cli(d, tmp_path / "out")
    assert s["n_predictions"] == 100


def test_fast_scorer_matches_sklearn():
    from automl.supervised import clf_score, fast_clf_score
    rng = np.random.default_rng(1)
    for trial in range(100):
        K = int(rng.integers(2, 10))
        y = rng.integers(0, K, 200)
        pred = rng.integers(0, K, 200)
        if trial % 3 == 0:
            pred = np.where(pred == 0, 1, pred)  # a class that is never predicted
        for m in ("macro_f1", "weighted_f1", "balanced_accuracy", "accuracy"):
            assert abs(fast_clf_score(m, y, pred, K, 1) - clf_score(m, y, pred, None, 1)) < 1e-12


def test_decision_tuning_is_fast_on_large_multiclass():
    import time
    from automl.supervised import tune_decision
    rng = np.random.default_rng(0)
    n, K = 30000, 20
    y = rng.integers(0, K, n)
    P = rng.dirichlet(np.ones(K), n)
    t = time.time()
    tune_decision("macro_f1", y, P, 1)
    assert time.time() - t < 10


def test_output_names_cannot_escape_the_output_folder(tmp_path):
    from automl.io_contract import _file_name, safe_name
    assert safe_name("../../etc") == "etc" and "/" not in safe_name("a/b\\c")
    assert _file_name("../x/preds.csv", "p.csv") == "preds.csv" and _file_name("..", "p.csv") == "p.csv"
