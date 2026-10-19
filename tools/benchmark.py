"""Local checker + scorer.

Runs the standard entry point on every dataset of a suite (one subprocess per
dataset, exactly like an external assessment agent would), validates the
outputs, scores them against the hidden labels and estimates the contest score
(85 model quality split equally across the three tasks, 10 profiling, 5 speed).

Usage::

    python tools/benchmark.py --data data/practice --labels data/private_practice
    python tools/benchmark.py --data data/bench --labels data/private_bench --out runs/bench
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, adjusted_rand_score, balanced_accuracy_score,
                             f1_score, mean_squared_error, normalized_mutual_info_score, r2_score)

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PROFILE_KEYS = ["task", "n_rows", "n_features", "feature_types", "missing_values",
                         "constant_columns", "duplicate_columns", "data_issues"]


def check_outputs(ds_dir: Path, out_dir: Path) -> tuple[list, pd.DataFrame | None, dict | None]:
    """Format checks an organiser's checker would plausibly run."""
    problems = []
    meta = json.loads((ds_dir / "metadata.json").read_text())
    pred_p, prof_p = out_dir / "predictions.csv", out_dir / "profile.json"
    if not pred_p.exists():
        return ["missing predictions.csv"], None, None
    if not prof_p.exists():
        problems.append("missing profile.json")
        prof = None
    else:
        try:
            prof = json.loads(prof_p.read_text())
            for k in REQUIRED_PROFILE_KEYS:
                if k not in prof:
                    problems.append(f"profile missing key {k}")
        except json.JSONDecodeError as exc:
            problems.append(f"profile invalid json: {exc}")
            prof = None
    pred = pd.read_csv(pred_p)
    ref = ds_dir / ("test.csv" if (ds_dir / "test.csv").exists() else "train.csv")
    ref_df = pd.read_csv(ref)
    if len(pred) != len(ref_df):
        problems.append(f"row count {len(pred)} != {len(ref_df)}")
    idc = meta.get("id_column")
    if idc:
        if idc not in pred.columns:
            problems.append(f"id column {idc} missing")
        elif not (pred[idc].astype(str).values == ref_df[idc].astype(str).values).all():
            problems.append("ids not aligned with input order")
    if "prediction" not in pred.columns:
        problems.append("prediction column missing")
    elif pred["prediction"].isna().any():
        problems.append("NaN predictions")
    if meta["task"] == "clustering" and "prediction" in pred.columns:
        k = meta.get("n_clusters")
        nun = pred["prediction"].nunique()
        if k and nun != k:
            problems.append(f"{nun} clusters != required {k}")
    if meta["task"] == "classification" and "prediction" in pred.columns:
        tr = pd.read_csv(ds_dir / "train.csv", usecols=[meta["target"]])
        allowed = set(tr[meta["target"]].dropna().astype(str))
        bad = set(pred["prediction"].astype(str)) - allowed
        if bad:
            problems.append(f"unknown class labels predicted: {sorted(bad)[:5]}")
    return problems, pred, prof


def score(task, y_true, y_pred):
    if task == "classification":
        yt, yp = y_true.astype(str), y_pred.astype(str)
        return {"macro_f1": f1_score(yt, yp, average="macro"), "accuracy": accuracy_score(yt, yp),
                "balanced_accuracy": balanced_accuracy_score(yt, yp)}
    if task == "regression":
        yt, yp = y_true.astype(float), y_pred.astype(float)
        return {"r2": r2_score(yt, yp), "rmse": float(np.sqrt(mean_squared_error(yt, yp)))}
    return {"ari": adjusted_rand_score(y_true, y_pred),
            "nmi": normalized_mutual_info_score(y_true, y_pred)}


def profile_score(prof: dict, truth: dict, ds_dir: Path, meta: dict) -> dict:
    """Fraction of verifiable profile facts that are correct."""
    if not prof:
        return {"profile_accuracy": 0.0}
    tr = pd.read_csv(ds_dir / "train.csv")
    checks = {}
    checks["n_rows"] = prof.get("n_rows") == len(tr)
    miss = tr.isna().sum()
    checks["missing_values"] = all(prof.get("missing_values", {}).get(c) == int(miss[c])
                                   for c in tr.columns)
    if truth.get("constant_columns") is not None:
        checks["constant_columns"] = set(truth["constant_columns"]) <= set(prof.get("constant_columns", []))
    if truth.get("duplicate_columns") is not None:
        want = {d["column"] for d in truth["duplicate_columns"]}
        checks["duplicate_columns"] = want <= set(prof.get("duplicate_columns", []))
    if truth.get("irrelevant_columns"):
        want = set(truth["irrelevant_columns"])
        got = set(prof.get("irrelevant_features", []))
        tp = len(want & got)
        checks["irrelevant_recall"] = tp / len(want)
        checks["irrelevant_precision"] = tp / len(got) if got else 0.0
    ft = prof.get("feature_types", {})
    declared = meta.get("feature_types", {})
    checks["feature_types"] = all(ft.get(c) == t for c, t in declared.items())
    vals = [float(v) for v in checks.values()]
    return {"profile_accuracy": float(np.mean(vals)), **{f"p_{k}": v for k, v in checks.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/practice")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="substring filter on dataset names")
    ap.add_argument("--entry", default=str(ROOT / "main.py"))
    a = ap.parse_args(argv)
    data = Path(a.data)
    labels = Path(a.labels or str(data.parent / f"private_{data.name}"))
    out_root = Path(a.out or (ROOT / "runs" / data.name))
    rows = []
    for ds_dir in sorted(p for p in data.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if a.only and a.only not in ds_dir.name:
            continue
        meta = json.loads((ds_dir / "metadata.json").read_text())
        out_dir = out_root / ds_dir.name
        t0 = time.time()
        cp = subprocess.run([sys.executable, a.entry, str(ds_dir), str(out_dir), "--quiet"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        secs = time.time() - t0
        problems, pred, prof = check_outputs(ds_dir, out_dir)
        if cp.returncode != 0:
            problems.append(f"exit code {cp.returncode}: {cp.stderr.decode()[-300:]}")
        row = {"dataset": ds_dir.name, "task": meta["task"], "seconds": round(secs, 2),
               "problems": "; ".join(problems)}
        lab = pd.read_csv(labels / ds_dir.name / "test_labels.csv")
        if pred is not None and "prediction" in pred.columns and len(pred) == len(lab):
            row.update(score(meta["task"], lab.iloc[:, 1].values, pred["prediction"].values))
            truth = json.loads((labels / ds_dir.name / "truth.json").read_text())
            row.update(profile_score(prof, truth, ds_dir, meta))
            if prof:
                row["model"] = prof.get("selected_model")
                row["cv"] = prof.get("cv_score")
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
               if not k.startswith("p_")}, flush=True)
    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "benchmark.csv", index=False)

    # contest-score estimate (our assumptions: macro-F1, R^2 clipped at 0, ARI)
    q = []
    for task, col in (("classification", "macro_f1"), ("regression", "r2"), ("clustering", "ari")):
        sub = df[df.task == task]
        if len(sub) and col in sub:
            q.append(np.clip(sub[col].fillna(0).values, 0, 1).mean())
    quality = 85 * float(np.mean(q)) if q else 0.0
    prof_pts = 10 * float(df["profile_accuracy"].fillna(0).mean()) if "profile_accuracy" in df else 0
    speed = 5 * float(np.clip((60 - df["seconds"]) / 60, 0, 1).mean())
    print("\nper-task means:")
    for task, cols in (("classification", ["macro_f1", "accuracy", "balanced_accuracy"]),
                       ("regression", ["r2"]), ("clustering", ["ari", "nmi"])):
        sub = df[df.task == task]
        if len(sub):
            print(f"  {task:15s}", {c: round(float(sub[c].mean()), 4) for c in cols if c in sub})
    print(f"  max runtime {df['seconds'].max():.1f}s, mean {df['seconds'].mean():.1f}s")
    print(f"  problems: {int((df['problems'] != '').sum())} datasets")
    print(f"ESTIMATED SCORE  quality {quality:.2f}/85  profile {prof_pts:.2f}/10  "
          f"speed {speed:.2f}/5  total {quality + prof_pts + speed:.2f}/100")
    return 0


if __name__ == "__main__":
    sys.exit(main())
