"""Generate local practice / benchmark datasets in the assumed hackathon contract.

The organisers' nine practice datasets were not available to us, so this
script builds our own stand-ins that mimic the announced scope:

* real tables matching the announcement's Kaggle examples (fetched from OpenML
  when a network connection is available, synthetic stand-ins otherwise);
* synthetic tables with every announced difficulty injected: missing values,
  differently scaled features, categorical columns, constant and duplicate
  columns, irrelevant features, imbalanced classes and nonlinear relationships.

Each dataset is written as::

    <out>/<name>/train.csv       features (+ target column for supervised tasks)
    <out>/<name>/test.csv        features only (omitted for some clustering sets)
    <out>/<name>/metadata.json   task, target, id column, feature types, n_clusters
    <labels>/<name>/test_labels.csv   hidden labels used only for local scoring
    <labels>/<name>/truth.json        injected data issues (profile ground truth)

Usage::

    python tools/make_practice_data.py --suite practice --out data/practice --labels data/private
    python tools/make_practice_data.py --suite bench --out data/bench --labels data/private_bench
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets as skd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Messiness injection
# --------------------------------------------------------------------------- #
def messify(df: pd.DataFrame, rng: np.random.Generator, *, protect=(), missing_rate=0.05,
            n_missing_cols=3, n_const=1, n_dup=1, n_noise_num=2, n_noise_cat=1,
            rescale=True):
    """Inject the announced data issues into a feature table.

    Returns the modified frame and a dict describing what was injected.
    """
    natural_noise = list(df.attrs.get("irrelevant", []))
    df = df.copy()
    truth = {"constant_columns": [], "duplicate_columns": [], "irrelevant_columns": natural_noise,
             "rescaled_columns": [], "missing_injected_columns": []}
    n = len(df)
    feat_cols = [c for c in df.columns if c not in protect]
    num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c])]

    if rescale and num_cols:
        for c in rng.choice(num_cols, size=min(len(num_cols), max(1, len(num_cols) // 3)),
                            replace=False):
            factor = float(10.0 ** rng.choice([-3, -2, 2, 3, 4]))
            df[c] = df[c].astype(float) * factor
            truth["rescaled_columns"].append(str(c))

    for i in range(n_noise_num):
        name = f"noise_num_{i}"
        df[name] = (rng.normal(0, 1, n) * 10 ** rng.uniform(-1, 3) if i % 2 == 0
                    else rng.uniform(-50, 50, n))
        truth["irrelevant_columns"].append(name)
    for i in range(n_noise_cat):
        name = f"noise_cat_{i}"
        df[name] = rng.choice(["alpha", "beta", "gamma", "delta"], size=n)
        truth["irrelevant_columns"].append(name)

    for i in range(n_const):
        name = f"const_{i}"
        df[name] = 1.0 if i % 2 == 0 else "same"
        truth["constant_columns"].append(name)

    cand = [c for c in df.columns if c not in protect and c not in truth["constant_columns"]]
    for i in range(min(n_dup, len(cand))):
        src = cand[int(rng.integers(len(cand)))]
        name = f"{src}_copy"
        df[name] = df[src]
        truth["duplicate_columns"].append({"column": name, "duplicate_of": str(src)})

    cand = [c for c in df.columns if c not in protect and c not in truth["constant_columns"]
            and not c.endswith("_copy")]
    if n_missing_cols and missing_rate > 0:
        for c in rng.choice(cand, size=min(n_missing_cols, len(cand)), replace=False):
            mask = rng.random(n) < missing_rate * rng.uniform(0.5, 2.0)
            df.loc[mask, c] = np.nan
            truth["missing_injected_columns"].append(str(c))
    # keep duplicate columns exact duplicates (copy missingness too)
    for d in truth["duplicate_columns"]:
        df[d["column"]] = df[d["duplicate_of"]]

    # shuffle column order so the target / noise are not always last
    cols = list(df.columns)
    order = rng.permutation(len(cols))
    df = df[[cols[i] for i in order]]
    return df, truth


def feature_types(df: pd.DataFrame, exclude=()):
    out = {}
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            out[str(c)] = "categorical"
        else:
            out[str(c)] = "numeric"
    return out


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def write_supervised(name, X, y, task, out, labels, rng, truth=None, target="target",
                     test_size=0.25, description=""):
    X = X.reset_index(drop=True).copy()
    y = pd.Series(np.asarray(y), name=target).reset_index(drop=True)
    X.insert(0, "id", np.arange(len(X)))
    strat = y if task == "classification" and y.value_counts().min() >= 2 else None
    tr_idx, te_idx = train_test_split(np.arange(len(X)), test_size=test_size,
                                      random_state=int(rng.integers(1_000_000)), stratify=strat)
    tr_idx, te_idx = np.sort(tr_idx), np.sort(te_idx)
    train = X.iloc[tr_idx].copy()
    train[target] = y.iloc[tr_idx].values
    test = X.iloc[te_idx].copy()
    d = Path(out) / name
    d.mkdir(parents=True, exist_ok=True)
    train.to_csv(d / "train.csv", index=False)
    test.to_csv(d / "test.csv", index=False)
    meta = {
        "dataset_id": name,
        "task": task,
        "target": target,
        "id_column": "id",
        "feature_types": feature_types(X, exclude=("id",)),
        "description": description,
    }
    (d / "metadata.json").write_text(json.dumps(meta, indent=2))
    ld = Path(labels) / name
    ld.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test["id"].values, target: y.iloc[te_idx].values}).to_csv(
        ld / "test_labels.csv", index=False)
    (ld / "truth.json").write_text(json.dumps(truth or {}, indent=2, default=str))
    print(f"[{task:14s}] {name:28s} train={train.shape} test={test.shape}")


def write_clustering(name, X, y, k, out, labels, rng, truth=None, with_test=False,
                     description=""):
    X = X.reset_index(drop=True).copy()
    y = np.asarray(y)
    X.insert(0, "id", np.arange(len(X)))
    d = Path(out) / name
    d.mkdir(parents=True, exist_ok=True)
    ld = Path(labels) / name
    ld.mkdir(parents=True, exist_ok=True)
    if with_test:
        tr_idx, te_idx = train_test_split(np.arange(len(X)), test_size=0.3,
                                          random_state=int(rng.integers(1_000_000)))
        tr_idx, te_idx = np.sort(tr_idx), np.sort(te_idx)
        X.iloc[tr_idx].to_csv(d / "train.csv", index=False)
        X.iloc[te_idx].to_csv(d / "test.csv", index=False)
        lab = pd.DataFrame({"id": X["id"].values[te_idx], "cluster": y[te_idx]})
    else:
        X.to_csv(d / "train.csv", index=False)
        lab = pd.DataFrame({"id": X["id"].values, "cluster": y})
    lab.to_csv(ld / "test_labels.csv", index=False)
    meta = {
        "dataset_id": name,
        "task": "clustering",
        "target": None,
        "id_column": "id",
        "n_clusters": int(k),
        "feature_types": feature_types(X, exclude=("id",)),
        "description": description,
    }
    (d / "metadata.json").write_text(json.dumps(meta, indent=2))
    (ld / "truth.json").write_text(json.dumps(truth or {}, indent=2, default=str))
    print(f"[clustering    ] {name:28s} rows={len(X)} k={k} test={'yes' if with_test else 'no'}")


# --------------------------------------------------------------------------- #
# Real tables (OpenML) with synthetic fall-backs
# --------------------------------------------------------------------------- #
def openml(data_id, cache):
    from sklearn.datasets import fetch_openml
    return fetch_openml(data_id=data_id, as_frame=True, data_home=cache, parser="auto").frame


def real_telco(cache):
    df = openml(42178, cache)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = df.pop("Churn").astype(str).str.strip("'")
    return df, y


def real_ai4i(cache):
    df = openml(42890, cache)
    y = df.pop("Machine failure").astype(int)
    df = df.drop(columns=["TWF", "HDF", "PWF", "OSF", "RNF"])  # failure modes leak the target
    return df, y


def real_house(cache):
    df = openml(42165, cache)
    df = df.drop(columns=["Id"])
    y = df.pop("SalePrice").astype(float)
    return df, y


def real_bike(cache):
    df = openml(42712, cache)
    y = df.pop("count").astype(float)
    for c in ["season", "holiday", "workingday", "weather"]:
        df[c] = df[c].astype(str)
    return df, y


def real_penguins(cache):
    df = openml(42585, cache)
    y = df.pop("species").astype(str)
    return df, pd.factorize(y)[0]


# --------------------------------------------------------------------------- #
# Synthetic generators
# --------------------------------------------------------------------------- #
def synth_classification(rng, n=4000, n_classes=3, imbalance=0.0, nonlinear=True):
    seed = int(rng.integers(1_000_000))
    if imbalance > 0:
        w = np.array([1.0] + [imbalance] * (n_classes - 1))
        w = (w / w.sum()).tolist()
    else:
        w = None
    X, y = skd.make_classification(n_samples=n, n_features=8, n_informative=5, n_redundant=1,
                                   n_classes=n_classes, n_clusters_per_class=2, weights=w,
                                   class_sep=1.0, flip_y=0.01, shuffle=False, random_state=seed)
    order = rng.permutation(n)
    X, y = X[order], y[order]
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if nonlinear:
        z = np.sin(2 * df["x0"]) + df["x1"] * df["x2"]
        df["x0"] = df["x0"] ** 3
        # categorical feature carrying signal
        y_shift = (z > np.median(z)).astype(int)
        y = np.where((rng.random(n) < 0.15) & (y_shift == 1), (y + 1) % n_classes, y)
    df["segment"] = pd.cut(df["x3"], bins=4, labels=["low", "mid", "high", "top"]).astype(str)
    df["region"] = rng.choice(["north", "south", "east", "west"], size=n)
    labels = np.array([f"class_{c}" for c in y])
    df.attrs["irrelevant"] = ["x6", "x7", "region"]  # useless make_classification features
    return df, labels


def synth_regression(rng, n=3000, kind="friedman", skewed=False):
    seed = int(rng.integers(1_000_000))
    if kind == "friedman":
        X, y = skd.make_friedman1(n_samples=n, n_features=7, noise=1.0, random_state=seed)
        noise = ["f5", "f6"]  # Friedman #1 only uses the first five inputs
    else:
        X, y, coef = skd.make_regression(n_samples=n, n_features=8, n_informative=5, noise=10.0,
                                         coef=True, random_state=seed)
        noise = [f"f{i}" for i in np.flatnonzero(coef == 0)]
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df.attrs["irrelevant"] = noise
    cat = rng.choice(["A", "B", "C", "D"], size=n)
    effect = pd.Series(cat).map({"A": 0.0, "B": 2.0, "C": -1.5, "D": 4.0}).values
    df["category"] = cat
    y = y + effect * (np.std(y) / 3)
    if skewed:
        y = np.exp((y - y.mean()) / y.std() * 0.7 + 10)
    return df, y


def synth_demand_datetime(rng, n=5000):
    """Bike-sharing-like table with a raw datetime column and count target."""
    start = pd.Timestamp("2021-01-01")
    hours = np.sort(rng.choice(24 * 730, size=n, replace=False))
    ts = start + pd.to_timedelta(hours, unit="h")
    hour = ts.hour.values
    dow = ts.dayofweek.values
    month = ts.month.values
    temp = 15 + 10 * np.sin((month - 4) / 12 * 2 * np.pi) + rng.normal(0, 3, n)
    humidity = np.clip(rng.normal(0.6, 0.15, n), 0, 1)
    weather = rng.choice(["clear", "mist", "rain", "storm"], p=[0.6, 0.25, 0.13, 0.02], size=n)
    workday = (dow < 5).astype(int)
    peak = np.where(workday == 1,
                    np.exp(-((hour - 8) ** 2) / 4) + np.exp(-((hour - 17.5) ** 2) / 5),
                    np.exp(-((hour - 14) ** 2) / 12))
    wmult = pd.Series(weather).map({"clear": 1.0, "mist": 0.85, "rain": 0.45, "storm": 0.1}).values
    lam = (5 + 300 * peak * (1 + 0.03 * (temp - 15)).clip(0.2)) * wmult * (1.2 - 0.5 * humidity)
    lam *= 1 + 0.4 * (ts.year.values - 2021)
    y = rng.poisson(lam).astype(float)
    df = pd.DataFrame({"datetime": ts.strftime("%Y-%m-%d %H:%M:%S"), "temp": temp.round(2),
                       "humidity": humidity.round(3), "weather": weather,
                       "workingday": workday})
    return df, y


def synth_blobs(rng, n=1500, k=4, n_inf=4, std=1.0, aniso=False, unequal=False):
    seed = int(rng.integers(1_000_000))
    stds = rng.uniform(0.5, 2.0, k) * std if unequal else std
    X, y = skd.make_blobs(n_samples=n, centers=k, n_features=n_inf, cluster_std=stds,
                          center_box=(-10, 10), random_state=seed)
    if aniso:
        A = rng.normal(0, 1, (n_inf, n_inf))
        X = X @ A
    df = pd.DataFrame(X, columns=[f"v{i}" for i in range(n_inf)])
    return df, y


def synth_mall(rng, n=600):
    """Mall-customer-like table: income and spending define five segments."""
    centers = np.array([[25, 20], [25, 80], [55, 50], [88, 18], [88, 82]])
    k = len(centers)
    y = rng.integers(0, k, n)
    X = centers[y] + rng.normal(0, [6, 8], (n, 2))
    df = pd.DataFrame({
        "CustomerID": np.arange(1, n + 1),
        "Gender": rng.choice(["Male", "Female"], size=n),
        "Age": rng.integers(18, 70, n),
        "Annual Income (k$)": X[:, 0].round(1),
        "Spending Score (1-100)": np.clip(X[:, 1], 1, 100).round(0),
    })
    return df, y, k


# --------------------------------------------------------------------------- #
# Suites
# --------------------------------------------------------------------------- #
def build_practice(out, labels, cache, offline):
    rng = np.random.default_rng(2026)

    def real_or(fn, fallback):
        if offline:
            return fallback()
        try:
            return fn(cache)
        except Exception as exc:  # network or OpenML failure -> synthetic stand-in
            print(f"  ! could not fetch real data ({exc.__class__.__name__}); using synthetic")
            return fallback()

    # --- classification
    X, y = real_or(real_telco, lambda: synth_classification(rng, n=7000, n_classes=2, imbalance=0.35))
    X, t = messify(X, rng, n_noise_num=1, n_noise_cat=0, n_const=1, n_dup=1, missing_rate=0.02,
                   rescale=False)
    write_supervised("clf_telco_churn", X, y, "classification", out, labels, rng, t, target="Churn")

    X, y = real_or(real_ai4i, lambda: synth_classification(rng, n=10000, n_classes=2, imbalance=0.04))
    X, t = messify(X, rng, n_noise_num=2, n_noise_cat=1, n_const=1, n_dup=1, missing_rate=0.03)
    write_supervised("clf_machine_failure", X, y, "classification", out, labels, rng, t,
                     target="Machine failure")

    X, y = synth_classification(rng, n=4000, n_classes=4, imbalance=0.3)
    X, t = messify(X, rng, n_noise_num=4, n_noise_cat=2, n_const=2, n_dup=2, missing_rate=0.08)
    write_supervised("clf_synthetic_multiclass", X, y, "classification", out, labels, rng, t,
                     target="label")

    # --- regression
    X, y = real_or(real_house, lambda: synth_regression(rng, n=1460, kind="linear", skewed=True))
    X, t = messify(X, rng, n_noise_num=2, n_noise_cat=1, n_const=1, n_dup=1, missing_rate=0.0,
                   n_missing_cols=0, rescale=False)
    write_supervised("reg_house_prices", X, y, "regression", out, labels, rng, t, target="SalePrice")

    X, y = synth_demand_datetime(rng, n=6000)
    X, t = messify(X, rng, n_noise_num=1, n_noise_cat=1, n_const=1, n_dup=0, missing_rate=0.03,
                   rescale=False)
    write_supervised("reg_bike_demand", X, y, "regression", out, labels, rng, t, target="count")

    X, y = synth_regression(rng, n=3000, kind="friedman")
    X, t = messify(X, rng, n_noise_num=4, n_noise_cat=1, n_const=1, n_dup=1, missing_rate=0.06)
    write_supervised("reg_synthetic_friedman", X, y, "regression", out, labels, rng, t,
                     target="y")

    # --- clustering
    X, y, k = synth_mall(rng)
    write_clustering("clu_mall_customers", X, y, k, out, labels, rng,
                     {"irrelevant_columns": ["Gender", "Age"], "id_like_columns": ["CustomerID"]})

    X, y = real_or(real_penguins, lambda: synth_blobs(rng, n=344, k=3, n_inf=4))
    k = len(np.unique(y))
    X, t = messify(X, rng, n_noise_num=0, n_noise_cat=0, n_const=1, n_dup=0, missing_rate=0.0,
                   n_missing_cols=0, rescale=False)
    write_clustering("clu_penguins", X, y, k, out, labels, rng, t)

    X, y = synth_blobs(rng, n=2000, k=5, n_inf=4, std=1.5, unequal=True)
    X, t = messify(X, rng, n_noise_num=3, n_noise_cat=1, n_const=1, n_dup=1, missing_rate=0.04)
    write_clustering("clu_synthetic_blobs", X, y, 5, out, labels, rng, t, with_test=True)


def build_bench(out, labels, cache, offline):
    """Eighteen extra hidden-like datasets (6 per task) with varied difficulties."""
    rng = np.random.default_rng(7)

    def real_or(fn, fallback):
        if offline:
            return fallback()
        try:
            return fn(cache)
        except Exception as exc:
            print(f"  ! could not fetch real data ({exc.__class__.__name__}); using synthetic")
            return fallback()

    # classification
    for i, (n, c, imb) in enumerate([(3000, 2, 0.1), (8000, 3, 0.0), (1500, 5, 0.5), (20000, 2, 0.02)]):
        X, y = synth_classification(rng, n=n, n_classes=c, imbalance=imb, nonlinear=i % 2 == 0)
        X, t = messify(X, rng, n_noise_num=int(rng.integers(0, 8)), n_noise_cat=int(rng.integers(0, 3)),
                       n_const=int(rng.integers(0, 3)), n_dup=int(rng.integers(0, 3)),
                       missing_rate=float(rng.uniform(0, 0.15)))
        write_supervised(f"b_clf_synth_{i}", X, y, "classification", out, labels, rng, t)

    def credit(cache):
        df = openml(31, cache)
        y = df.pop("class").astype(str)
        return df, y
    X, y = real_or(credit, lambda: synth_classification(rng, n=1000, n_classes=2, imbalance=0.4))
    X, t = messify(X, rng, n_noise_num=2, n_noise_cat=1, missing_rate=0.05)
    write_supervised("b_clf_credit_g", X, y, "classification", out, labels, rng, t, target="class")

    def adult(cache):
        df = openml(179, cache).sample(15000, random_state=0)
        y = df.pop("class").astype(str)
        return df, y
    X, y = real_or(adult, lambda: synth_classification(rng, n=15000, n_classes=2, imbalance=0.3))
    X, t = messify(X, rng, n_noise_num=1, n_noise_cat=1, missing_rate=0.02, rescale=False)
    write_supervised("b_clf_adult", X, y, "classification", out, labels, rng, t, target="income")

    # regression
    for i, (n, kind, skew) in enumerate([(2000, "friedman", False), (10000, "linear", True),
                                         (800, "friedman", True), (30000, "linear", False)]):
        X, y = synth_regression(rng, n=n, kind=kind, skewed=skew)
        X, t = messify(X, rng, n_noise_num=int(rng.integers(0, 8)), n_noise_cat=int(rng.integers(0, 3)),
                       n_const=int(rng.integers(0, 3)), n_dup=int(rng.integers(0, 3)),
                       missing_rate=float(rng.uniform(0, 0.15)))
        write_supervised(f"b_reg_synth_{i}", X, y, "regression", out, labels, rng, t)

    X, y = real_or(real_bike, lambda: synth_demand_datetime(rng, n=8000))
    X, t = messify(X, rng, n_noise_num=1, n_noise_cat=0, missing_rate=0.02, rescale=False)
    write_supervised("b_reg_bike_hourly", X, y, "regression", out, labels, rng, t, target="count")

    def wine_quality(cache):
        df = openml(40691, cache)
        y = df.pop("class").astype(float)
        return df, y
    X, y = real_or(wine_quality, lambda: synth_regression(rng, n=1600))
    X, t = messify(X, rng, n_noise_num=3, n_noise_cat=1, missing_rate=0.05)
    write_supervised("b_reg_wine_quality", X, y, "regression", out, labels, rng, t, target="quality")

    # clustering
    configs = [dict(n=1000, k=3, n_inf=2, std=1.0), dict(n=3000, k=6, n_inf=5, std=2.0, unequal=True),
               dict(n=800, k=4, n_inf=3, std=1.0, aniso=True), dict(n=5000, k=8, n_inf=6, std=1.5)]
    for i, cfg in enumerate(configs):
        X, y = synth_blobs(rng, **cfg)
        X, t = messify(X, rng, n_noise_num=int(rng.integers(0, 6)), n_noise_cat=int(rng.integers(0, 2)),
                       n_const=int(rng.integers(0, 2)), n_dup=int(rng.integers(0, 2)),
                       missing_rate=float(rng.uniform(0, 0.1)))
        write_clustering(f"b_clu_blobs_{i}", X, y, cfg["k"], out, labels, rng, t, with_test=i % 2 == 1)

    for name, loader in [("b_clu_wine", skd.load_wine), ("b_clu_iris", skd.load_iris)]:
        b = loader(as_frame=True)
        X, y = b.data, b.target.values
        X, t = messify(X, rng, n_noise_num=1, n_noise_cat=0, n_const=1, n_dup=0, missing_rate=0.03,
                       rescale=False)
        write_clustering(name, X, y, len(np.unique(y)), out, labels, rng, t)


def build_examples(out, labels, cache, offline):
    """Three small offline datasets (one per task) shipped in the repository."""
    rng = np.random.default_rng(42)
    X, y = synth_classification(rng, n=800, n_classes=2, imbalance=0.25)
    X, t = messify(X, rng, n_noise_num=2, n_noise_cat=1, n_const=1, n_dup=1, missing_rate=0.05)
    write_supervised("classification_demo", X, y, "classification", out, labels, rng, t,
                     target="label")
    X, y = synth_demand_datetime(rng, n=800)
    X, t = messify(X, rng, n_noise_num=1, n_noise_cat=1, n_const=1, n_dup=1, missing_rate=0.03,
                   rescale=False)
    write_supervised("regression_demo", X, y, "regression", out, labels, rng, t, target="count")
    X, y = synth_blobs(rng, n=600, k=4, n_inf=3, std=1.0)
    X, t = messify(X, rng, n_noise_num=2, n_noise_cat=1, n_const=1, n_dup=1, missing_rate=0.03)
    write_clustering("clustering_demo", X, y, 4, out, labels, rng, t)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["practice", "bench", "examples"], default="practice")
    ap.add_argument("--out", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--cache", default=os.path.join("data", ".openml_cache"))
    ap.add_argument("--offline", action="store_true", help="never download; synthetic stand-ins only")
    a = ap.parse_args(argv)
    out = a.out or os.path.join("data", a.suite)
    labels = a.labels or os.path.join("data", f"private_{a.suite}")
    if a.suite == "practice":
        build_practice(out, labels, a.cache, a.offline)
    elif a.suite == "examples":
        build_examples(out, labels, a.cache, True)
    else:
        build_bench(out, labels, a.cache, a.offline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
