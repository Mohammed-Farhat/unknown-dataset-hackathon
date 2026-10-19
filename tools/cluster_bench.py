"""Clustering strategy benchmark (used to choose the defaults in automl/clustering.py).

Generates a varied collection of labelled clustering problems (Gaussian blobs
with different shapes / overlaps / sizes, injected noise features, categorical
columns, missing values, rescaled columns, constant and duplicate columns, plus
real labelled tables) and reports the mean ARI of each strategy variant.

    python tools/cluster_bench.py            # all variants
    python tools/cluster_bench.py --quick    # fewer datasets
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets as skd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from automl import clustering as C  # noqa: E402
from automl.features import FeatureBuilder, plan_columns  # noqa: E402
from make_practice_data import messify  # noqa: E402


def make_problems(n_sets: int, seed: int = 11):
    rng = np.random.default_rng(seed)
    probs = []
    kinds = ["iso", "aniso", "unequal", "imbalanced", "iso", "unequal"]
    for i in range(n_sets):
        kind = kinds[i % len(kinds)]
        k = int(rng.integers(2, 9))
        d = int(rng.integers(2, 8))
        n = int(rng.integers(300, 3000))
        std = float(rng.uniform(0.6, 2.0))
        if kind == "imbalanced":
            w = rng.dirichlet(np.ones(k) * 0.7)
            sizes = np.maximum((w * n).astype(int), 15)
            X, y = skd.make_blobs(n_samples=sizes, n_features=d, cluster_std=std,
                                  center_box=(-10, 10), random_state=int(rng.integers(1e6)))
        else:
            stds = rng.uniform(0.5, 2.0, k) * std if kind == "unequal" else std
            X, y = skd.make_blobs(n_samples=n, centers=k, n_features=d, cluster_std=stds,
                                  center_box=(-10, 10), random_state=int(rng.integers(1e6)))
        if kind == "aniso":
            X = X @ rng.normal(0, 1, (d, d))
        df = pd.DataFrame(X, columns=[f"v{j}" for j in range(d)])
        if rng.random() < 0.3:  # an informative categorical derived from the label
            df["segment"] = np.where(rng.random(len(y)) < 0.8, y, rng.integers(0, k, len(y)))
            df["segment"] = "s" + df["segment"].astype(str)
        df, truth = messify(df, rng, n_noise_num=int(rng.integers(0, 7)),
                            n_noise_cat=int(rng.integers(0, 2)), n_const=int(rng.integers(0, 2)),
                            n_dup=int(rng.integers(0, 2)), missing_rate=float(rng.uniform(0, 0.1)))
        probs.append((f"synth_{i}_{kind}_k{k}", df, y, k))
    # real labelled tables (labels hidden)
    for name, loader in [("iris", skd.load_iris), ("wine", skd.load_wine),
                         ("breast_cancer", skd.load_breast_cancer)]:
        b = loader(as_frame=True)
        df = b.data.copy()
        df, _ = messify(df, rng, n_noise_num=1, n_noise_cat=0, n_const=1, n_dup=0,
                        missing_rate=0.02, rescale=False)
        probs.append((name, df, b.target.values, len(np.unique(b.target))))
    # mall-like (two informative features defining five groups + irrelevant age/gender)
    centers = np.array([[25, 20], [25, 80], [55, 50], [88, 18], [88, 82]])
    y = rng.integers(0, 5, 600)
    Xm = centers[y] + rng.normal(0, [6, 8], (600, 2))
    mall = pd.DataFrame({"Gender": rng.choice(["Male", "Female"], 600), "Age": rng.integers(18, 70, 600),
                         "Income": Xm[:, 0], "Spending": Xm[:, 1]})
    probs.append(("mall_like", mall, y, 5))
    for nm, (Xs, ys) in {"moons": skd.make_moons(800, noise=0.07, random_state=1),
                         "circles": skd.make_circles(800, noise=0.05, factor=0.4, random_state=1)}.items():
        df = pd.DataFrame(Xs * [1, 100], columns=["a", "b"])
        df["noise"] = rng.normal(0, 1, len(df))
        probs.append((nm, df, ys, 2))
    return probs


def run_variant(probs, opts):
    C.OPTIONS.update(opts)
    scores, secs = [], []
    for name, df, y, k in probs:
        plan = plan_columns(df, None, None, None, {})
        fb = FeatureBuilder(plan).fit(df)
        F = fb.transform(df)
        t = time.time()
        lab, info = C.cluster(F, plan.numeric, plan.categorical, k)
        secs.append(time.time() - t)
        scores.append(adjusted_rand_score(y, lab))
    return np.array(scores), np.array(secs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sets", type=int, default=36)
    ap.add_argument("--variants", default="all", choices=["all", "fs", "model", "final"])
    a = ap.parse_args(argv)
    probs = make_problems(12 if a.quick else a.sets)
    names = [p[0] for p in probs]
    grid = {
        "fs": ["eta_loo_dep", "eta_loo", "eta_loo_guard", "eta", "none"],
        "model": ["auto", "gmm_tied", "gmm_bic", "gmm_ft", "kmeans"],
        "spectral": ["numeric", "off"],
    }
    if a.variants == "fs":
        grid = {"fs": ["eta_loo_dep", "eta_loo"], "model": ["auto"], "spectral": ["numeric"]}
    if a.variants == "final":
        grid = None
    if a.variants == "model":
        grid = {"fs": ["eta_loo_dep"], "model": ["auto", "auto_tied", "auto_bic3", "kmeans"],
                "spectral": ["numeric"]}
    rows, per = [], {}
    base = dict(C.OPTIONS)
    combos = (list(itertools.product(*grid.values())) if grid is not None else
              [("none", "kmeans", "off"), ("eta_loo_dep", "auto", "numeric")])
    for fs, model, sp in combos:
        if a.variants == "all" and fs not in ("eta_loo", "eta_loo_dep") and                 (model not in ("auto", "gmm_tied") or sp != "numeric"):
            continue
        opts = {"fs": fs, "model": model, "spectral": sp}
        sc, se = run_variant(probs, opts)
        key = f"fs={fs} model={model} spectral={sp}"
        per[key] = sc
        synth = np.array([n.startswith("synth") for n in names])
        rows.append({"variant": key, "mean_ari": sc.mean(), "synthetic": sc[synth].mean(),
                     "real+shapes": sc[~synth].mean(), "worst": sc.min(), "sec_mean": se.mean(),
                     "sec_max": se.max()})
        print(rows[-1], flush=True)
    C.OPTIONS.update(base)
    df = pd.DataFrame(rows).sort_values("mean_ari", ascending=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    best = df.iloc[0]["variant"]
    print("\nper-dataset ARI of the best variant:")
    print(pd.Series(per[best], index=names).round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
