"""Clustering with a known number of groups.

Recipe (chosen from a benchmark of 40 synthetic + 4 real labelled tables):

1. Numeric columns: median imputation + standardisation. Categorical columns:
   most-frequent imputation + unscaled one-hot (each block has variance <= 1,
   comparable to one standardised numeric column).
2. Irrelevant-feature removal without labels: iterative eta^2 saliency (share
   of a feature's variance explained by the current clusters), followed by a
   leave-one-out check (a pure-noise feature cannot be predicted by clusters
   found without it). Guarded so that features which only matter jointly
   (e.g. income x spending) are never discarded.
3. Model: Gaussian mixture (full vs diagonal covariance, chosen by BIC); KMeans
   when categorical / low-cardinality features remain; spectral clustering
   replaces it when it gives a much cleaner k-NN graph cut (non-convex groups).
4. Labels are 0..k-1 integers, exactly k non-empty groups, deterministic.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SEED = 0
# Tunable strategy switches (defaults chosen by tools/cluster_bench.py).
OPTIONS = {
    "fs": "eta_loo_dep",      # none | eta | eta_loo | eta_loo_guard | eta_loo_dep
    "model": "auto",          # auto | kmeans | gmm_full | gmm_tied | gmm_bic
    "spectral": "numeric",    # off | numeric | on
}
FS_SAMPLE = 4000        # rows used for feature selection
SPECTRAL_MAX = 3000     # rows used to fit spectral clustering
MAX_LEVELS = 20


# --------------------------------------------------------------------------- #
# Matrix construction
# --------------------------------------------------------------------------- #
def build_matrix(F: pd.DataFrame, numeric: list, categorical: list):
    """Return (X, groups, names). groups maps original column -> matrix columns."""
    blocks, groups, names = [], {}, []
    col = 0
    for c in numeric:
        v = F[c].astype(float).to_numpy()
        med = np.nanmedian(v) if np.isfinite(v).any() else 0.0
        v = np.where(np.isfinite(v), v, med)
        sd = v.std()
        if sd <= 1e-12:
            continue
        blocks.append(((v - v.mean()) / sd)[:, None])
        groups[c] = [col]
        names.append(c)
        col += 1
    for c in categorical:
        s = F[c].astype(object)
        s = s.where(s.notna(), None)
        if s.isna().all():
            continue
        mode = s.mode(dropna=True)
        s = s.fillna(mode.iloc[0] if len(mode) else "missing").astype(str)
        vc = s.value_counts()
        levels = list(vc.index[:MAX_LEVELS])
        if len(levels) < 2:
            continue
        s = s.where(s.isin(levels), "__other__")
        d = pd.get_dummies(s).astype(float)
        if d.shape[1] < 2:
            continue
        blocks.append(d.to_numpy())
        groups[c] = list(range(col, col + d.shape[1]))
        names.append(c)
        col += d.shape[1]
    if not blocks:
        return np.zeros((len(F), 0)), {}, []
    return np.hstack(blocks), groups, names


def eta2(X: np.ndarray, labels: np.ndarray, groups: dict) -> dict:
    """Between-cluster share of variance per original column (one-hot blocks summed)."""
    mu = X.mean(axis=0)
    tss = ((X - mu) ** 2).sum(axis=0)
    bss = np.zeros(X.shape[1])
    for lab in np.unique(labels):
        m = labels == lab
        bss += m.sum() * (X[m].mean(axis=0) - mu) ** 2
    out = {}
    for g, cols in groups.items():
        t = tss[cols].sum()
        out[g] = float(bss[cols].sum() / t) if t > 1e-12 else 0.0
    return out


def _cols(groups, keep):
    return [i for g in keep for i in groups[g]]


def _kmeans(X, k, n_init=10, seed=SEED):
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(X)


DEP_PROTECT = 0.12   # pairwise-MI level above which a feature is never dropped
DEP_MAX_FEATURES = 40


def dependence_scores(X: np.ndarray, groups: dict, max_rows: int = 1000) -> dict:
    """Strongest mutual information of each original column with any other column.

    Cluster structure makes informative columns statistically dependent on each
    other (also for non-convex shapes such as rings, where no single column is
    informative on its own), while injected noise columns are independent of
    everything. On the benchmark, noise never exceeded 0.09 and informative
    columns of real tables never fell below 0.13.
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    names = list(groups)
    if len(names) < 2 or len(names) > DEP_MAX_FEATURES:
        return {}
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(X)) if len(X) <= max_rows else rng.choice(len(X), max_rows, replace=False)
    Xs = X[idx]
    rep = np.zeros((len(idx), len(names)))
    disc = np.zeros(len(names), dtype=bool)
    for j, g in enumerate(names):
        cols = groups[g]
        if len(cols) > 1:
            rep[:, j] = Xs[:, cols].argmax(1)
            disc[j] = True
        else:
            rep[:, j] = Xs[:, cols[0]]
    M = np.zeros((len(names), len(names)))
    for j in range(len(names)):
        others = [i for i in range(len(names)) if i != j]
        if disc[j]:
            mi = mutual_info_classif(rep[:, others], rep[:, j].astype(int),
                                     discrete_features=disc[others], random_state=SEED)
        else:
            mi = mutual_info_regression(rep[:, others], rep[:, j],
                                        discrete_features=disc[others], random_state=SEED)
        M[j, others] = mi
    S = np.maximum(M, M.T)
    return {g: float(S[j].max()) for j, g in enumerate(names)}


def select_features(X: np.ndarray, groups: dict, k: int, deadline: float) -> tuple[list, dict]:
    """Iterative eta^2 saliency + leave-one-out check (+ dependence protection)."""
    names = list(groups)
    info = {"eta2": {}, "loo_eta2": {}, "removed": []}
    if len(names) <= 1 or OPTIONS.get("fs") == "none":
        return names, info
    rng = np.random.default_rng(SEED)
    Xs = X if len(X) <= FS_SAMPLE else X[rng.choice(len(X), FS_SAMPLE, replace=False)]
    protected: set = set()
    if OPTIONS.get("fs") == "eta_loo_dep":
        try:
            dep = dependence_scores(X, groups)
        except Exception:
            dep = {}
        info["dependence"] = dep
        protected = {g for g, v in dep.items() if v >= DEP_PROTECT}

    keep = list(names)
    for _ in range(5):
        lab = _kmeans(Xs[:, _cols(groups, keep)], k)
        e = eta2(Xs, lab, groups)
        mx = max(e[g] for g in keep)
        new = [g for g in names if (e[g] >= 0.05 and e[g] >= 0.25 * mx) or g in protected]
        info["eta2"] = e
        if not new:
            break
        if set(new) == set(keep):
            break
        keep = new
        if time.time() > deadline:
            break

    if len(keep) >= 3 and time.time() < deadline and OPTIONS.get("fs") != "eta":
        lab_all = _kmeans(Xs[:, _cols(groups, keep)], k, n_init=4)
        loo, stable = {}, {}
        for g in keep:
            others = [h for h in keep if h != g]
            lab = _kmeans(Xs[:, _cols(groups, others)], k, n_init=4)
            loo[g] = eta2(Xs[:, groups[g]], lab, {g: list(range(len(groups[g])))})[g]
            # how much the partition changes when g is removed: a feature that only
            # matters jointly with another (XOR-like) changes it a lot -> keep it.
            stable[g] = adjusted_rand_score(lab_all, lab)
            if time.time() > deadline:
                break
        if len(loo) == len(keep):
            mx = max(loo.values())
            thr = max(0.05, 0.2 * mx)
            guard = OPTIONS.get("fs") == "eta_loo_guard"
            new = [g for g in keep if loo[g] >= thr or (guard and stable[g] < 0.6)
                   or g in protected]
            info["loo_eta2"] = loo
            info["loo_partition_stability"] = stable
            if new:
                keep = new
    info["removed"] = [g for g in names if g not in keep]
    return keep, info


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def relabel(labels: np.ndarray) -> np.ndarray:
    """Map labels to 0..k-1 in order of first appearance."""
    labels = np.asarray(labels)
    _, first = np.unique(labels, return_index=True)
    order = labels[np.sort(first)]
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[v] for v in labels], dtype=int)


def knn_cut_fraction(X: np.ndarray, labels: np.ndarray, n_neighbors: int = 10,
                     max_rows: int = 3000) -> float:
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(X)) if len(X) <= max_rows else rng.choice(len(X), max_rows, replace=False)
    Xs, ls = X[idx], labels[idx]
    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(Xs))).fit(Xs)
    _, ind = nn.kneighbors(Xs)
    return float((ls[ind[:, 1:]] != ls[:, None]).mean())


def fit_gmm(X, k, deadline, covs=("full", "diag")):
    best = None
    for cov in covs:
        if time.time() > deadline and best is not None:
            break
        try:
            g = GaussianMixture(n_components=k, covariance_type=cov, n_init=3, reg_covar=1e-4,
                                random_state=SEED, max_iter=200).fit(X)
            bic = g.bic(X)
            lab = g.predict(X)
        except Exception:
            continue
        if len(np.unique(lab)) < k:
            continue
        if best is None or bic < best[0]:
            best = (bic, cov, lab)
    return best


def fit_spectral(X, k):
    n = len(X)
    rng = np.random.default_rng(SEED)
    idx = np.arange(n) if n <= SPECTRAL_MAX else np.sort(rng.choice(n, SPECTRAL_MAX, replace=False))
    sc = SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                            n_neighbors=min(15, len(idx) - 1), assign_labels="cluster_qr",
                            random_state=SEED)
    lab_s = sc.fit_predict(X[idx])
    if n == len(idx):
        return lab_s
    knn = KNeighborsClassifier(n_neighbors=5).fit(X[idx], lab_s)
    return knn.predict(X)


def choose_k(X, k_max=10):
    """Only used when the metadata does not give the number of groups."""
    rng = np.random.default_rng(SEED)
    Xs = X if len(X) <= 3000 else X[rng.choice(len(X), 3000, replace=False)]
    best_k, best_s = 2, -1.0
    for k in range(2, min(k_max, len(Xs) - 1) + 1):
        lab = _kmeans(Xs, k, n_init=4)
        if len(np.unique(lab)) < 2:
            continue
        s = silhouette_score(Xs, lab)
        if s > best_s:
            best_k, best_s = k, s
    return best_k


def cluster(F: pd.DataFrame, numeric: list, categorical: list, k: int | None,
            budget=None) -> tuple[np.ndarray, dict]:
    t0 = time.time()
    remaining = budget.remaining() if budget is not None else 30.0
    deadline = t0 + max(2.0, 0.6 * remaining)
    n = len(F)
    X, groups, names = build_matrix(F, numeric, categorical)
    info: dict = {"n_input_features": len(names)}
    if k is None:
        k = choose_k(X) if X.shape[1] else 2
        info["k_chosen_by"] = "silhouette"
    k = int(max(1, k))
    if n == 0:
        return np.array([], dtype=int), info
    if X.shape[1] == 0 or n <= k:
        lab = np.arange(n) % k
        info["algorithm"] = "trivial"
        return lab, info

    keep, fs = select_features(X, groups, k, deadline=t0 + max(1.5, 0.35 * remaining))
    info["feature_selection"] = fs
    info["selected_features"] = keep
    info["irrelevant_features"] = fs.get("removed", [])
    Xk = X[:, _cols(groups, keep)]

    has_cat = any(len(groups[g]) > 1 or g in categorical for g in keep)
    low_card = any(len(np.unique(Xk[:, i])) < 20 for i in range(Xk.shape[1]))
    labels, algo = None, None
    mode = OPTIONS.get("model", "auto")
    mixed = has_cat or low_card
    covs = {"auto": ("full", "diag"), "gmm_full": ("full",), "gmm_tied": ("tied",),
            "gmm_bic": ("full", "tied", "diag", "spherical"),
            "gmm_ft": ("full", "tied"),
            "auto_tied": ("tied",) if mixed else ("full", "diag"),
            "auto_bic3": ("tied",) if mixed else ("full", "diag", "tied")}.get(mode, ())
    if covs and (mode not in ("auto",) or not mixed):
        g = fit_gmm(Xk, k, deadline, covs)
        if g is not None:
            labels, algo = g[2], f"gaussian_mixture_{g[1]}"
            info["bic"] = g[0]
    if labels is None:
        labels, algo = _kmeans(Xk, k), "kmeans"

    sp = OPTIONS.get("spectral", "numeric")
    allow_spectral = sp == "on" or (sp == "numeric" and not has_cat and not low_card)
    if allow_spectral and time.time() < deadline and k >= 2:
        try:
            cut_main = knn_cut_fraction(Xk, labels)
            lab_s = fit_spectral(Xk, k)
            if len(np.unique(lab_s)) == k:
                cut_s = knn_cut_fraction(Xk, lab_s)
                info["knn_cut_fraction"] = {algo: cut_main, "spectral": cut_s}
                # replace only when the mixture solution genuinely cuts through dense
                # neighbourhoods (non-convex groups such as moons / rings) and the
                # graph partition is much cleaner; overlapping blobs keep the mixture.
                if cut_main >= 0.01 and cut_s <= 0.25 * cut_main:
                    labels, algo = lab_s, "spectral_knn_graph"
        except Exception as exc:  # spectral is optional
            info["spectral_error"] = str(exc)[:200]

    if len(np.unique(labels)) < k:
        labels, algo = _kmeans(Xk, k), "kmeans"
    labels = relabel(labels)
    info["algorithm"] = algo
    info["n_clusters"] = int(len(np.unique(labels)))
    info["cluster_sizes"] = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    try:
        rng = np.random.default_rng(SEED)
        idx = np.arange(n) if n <= 3000 else rng.choice(n, 3000, replace=False)
        if len(np.unique(labels[idx])) > 1:
            info["silhouette"] = float(silhouette_score(Xk[idx], labels[idx]))
    except Exception:
        pass
    info["fit_seconds"] = round(time.time() - t0, 3)
    return labels, info
