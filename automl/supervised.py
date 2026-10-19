"""Classification and regression: candidate models, CV, blending, decision tuning.

Design (see README for the rationale):

* candidates: HistGradientBoosting (native categoricals + NaN), a regularised
  HGB variant, a linear model on one-hot + scaled features, ExtraTrees, and for
  skewed non-negative regression targets log1p-target variants;
* K-fold out-of-fold (OOF) predictions for every candidate; test predictions are
  the average of the fold models (no refit needed, saves time);
* greedy (Caruana) ensemble selection on the OOF predictions;
* classification: decision rule tuned on OOF probabilities for the target
  metric (threshold for binary, per-class weights for multiclass);
* regression: clipping to a sane range, smearing correction for log models;
* a time projection after the first fold skips candidates that would not fit
  in the budget.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, log_loss,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore")

SEED = 0


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def normalize_metric(name: str | None, task: str) -> str:
    """Map a free-text metric name to one of our metric ids."""
    default = "macro_f1" if task == "classification" else "r2"
    if not name:
        return default
    m = str(name).lower().replace("-", "_").replace(" ", "_")
    if task == "classification":
        if "balanced" in m:
            return "balanced_accuracy"
        if "auc" in m or "roc" in m:
            return "roc_auc"
        if "log" in m and ("loss" in m or "likelihood" in m) or "cross_entropy" in m:
            return "log_loss"
        if "f1" in m or "f_1" in m or "fscore" in m or "f_score" in m:
            if "weighted" in m:
                return "weighted_f1"
            if "micro" in m:
                return "accuracy"
            if "binary" in m:
                return "binary_f1"
            return "macro_f1"
        if "acc" in m:
            return "accuracy"
        if "mcc" in m or "matthews" in m or "kappa" in m:
            return "macro_f1"
        return default
    if "rmsle" in m or "msle" in m or "squared_log" in m or "log_error" in m:
        return "rmsle"
    if "mape" in m or "percentage" in m:
        return "mape"
    if "mae" in m or "absolute" in m or "median" in m:
        return "mae"
    if "r2" in m or "r^2" in m or "r_squared" in m or "rsquared" in m or "explained" in m:
        return "r2"
    if "rmse" in m or "mse" in m or "squared" in m:
        return "rmse"
    return default


def _norm_proba(P: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip away exact 0/1 and renormalise rows (log-loss needs proper probabilities)."""
    P = np.clip(np.asarray(P, dtype=float), eps, 1.0)
    return P / P.sum(axis=1, keepdims=True)


def clf_score(metric: str, y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None = None,
              pos: int = 1) -> float:
    """Higher is better for every metric id."""
    if metric == "accuracy":
        return accuracy_score(y, pred)
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y, pred)
    if metric == "weighted_f1":
        return f1_score(y, pred, average="weighted", zero_division=0)
    if metric == "binary_f1":
        return f1_score(y, pred, pos_label=pos, average="binary", zero_division=0)
    if metric == "roc_auc" and proba is not None:
        try:
            if proba.shape[1] == 2:
                return roc_auc_score(y, proba[:, 1])
            return roc_auc_score(y, proba, multi_class="ovr")
        except ValueError:
            return 0.5
    if metric == "log_loss" and proba is not None:
        return -log_loss(y, _norm_proba(proba), labels=np.arange(proba.shape[1]))
    return f1_score(y, pred, average="macro", zero_division=0)


def reg_score(metric: str, y: np.ndarray, pred: np.ndarray) -> float:
    if metric == "mae":
        return -mean_absolute_error(y, pred)
    if metric == "rmsle":
        return -float(np.sqrt(np.mean((np.log1p(np.clip(pred, 0, None)) - np.log1p(np.clip(y, 0, None))) ** 2)))
    if metric == "mape":
        return -float(np.mean(np.abs(y - pred) / np.maximum(np.abs(y), 1e-9)))
    if metric == "rmse":
        return -float(np.sqrt(mean_squared_error(y, pred)))
    return r2_score(y, pred)


# --------------------------------------------------------------------------- #
# Design matrices
# --------------------------------------------------------------------------- #
class Design:
    """Builds the three matrix flavours from the typed frame of FeatureBuilder."""

    def __init__(self, F_train: pd.DataFrame, cat_cols: list):
        self.cat_cols = [c for c in cat_cols if c in F_train.columns]
        self.num_cols = [c for c in F_train.columns if c not in self.cat_cols]
        num = F_train[self.num_cols].astype(float) if self.num_cols else pd.DataFrame(index=F_train.index)
        self.medians = num.median().fillna(0.0) if self.num_cols else pd.Series(dtype=float)
        self.miss_cols = [c for c in self.num_cols if F_train[c].isna().any()]
        # linear-model numeric transform
        self.log_cols = []
        for c in self.num_cols:
            v = num[c].dropna()
            if len(v) > 10 and v.min() >= 0 and v.nunique() > 10 and abs(v.skew()) > 1.0:
                self.log_cols.append(c)
        lin = self._lin_numeric(num.fillna(self.medians))
        self.lin_mean = lin.mean()
        self.lin_std = lin.std().replace(0, 1.0).fillna(1.0)
        # one-hot levels for the linear model (top 30 per column)
        self.oh_levels = {}
        for c in self.cat_cols:
            codes = F_train[c].cat.codes
            vc = codes[codes >= 0].value_counts()
            self.oh_levels[c] = list(vc[vc >= 3].index[:30])

    def _lin_numeric(self, num: pd.DataFrame) -> pd.DataFrame:
        num = num.copy()
        for c in self.log_cols:
            num[c] = np.log1p(num[c].clip(lower=0))
        return num

    def hgb(self, F: pd.DataFrame):
        cols = []
        for c in self.num_cols:
            cols.append(F[c].astype(float).to_numpy())
        for c in self.cat_cols:
            codes = F[c].cat.codes.to_numpy().astype(float)
            codes[codes < 0] = np.nan
            cols.append(codes)
        X = np.column_stack(cols) if cols else np.zeros((len(F), 1))
        mask = np.array([False] * len(self.num_cols) + [True] * len(self.cat_cols))
        if not cols:
            mask = np.array([False])
        return X, mask

    def trees(self, F: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.num_cols:
            v = F[c].astype(float)
            cols.append(v.fillna(self.medians.get(c, 0.0)).to_numpy())
        for c in self.miss_cols:
            cols.append(F[c].isna().to_numpy().astype(float))
        for c in self.cat_cols:
            cols.append(F[c].cat.codes.to_numpy().astype(float))
        return np.column_stack(cols) if cols else np.zeros((len(F), 1))

    def linear(self, F: pd.DataFrame):
        parts = []
        if self.num_cols:
            num = F[self.num_cols].astype(float).fillna(self.medians)
            num = self._lin_numeric(num)
            z = ((num - self.lin_mean) / self.lin_std).clip(-5, 5).fillna(0.0).to_numpy()
            parts.append(sparse.csr_matrix(z))
        if self.miss_cols:
            parts.append(sparse.csr_matrix(F[self.miss_cols].isna().to_numpy().astype(float)))
        n = len(F)
        for c in self.cat_cols:
            codes = F[c].cat.codes.to_numpy()
            levels = self.oh_levels.get(c, [])
            if not levels:
                continue
            pos = {lv: i for i, lv in enumerate(levels)}
            col_idx = np.array([pos.get(v, len(levels)) for v in codes])
            m = sparse.csr_matrix((np.ones(n), (np.arange(n), col_idx)), shape=(n, len(levels) + 1))
            parts.append(m)
        if not parts:
            return sparse.csr_matrix(np.zeros((n, 1)))
        X = sparse.hstack(parts).tocsr()
        if X.shape[0] * X.shape[1] <= 5_000_000:
            return X.toarray()
        return X


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    name: str
    kind: str                      # hgb | linear | trees
    make: object                   # callable -> estimator
    log_target: bool = False
    oof: np.ndarray | None = None
    test: np.ndarray | None = None
    score: float = -np.inf
    seconds: float = 0.0
    n_iters: list = field(default_factory=list)


def _forest_size(n: int) -> dict:
    """Fully grown trees on small data; bounded tree size (memory/time) on large data."""
    if n <= 1000:  # tiny tables: 150 trees are as good in a blend and twice as fast
        return {"n_estimators": 150, "min_samples_leaf": 1}
    if n <= 5000:
        return {"n_estimators": 300, "min_samples_leaf": 1}
    if n <= 20000:
        return {"n_estimators": 250, "min_samples_leaf": 2}
    return {"n_estimators": 150, "min_samples_leaf": 5}


def _hgb_params(n: int, p: int, min_class_count: int | None, regularized: bool) -> dict:
    """HistGradientBoosting settings.

    * large tables (n*p > 2M cells): faster learning rate / patience
    * early stopping needs a stratified validation split, which fails when a
      class has very few rows; tiny tables are too small for it anyway
    """
    big = n * p > 2_000_000
    if regularized:
        kw = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0)
    else:
        kw = dict(learning_rate=0.1 if big else 0.06, max_leaf_nodes=31, min_samples_leaf=20,
                  l2_regularization=0.1)
    if n >= 200 and (min_class_count is None or min_class_count >= 10):
        kw.update(max_iter=1000, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=15 if big else 30)
    else:
        kw.update(max_iter=150, early_stopping=False)
    kw["min_samples_leaf"] = int(max(2, min(kw["min_samples_leaf"], n // 20)))
    return kw


def make_candidates(task: str, n: int, imbalance_minority: float | None, log_ok: bool,
                    n_threads: int, n_classes: int = 2, n_features: int = 10,
                    min_class_count: int | None = None):
    C = []
    fsz = _forest_size(n)
    mcc = min_class_count if task == "classification" else None
    kw_a = _hgb_params(n, n_features, mcc, regularized=False)
    kw_b = _hgb_params(n, n_features, mcc, regularized=True)
    try:  # feature subsampling only exists in recent scikit-learn
        HistGradientBoostingRegressor(max_features=0.6)
        kw_b["max_features"] = 0.6
    except TypeError:
        pass
    if task == "classification":
        def hgb_a(mask):
            return HistGradientBoostingClassifier(categorical_features=mask, random_state=SEED,
                                                  **kw_a)

        def hgb_b(mask):
            kw = dict(kw_b)
            if imbalance_minority is not None and imbalance_minority < 0.10:
                kw["class_weight"] = "balanced"
            return HistGradientBoostingClassifier(categorical_features=mask, random_state=SEED,
                                                  **kw)

        def lin(_):
            return LogisticRegression(C=1.0 if n > 2000 else 0.3, max_iter=3000)

        def rf(_):
            return RandomForestClassifier(max_features="sqrt", n_jobs=n_threads, random_state=SEED,
                                          **fsz)

        def et(_):
            return ExtraTreesClassifier(max_features="sqrt", n_jobs=n_threads, random_state=SEED,
                                        **fsz)
        C += [Candidate("hgb", "hgb", hgb_a), Candidate("logistic", "linear", lin),
              Candidate("random_forest", "trees", rf), Candidate("hgb_regularized", "hgb", hgb_b),
              Candidate("extra_trees", "trees", et)]
        if n_classes >= 5:  # boosting cost grows with the number of classes, forests' does not
            C[3], C[4] = C[4], C[3]
    else:
        def hgb_a(mask):
            return HistGradientBoostingRegressor(categorical_features=mask, random_state=SEED,
                                                 **kw_a)

        def hgb_b(mask):
            return HistGradientBoostingRegressor(categorical_features=mask, random_state=SEED,
                                                 **kw_b)

        def ridge(_):
            return _RidgeAuto()

        def et(_):
            return ExtraTreesRegressor(max_features=0.7, n_jobs=n_threads, random_state=SEED, **fsz)
        C += [Candidate("hgb", "hgb", hgb_a), Candidate("ridge", "linear", ridge)]
        if log_ok:
            C += [Candidate("hgb_log", "hgb", hgb_a, log_target=True),
                  Candidate("ridge_log", "linear", ridge, log_target=True)]
        C += [Candidate("hgb_regularized", "hgb", hgb_b), Candidate("extra_trees", "trees", et)]
    return C


class _RidgeAuto:
    """Ridge with alpha chosen by efficient leave-one-out / GCV on the fold."""

    def fit(self, X, y):
        from sklearn.linear_model import RidgeCV
        alphas = np.logspace(-2, 3, 12)
        if sparse.issparse(X):
            best, best_s = alphas[0], -np.inf
            n = X.shape[0]
            cut = int(n * 0.8)
            for a in alphas[::2]:
                m = Ridge(alpha=a).fit(X[:cut], y[:cut])
                s = -np.mean((m.predict(X[cut:]) - y[cut:]) ** 2)
                if s > best_s:
                    best, best_s = a, s
            self.model = Ridge(alpha=best).fit(X, y)
        else:
            self.model = RidgeCV(alphas=alphas).fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


# --------------------------------------------------------------------------- #
# Decision rules
# --------------------------------------------------------------------------- #
def tune_decision(metric: str, y: np.ndarray, P: np.ndarray, pos: int):
    """Return (rule, oof_score_before, oof_score_after).

    rule = ("argmax",) | ("threshold", t) | ("weights", w)
    """
    base_pred = P.argmax(1)
    base = clf_score(metric, y, base_pred, P, pos)
    if metric in ("accuracy", "roc_auc", "log_loss"):
        return ("argmax",), base, base
    K = P.shape[1]
    if K == 2:
        p = P[:, pos]
        ts = np.unique(np.quantile(p, np.linspace(0.0, 1.0, 201)))
        scores = []
        for t in ts:
            pred = np.where(p >= t, pos, 1 - pos)
            scores.append(clf_score(metric, y, pred, None, pos))
        scores = np.array(scores)
        best = scores.max()
        plateau = ts[scores >= best - 0.002]
        t = float(np.median(plateau))
        after = clf_score(metric, y, np.where(p >= t, pos, 1 - pos), None, pos)
        if after > base + 0.002:
            return ("threshold", t, pos), base, after
        return ("argmax",), base, base
    w = np.ones(K)
    grid = np.exp(np.linspace(np.log(0.2), np.log(8), 41))
    cur = base
    for _ in range(2):
        for k in range(1, K):
            best_g, best_s = w[k], cur
            for g in grid:
                w2 = w.copy()
                w2[k] = g
                s = clf_score(metric, y, (P * w2).argmax(1), None, pos)
                if s > best_s + 1e-9:
                    best_g, best_s = g, s
            w[k] = best_g
            cur = best_s
    counts = np.bincount(y, minlength=K)
    if counts.min() < 15:
        w = np.exp(0.5 * np.log(w))
    after = clf_score(metric, y, (P * w).argmax(1), None, pos)
    if after > base + 0.002:
        return ("weights", w), base, after
    return ("argmax",), base, base


def apply_decision(rule, P: np.ndarray) -> np.ndarray:
    if rule[0] == "threshold":
        t, pos = rule[1], rule[2]
        return np.where(P[:, pos] >= t, pos, 1 - pos)
    if rule[0] == "weights":
        return (P * rule[1]).argmax(1)
    return P.argmax(1)


# --------------------------------------------------------------------------- #
# Main solver
# --------------------------------------------------------------------------- #
def _folds(task, y, n, seed=SEED):
    """Fold plan: (splits, repeats, subsample_size)."""
    if n < 600:
        k, reps = 5, 2
    elif n <= 30000:
        k, reps = 5, 1
    else:
        k, reps = 3, 1
    if task == "classification":
        min_count = np.bincount(y).min() if len(y) else 0
        k = int(max(2, min(k, min_count))) if min_count >= 2 else 0
    out = []
    for r in range(reps):
        if task == "classification" and k >= 2:
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed + r)
            out += list(cv.split(np.zeros(n), y))
        else:
            cv = KFold(n_splits=max(2, k or 5), shuffle=True, random_state=seed + r)
            out += list(cv.split(np.zeros(n)))
    return out, reps


def _expand_proba(est, P, n_classes):
    out = np.zeros((P.shape[0], n_classes))
    out[:, est.classes_.astype(int)] = P
    return out


def _caruana(oofs: list, y, task, metric, pos, steps=15):
    """Greedy ensemble selection with replacement; returns weights."""
    m = len(oofs)
    if m == 1:
        return np.array([1.0])

    def score(pred):
        if task == "classification":
            # the target metric decides; log-loss only breaks ties (it alone would
            # reject accurate but over-confident models such as fully grown forests)
            ll = log_loss(y, _norm_proba(pred), labels=np.arange(pred.shape[1]))
            if metric in ("log_loss",):
                return -ll
            return clf_score(metric, y, pred.argmax(1), pred, pos) - 1e-4 * ll
        return reg_score(metric if metric in ("mae", "rmsle", "mape") else "rmse", y, pred)

    counts = np.zeros(m)
    cur = None
    best_overall, best_counts = -np.inf, None
    for _ in range(steps):
        best_s, best_j = -np.inf, None
        for j in range(m):
            cand = oofs[j] if cur is None else (cur * counts.sum() + oofs[j]) / (counts.sum() + 1)
            s = score(cand)
            if s > best_s:
                best_s, best_j = s, j
        counts[best_j] += 1
        cur = oofs[best_j] if cur is None else (cur * (counts.sum() - 1) + oofs[best_j]) / counts.sum()
        if best_s > best_overall + 1e-12:
            best_overall, best_counts = best_s, counts.copy()
    return best_counts / best_counts.sum()


class SupervisedSolver:
    def __init__(self, task: str, metric: str | None, budget, n_threads: int = 2,
                 search_until: float = 22.0, on_progress=None):
        self.task = task
        self.metric = normalize_metric(metric, task)
        self.budget = budget
        self.n_threads = n_threads
        self.search_until = search_until   # seconds since process start
        self.on_progress = on_progress     # callback(test_predictions) for anytime outputs
        self.info: dict = {"metric": self.metric}

    # --------------------------------------------------------------- utils
    def _time_left(self):
        return self.search_until - self.budget.elapsed()

    def fit_predict(self, F_train: pd.DataFrame, y_raw: pd.Series, F_test: pd.DataFrame,
                    cat_cols: list):
        task = self.task
        t_start = time.time()
        y_raw = pd.Series(y_raw).reset_index(drop=True)
        F_train = F_train.reset_index(drop=True)
        if task == "classification":
            keep = y_raw.notna()
        else:
            y_num = pd.to_numeric(y_raw, errors="coerce")
            keep = y_num.notna() & np.isfinite(y_num)
        if not keep.all():
            F_train = F_train[keep.values].reset_index(drop=True)
            y_raw = y_raw[keep.values].reset_index(drop=True)
        n = len(F_train)

        if task == "classification":
            yv = y_raw.to_numpy()
            if y_raw.dtype == object or isinstance(y_raw.dtype, pd.CategoricalDtype):
                yv = y_raw.astype(str).to_numpy()
            classes, y = np.unique(yv, return_inverse=True)
            # map back to original (typed) labels
            first = {}
            yv = y_raw.to_numpy()
            for i, code in enumerate(y):
                first.setdefault(code, yv[i])
            labels = [first[i] for i in range(len(classes))]
            if all(isinstance(v, (float, np.floating)) and float(v).is_integer() for v in labels):
                labels = [int(v) for v in labels]  # 1.0 -> 1 (target read as float due to NaN)
            self.labels_ = np.array(labels, dtype=object)
            K = len(classes)
            counts = np.bincount(y, minlength=K)
            self.info["n_classes"] = K
            if K == 1:
                pred = np.repeat(self.labels_[0], len(F_test))
                self.info["model"] = "constant_single_class"
                return pred
            pos = self._positive_index()
            minority = counts.min() / counts.sum()
        else:
            y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
            K, pos, minority = 0, 1, None
        self.pos_ = pos

        design = Design(F_train, cat_cols)
        self.hgb_columns_ = design.num_cols + design.cat_cols
        Xh, mask = design.hgb(F_train)
        Xh_te, _ = design.hgb(F_test)
        mats = {"hgb": (Xh, Xh_te)}
        log_ok = False
        if task == "regression":
            ymin = float(np.min(y))
            skew = float(pd.Series(y).skew()) if n > 2 else 0.0
            is_int = np.allclose(y, np.round(y))
            med = float(np.median(y))
            log_ok = ymin >= 0 and (skew > 0.75 or (is_int and med > 0 and np.max(y) / med >= 5))
            self.info["target_skew"] = skew
            self.info["log_target_considered"] = bool(log_ok)

        min_count = int(counts.min()) if task == "classification" else None
        cands = make_candidates(task, n, minority, log_ok, self.n_threads, K,
                                n_features=Xh.shape[1], min_class_count=min_count)
        # the search window starts when modelling starts (loading big tables can be slow),
        # but never runs into the reserve needed for refits and writing
        self.search_until = min(max(self.search_until, self.budget.elapsed() + 12.0),
                                self.budget.total - 8.0)
        folds, reps = _folds(task, y, n)
        self.info["cv"] = {"n_splits": len(folds) // reps, "repeats": reps}

        sub_idx = None
        if n > 30000:
            rng = np.random.default_rng(SEED)
            sub_idx = np.sort(rng.choice(n, 30000, replace=False))
            folds, reps = _folds(task, y[sub_idx], len(sub_idx))
            self.info["cv"]["subsample"] = int(len(sub_idx))

        done = []
        self.failed_ = {}
        for cand in cands:
            if self._time_left() <= 0.5 and done:
                break
            try:
                mat = self._matrices(cand.kind, design, F_train, F_test, mats)
                ok = self._run_cv(cand, mat, mask, y, folds, reps, K, sub_idx, n,
                                  first=not done)
            except Exception as exc:  # one failing model must never sink the others
                self.failed_[cand.name] = f"{type(exc).__name__}: {exc}"[:200]
                ok = False
            if ok:
                done.append(cand)
                if self.on_progress is not None and len(done) == 1:
                    try:
                        self.on_progress(self._finalize([cand], [1.0], y, K, quick=True))
                    except Exception:
                        pass
        if self.failed_:
            self.info["failed_candidates"] = self.failed_
        if not done:
            raise RuntimeError(f"no candidate model finished ({self.failed_})")

        # scores + blend
        y_eval = y if sub_idx is None else y[sub_idx]
        self.info["candidates"] = {c.name: round(float(c.score), 5) for c in done}
        self.info["candidate_seconds"] = {c.name: round(c.seconds, 2) for c in done}
        weights = _caruana([c.oof for c in done], y_eval, task, self.metric, pos)
        sel = [(c, w) for c, w in zip(done, weights) if w > 0]
        self.info["ensemble"] = {c.name: round(float(w), 4) for c, w in sel}
        best_single = max(done, key=lambda c: c.score)
        self.info["best_single_model"] = best_single.name

        # nonlinearity evidence: best tree model vs linear model OOF score
        lin = [c for c in done if c.kind == "linear" and not c.log_target]
        tree = [c for c in done if c.kind != "linear"]
        if lin and tree:
            gap = max(c.score for c in tree) - lin[0].score
            if task == "regression" and self.metric not in ("r2",):
                # express the gap on the R^2 scale for comparability
                gap = (max(r2_score(y_eval, c.oof) for c in tree) - r2_score(y_eval, lin[0].oof))
            self.info["tree_vs_linear_gap"] = float(gap)
            self.info["nonlinear"] = bool(gap > 0.03)

        blend_oof = sum(w * c.oof for c, w in sel)
        if task == "classification":
            rule, before, after = tune_decision(self.metric, y_eval, blend_oof, pos)
            self.rule_ = rule
            self.info["decision_rule"] = rule[0]
            self.info["cv_score_argmax"] = float(before)
            self.info["cv_score"] = float(after)
        else:
            self.info["cv_score"] = float(reg_score(self.metric, y_eval, blend_oof))
        self.info["model"] = "+".join(c.name for c, _ in sel)
        self.info["fit_seconds"] = round(time.time() - t_start, 2)

        # optional full refit when CV ran on a subsample of a large table
        if sub_idx is not None:
            self._refit_full(sel, design, F_train, F_test, mats, mask, y, K)
        return self._finalize([c for c, _ in sel], [w for _, w in sel], y, K)

    # ------------------------------------------------------------ internals
    def _positive_index(self):
        labels = [str(v).strip().lower() for v in self.labels_]
        for good in ("1", "1.0", "true", "yes", "y", "positive", "pos"):
            if good in labels:
                return labels.index(good)
        return 1

    def _matrices(self, kind, design, F_train, F_test, mats):
        if kind not in mats:
            if kind == "linear":
                mats[kind] = (design.linear(F_train), design.linear(F_test))
            else:
                mats[kind] = (design.trees(F_train), design.trees(F_test))
        return mats[kind]

    def _transform_y(self, cand, y):
        return np.log1p(np.clip(y, 0, None)) if cand.log_target else y

    def _inverse_y(self, cand, p):
        return np.expm1(p) if cand.log_target else p

    def _run_cv(self, cand, mat, mask, y, folds, reps, K, sub_idx, n_full, first=False):
        X, X_te = mat
        if sub_idx is not None:
            X = X[sub_idx]
            y = y[sub_idx]
        n = X.shape[0]
        task = self.task
        oof = np.zeros((n, K)) if task == "classification" else np.zeros(n)
        cnt = np.zeros(n)
        test = None
        t0 = time.time()
        n_folds = len(folds)
        for i, (tr, va) in enumerate(list(folds)):
            if i >= n_folds:  # fold list shortened by the time projection below
                break
            ts = time.time()
            est = cand.make(mask) if cand.kind == "hgb" else cand.make(None)
            yt = self._transform_y(cand, y[tr])
            est.fit(X[tr], yt)
            if task == "classification":
                pv = _expand_proba(est, est.predict_proba(X[va]), K)
                pt = _expand_proba(est, est.predict_proba(X_te), K)
                oof[va] += pv
            else:
                pv = self._inverse_y(cand, est.predict(X[va]))
                pt = self._inverse_y(cand, est.predict(X_te))
                oof[va] += pv
            cnt[va] += 1
            test = pt if test is None else test + pt
            if cand.kind == "hgb" and hasattr(est, "n_iter_"):
                cand.n_iters.append(int(np.max(np.atleast_1d(est.n_iter_))))
            if i == 0:
                per_fold = time.time() - ts
                # the first model may use the whole budget (minus a reserve); later
                # candidates must fit in the search window
                left = (self.budget.remaining() - 6.0) if first else self._time_left()
                projected = per_fold * (n_folds - 1) * 1.1
                if projected > left and i + 1 < n_folds:
                    # cannot afford all folds: keep the folds that still fit (partial
                    # out-of-fold coverage), or give up on this candidate
                    afford = int(left / max(per_fold * 1.1, 1e-3))
                    if afford < 1 and not first:
                        cand.seconds = time.time() - t0
                        return False
                    folds = folds[: max(1, afford) + 1] if afford >= 1 else folds[:1]
                    n_folds = len(folds)
                if cand.kind == "hgb" and not hasattr(self, "perm_importance_"):
                    try:
                        self.perm_importance_ = self._permutation_importance(est, X[va], y[va], K)
                    except Exception:
                        self.perm_importance_ = []
        covered = cnt > 0
        safe = np.where(covered, cnt, 1.0)
        oof = oof / (safe[:, None] if task == "classification" else safe)
        if not covered.all():
            # partial CV (time budget): rows never validated get the mean prediction
            oof[~covered] = oof[covered].mean(axis=0)
        test = test / n_folds
        if task == "regression":
            if cand.log_target and self.metric not in ("rmsle", "mape", "mae"):
                s = np.clip(np.mean(y) / max(np.mean(oof), 1e-9), 0.9, 1.1)
                oof, test = oof * s, test * s
            oof, test = self._clip_reg(oof, y), self._clip_reg(test, y)
            cand.score = reg_score(self.metric, y[covered], oof[covered])
        else:
            P = oof
            cand.score = clf_score(self.metric, y[covered], P[covered].argmax(1), P[covered],
                                   self.pos_)
        cand.oof, cand.test = oof, test
        cand.seconds = time.time() - t0
        return True

    def _permutation_importance(self, est, Xv, yv, K, max_rows=2000):
        """Held-out permutation importance (loss increase) per HGB matrix column."""
        rng = np.random.default_rng(SEED)
        if len(Xv) > max_rows:
            sel = rng.choice(len(Xv), max_rows, replace=False)
            Xv, yv = Xv[sel], yv[sel]
        if self.task == "classification":
            def loss(M):
                P = _expand_proba(est, est.predict_proba(M), K)
                return log_loss(yv, _norm_proba(P), labels=np.arange(K))
        else:
            yt = yv
            var = max(float(np.var(yt)), 1e-12)

            def loss(M):
                return float(np.mean((est.predict(M) - yt) ** 2)) / var
        base = loss(Xv)
        reps = 3 if Xv.shape[1] <= 40 else 2
        out = []
        t_end = time.time() + min(3.0, max(0.5, 0.1 * self._time_left()))
        for j in range(Xv.shape[1]):
            if time.time() > t_end:  # out of time: unknown importance (never "irrelevant")
                out.append((np.inf, 0.0))
                continue
            deltas = []
            for _ in range(reps):
                Xp = Xv.copy()
                Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                deltas.append(loss(Xp) - base)
            out.append((float(np.mean(deltas)), float(np.std(deltas))))
        return out

    def _clip_reg(self, p, y):
        lo, hi = float(np.min(y)), float(np.max(y))
        rng = hi - lo
        lo2, hi2 = lo - 0.1 * rng, hi + 0.1 * rng
        if lo >= 0:
            lo2 = max(lo2, 0.0)
        return np.clip(p, lo2, hi2)

    def _refit_full(self, sel, design, F_train, F_test, mats, mask, y, K):
        """Refit selected models on all rows if the budget allows (big tables)."""
        for cand, _w in sel:
            est_t = cand.seconds / max(1, len(cand.n_iters) or 3) * 3.5
            if self.budget.remaining() - 8 < est_t:
                continue
            X, X_te = self._matrices(cand.kind, design, F_train, F_test, mats)
            est = cand.make(mask) if cand.kind == "hgb" else cand.make(None)
            if cand.kind == "hgb" and cand.n_iters:
                est.set_params(early_stopping=False,
                               max_iter=int(max(20, 1.15 * np.median(cand.n_iters))))
            est.fit(X, self._transform_y(cand, y))
            if self.task == "classification":
                cand.test = _expand_proba(est, est.predict_proba(X_te), K)
            else:
                p = self._inverse_y(cand, est.predict(X_te))
                cand.test = self._clip_reg(p, y)
            self.info.setdefault("refit_full", []).append(cand.name)

    def _finalize(self, cands, weights, y, K, quick=False):
        T = sum(w * c.test for c, w in zip(cands, weights))
        if self.task == "classification":
            rule = ("argmax",) if quick else getattr(self, "rule_", ("argmax",))
            idx = apply_decision(rule, T)
            self.test_proba_ = T
            return self.labels_[idx]
        return T
