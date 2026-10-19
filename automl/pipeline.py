"""End-to-end orchestration: load -> profile -> prepare -> model -> write."""
from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .budget import Budget, n_jobs
from .features import FeatureBuilder, plan_columns
from .io_contract import (Dataset, Metadata, load_dataset, prediction_frame,
                          resolve_output_paths, write_outputs, write_profile)
from .profiling import build_profile, irrelevant_supervised

# The model modules import heavy parts of scikit-learn; they are imported lazily so a
# clustering run never pays for sklearn.ensemble and vice versa (~1.5 s each).

log = logging.getLogger("automl")

SEARCH_SECONDS = float(os.environ.get("AUTOML_SEARCH_SECONDS", 20.0))


def _threadpool_limits(n):
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=n)
    except Exception:  # pragma: no cover - threadpoolctl ships with scikit-learn
        import contextlib
        return contextlib.nullcontext()


def _cluster_metric(name):
    if not name:
        return "adjusted_rand_index"
    m = str(name).lower()
    if "nmi" in m or "mutual" in m:
        return "normalized_mutual_info"
    if "silhouette" in m:
        return "silhouette"
    return "adjusted_rand_index"


class Run:
    """One dataset: keeps the latest predictions/profile and writes them."""

    def __init__(self, ds: Dataset, pred_path: Path | None, prof_path: Path | None,
                 budget: Budget, write: bool = True):
        self.ds = ds
        self.pred_path, self.prof_path = pred_path, prof_path
        self.budget = budget
        self.write = write and pred_path is not None
        self.profile: dict = {}
        self.pred_df: pd.DataFrame | None = None

    def emit(self, preds, stage: str):
        self.pred_df = prediction_frame(self.ds, self.ids, preds)
        self.profile["status"] = stage
        self.profile["runtime_seconds"] = round(self.budget.elapsed(), 3)
        if self.write:
            write_outputs(self.pred_df, self.profile, self.pred_path, self.prof_path)
        log.info("[%s] wrote %s predictions (%d rows) at %.1fs", self.ds.name, stage,
                 len(self.pred_df), self.budget.elapsed())

    # ------------------------------------------------------------------ main
    def execute(self) -> dict:
        ds, meta = self.ds, self.ds.meta
        task, target, id_col = meta.task, meta.target, meta.id_column
        train, test = ds.train, ds.test
        log.info("[%s] task=%s target=%s id=%s train=%s test=%s k=%s", ds.name, task, target,
                 id_col, train.shape, None if test is None else test.shape, meta.n_clusters)

        # rows we must predict
        if task == "clustering":
            pred_rows = test if test is not None else train
        else:
            pred_rows = test if test is not None else train.drop(columns=[target])
        self.ids = pred_rows[id_col].to_numpy() if id_col and id_col in pred_rows.columns else None

        declared = {c: t for c, t in meta.feature_types.items() if c in train.columns}
        declared_raw = self._declared_raw(meta, train)
        plan_test = test if task != "clustering" else None
        plan = plan_columns(train, plan_test if plan_test is not None else None,
                            target if task != "clustering" else None, id_col, declared)
        if task == "clustering" and target and target in train.columns:
            plan.dropped[target] = "target_not_used_for_clustering"
            for lst in (plan.numeric, plan.categorical, plan.datetime, plan.text):
                if target in lst:
                    lst.remove(target)
        if task != "clustering":
            from .supervised import normalize_metric
            metric = normalize_metric(meta.metric, task)
        else:
            metric = _cluster_metric(meta.metric)
        self.profile = build_profile(ds.name, task, train, test,
                                     target if task != "clustering" else None,
                                     id_col, meta.n_clusters, plan, declared_raw, metric)
        self.profile["preprocessing"] = {"dropped_columns": dict(plan.dropped),
                                         "used_columns": plan.used}
        self.profile["environment"] = {"automl_version": __version__,
                                       "python": sys.version.split()[0],
                                       "n_threads": n_jobs()}

        # instant fallback so a valid submission exists from the first seconds
        self.emit(self._trivial(task, train, target, len(pred_rows), meta.n_clusters), "fallback")

        with _threadpool_limits(n_jobs()):
            if task == "clustering":
                self._clustering(train, test, plan)
            else:
                self._supervised(task, train, test, pred_rows, target, plan, metric)
        return {"predictions": self.pred_df, "profile": self.profile}

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _declared_raw(meta: Metadata, train: pd.DataFrame) -> dict:
        """Declared feature types exactly as written in the metadata."""
        raw = meta.raw or {}
        out = {}
        from .io_contract import KEYS, _flatten
        flat = _flatten(raw)
        for k in KEYS["feature_types"]:
            v = flat.get(k)
            if isinstance(v, dict):
                for c, t in v.items():
                    if c in train.columns and isinstance(t, (str, int, float)):
                        out[c] = t
                if out:
                    break
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        nm = item.get("name", item.get("column"))
                        t = item.get("type", item.get("dtype"))
                        if nm in train.columns and t is not None:
                            out[nm] = t
                if out:
                    break
        for c, t in meta.feature_types.items():
            if c in train.columns:
                out.setdefault(c, t)
        return out

    @staticmethod
    def _trivial(task, train, target, n, k):
        if task == "classification":
            mode = train[target].mode(dropna=True)
            v = mode.iloc[0] if len(mode) else 0
            if isinstance(v, (float, np.floating)) and float(v).is_integer():
                v = int(v)
            return np.repeat(v, n)
        if task == "regression":
            y = pd.to_numeric(train[target], errors="coerce")
            return np.repeat(float(y.mean()) if y.notna().any() else 0.0, n)
        k = int(k or 2)
        return np.arange(n) % max(k, 1)

    def _supervised(self, task, train, test, pred_rows, target, plan, metric):
        from .supervised import SupervisedSolver
        X_tr = train.drop(columns=[target])
        y = train[target]
        fb = FeatureBuilder(plan).fit(X_tr)
        F_tr = fb.transform(X_tr)
        F_te = fb.transform(pred_rows)
        solver = SupervisedSolver(task, self.ds.meta.metric, self.budget, n_threads=n_jobs(),
                                  search_until=SEARCH_SECONDS,
                                  on_progress=lambda p: self.emit(p, "first_model"))
        preds = solver.fit_predict(F_tr, y, F_te, fb.categorical_out)
        info = solver.info

        # post-model profile fields
        plan_map = {}
        for c in F_tr.columns:
            plan_map[c] = c.split("__")[0] if "__" in c and c.split("__")[0] in plan.used else c
        try:
            ymask = y.notna().to_numpy()
            if task == "classification":
                yy = pd.factorize(y[ymask].astype(str))[0]
            else:
                yy = pd.to_numeric(y[ymask], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(yy)
                yy, ymask_idx = yy[ok], np.flatnonzero(ymask)[ok]
                ymask = np.zeros(len(y), dtype=bool)
                ymask[ymask_idx] = True
            perm = {}
            pi = getattr(solver, "perm_importance_", None) or []
            for name, val in zip(getattr(solver, "hgb_columns_", []), pi):
                perm[name] = val
            irr, mi = irrelevant_supervised(F_tr[ymask].reset_index(drop=True), yy, task,
                                            fb.categorical_out, plan_map, perm)
        except Exception as exc:  # profiling must never break predictions
            log.warning("mutual information failed: %s", exc)
            irr, mi = [], {}
        self._finish_profile(irr, mi, info.get("nonlinear"), info)
        self.emit(preds, "final")

    def _clustering(self, train, test, plan):
        from .clustering import cluster
        frames = [train] + ([test] if test is not None else [])
        allX = pd.concat(frames, ignore_index=True)
        fb = FeatureBuilder(plan).fit(allX)
        F = fb.transform(allX)
        labels, info = cluster(F, list(plan.numeric), list(plan.categorical),
                               self.ds.meta.n_clusters, self.budget)
        if test is not None:
            labels = labels[len(train):]
        nonlinear = info.get("algorithm") == "spectral_knn_graph"
        irr = info.get("irrelevant_features", [])
        self.profile["n_clusters_used"] = info.get("n_clusters")
        self._finish_profile(irr, None, nonlinear, info)
        self.emit(labels.astype(int), "final")

    def _finish_profile(self, irrelevant, mi, nonlinear, info):
        p = self.profile
        p["irrelevant_features"] = list(irrelevant)
        p["suspected_irrelevant_features"] = list(irrelevant)
        if mi:
            p["feature_relevance"] = mi
        p["data_issues"]["irrelevant_features"] = bool(irrelevant)
        p["data_issues"]["nonlinear_relationships"] = None if nonlinear is None else bool(nonlinear)
        p["nonlinearity_detected"] = None if nonlinear is None else bool(nonlinear)
        p["model"] = info
        p["selected_model"] = info.get("model", info.get("algorithm"))
        p["cv_score"] = info.get("cv_score")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def solve_dataset(ds: Dataset, output=None, budget: Budget | None = None,
                  write: bool = True, state: dict | None = None) -> dict:
    budget = budget or Budget(start=time.time())  # API use: the clock starts now
    pred_path, prof_path = resolve_output_paths(output, ds) if write else (None, None)
    if state is not None:
        state.update(pred=str(pred_path) if pred_path else None,
                     prof=str(prof_path) if prof_path else None, name=ds.name)
    run = Run(ds, pred_path, prof_path, budget, write=write)
    try:
        res = run.execute()
    except Exception as exc:
        log.error("[%s] pipeline error: %s\n%s", ds.name, exc, traceback.format_exc())
        run.profile.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}")
        if run.pred_df is None:
            # even the fallback failed: emit something valid
            n = len(ds.test) if ds.test is not None else len(ds.train)
            run.ids = None
            if ds.meta.id_column:
                src = ds.test if ds.test is not None else ds.train
                if ds.meta.id_column in src.columns:
                    run.ids = src[ds.meta.id_column].to_numpy()
            run.emit(np.zeros(n, dtype=int), "error_fallback")
        else:
            run.profile["status"] = "fallback_after_error"
            if run.write:
                write_profile(run.profile, run.prof_path)
        res = {"predictions": run.pred_df, "profile": run.profile}
    res["predictions_path"] = str(pred_path) if pred_path else None
    res["profile_path"] = str(prof_path) if prof_path else None
    return res


def run(data_dir=None, output_dir=None, *, train=None, test=None, metadata=None, task=None,
        target=None, n_clusters=None, id_column=None, metric=None, write=True, budget=None,
        state=None, description=None) -> dict:
    """Importable entry point: ``run("path/to/dataset", "path/to/output")``.

    Returns ``{"predictions": DataFrame, "profile": dict, "predictions_path", "profile_path"}``.
    """
    overrides = {"task": task, "target": target, "n_clusters": n_clusters, "id_column": id_column,
                 "metric": metric}
    budget = budget or Budget(start=time.time())  # API use: the clock starts now
    ds = load_dataset(data_dir, train=train, test=test, metadata=metadata, overrides=overrides,
                      description=description)
    res = solve_dataset(ds, output_dir, budget=budget, write=write, state=state)
    res["resolved"] = resolved_summary(ds)
    return res


def resolved_summary(ds: Dataset) -> dict:
    """How the dataset was understood (printed by the CLI so a caller can check it)."""
    m = ds.meta
    return {
        "dataset": ds.name,
        "files": {k: str(Path(v).resolve()) for k, v in ds.files.items() if v},
        "task": m.task, "target": m.target, "id_column": m.id_column,
        "n_clusters": m.n_clusters, "metric": m.metric,
        "n_train_rows": int(len(ds.train)),
        "n_test_rows": int(len(ds.test)) if ds.test is not None else 0,
        "n_declared_feature_types": len(m.feature_types),
    }


def solve(train_df: pd.DataFrame, test_df: pd.DataFrame | None = None, metadata: dict | None = None,
          **overrides):
    """In-memory API: returns (predictions ndarray, profile dict)."""
    from .io_contract import infer_missing_meta
    meta = Metadata.from_dict(metadata or {})
    for k, v in overrides.items():
        if v is not None and hasattr(meta, k):
            setattr(meta, k, v)
    train_df = train_df.copy()
    train_df.columns = [str(c) for c in train_df.columns]
    if test_df is not None:
        test_df = test_df.copy()
        test_df.columns = [str(c) for c in test_df.columns]
    infer_missing_meta(train_df, test_df, meta)
    ds = Dataset(name=meta.name or "dataset", train=train_df, test=test_df, meta=meta)
    res = solve_dataset(ds, None, budget=Budget(start=time.time()), write=False)
    pred_df = res["predictions"]
    return pred_df.iloc[:, -1].to_numpy(), res["profile"]
