"""Input/output contract: locating the tables, reading metadata, writing outputs.

The announcement does not specify the exact contract, so this module is
deliberately lenient. It accepts the most likely layout::

    <dataset_dir>/train.csv  test.csv  metadata.json

and many variants of it (other file names, separate label files, YAML metadata,
synonyms for every metadata key, no metadata at all). Everything that depends on
the contract lives here so it can be adjusted in one place.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TABLE_EXTS = (".csv", ".tsv", ".txt", ".data", ".parquet", ".pq", ".feather", ".json", ".jsonl",
              ".xlsx", ".xls", ".gz", ".bz2", ".xz", ".zip")
COMPRESSED = (".gz", ".bz2", ".xz", ".zip")
TARGET_ROLE_WORDS = ("target", "label", "response", "outcome", "y", "class_label", "dependent",
                     "predict", "prediction_target")
ID_ROLE_WORDS = ("id", "identifier", "key", "index", "row_id")
IGNORE_ROLE_WORDS = ("ignore", "drop", "exclude", "skip", "unused", "metadata")
META_EXTS = (".json", ".yaml", ".yml")
META_NAME_HINTS = ("metadata", "meta", "task", "spec", "info", "schema", "config", "dataset",
                   "manifest", "description")

# ---- metadata key synonyms (matched case-insensitively, recursively) -------------------
KEYS = {
    "task": ("task", "task_type", "tasktype", "problem_type", "problem", "type", "ml_task",
             "learning_task", "objective_type"),
    "target": ("target", "target_column", "target_col", "target_name", "label", "label_column",
               "label_col", "y", "y_column", "outcome", "response", "dependent_variable"),
    "id_column": ("id_column", "id_col", "id", "index_column", "row_id_column", "id_field",
                  "key_column", "identifier"),
    "n_clusters": ("n_clusters", "num_clusters", "number_of_clusters", "k", "n_groups",
                   "num_groups", "number_of_groups", "required_groups", "n_components",
                   "clusters", "groups", "n_segments"),
    "feature_types": ("feature_types", "column_types", "featuretypes", "columns_types", "dtypes",
                      "types", "schema", "features", "columns"),
    "metric": ("metric", "eval_metric", "evaluation_metric", "scoring", "score", "metrics",
               "evaluation"),
    "name": ("dataset_id", "dataset", "dataset_name", "name", "id_dataset", "title"),
    "prediction_column": ("prediction_column", "prediction_col", "output_column", "pred_column",
                          "submission_column"),
    "predictions_file": ("predictions_file", "prediction_file", "output_file",
                         "predictions_path", "submission_file"),
    "profile_file": ("profile_file", "profile_path", "data_profile_file"),
}

TASK_SYNONYMS = {
    "classification": ("classification", "classify", "classifier", "binary", "multiclass",
                       "multi-class", "binary_classification", "multiclass_classification",
                       "clf"),
    "regression": ("regression", "regress", "regressor", "continuous", "reg", "forecast",
                   "forecasting"),
    "clustering": ("clustering", "cluster", "unsupervised", "segmentation", "grouping",
                   "groups", "clu", "kmeans"),
}

FEATURE_TYPE_SYNONYMS = {
    "numeric": ("numeric", "numerical", "number", "num", "float", "int", "integer", "continuous",
                "real", "double", "float64", "int64", "quantitative", "interval", "ratio"),
    "categorical": ("categorical", "category", "cat", "nominal", "string", "str", "object",
                    "factor", "label", "enum", "discrete", "binary", "bool", "boolean",
                    "ordinal", "ordered"),
    "datetime": ("datetime", "date", "time", "timestamp", "datetime64", "temporal"),
    "text": ("text", "free_text", "freetext", "sentence", "document"),
    "id": ("id", "identifier", "key", "index", "ignore", "drop"),
}


def normalize_task(value: Any) -> str | None:
    if value is None:
        return None
    v = str(value).strip().lower().replace(" ", "_")
    for task, syns in TASK_SYNONYMS.items():
        if v in syns:
            return task
    for task, syns in TASK_SYNONYMS.items():
        if any(s in v for s in syns if len(s) > 3):
            return task
    return None


def normalize_feature_type(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for k in ("type", "dtype", "kind", "feature_type"):
            if k in value:
                return normalize_feature_type(value[k])
        return None
    v = str(value).strip().lower()
    for t, syns in FEATURE_TYPE_SYNONYMS.items():
        if v in syns:
            return t
    for t, syns in FEATURE_TYPE_SYNONYMS.items():
        if any(v.startswith(s) for s in syns if len(s) > 2):
            return t
    return None


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def _flatten(d: Any, out: dict | None = None, depth: int = 0) -> dict:
    """Collect all (lower-cased key -> value) pairs, shallow keys win.

    Containers of per-column information (feature types, column stats) are not
    descended into, so a column called e.g. ``type`` or ``id`` is never mistaken
    for a metadata key.
    """
    out = {} if out is None else out
    if isinstance(d, dict) and depth < 4:
        for k, v in d.items():
            key = str(k).strip().lower()
            if key not in out:
                out[key] = v
        for k, v in d.items():
            key = str(k).strip().lower()
            skip = key in KEYS["feature_types"] or key.endswith(
                ("_types", "_type", "columns", "features", "stats"))
            if isinstance(v, dict) and not skip:
                _flatten(v, out, depth + 1)
    return out


def _lookup(flat: dict, name: str):
    for k in KEYS[name]:
        if k in flat and flat[k] is not None:
            return flat[k]
    return None


def parse_feature_types(raw: Any, flat: dict) -> dict[str, str]:
    """Accepts {col: type}, [{name, type}], or numeric/categorical column lists."""
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        # {col: type} or {"numeric": [...], "categorical": [...]}
        if all(isinstance(v, (list, tuple)) for v in raw.values()) and raw:
            for t, cols in raw.items():
                nt = normalize_feature_type(t)
                for c in cols:
                    if nt:
                        out[str(c)] = nt
        else:
            for c, t in raw.items():
                nt = normalize_feature_type(t)
                if nt:
                    out[str(c)] = nt
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict):
                name = item.get("name", item.get("column", item.get("feature")))
                nt = normalize_feature_type(item)
                if name is not None and nt:
                    out[str(name)] = nt
    # explicit lists anywhere in the metadata
    for key, t in (("numeric_features", "numeric"), ("numerical_features", "numeric"),
                   ("numeric_columns", "numeric"), ("num_features", "numeric"),
                   ("continuous_features", "numeric"),
                   ("categorical_features", "categorical"), ("categorical_columns", "categorical"),
                   ("cat_features", "categorical"), ("nominal_features", "categorical"),
                   ("ordinal_features", "categorical"), ("binary_features", "categorical"),
                   ("datetime_features", "datetime"), ("date_features", "datetime"),
                   ("datetime_columns", "datetime"), ("text_features", "text"),
                   ("text_columns", "text")):
        cols = flat.get(key)
        if isinstance(cols, (list, tuple)):
            for c in cols:
                out.setdefault(str(c), t)
    return out


def read_metadata_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # optional
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return _mini_yaml(text)


def _yaml_scalar(v: str):
    v = v.strip()
    if v.startswith(("'", '"')) and v.endswith(("'", '"')) and len(v) >= 2:
        return v[1:-1]
    low = v.lower()
    if low in ("null", "~", "none", ""):
        return None
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if v.startswith("[") and v.endswith("]"):
        return [_yaml_scalar(x) for x in v[1:-1].split(",") if x.strip()]
    return v


def _mini_yaml(text: str) -> dict:
    """Tiny YAML subset (nested mappings, '- item' lists, scalars) used when PyYAML
    is not installed."""
    root: dict = {}
    stack = [(-1, root)]          # (indent, container)
    pending = None                # (indent, parent dict, key) waiting for a child block
    for raw in text.splitlines():
        line = raw.split(" #")[0].rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        ind = len(line) - len(line.lstrip())
        body = line.strip()
        if pending is not None:
            p_ind, parent, key = pending
            pending = None
            if ind > p_ind:
                child = [] if body.startswith("- ") or body == "-" else {}
                parent[key] = child
                stack.append((p_ind, child))
        while len(stack) > 1 and ind <= stack[-1][0]:
            stack.pop()
        cont = stack[-1][1]
        if body.startswith("- ") or body == "-":
            item = body[2:].strip() if body != "-" else ""
            if isinstance(cont, list):
                m = re.match(r"^([^:]+):\s*(.*)$", item)
                if m and not item.startswith(("'", '"')):
                    d = {m.group(1).strip(): _yaml_scalar(m.group(2))}
                    cont.append(d)
                    stack.append((ind, d))
                else:
                    cont.append(_yaml_scalar(item))
            continue
        m = re.match(r"^([^:]+):\s*(.*)$", body)
        if not m or not isinstance(cont, dict):
            continue
        key, val = m.group(1).strip().strip("'\""), m.group(2)
        if val.strip() == "":
            cont[key] = None
            pending = (ind, cont, key)
        else:
            cont[key] = _yaml_scalar(val)
    return root


@dataclass
class Metadata:
    task: str | None = None
    target: str | None = None
    id_column: str | None = None
    n_clusters: int | None = None
    feature_types: dict[str, str] = field(default_factory=dict)
    metric: str | None = None
    name: str | None = None
    prediction_column: str | None = None
    predictions_file: str | None = None
    profile_file: str | None = None
    raw: dict = field(default_factory=dict)
    source: str | None = None

    @classmethod
    def from_dict(cls, d: dict | None, source: str | None = None) -> "Metadata":
        d = d or {}
        flat = _flatten(d)
        task_raw = _lookup(flat, "task")
        if isinstance(task_raw, dict):
            task_raw = task_raw.get("type", task_raw.get("name"))
        task = normalize_task(task_raw)
        target = _lookup(flat, "target")
        if isinstance(target, (list, tuple)):
            target = target[0] if len(target) == 1 else None
        if isinstance(target, dict):
            target = target.get("name", target.get("column"))
        id_col = _lookup(flat, "id_column")
        if isinstance(id_col, (list, tuple)):
            id_col = id_col[0] if id_col else None
        if isinstance(id_col, dict):
            id_col = id_col.get("name", id_col.get("column"))
        k = _lookup(flat, "n_clusters")
        try:
            k = int(k) if k is not None and not isinstance(k, (list, dict)) else None
        except (TypeError, ValueError):
            k = None
        ft = parse_feature_types(_lookup(flat, "feature_types"), flat)
        metric = _lookup(flat, "metric")
        if isinstance(metric, (list, tuple)):
            metric = metric[0] if metric else None
        if isinstance(metric, dict):
            metric = metric.get("name", metric.get("metric"))
        name = _lookup(flat, "name")
        # column roles inside a schema ({"Churn": "target"}, [{name, type, role}])
        roles = _column_roles(_lookup(flat, "feature_types"))
        if target is None and roles.get("target"):
            target = roles["target"]
        if id_col is None and roles.get("id"):
            id_col = roles["id"]
        for c in roles.get("ignore", []):
            ft[c] = "id"
        # unusual key names ("task_family", "required_number_of_groups", "target_variable")
        if task is None:
            task = _fuzzy_task(flat)
        if k is None:
            k = _fuzzy_k(flat)
        if target is None:
            target = _fuzzy_target(flat)
        return cls(task=task, target=None if target is None else str(target),
                   id_column=None if id_col is None or isinstance(id_col, bool) else str(id_col),
                   n_clusters=k, feature_types=ft,
                   metric=None if metric is None else str(metric).lower(),
                   name=None if name is None or isinstance(name, (dict, list)) else str(name),
                   prediction_column=_as_str(_lookup(flat, "prediction_column")),
                   predictions_file=_as_str(_lookup(flat, "predictions_file")),
                   profile_file=_as_str(_lookup(flat, "profile_file")),
                   raw=d, source=source)


def _as_str(v):
    return None if v is None or isinstance(v, (dict, list)) else str(v)


def _column_roles(raw) -> dict:
    """Target / id / ignored columns declared through a role in the schema."""
    out: dict = {"ignore": []}

    def assign(col, val):
        v = str(val).strip().lower()
        if v in TARGET_ROLE_WORDS and "target" not in out:
            out["target"] = str(col)
        elif v in ID_ROLE_WORDS and "id" not in out:
            out["id"] = str(col)
        elif v in IGNORE_ROLE_WORDS:
            out["ignore"].append(str(col))

    if isinstance(raw, dict):
        for c, t in raw.items():
            if isinstance(t, dict):
                for key in ("role", "usage", "purpose", "kind"):
                    if key in t:
                        assign(c, t[key])
                        break
            elif isinstance(t, str) and not isinstance(c, (list, tuple)):
                if t.strip().lower() in TARGET_ROLE_WORDS + ID_ROLE_WORDS + IGNORE_ROLE_WORDS:
                    assign(c, t)
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict):
                name = item.get("name", item.get("column", item.get("feature")))
                if name is None:
                    continue
                for key in ("role", "usage", "purpose", "kind"):
                    if key in item:
                        assign(name, item[key])
                        break
                if item.get("is_target") is True and "target" not in out:
                    out["target"] = str(name)
    return out


def _fuzzy_task(flat: dict):
    for k, v in flat.items():
        if isinstance(v, str) and any(w in k for w in ("task", "problem", "family", "type", "kind",
                                                       "objective", "mode", "learning")):
            t = normalize_task(v)
            if t:
                return t
    return None


def _fuzzy_k(flat: dict):
    for k, v in flat.items():
        if any(w in k for w in ("cluster", "group", "segment")) and not k.endswith(
                ("column", "col", "name", "names", "columns")):
            if isinstance(v, bool):
                continue
            try:
                kv = int(str(v).strip())
            except (TypeError, ValueError):
                continue
            if 1 <= kv <= 1000:
                return kv
    return None


def _fuzzy_target(flat: dict):
    for k, v in flat.items():
        if isinstance(v, str) and len(v) < 200 and "/" not in v and "\\" not in v and any(
                w in k for w in ("target", "label", "response", "outcome", "predict", "dependent")):
            if not k.endswith(("type", "types", "metric", "file", "path")):
                return v
    return None


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _open_text(path: Path):
    suf = path.suffix.lower()
    if suf == ".gz":
        import gzip
        return gzip.open(path, "rt", encoding="utf-8-sig", errors="replace")
    if suf == ".bz2":
        import bz2
        return bz2.open(path, "rt", encoding="utf-8-sig", errors="replace")
    if suf == ".xz":
        import lzma
        return lzma.open(path, "rt", encoding="utf-8-sig", errors="replace")
    if suf == ".zip":
        import io
        import zipfile
        with zipfile.ZipFile(path) as zf:  # small header sample only; archive closed at once
            name = [n for n in zf.namelist() if not n.endswith("/")][0]
            with zf.open(name) as fh:
                head = fh.read(65536)
        return io.StringIO(head.decode("utf-8-sig", errors="replace"))
    return open(path, "r", encoding="utf-8-sig", errors="replace")


def _sniff_sep(path: Path) -> str:
    try:
        with _open_text(path) as fh:
            head = "".join([fh.readline() for _ in range(5)])
    except (OSError, IndexError, ValueError):
        return ","
    counts = {s: head.count(s) for s in (",", ";", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


_DECIMAL_COMMA = re.compile(r"^\s*-?\d+,\d+\s*$")


def _read_csv_robust(path: Path, sep: str, dtype) -> pd.DataFrame:
    """read_csv with encoding fall-backs and European decimal commas (``1,5``)."""
    last_exc = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            try:
                df = pd.read_csv(path, sep=sep, dtype=dtype, low_memory=False, encoding=enc)
            except (ValueError, KeyError):
                df = pd.read_csv(path, sep=sep, low_memory=False, encoding=enc)
            break
        except UnicodeDecodeError as exc:
            last_exc = exc
    else:
        raise last_exc
    if sep in (";", "\t", "|"):
        obj = [c for c in df.columns if df[c].dtype == object]
        hits = 0
        for c in obj[:50]:
            v = df[c].dropna().astype(str).head(200)
            if len(v) and v.str.match(_DECIMAL_COMMA).mean() > 0.8:
                hits += 1
        if hits:
            df = pd.read_csv(path, sep=sep, dtype=dtype, low_memory=False, decimal=",",
                             encoding=enc)
    return df


def read_table(path: Path, id_column: str | None = None) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in COMPRESSED:  # data.csv.gz -> judged by the inner extension
        inner = Path(path.stem).suffix.lower()
        suf = inner if inner in (".csv", ".tsv", ".txt", ".data", ".json", ".jsonl") else ".csv"
        if suf in (".json", ".jsonl"):
            df = pd.read_json(path, lines=suf == ".jsonl")
            df.columns = [str(c) for c in df.columns]
            return df
    dtype = {id_column: str} if id_column else None
    if suf in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif suf == ".feather":
        df = pd.read_feather(path)
    elif suf in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suf == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif suf == ".json":
        try:
            df = pd.read_json(path)
        except ValueError:
            df = pd.read_json(path, lines=True)
    else:
        sep = "\t" if suf == ".tsv" else _sniff_sep(path)
        df = _read_csv_robust(path, sep, dtype)
    df.columns = [str(c) for c in df.columns]
    if id_column and id_column in df.columns and suf not in (".csv", ".tsv", ".txt", ".data"):
        df[id_column] = df[id_column].astype(str)
    return df


def _is_table(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in TABLE_EXTS and not p.name.startswith(".")


def _looks_like_metadata_json(p: Path) -> bool:
    """A .json file is metadata if it is a dict (not records) with known keys."""
    try:
        if p.stat().st_size > 2_000_000:
            return False
        d = json.loads(p.read_text(encoding="utf-8-sig", errors="replace"))
    except Exception:
        return False
    if not isinstance(d, dict) or not d:
        return False
    flat = _flatten(d)
    known = {k for v in KEYS.values() for k in v}
    if any(k in known for k in flat):
        return True
    # a pandas-style data table ({"col": {"0": .., "1": ..}, ...}) has only long containers;
    # anything else that is a small JSON object is a description of the dataset
    vals = list(d.values())
    looks_like_table = len(vals) >= 2 and all(isinstance(v, (list, dict)) and len(v) >= 10
                                              for v in vals)
    hint = re.compile(r"task|target|label|cluster|group|feature|column|type|metric|problem")
    return not looks_like_table and any(hint.search(str(k).lower()) for k in flat)


@dataclass
class Dataset:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame | None
    meta: Metadata
    root: Path | None = None
    files: dict = field(default_factory=dict)
    submission_columns: list | None = None


OWN_OUTPUTS = {"predictions.csv", "profile.json", "data_profile.json", "batch_summary.json"}


def find_files(root: Path) -> dict:
    """Map roles (train/test/meta/labels/sample) to files inside a dataset dir."""
    files = sorted([p for p in root.iterdir() if p.is_file()
                    and p.name.lower() not in OWN_OUTPUTS and not p.name.endswith(".tmp")],
                   key=lambda p: p.name.lower())
    roles: dict[str, Path] = {}
    metas = [p for p in files if p.suffix.lower() in META_EXTS and
             (p.suffix.lower() != ".json" or _looks_like_metadata_json(p))]
    if metas:
        metas.sort(key=lambda p: (not any(h in p.stem.lower() for h in META_NAME_HINTS),
                                  p.name.lower()))
        roles["meta"] = metas[0]
    tables = [p for p in files if _is_table(p) and p not in metas]

    def pick(cands, *words, exclude=()):
        for w in words:
            for p in cands:
                s = p.stem.lower()
                if re.search(w, s) and not any(re.search(e, s) for e in exclude):
                    return p
        return None

    lab = ("label", "target", r"(^|_)y(_|$)", "solution", "answer", "truth", "submission")
    roles_sample = pick(tables, r"sample_?submission", r"submission_?(format|template|example)",
                        r"^submission$")
    if roles_sample:
        roles["sample"] = roles_sample
        tables = [p for p in tables if p != roles_sample]
    tr = pick(tables, r"(^|[_\-\s])train", r"train", r"(^|_)x_?train", r"^data$", r"^dataset$",
              r"^features$", r"^x$", exclude=lab)
    te = pick(tables, r"(^|[_\-\s])test", r"test", r"eval", r"holdout", r"to_?predict",
              r"inference", r"score", r"unlabel", exclude=lab + ("predictions?$",))
    if tr is None:
        rest = [p for p in tables if p != te and not any(re.search(e, p.stem.lower()) for e in lab)]
        if len(rest) >= 1:
            rest.sort(key=lambda p: -p.stat().st_size)
            tr = rest[0]
    if tr is not None:
        roles["train"] = tr
    if te is not None and te != tr:
        roles["test"] = te
    ytr = pick(tables, r"y_?train", r"train_?(labels?|targets?|y)", r"^labels?$", r"^targets?$",
               r"^y$")
    if ytr is not None and ytr not in (tr, te):
        roles["train_labels"] = ytr
    val = pick(tables, r"valid", r"(^|_)val($|_)", r"dev", exclude=lab)
    if val is not None and val not in (tr, te):
        roles["valid"] = val
    return roles


PATH_KEYS = {
    "train": ("train_path", "train_file", "train_csv", "training_file", "training_data",
              "train_data", "train_table", "train"),
    "test": ("test_path", "test_file", "test_csv", "test_data", "test_table", "eval_path",
             "evaluation_file", "predict_path", "test"),
    "data": ("data_path", "data_file", "dataset_path", "dataset_file", "table", "file", "path",
             "csv", "data"),
}
TEXT_HINTS = ("task", "readme", "description", "info", "instructions", "spec", "metadata",
              "dataset", "about", "notes")


def _paths_from_metadata(flat: dict, bases: list) -> dict:
    """File locations named inside the metadata (``"train_path": "data/train.csv"``)."""
    out = {}
    for role, keys in PATH_KEYS.items():
        for k in keys:
            v = flat.get(k)
            if not isinstance(v, str) or not v.strip() or len(v) > 400:
                continue
            for b in bases:
                cand = Path(v) if Path(v).is_absolute() else Path(b) / v
                if cand.is_file() and cand.suffix.lower() in TABLE_EXTS and not (
                        cand.suffix.lower() in META_EXTS and (cand.suffix.lower() != ".json"
                                                              or _looks_like_metadata_json(cand))):
                    out[role] = cand
                    break
            if role in out:
                break
    return out


_TASK_WORDS = [("clustering", r"\b(clustering|cluster(s|ed)?|segment(ation|s)?|unsupervised)\b"),
               ("regression", r"\b(regression|regress|predict (a|the) (number|value|price|count))\b"),
               ("classification", r"\b(classification|classif(y|ier)|binary|multi-?class)\b")]


def parse_text_description(text: str, columns: list) -> dict:
    """Task / target / number of groups from a free-text description (README.md, task.txt)."""
    out: dict = {}
    low = text.lower()
    m = re.search(r"\btask\s*[:=]\s*\**\s*([a-z_\- ]+)", low)
    if m:
        t = normalize_task(m.group(1).strip().split()[0] if m.group(1).strip() else "")
        if t:
            out["task"] = t
    if "task" not in out:
        hits = [(re.search(rx, low).start(), t) for t, rx in _TASK_WORDS if re.search(rx, low)]
        if hits:
            out["task"] = min(hits)[1]
    m = (re.search(r"\b(?:n_clusters|num_clusters|k|number of (?:clusters|groups|segments))"
                   r"\s*[:=]?\s*(\d{1,3})\b", low)
         or re.search(r"\b(\d{1,3})\s+(?:clusters|groups|segments)\b", low)
         or re.search(r"\b(two|three|four|five|six|seven|eight|nine|ten)\s+(?:clusters|groups|segments)\b",
                      low))
    if m:
        words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10}
        g = m.group(1)
        out["n_clusters"] = int(words.get(g, g))
    lower_cols = {str(c).lower(): str(c) for c in columns}
    for rx in (r"(?:target|label)(?:\s+column)?\s*(?:is|[:=])\s*[`'\"*]*([^`'\"*\n,;]+)",
               r"predict(?:s|ing)?\s+(?:the\s+)?[`'\"*]+([^`'\"*\n]+)[`'\"*]+"):
        for m in re.finditer(rx, text, flags=re.IGNORECASE):
            name = m.group(1).strip().rstrip(".")
            if name.lower() in lower_cols:
                out["target"] = lower_cols[name.lower()]
                break
        if "target" in out:
            break
    return out


def _text_files(root: Path, exclude=()) -> list:
    if root is None or not root.is_dir():
        return []
    excl = {Path(e).resolve() for e in exclude if e}
    cands = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in (".txt", ".md")
             and p.stat().st_size < 200_000 and p.resolve() not in excl
             and not re.search(r"train|test|data|valid", p.stem, re.I)]
    cands.sort(key=lambda p: (not any(h in p.stem.lower() for h in TEXT_HINTS), p.name.lower()))
    return cands


_NAME_TASK = [("classification", r"(^|[_\-\s.])(clf|cls|classif\w*)([_\-\s.\d]|$)"),
              ("regression", r"(^|[_\-\s.])(reg|regr|regression)([_\-\s.\d]|$)"),
              ("clustering", r"(^|[_\-\s.])(clu|clus|clust\w*|cluster\w*|segment\w*)([_\-\s.\d]|$)")]


def task_from_name(*names) -> str | None:
    for n in names:
        if not n:
            continue
        low = str(n).lower()
        for t, rx in _NAME_TASK:
            if re.search(rx, low):
                return t
    return None


def load_dataset(root: str | Path | None = None, *, train: str | None = None,
                 test: str | None = None, metadata: str | dict | None = None,
                 overrides: dict | None = None, description: str | None = None) -> Dataset:
    """Load a dataset folder (or explicit files) and normalise its metadata."""
    root_p = Path(root) if root else None
    files: dict[str, Path] = {}
    if root_p is not None and root_p.is_file():
        # a single file was given: either the train table or the metadata
        if root_p.suffix.lower() in META_EXTS and (root_p.suffix.lower() != ".json"
                                                    or _looks_like_metadata_json(root_p)):
            files["meta"] = root_p
            root_p = root_p.parent
            files = {**find_files(root_p), **files}
        else:
            files["train"] = root_p
            sib = find_files(root_p.parent)
            for role in ("meta", "test", "sample"):
                if role in sib and sib[role] != root_p:
                    files[role] = sib[role]
            root_p = root_p.parent
    elif root_p is not None:
        files = find_files(root_p)
    if train:
        files["train"] = Path(train)
    if test:
        files["test"] = Path(test)
    if isinstance(metadata, str) and metadata.strip().startswith("{"):
        try:  # metadata passed inline as JSON text
            metadata = json.loads(metadata)
        except ValueError:
            pass
    if root_p is None and train:
        # explicit files: pick up the description / test file sitting next to them
        tp = Path(train)
        sib = find_files(tp.parent) if tp.parent.is_dir() else {}
        if not metadata and "meta" in sib:
            files["meta"] = sib["meta"]
        if not test:
            guess = tp.with_name(re.sub(r"train", "test", tp.name, flags=re.I))
            if guess != tp and guess.is_file():
                files["test"] = guess
    meta_dict: dict = {}
    if isinstance(metadata, dict):
        meta_dict = metadata
    elif metadata:
        files["meta"] = Path(metadata)
    if "meta" in files and not meta_dict:
        try:
            meta_dict = read_metadata_file(files["meta"])
        except Exception:
            meta_dict = {}
    meta = Metadata.from_dict(meta_dict, source=str(files.get("meta")) if files.get("meta") else None)

    # file locations written inside the metadata win over guessed ones (not over CLI flags)
    if meta_dict:
        bases = [files["meta"].parent] if files.get("meta") else []
        bases += [root_p] if root_p is not None else []
        bases += [Path.cwd()]
        named = _paths_from_metadata(_flatten(meta_dict), bases)
        if "train" in named and not train:
            files["train"] = named["train"]
        elif "data" in named and not train:
            files["train"] = named["data"]
        if "test" in named and not test:
            files["test"] = named["test"]

    for k, v in (overrides or {}).items():
        if v is None:
            continue
        if k == "task":
            v = normalize_task(v) or v
        setattr(meta, k, v)

    if "train" not in files:
        raise FileNotFoundError(f"No training/data table found in {root}")
    tr = read_table(files["train"], meta.id_column)

    # free-text task descriptions (README.md / task.txt) fill what the metadata lacks
    if description:
        hints = parse_text_description(description, list(tr.columns))
        for key in ("task", "target", "n_clusters"):
            if getattr(meta, key) is None and hints.get(key):
                setattr(meta, key, hints[key])
    if meta.task is None or meta.target is None or meta.n_clusters is None:
        text_root = root_p if root_p is not None else files["train"].parent
        for tf in _text_files(text_root, exclude=list(files.values())):
            try:
                hints = parse_text_description(
                    tf.read_text(encoding="utf-8-sig", errors="replace"), list(tr.columns))
            except Exception:
                continue
            if meta.task is None and hints.get("task"):
                meta.task = hints["task"]
            if meta.target is None and hints.get("target"):
                meta.target = hints["target"]
            if meta.n_clusters is None and hints.get("n_clusters"):
                meta.n_clusters = hints["n_clusters"]
            if hints:
                files.setdefault("description", tf)
                break
    # last resort: the task encoded in the folder / file name (clf_01, reg_house, cluster_3)
    if meta.task is None:
        meta.task = task_from_name(root_p.name if root_p is not None else None,
                                   files["train"].stem, meta.name)
    if "train_labels" in files:
        y = read_table(files["train_labels"], meta.id_column)
        if meta.id_column and meta.id_column in y.columns and meta.id_column in tr.columns \
                and y.shape[1] == 2:
            tr = tr.merge(y, on=meta.id_column, how="left")
        elif len(y) == len(tr):
            ycol = [c for c in y.columns if c != meta.id_column][-1]
            name = meta.target or ycol
            tr[name] = y[ycol].values
            meta.target = meta.target or name
    if "valid" in files:
        try:
            va = read_table(files["valid"], meta.id_column)
            if meta.target and meta.target in va.columns and set(tr.columns) == set(va.columns):
                tr = pd.concat([tr, va[tr.columns]], ignore_index=True)
        except Exception:
            pass
    te = read_table(files["test"], meta.id_column) if "test" in files else None

    # a single table with an explicit split column
    if te is None:
        for sc in ("split", "set", "subset", "is_train", "is_test", "partition"):
            if sc in tr.columns:
                vals = tr[sc].astype(str).str.lower()
                is_test = vals.isin(["test", "1", "true", "eval", "holdout"]) if sc == "is_test" \
                    else (~vals.isin(["train", "1", "true"]) if sc == "is_train"
                          else vals.isin(["test", "eval", "holdout"]))
                if 0 < is_test.sum() < len(tr):
                    te = tr[is_test].drop(columns=[sc]).reset_index(drop=True)
                    tr = tr[~is_test].drop(columns=[sc]).reset_index(drop=True)
                break

    sub_cols = None
    if "sample" in files:
        try:
            sub_cols = list(pd.read_csv(files["sample"], nrows=5).columns)
        except Exception:
            sub_cols = None

    stem = re.sub(r"\.(csv|tsv|txt|json|jsonl|parquet)$", "", files["train"].stem, flags=re.I)
    if re.fullmatch(r"(x_?)?(train|training|data|dataset|features|input|table)(_?data|_?set)?", stem,
                    flags=re.I):
        stem = files["train"].parent.name or stem  # generic file name: use the folder name
    name = meta.name or (root_p.name if root_p is not None else stem)
    infer_missing_meta(tr, te, meta)
    return Dataset(name=str(name), train=tr, test=te, meta=meta, root=root_p, files=files,
                   submission_columns=sub_cols)


def infer_missing_meta(train: pd.DataFrame, test: pd.DataFrame | None, meta: Metadata) -> None:
    """Fill task / target / id when the metadata does not provide them."""
    cols = list(train.columns)
    # case-insensitive match of declared names
    lower = {c.lower(): c for c in cols}
    if meta.target and meta.target not in cols and meta.target.lower() in lower:
        meta.target = lower[meta.target.lower()]
    if meta.id_column and meta.id_column not in cols and meta.id_column.lower() in lower:
        meta.id_column = lower[meta.id_column.lower()]
    if meta.id_column and meta.id_column not in cols and \
            (test is None or meta.id_column not in test.columns):
        meta.id_column = None
    if meta.target and meta.target not in cols:
        meta.target = None

    if meta.target is None and meta.task != "clustering":
        if test is not None:
            extra = [c for c in cols if c not in test.columns]
            if len(extra) == 1:
                meta.target = extra[0]
        if meta.target is None:
            for c in cols:
                if c.lower() in ("target", "label", "class", "y", "outcome"):
                    meta.target = c
                    break
    if meta.task is None:
        if meta.n_clusters is not None and meta.target is None:
            meta.task = "clustering"
        elif meta.target is None:
            meta.task = "clustering"
        else:
            meta.task = infer_supervised_task(train[meta.target])
    if meta.task in ("classification", "regression") and meta.target is None:
        meta.task = "clustering"
    if meta.id_column is None:
        for c in cols:
            if c.lower() in ("id", "row_id", "rowid", "index", "idx", "sample_id", "record_id"):
                if train[c].is_unique and (test is None or c in test.columns):
                    meta.id_column = c
                    break


def infer_supervised_task(y: pd.Series) -> str:
    y = y.dropna()
    if y.empty:
        return "regression"
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        num = pd.to_numeric(y, errors="coerce")
        if num.notna().mean() > 0.99 and num.nunique() > 20:
            return "regression"
        return "classification"
    if pd.api.types.is_float_dtype(y) and not np.allclose(y, np.round(y)):
        return "regression"
    nun = y.nunique()
    if nun <= 2:
        return "classification"
    if nun <= 20 and nun / max(len(y), 1) < 0.05:
        return "classification"
    return "regression"


# --------------------------------------------------------------------------- #
# Discovery of several datasets (batch mode)
# --------------------------------------------------------------------------- #
def is_dataset_dir(p: Path) -> bool:
    return p.is_dir() and any(_is_table(f) for f in p.iterdir())


def list_dataset_dirs(root: Path) -> list[Path]:
    """Sub-directories that look like datasets (used when root holds several)."""
    if not root.is_dir():
        return []
    subs = sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))])
    return [d for d in subs if is_dataset_dir(d)]


def list_flat_datasets(root: Path) -> list[dict]:
    """Several datasets stored as prefixed files in one folder.

    e.g. ``clf_01_train.csv, clf_01_test.csv, clf_01_metadata.json, reg_02_train.csv ...``
    Returns ``[{"name", "train", "test", "metadata"}]`` (only when >= 2 train files).
    """
    if not root.is_dir():
        return []
    files = [p for p in root.iterdir() if p.is_file()]
    trains = [p for p in files if _is_table(p) and re.search(r"train", p.stem, re.I)
              and not re.search(r"label|target|(^|_)y(_|$)", p.stem, re.I)]
    if len(trains) < 2:
        return []
    out = []
    for tr in sorted(trains):
        prefix = re.split(r"[_\-\s.]*train", tr.stem, flags=re.I)[0]
        if not prefix:
            continue
        same = [p for p in files if p.stem.lower().startswith(prefix.lower()) and p != tr]
        te = next((p for p in same if _is_table(p) and re.search(r"test|eval", p.stem, re.I)), None)
        meta = next((p for p in same if p.suffix.lower() in META_EXTS
                     and (p.suffix.lower() != ".json" or _looks_like_metadata_json(p))), None)
        out.append({"name": prefix.strip("_-. "), "train": tr, "test": te, "metadata": meta})
    return out


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if o is None or isinstance(o, (str, int, bool)):
        return o
    if isinstance(o, np.ndarray):
        return [to_jsonable(v) for v in o.tolist()]
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    return str(o)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def safe_name(name: str) -> str:
    """Dataset name usable as a single folder name (no separators, no '..')."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._") or "dataset"
    return s[:100]


def _file_name(value: str | None, default: str) -> str:
    """File names taken from metadata are reduced to a bare name (no directories)."""
    n = Path(str(value)).name if value else ""
    return n if n and n not in (".", "..") else default


def resolve_output_paths(output: str | Path | None, ds: Dataset) -> tuple[Path, Path]:
    """Return (predictions_csv, profile_json) paths."""
    meta = ds.meta
    pred_name = _file_name(meta.predictions_file, "predictions.csv")
    prof_name = _file_name(meta.profile_file, "profile.json")
    if output is None:
        output = Path("outputs") / safe_name(ds.name)
    out = Path(output)
    if out.suffix.lower() == ".csv":
        pred_path = out
        prof_path = out.with_name(prof_name)
    elif out.suffix.lower() == ".json":
        prof_path = out
        pred_path = out.with_name(pred_name)
    else:
        pred_path = out / pred_name
        prof_path = out / prof_name
    inputs = {Path(f).resolve() for f in ds.files.values() if f}
    if pred_path.resolve() in inputs or prof_path.resolve() in inputs:
        safe = Path("outputs") / safe_name(ds.name)  # never overwrite an input file
        pred_path, prof_path = safe / "predictions.csv", safe / "profile.json"
    return pred_path, prof_path


def prediction_frame(ds: Dataset, ids, preds) -> pd.DataFrame:
    """Build the predictions table: [id column, prediction column]."""
    meta = ds.meta
    pred_col = meta.prediction_column or "prediction"
    id_col = meta.id_column
    if ds.submission_columns and len(ds.submission_columns) >= 2:
        id_col = ds.submission_columns[0]
        pred_col = ds.submission_columns[1]
    elif ds.submission_columns and len(ds.submission_columns) == 1:
        pred_col = ds.submission_columns[0]
        id_col = None
    preds = np.asarray(preds)
    if id_col is not None and ids is not None:
        return pd.DataFrame({id_col: np.asarray(ids), pred_col: preds})
    return pd.DataFrame({pred_col: preds})


def write_outputs(pred_df: pd.DataFrame, profile: dict, pred_path: Path, prof_path: Path) -> None:
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pred_path.with_name(pred_path.name + ".tmp")
    pred_df.to_csv(tmp, index=False)
    os.replace(tmp, pred_path)
    write_profile(profile, prof_path)


def write_profile(profile: dict, prof_path: Path) -> None:
    text = json.dumps(to_jsonable(profile), indent=2, allow_nan=False)
    _atomic_write_text(prof_path, text)
    alias = prof_path.with_name("data_profile.json")
    if alias != prof_path:
        _atomic_write_text(alias, text)
