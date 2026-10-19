"""Column typing, cleaning and feature construction shared by all tasks."""
from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ID_NAME_RE = re.compile(r"(^|[_\s\-.])(id|uuid|guid|key|index|idx|identifier|code)$|^(id|uuid|guid)"
                        r"([_\s\-.]|$)|^(row|record|sample|customer|client|user|product|"
                        r"transaction|order|account|passenger|patient|employee)[_\s\-.]?(id|no|"
                        r"number|num|code)$|^unnamed:\s*\d+$|^udi$", re.IGNORECASE)
LABEL_NAME_RE = re.compile(r"^(label|labels|target|class|cluster|clusters|group|segment|y|"
                           r"truth|true_label|ground_truth|outcome)$", re.IGNORECASE)
MISSING_TOKENS = {"", "?", "na", "n/a", "nan", "none", "null", "-", "--", "missing", "unknown?"}


# --------------------------------------------------------------------------- #
# Detection helpers
# --------------------------------------------------------------------------- #
def constant_columns(df: pd.DataFrame) -> list[str]:
    """Columns with at most one distinct non-missing value (all-NaN included)."""
    return [c for c in df.columns if df[c].nunique(dropna=True) <= 1]


def duplicate_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(original, duplicate) pairs of identical columns.

    Equality follows ``df.T.duplicated()`` (missing values compare equal, dtype
    ignored) but is computed with per-column hashes so it stays fast on wide
    tables. The copy is the later column unless the names say otherwise
    (``x_copy`` / ``x_dup`` / ``x.1`` is the copy of ``x``), see ``_orient_pair``.
    """
    groups: dict[str, list[str]] = {}
    norm: dict[str, np.ndarray] = {}
    pairs = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            arr = s.astype("float64").to_numpy()  # dtype-agnostic: 1 == 1.0 == True
            key = "n" + hashlib.md5(np.where(np.isnan(arr), np.inf, arr).tobytes()).hexdigest()
        else:
            arr = s.astype(object).where(s.notna(), "\x00<NA>").astype(str).to_numpy()
            key = "s" + hashlib.md5("\x1f".join(arr.tolist()).encode("utf-8", "replace")).hexdigest()
        norm[c] = arr
        cands = groups.setdefault(key, [])
        for o in cands:
            a, b = norm[o], arr
            same = (np.array_equal(a, b, equal_nan=True) if key[0] == "n"
                    else np.array_equal(a, b))
            if same:
                pairs.append(_orient_pair(o, c))
                break
        else:
            cands.append(c)
    return pairs


_COPY_HINT = re.compile(r"(copy|dup|duplicate|clone|replica|redundant|mirror|_bis$|\.\d+$|_\d+$)",
                        re.IGNORECASE)


def _orient_pair(first: str, second: str) -> tuple[str, str]:
    """Return (original, duplicate). Default: the later column is the duplicate,
    unless the names make clear which one is the copy (``x_copy``, ``x_dup``, ``x.1``)."""
    a, b = str(first), str(second)
    if a != b and a.lower().startswith(b.lower()) and _COPY_HINT.search(a[len(b):] or a):
        return second, first
    if a != b and b.lower().startswith(a.lower()):
        return first, second
    if _COPY_HINT.search(a) and not _COPY_HINT.search(b):
        return second, first
    return first, second


def is_id_like(s: pd.Series, name: str, n_rows: int) -> bool:
    """Row identifiers carry no signal and must not be used as features."""
    nn = s.dropna()
    if len(nn) < 20:
        return False
    nun = nn.nunique()
    uniq_ratio = nun / len(nn)
    if pd.api.types.is_float_dtype(s):
        vals = nn.values
        if not np.allclose(vals, np.round(vals)):
            return False
    if ID_NAME_RE.search(str(name).strip()) and uniq_ratio >= 0.95:
        return True
    if pd.api.types.is_integer_dtype(s) and uniq_ratio == 1.0 and len(nn) == n_rows:
        v = np.sort(nn.values.astype(np.int64))
        d = np.diff(v)
        if len(d) and (d == 1).all():  # 0..n-1 / 1..n style row counter
            return True
    if (s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype)) and uniq_ratio >= 0.95 \
            and len(nn) >= 50:
        if s.dtype == object and (looks_datetime(s, name) or looks_numeric_text(s)):
            return False  # timestamps / numbers stored as text are features, not ids
        sample = nn.astype(str).head(200)
        avg_words = sample.str.split().str.len().mean()
        if avg_words <= 2:  # short codes like "M14860", not free text
            return True
    return False


def _strip_missing_tokens(s: pd.Series) -> pd.Series:
    st = s.astype(str).str.strip()
    return st.mask(st.str.lower().isin(MISSING_TOKENS) | s.isna())


_THOUSANDS = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_DEC_COMMA = re.compile(r"^-?\d+,\d+$")


def _normalise_number_text(st: pd.Series) -> pd.Series:
    """'1,234.5' -> '1234.5' (thousands), '1,05' -> '1.05' (decimal comma), '$5'/'5%' -> '5'."""
    st = st.str.replace(r"^\$|%$", "", regex=True).str.strip()
    thou = st.str.match(_THOUSANDS, na=False).astype(bool)
    dec = (~thou) & st.str.match(_DEC_COMMA, na=False).astype(bool)
    st = st.where(~thou, st.str.replace(",", "", regex=False))
    st = st.where(~dec, st.str.replace(",", ".", regex=False))
    return st


def looks_numeric_text(s: pd.Series) -> bool:
    if s.dtype != object:
        return False
    st = _strip_missing_tokens(s).dropna()
    if st.empty:
        return False
    conv = pd.to_numeric(_normalise_number_text(st), errors="coerce")
    return conv.notna().mean() >= 0.95


def to_numeric_text(s: pd.Series) -> pd.Series:
    st = _strip_missing_tokens(s)
    return pd.to_numeric(_normalise_number_text(st), errors="coerce")


_DT_HINT = re.compile(r"\d{1,4}[-/.:]\d{1,2}([-/.:]\d{1,4})?")


def looks_datetime(s: pd.Series, name: str = "") -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype != object:
        return False
    nn = s.dropna().astype(str).str.strip()
    if nn.empty:
        return False
    sample = nn.sample(min(len(nn), 300), random_state=0) if len(nn) > 300 else nn
    if sample.str.match(_DT_HINT).mean() < 0.9:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.9


def parse_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                out = pd.to_datetime(s, errors="coerce", format="mixed")
            except (TypeError, ValueError):
                out = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_convert(None)
    except (AttributeError, TypeError):
        pass
    return out


def looks_text(s: pd.Series) -> bool:
    if s.dtype != object:
        return False
    nn = s.dropna().astype(str)
    if len(nn) < 20:
        return False
    sample = nn.head(300)
    return sample.str.split().str.len().mean() >= 4 and nn.nunique() / len(nn) > 0.5


def infer_type(s: pd.Series, name: str = "") -> str:
    """numeric | categorical | datetime | text (bool -> categorical)."""
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if isinstance(s.dtype, pd.CategoricalDtype):
        return "categorical"
    if looks_numeric_text(s):
        return "numeric"
    if looks_datetime(s, name):
        return "datetime"
    if looks_text(s):
        return "text"
    return "categorical"


# --------------------------------------------------------------------------- #
# Column plan
# --------------------------------------------------------------------------- #
@dataclass
class ColumnPlan:
    numeric: list = field(default_factory=list)
    categorical: list = field(default_factory=list)
    datetime: list = field(default_factory=list)
    text: list = field(default_factory=list)
    dropped: dict = field(default_factory=dict)        # column -> reason
    declared_types: dict = field(default_factory=dict)  # from metadata (normalised)
    inferred_types: dict = field(default_factory=dict)  # final type per kept/typed column
    constant: list = field(default_factory=list)
    duplicates: list = field(default_factory=list)      # (original, duplicate)
    id_like: list = field(default_factory=list)

    @property
    def used(self) -> list:
        return self.numeric + self.categorical + self.datetime + self.text


def plan_columns(train: pd.DataFrame, test: pd.DataFrame | None, target: str | None,
                 id_column: str | None, declared: dict | None = None) -> ColumnPlan:
    """Decide which columns to use and how to treat them."""
    declared = declared or {}
    plan = ColumnPlan(declared_types=dict(declared))
    feats = [c for c in train.columns if c != target and c != id_column]
    if id_column:
        plan.dropped[id_column] = "id_column"
    n = len(train)

    for c in feats:
        if test is not None and c not in test.columns:
            plan.dropped[c] = "absent_from_test"
    # When the metadata lists (nearly) every column with its type, a column it leaves
    # out (e.g. a "label" column next to clustering features) is not a feature.
    if declared:
        covered = [c for c in feats if c in declared]
        complete = len(covered) >= 0.8 * len(feats) and any(t == "numeric"
                                                            for t in declared.values())
        mostly = len(covered) >= 0.5 * len(feats)
        for c in feats:
            if c in declared or c in plan.dropped:
                continue
            if complete or (mostly and LABEL_NAME_RE.search(str(c))):
                plan.dropped[c] = "not_in_declared_feature_types"
    feats = [c for c in feats if c not in plan.dropped]

    X = train[feats]
    plan.constant = constant_columns(X)
    for c in plan.constant:
        plan.dropped[c] = "constant"
    plan.duplicates = duplicate_columns(X)
    for orig, dup in plan.duplicates:
        if dup not in plan.dropped:
            plan.dropped[dup] = f"duplicate_of:{orig}"

    for c in feats:
        if c in plan.dropped:
            continue
        dt = declared.get(c)
        if dt == "id":
            plan.dropped[c] = "declared_id"
            plan.id_like.append(c)
            continue
        if is_id_like(train[c], c, n):  # also when declared numeric (e.g. CustomerID)
            plan.dropped[c] = "id_like"
            plan.id_like.append(c)
            continue
        if dt == "numeric":
            t = "numeric"
        elif dt == "categorical":
            # timestamps declared as categorical are far more useful expanded
            t = "datetime" if looks_datetime(train[c], c) else "categorical"
        elif dt == "datetime":
            t = "datetime"
        elif dt == "text":
            t = "text" if looks_text(train[c]) else "categorical"
        else:
            t = infer_type(train[c], c)
        if t == "datetime" and not (pd.api.types.is_datetime64_any_dtype(train[c])
                                    or looks_datetime(train[c].astype(object), c)):
            t = "categorical"
        plan.inferred_types[c] = t
        getattr(plan, t).append(c)
    return plan


# --------------------------------------------------------------------------- #
# Frame construction (raw -> typed matrix used by the models)
# --------------------------------------------------------------------------- #
class FeatureBuilder:
    """Turns raw tables into a clean typed frame.

    * numeric: coerced to float (text numbers such as " 1,234" handled)
    * categorical: strings, rare levels merged, missing -> NaN
    * datetime: expanded to year / month / day / weekday / hour / elapsed days
    * text: length + word count (+ treated as categorical when short)
    """

    def __init__(self, plan: ColumnPlan, max_levels: int = 60, min_count: int = 5):
        self.plan = plan
        self.max_levels = max_levels
        self.min_count = min_count

    def fit(self, X: pd.DataFrame):
        self.levels_: dict[str, list] = {}
        self.freq_: dict[str, pd.Series] = {}
        self.dt_parts_: dict[str, list] = {}
        self.dt_origin_: dict[str, pd.Timestamp] = {}
        for c in self.plan.categorical:
            s = self._cat_series(X[c])
            vc = s.value_counts()
            keep = vc[vc >= min(self.min_count, max(2, int(0.001 * len(s))))].index[: self.max_levels]
            self.levels_[c] = list(keep)
            if len(vc) > self.max_levels:  # high cardinality: also give the level frequency
                self.freq_[c] = vc / max(len(s), 1)
        for c in self.plan.datetime:
            d = parse_datetime(X[c])
            parts = []
            for part in ("year", "month", "day", "dayofweek", "hour", "minute"):
                v = getattr(d.dt, part)
                if v.nunique(dropna=True) > 1:
                    parts.append(part)
            if d.dt.dayofweek.nunique(dropna=True) > 1:
                parts.append("is_weekend")
            if d.dt.dayofyear.nunique(dropna=True) > 1:
                parts.append("dayofyear")
            self.dt_parts_[c] = parts
            self.dt_origin_[c] = d.min() if d.notna().any() else pd.Timestamp("1970-01-01")
        for c in self.plan.text:
            s = X[c].astype(str)
            vc = s.value_counts()
            self.levels_[c] = list(vc[vc >= self.min_count].index[: self.max_levels])
        return self

    @staticmethod
    def _cat_series(s: pd.Series) -> pd.Series:
        """Vectorised: values -> stripped strings (1.0 -> "1"), missing tokens -> None."""
        notna = s.notna()
        if pd.api.types.is_float_dtype(s):
            v = s.to_numpy(dtype=float)
            isint = notna.to_numpy() & np.isfinite(v) & (np.round(v) == v)
            txt = s.astype(str).to_numpy(dtype=object)
            if isint.any():
                txt[isint] = v[isint].astype(np.int64).astype(str)
            out = pd.Series(txt, index=s.index, dtype=object)
        elif pd.api.types.is_bool_dtype(s):
            out = s.astype(str)
        else:
            out = s.astype(str)
        out = out.str.strip()
        bad = ~notna | out.str.lower().isin(MISSING_TOKENS)
        return out.where(~bad, None)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for c in self.plan.numeric:
            s = X[c]
            if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                s = to_numeric_text(s.astype(object)) if s.dtype == object else \
                    pd.to_numeric(s, errors="coerce")
            cols[c] = s.astype(float).replace([np.inf, -np.inf], np.nan)
        for c in self.plan.datetime:
            d = parse_datetime(X[c])
            for part in self.dt_parts_.get(c, []):
                if part == "is_weekend":
                    v = (d.dt.dayofweek >= 5).astype(float).where(d.notna())
                else:
                    v = getattr(d.dt, part).astype(float)
                cols[f"{c}__{part}"] = v
            cols[f"{c}__elapsed_days"] = (d - self.dt_origin_[c]).dt.total_seconds() / 86400.0
        for c in self.plan.text:
            s = X[c].astype(str).where(X[c].notna(), "")
            cols[f"{c}__len"] = s.str.len().astype(float)
            cols[f"{c}__words"] = s.str.split().str.len().astype(float)
        for c, freq in self.freq_.items():
            cols[f"{c}__freq"] = self._cat_series(X[c]).map(freq).astype(float).fillna(0.0)
        out = pd.DataFrame(cols, index=X.index)
        for c in self.plan.categorical + self.plan.text:
            s = self._cat_series(X[c]) if c in self.plan.categorical else X[c].astype(str)
            lv = self.levels_.get(c, [])
            s = s.where(s.isin(lv) | s.isna(), "__other__")
            out[c] = pd.Categorical(s, categories=lv + ["__other__"])
        return out

    @property
    def categorical_out(self) -> list:
        return list(self.plan.categorical) + list(self.plan.text)
