"""End-to-end run of the standard entry point on the bundled example datasets."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score, f1_score, r2_score

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


@pytest.mark.parametrize("name,metric,minimum", [
    ("classification_demo", "macro_f1", 0.70),
    ("regression_demo", "r2", 0.70),
    ("clustering_demo", "ari", 0.80),
])
def test_example(tmp_path, name, metric, minimum):
    t0 = time.time()
    cp = subprocess.run([sys.executable, str(ROOT / "main.py"), str(EXAMPLES / name),
                         str(tmp_path), "--quiet"], capture_output=True, text=True, timeout=120)
    secs = time.time() - t0
    assert cp.returncode == 0, cp.stderr
    assert secs < 60
    pred = pd.read_csv(tmp_path / "predictions.csv")
    prof = json.loads((tmp_path / "profile.json").read_text())
    lab = pd.read_csv(EXAMPLES / "_labels" / name / "test_labels.csv")
    assert len(pred) == len(lab) and (pred["id"].values == lab["id"].values).all()
    y, p = lab.iloc[:, 1].values, pred["prediction"].values
    score = {"macro_f1": lambda: f1_score(y.astype(str), p.astype(str), average="macro"),
             "r2": lambda: r2_score(y, p), "ari": lambda: adjusted_rand_score(y, p)}[metric]()
    assert score >= minimum, f"{name}: {metric}={score:.3f}"
    assert prof["task"] == json.loads((EXAMPLES / name / "metadata.json").read_text())["task"]
