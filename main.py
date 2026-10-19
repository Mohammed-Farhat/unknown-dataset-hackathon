"""Standard entry point.

    python main.py <dataset_dir> [<output_dir>]
    python main.py --data-dir <dataset_dir> --output-dir <output_dir>

Writes ``predictions.csv`` and ``profile.json`` into the output directory
(default ``outputs/<dataset_name>/``) and prints a one-line JSON summary.
See README.md for the full input/output contract.
"""
import os
import sys
import time

# Start the clock before the heavy imports: the 60 s limit includes them.
os.environ.setdefault("AUTOML_T0", str(time.time()))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automl.cli import main  # noqa: E402
from automl import run, solve, predict  # noqa: E402,F401  (importable API)

if __name__ == "__main__":
    sys.exit(main())
