"""Alias of main.py for graders that expect `python solution.py ...` (same CLI and API)."""
import os
import sys
import time

os.environ.setdefault("AUTOML_T0", str(time.time()))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automl.cli import main  # noqa: E402
from automl import run, solve, predict  # noqa: E402,F401

if __name__ == "__main__":
    sys.exit(main())
