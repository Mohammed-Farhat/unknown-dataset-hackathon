"""``python -m automl <dataset_dir> [<output_dir>]``"""
import os
import sys
import time

os.environ.setdefault("AUTOML_T0", str(time.time()))

from automl.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
