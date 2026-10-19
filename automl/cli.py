"""Command-line interface (used by main.py / run.py / solution.py / python -m automl)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from .budget import HARD_LIMIT, Budget, Watchdog

USAGE = """\
python main.py <dataset_dir> [<output_dir>]
python main.py --data-dir <dataset_dir> --output-dir <output_dir>
python main.py --train train.csv --test test.csv --metadata metadata.json --output-dir out/
python main.py <folder_with_many_dataset_dirs> <output_root>      (batch mode)
"""


EPILOG = """\
examples:
  python main.py data/clf_01 out/clf_01
  python main.py train.csv test.csv metadata.json out/
  python main.py --train data.csv --task clustering --n-clusters 5 --output-dir out/
  python main.py --train train.csv --test test.csv --task regression --target price -o out/
  python main.py data/clf_01 --dry-run          # show how the dataset is understood, no training

outputs (in the output folder, default outputs/<dataset_name>/):
  predictions.csv  <id column>, prediction  (one row per test row, input order)
  profile.json     structured data profile (copy: data_profile.json)
  stdout           one JSON line: status, resolved task/target/files, output paths,
                   profile summary, runtime
flags always win over metadata; anything missing is inferred from the data.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py", usage=USAGE, epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Unknown-dataset AutoML: solves a classification, regression or clustering "
                    "table and writes predictions.csv + profile.json.")
    p.add_argument("paths", nargs="*",
                   help="dataset folder and/or data files (train, test, metadata), then output folder")
    p.add_argument("--description", "--task-description", dest="description",
                   help="free-text task description, e.g. 'cluster the rows into 4 groups'")
    p.add_argument("--dry-run", "--inspect", dest="dry_run", action="store_true",
                   help="only print how the dataset is understood (no training)")
    p.add_argument("--data-dir", "--data", "--dataset", "--dataset-dir", "--input", "--input-dir",
                   "--in", "--dataset-path", "--data-path", "-i", "-d", dest="data")
    p.add_argument("--output-dir", "--output", "--out", "--out-dir", "--output-path",
                   "--predictions", "--predictions-path", "--pred", "--prediction-file",
                   "--output-file", "--submission", "-o", dest="output")
    p.add_argument("--profile", "--profile-path", "--profile-file", dest="profile")
    p.add_argument("--train", "--train-file", "--train-path", dest="train")
    p.add_argument("--test", "--test-file", "--test-path", dest="test")
    p.add_argument("--metadata", "--meta", "--spec", "--task-file", "--config", "--schema",
                   "--metadata-path", dest="metadata")
    p.add_argument("--task", "--task-type", "--problem-type", dest="task")
    p.add_argument("--target", "--label", "--target-column", "--target-col", dest="target")
    p.add_argument("--n-clusters", "--n_clusters", "--k", "--clusters", "--num-clusters",
                   "--n-groups", dest="n_clusters", type=int)
    p.add_argument("--id-column", "--id-col", "--id", dest="id_column")
    p.add_argument("--metric", dest="metric")
    p.add_argument("--time-budget", dest="time_budget", type=float,
                   help="internal deadline in seconds (default 40, hard limit is 60)")
    p.add_argument("--time-limit", "--time_limit", "--timeout", "--max-time", "--time",
                   dest="time_limit", type=float,
                   help="external per-dataset limit in seconds (default 60)")
    p.add_argument("--seed", "--random-state", type=int, default=0,
                   help="kept for compatibility (always 0)")
    p.add_argument("--n-jobs", "--threads", dest="n_jobs", type=int)
    p.add_argument("--quiet", "-q", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def _setup_logging(quiet: bool, verbose: bool):
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(level=level, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def _batch(data_root: Path, out_root: Path, args, unknown) -> int:
    """Several datasets: each one runs in its own process with its own 60 s clock."""
    from .io_contract import list_dataset_dirs, list_flat_datasets
    script = Path(__file__).resolve().parents[1] / "main.py"
    jobs = []
    for d in list_dataset_dirs(data_root):
        jobs.append((d.name, [str(d), str(out_root / d.name)]))
    if not jobs:
        for spec in list_flat_datasets(data_root):
            a = ["--train", str(spec["train"]), "--output-dir", str(out_root / spec["name"])]
            if spec["test"]:
                a += ["--test", str(spec["test"])]
            if spec["metadata"]:
                a += ["--metadata", str(spec["metadata"])]
            jobs.append((spec["name"], a))
    env = {k: v for k, v in os.environ.items() if k != "AUTOML_T0"}  # each child: own clock
    results = []
    for name, a in jobs:
        cmd = [sys.executable, str(script)] + a
        if args.quiet:
            cmd.append("--quiet")
        t0 = time.time()
        try:
            cp = subprocess.run(cmd, timeout=HARD_LIMIT + 5, stdout=subprocess.PIPE, env=env)
            status = "ok" if cp.returncode == 0 else f"exit_{cp.returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
        if status == "ok" and not (out_root / name / "predictions.csv").exists():
            status = "no_output"
        results.append({"dataset": name, "status": status,
                        "seconds": round(time.time() - t0, 2),
                        "output_dir": str(out_root / name)})
        logging.info("batch: %s -> %s (%.1fs)", name, status, time.time() - t0)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "batch_summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"status": "ok", "mode": "batch", "datasets": results}))
    return 0


def _emit_summary(summary: dict) -> None:
    print(json.dumps(summary, default=str))
    sys.stdout.flush()


def main(argv=None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    _setup_logging(args.quiet, args.verbose)
    if unknown:
        logging.info("ignoring unrecognised arguments: %s", unknown)
    if args.n_jobs:
        os.environ["AUTOML_N_JOBS"] = str(max(1, args.n_jobs))

    data = args.data
    output = args.output
    from .io_contract import META_EXTS, TABLE_EXTS, _looks_like_metadata_json
    # positionals are sorted by what they are: a folder is the dataset, existing tables are
    # train then test, .json/.yaml is metadata, and the output is a path that does not exist
    # yet (or a second folder). Inputs are therefore never overwritten.
    dirs, tables, metas, missing = [], [], [], []
    for q in args.paths:
        qp = Path(q)
        if qp.is_dir():
            dirs.append(q)
        elif qp.is_file():
            suf = qp.suffix.lower()
            if suf in META_EXTS and (suf != ".json" or _looks_like_metadata_json(qp)):
                metas.append(q)
            elif suf in TABLE_EXTS:
                tables.append(q)
        elif not q.replace(".", "", 1).isdigit():  # a stray number is not a path
            missing.append(q)
    if data is None and dirs and not (args.train or tables):
        data = dirs.pop(0)
    if tables and args.train is None and data is None:
        if len(tables) == 1 and not metas:
            data = tables[0]            # single file: siblings (test, metadata) are detected
        else:
            args.train = tables[0]
            if len(tables) > 1 and args.test is None:
                args.test = tables[1]
    if metas and args.metadata is None:
        args.metadata = metas[0]
    if output is None:
        if missing:
            output = missing[-1]
        elif dirs:
            output = dirs[-1]
    data = data or os.environ.get("DATA_DIR") or os.environ.get("INPUT_DIR") \
        or os.environ.get("DATASET_DIR")
    output = output or os.environ.get("OUTPUT_DIR")
    if data is None and args.train is None:
        parser.print_help(sys.stderr)
        _emit_summary({"status": "error", "error": "no existing dataset path given"})
        return 2

    from .io_contract import is_dataset_dir, list_dataset_dirs, list_flat_datasets
    if data is not None and args.train is None:
        dp = Path(data)
        if dp.is_dir() and ((not is_dataset_dir(dp) and list_dataset_dirs(dp))
                            or list_flat_datasets(dp)):
            return _batch(dp, Path(output or "outputs"), args, unknown)

    if args.dry_run:
        from .io_contract import load_dataset
        from .pipeline import resolved_summary
        try:
            ds = load_dataset(data, train=args.train, test=args.test, metadata=args.metadata,
                              overrides={"task": args.task, "target": args.target,
                                         "n_clusters": args.n_clusters,
                                         "id_column": args.id_column, "metric": args.metric},
                              description=args.description)
        except Exception as exc:
            _emit_summary({"status": "error", "error": f"{type(exc).__name__}: {exc}",
                           "hint": _hint(exc)})
            return 1
        _emit_summary({"status": "dry_run", "resolved": resolved_summary(ds)})
        return 0

    total = args.time_budget
    if total is None and args.time_limit:
        total = max(5.0, min(args.time_limit - 20.0, 0.67 * args.time_limit))
    budget = Budget(total)
    state = {"pred": None, "prof": None, "name": None}

    def on_timeout():
        _emit_summary({"status": "timeout_fallback", "dataset": state["name"],
                       "predictions_path": state["pred"], "profile_path": state["prof"],
                       "runtime_seconds": round(budget.elapsed(), 3)})

    hard = args.time_limit or HARD_LIMIT
    watchdog = Watchdog(at_seconds=max(5.0, hard - 8.0), on_timeout=on_timeout).start()
    from .pipeline import run
    try:
        res = run(data, output, train=args.train, test=args.test, metadata=args.metadata,
                  task=args.task, target=args.target, n_clusters=args.n_clusters,
                  id_column=args.id_column, metric=args.metric, budget=budget,
                  state=state, description=args.description)
    except Exception as exc:  # unreadable input: nothing sensible can be predicted
        watchdog.cancel()
        logging.error("could not process %s: %s", data or args.train, exc)
        _emit_summary({"status": "error", "error": f"{type(exc).__name__}: {exc}",
                       "hint": _hint(exc), "runtime_seconds": round(budget.elapsed(), 3)})
        return 1
    watchdog.cancel()
    prof = res["profile"]
    if args.profile and res.get("profile_path"):
        from .io_contract import write_profile
        write_profile(prof, Path(args.profile))
    cb = prof.get("class_balance") or {}
    _emit_summary({
        "status": "ok" if not prof.get("errors") else "fallback",
        "dataset": prof.get("dataset"),
        "task": prof.get("task"),
        "predictions_path": str(Path(res["predictions_path"]).resolve())
        if res.get("predictions_path") else None,
        "profile_path": str(Path(res["profile_path"]).resolve()) if res.get("profile_path") else None,
        "n_predictions": None if res["predictions"] is None else int(len(res["predictions"])),
        "model": prof.get("selected_model"),
        "cv_score": prof.get("cv_score"),
        "resolved": res.get("resolved"),
        "profile_summary": {
            "n_rows": prof.get("n_rows"), "n_features": prof.get("n_features"),
            "total_missing": prof.get("total_missing"),
            "constant_columns": prof.get("constant_columns"),
            "duplicate_columns": prof.get("duplicate_columns"),
            "id_like_columns": prof.get("id_like_columns"),
            "irrelevant_features": prof.get("irrelevant_features"),
            "imbalance_ratio": cb.get("imbalance_ratio"),
            "data_issues": prof.get("data_issues"),
        },
        "runtime_seconds": round(budget.elapsed(), 3),
    })
    return 0


def _hint(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return ("no data table found: pass the dataset folder, or --train <file> "
                "[--test <file>] [--metadata <file>]")
    return ("if the task description was not understood, pass --task, --target, "
            "--n-clusters or --id-column (flags always win)")
