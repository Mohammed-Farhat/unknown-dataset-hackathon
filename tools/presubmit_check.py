"""Pre-submission checklist: run before pushing the final commit.

    python tools/presubmit_check.py

Checks that the repository satisfies every requirement of the announcement:
runnable code + standard entry point, pinned dependencies, team names, README
sections (approach, reproduction, assumptions), and that every example dataset
runs well inside the 60-second limit.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ok = True

    def check(cond: bool, msg: str):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok = ok and cond

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    team = readme.split("## Team", 1)[1].split("\n## ", 1)[0] if "## Team" in readme else ""
    check("## Team" in readme, "README has a Team section")
    check("TODO" not in team and "<" not in team, "team names filled in (no TODO / <placeholder>)")
    for section in ("How to run", "Approach", "Assumptions", "Reproduc", "Input / output contract"):
        check(section.lower() in readme.lower(), f"README mentions '{section}'")

    req = (ROOT / "requirements.txt").read_text().splitlines()
    pkgs = [ln for ln in req if ln.strip() and not ln.startswith("#")]
    check(all("==" in ln for ln in pkgs), "every dependency pinned with ==")
    check((ROOT / "main.py").exists(), "standard entry point main.py exists")

    for ex in sorted((ROOT / "examples").iterdir()):
        if not ex.is_dir() or ex.name.startswith("_"):
            continue
        out = ROOT / "runs" / "presubmit" / ex.name
        t0 = time.time()
        cp = subprocess.run([sys.executable, str(ROOT / "main.py"), str(ex), str(out), "--quiet"],
                            capture_output=True, text=True, timeout=120)
        secs = time.time() - t0
        good = cp.returncode == 0 and (out / "predictions.csv").exists() and \
            (out / "profile.json").exists()
        if good:
            json.loads((out / "profile.json").read_text())
        check(good and secs < 60, f"example {ex.name} runs ({secs:.1f}s)")

    big = [p for p in ROOT.rglob("*") if p.is_file() and p.stat().st_size > 5_000_000
           and ".git" not in p.parts and "data" not in p.parts and "runs" not in p.parts]
    check(not big, "no large files outside data/ and runs/ (ignored by git)")
    try:
        st = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                            text=True, timeout=30)
        check(st.returncode == 0 and not st.stdout.strip(), "git working tree clean (all committed)")
    except (OSError, subprocess.SubprocessError):
        check(False, "git available")
    print("\nREADY TO SUBMIT" if ok else "\nFix the FAIL items before submitting.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
