"""Wall-clock budget, CPU detection and the watchdog that guarantees we finish."""
from __future__ import annotations

import os
import threading
import time

# Process start reference. main.py sets AUTOML_T0 before the heavy imports so the
# budget also covers interpreter start-up and library import time.
_T0 = float(os.environ.get("AUTOML_T0", time.time()))

HARD_LIMIT = 60.0          # organisers' per-dataset limit
DEFAULT_BUDGET = 40.0      # our own deadline for the whole run (seconds since start)
WATCHDOG_AT = 52.0         # emergency exit, fallback outputs are already on disk


def process_start() -> float:
    return _T0


class Budget:
    """Tracks remaining time against a deadline measured from process start."""

    def __init__(self, total: float | None = None, start: float | None = None):
        env = os.environ.get("AUTOML_TIME_BUDGET")
        self.total = float(total if total is not None else (env or DEFAULT_BUDGET))
        self.start = float(start if start is not None else _T0)

    def elapsed(self) -> float:
        return time.time() - self.start

    def remaining(self) -> float:
        return self.total - self.elapsed()

    def fraction_used(self) -> float:
        return self.elapsed() / max(self.total, 1e-9)

    def can_afford(self, seconds: float, reserve: float = 0.0) -> bool:
        return self.remaining() - reserve > seconds


def effective_cpu_count() -> int:
    """CPUs we may really use: min of cpu_count, affinity mask and cgroup quota."""
    counts = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        try:
            counts.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass
    try:  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()[:2]
            if quota != "max":
                counts.append(max(1, int(float(quota) / float(period))))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fq, \
                open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            quota, period = int(fq.read()), int(fp.read())
            if quota > 0 and period > 0:
                counts.append(max(1, quota // period))
    except (OSError, ValueError):
        pass
    return max(1, min(counts))


def n_jobs() -> int:
    env = os.environ.get("AUTOML_N_JOBS")
    if env:
        return max(1, int(env))
    return max(1, min(effective_cpu_count(), 8))


class Watchdog:
    """Calls ``on_timeout`` and hard-exits the process if the deadline passes.

    The pipeline writes valid fallback outputs early, so an emergency exit still
    leaves a complete (if weaker) submission on disk instead of a timeout.
    """

    def __init__(self, at_seconds: float = WATCHDOG_AT, on_timeout=None):
        delay = max(1.0, at_seconds - (time.time() - _T0))
        self._on_timeout = on_timeout
        self._timer = threading.Timer(delay, self._fire)
        self._timer.daemon = True

    def _fire(self):
        try:
            if self._on_timeout is not None:
                self._on_timeout()
        finally:
            os._exit(0)

    def start(self):
        self._timer.start()
        return self

    def cancel(self):
        self._timer.cancel()
