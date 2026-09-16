"""Anti-leakage checks for exported scenario packs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmarks.largest.prep.mask import SplitMasks, crosses_gap
from benchmarks.largest.prep.types import ScenarioSeries, WindowIndices


@dataclass
class ValidationResult:
    passed: bool
    checks: dict[str, dict]


def validate_export(
    series: ScenarioSeries,
    splits: SplitMasks,
    windows: WindowIndices,
    *,
    lookback: int = 12,
    horizon: int = 12,
) -> ValidationResult:
    checks: dict[str, dict] = {}
    ok = True
    ts = series.timestamps

    if windows.test.size and (windows.train.size or windows.val.size):
        train_val_max = max(
            int(windows.train.max()) if windows.train.size else -1,
            int(windows.val.max()) if windows.val.size else -1,
        )
        test_min = int(windows.test.min())
        c_ok = ts[test_min] > ts[train_val_max]
        checks["test_after_train_val"] = {
            "status": "PASS" if c_ok else "FAIL",
            "test_min_ts": str(ts[test_min]),
            "train_val_max_ts": str(ts[train_val_max]),
        }
        ok = ok and c_ok

    if windows.val.size:
        c_ok = all(splits.val[i] for i in windows.val)
        checks["val_subset_of_early"] = {"status": "PASS" if c_ok else "FAIL"}
        ok = ok and c_ok

    if windows.test.size and windows.val.size:
        overlap = np.intersect1d(windows.test, windows.val)
        c_ok = overlap.size == 0
        checks["disjoint_val_test"] = {
            "status": "PASS" if c_ok else "FAIL",
            "n_overlap": int(overlap.size),
        }
        ok = ok and c_ok

    gap_fail = 0
    for idx_arr in (windows.train, windows.val, windows.test):
        for t in idx_arr:
            lb_start = int(t) - lookback + 1
            fc_end = int(t) + horizon
            if crosses_gap(lb_start, fc_end, series.segment_starts, series.gap_after_segment):
                gap_fail += 1
    c_ok = gap_fail == 0
    checks["no_gap_windows"] = {"status": "PASS" if c_ok else "FAIL", "n_fail": gap_fail}
    ok = ok and c_ok

    c_ok = all(v > 0 for v in windows.stats.values())
    checks["nonempty_splits"] = {"status": "PASS" if c_ok else "FAIL", **windows.stats}
    ok = ok and c_ok

    checks["overall"] = {"status": "PASS" if ok else "FAIL"}
    return ValidationResult(passed=ok, checks=checks)
