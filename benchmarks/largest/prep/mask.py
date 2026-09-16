"""Common-mask alignment and temporal splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _phase_key(ts: pd.Timestamp) -> tuple[int, int, int, int]:
    return (ts.month, ts.day, ts.hour, ts.minute)


def build_common_mask(raw_paths: dict[int, Path], node_ids: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int, int, int], int]]:
    from benchmarks.largest.prep.io import load_view_npz

    years = sorted(raw_paths.keys())
    phase_to_idx_by_year: dict[int, dict[tuple[int, int, int, int], int]] = {}
    masks_by_year: dict[int, np.ndarray] = {}

    for year in years:
        _data, mask, timestamps, _nids = load_view_npz(raw_paths[year])
        mapping: dict[tuple[int, int, int, int], int] = {}
        for i, ts in enumerate(timestamps):
            mapping[_phase_key(ts)] = i
        phase_to_idx_by_year[year] = mapping
        masks_by_year[year] = mask

    common_phases = set(phase_to_idx_by_year[years[0]].keys())
    for year in years[1:]:
        common_phases &= set(phase_to_idx_by_year[year].keys())
    common_phases_sorted = sorted(common_phases)

    n_phases = len(common_phases_sorted)
    n_nodes = masks_by_year[years[0]].shape[1]
    common = np.ones((n_phases, n_nodes), dtype=bool)
    for pi, phase in enumerate(common_phases_sorted):
        for year in years:
            row = phase_to_idx_by_year[year][phase]
            common[pi] &= masks_by_year[year][row]

    phase_to_idx = {phase: pi for pi, phase in enumerate(common_phases_sorted)}
    return common, phase_to_idx


def align_common_mask(
    timestamps: pd.DatetimeIndex,
    common_mask: np.ndarray,
    phase_to_idx: dict[tuple[int, int, int, int], int],
    col_indices: np.ndarray,
) -> np.ndarray:
    cm = common_mask[:, col_indices]
    aligned = np.zeros((len(timestamps), cm.shape[1]), dtype=bool)
    for t, ts in enumerate(timestamps):
        idx = phase_to_idx.get(_phase_key(ts))
        if idx is not None:
            aligned[t] = cm[idx]
    return aligned


def apply_common_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = data.astype(np.float32, copy=True)
    out[~mask] = np.nan
    return out


@dataclass
class SplitMasks:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_flow_only: np.ndarray
    train_end_idx: int
    val_start_idx: int
    test_start_idx: int
    test_end_idx: int


def build_period_mask(
    timestamps: pd.DatetimeIndex,
    start: str,
    end: str,
    *,
    exclude_years: tuple[int, ...] = (2017,),
) -> np.ndarray:
    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=45)
    mask = (timestamps >= t0) & (timestamps <= t1)
    for y in exclude_years:
        mask &= timestamps.year != y
    return np.asarray(mask, dtype=bool)


def build_split_masks(
    timestamps: pd.DatetimeIndex,
    early: list[str],
    eval_period: list[str],
    *,
    train_frac: float = 0.8,
    exclude_years: tuple[int, ...] = (2017,),
) -> SplitMasks:
    early_mask = build_period_mask(timestamps, early[0], early[1], exclude_years=exclude_years)
    test_mask = build_period_mask(timestamps, eval_period[0], eval_period[1], exclude_years=exclude_years)
    early_idx = np.where(early_mask)[0]
    if len(early_idx) == 0:
        raise ValueError("empty early period")
    n_train = max(int(len(early_idx) * train_frac), 1)
    if len(early_idx) - n_train < 1:
        n_train = len(early_idx) - 1
    train_idx_set = set(early_idx[:n_train].tolist())
    val_idx_set = set(early_idx[n_train:].tolist())
    train = np.zeros(len(timestamps), dtype=bool)
    val = np.zeros(len(timestamps), dtype=bool)
    for i in train_idx_set:
        train[i] = True
    for i in val_idx_set:
        val[i] = True
    return SplitMasks(
        train=train,
        val=val,
        test=test_mask,
        train_flow_only=train.copy(),
        train_end_idx=int(early_idx[n_train - 1]),
        val_start_idx=int(early_idx[n_train]),
        test_start_idx=int(np.where(test_mask)[0][0]) if test_mask.any() else -1,
        test_end_idx=int(np.where(test_mask)[0][-1]) if test_mask.any() else -1,
    )


def crosses_gap(
    lb_start: int,
    fc_end: int,
    segment_starts: list[int],
    gap_after_segment: list[bool],
) -> bool:
    for seg_i, has_gap in enumerate(gap_after_segment):
        if not has_gap:
            continue
        if seg_i + 1 >= len(segment_starts):
            continue
        boundary = segment_starts[seg_i + 1]
        if lb_start < boundary <= fc_end:
            return True
    return False
