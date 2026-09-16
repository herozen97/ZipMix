"""Build LATEST training packs from resampled view caches."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from benchmarks.largest.prep.io import (
    SUBSET_SCOPES,
    cache_view_path,
    load_meta,
    load_reliable_ids,
    load_view_stack,
)
from benchmarks.largest.prep.mask import (
    SplitMasks,
    align_common_mask,
    apply_common_mask,
    build_common_mask,
    build_split_masks,
    crosses_gap,
)
from benchmarks.largest.prep.types import ScenarioSeries, WindowIndices
from benchmarks.largest.prep.validate import validate_export


def load_prep_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _map_reliable_to_cache_cols(reliable_ids: np.ndarray, cache_node_ids: np.ndarray) -> np.ndarray:
    id_to_col = {int(n): i for i, n in enumerate(cache_node_ids)}
    missing = [int(n) for n in reliable_ids if int(n) not in id_to_col]
    if missing:
        raise ValueError(f"reliable nodes missing from cache: {missing[:5]}")
    return np.array([id_to_col[int(n)] for n in reliable_ids], dtype=np.int64)


def load_scenario_series(
    cache_dir: Path,
    scope: str,
    timeline_years: list[int],
    reliable_ids: np.ndarray,
) -> ScenarioSeries:
    full, timestamps, cache_node_ids, boundaries = load_view_stack(cache_dir, scope, timeline_years)
    cache_cols = _map_reliable_to_cache_cols(reliable_ids, cache_node_ids)

    raw_paths = {y: cache_view_path(cache_dir, y, scope) for y in timeline_years}
    common, phase_to_idx = build_common_mask(raw_paths, cache_node_ids)
    cm_ts = align_common_mask(timestamps, common, phase_to_idx, cache_cols)

    raw = full[:, cache_cols].astype(np.float32, copy=False)
    flow = apply_common_mask(raw, cm_ts)
    obs = np.isfinite(flow)

    gap_after: list[bool] = []
    for i, year in enumerate(timeline_years[:-1]):
        gap_after.append(timeline_years[i + 1] - year > 1)

    return ScenarioSeries(
        data=flow,
        obs_mask=obs,
        timestamps=timestamps,
        node_ids=reliable_ids.astype(np.int64),
        segment_starts=boundaries,
        segment_years=list(timeline_years),
        gap_after_segment=gap_after + [False],
    )


def build_window_indices(
    series: ScenarioSeries,
    splits: SplitMasks,
    *,
    lookback: int,
    horizon: int,
    stride: int,
    min_valid_frac: float,
) -> WindowIndices:
    T = len(series.timestamps)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    min_t = lookback - 1
    max_t = T - horizon - 1

    for t in range(min_t, max_t + 1, stride):
        lb_start = t - lookback + 1
        fc_end = t + horizon
        if crosses_gap(lb_start, fc_end, series.segment_starts, series.gap_after_segment):
            continue
        window = series.obs_mask[lb_start : fc_end + 1]
        if float(window.mean()) < min_valid_frac:
            continue
        in_train = splits.train[t] and np.all(splits.train[lb_start : fc_end + 1])
        in_val = splits.val[t] and np.all(splits.val[lb_start : fc_end + 1])
        in_test = splits.test[t] and np.all(splits.test[t + 1 : fc_end + 1])
        if in_train:
            train_idx.append(t)
        elif in_val:
            val_idx.append(t)
        elif in_test:
            if crosses_gap(lb_start, t, series.segment_starts, series.gap_after_segment):
                continue
            test_idx.append(t)

    stats = {"n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx)}
    return WindowIndices(
        train=np.asarray(train_idx, dtype=np.int64),
        val=np.asarray(val_idx, dtype=np.int64),
        test=np.asarray(test_idx, dtype=np.int64),
        stats=stats,
    )


def _build_features(flow: np.ndarray, timestamps: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    T, N = flow.shape
    parts = [np.expand_dims(flow.astype(np.float32), axis=-1)]
    if cfg.get("add_tod", True):
        time_ind = (timestamps.values - timestamps.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        tod = np.tile(time_ind, (N, 1)).T.astype(np.float32)
        parts.append(np.expand_dims(tod, axis=-1))
    if cfg.get("add_dow", True):
        dow = timestamps.dayofweek.values.astype(np.float32) / 7.0
        parts.append(np.expand_dims(np.tile(dow, (N, 1)).T, axis=-1))
    return np.concatenate(parts, axis=-1)


def _normalize_flow(data: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    flow = data[..., 0]
    train_vals = flow[train_mask]
    finite = train_vals[np.isfinite(train_vals)]
    if finite.size == 0:
        raise ValueError("no finite train values for normalization")
    mean = float(finite.mean())
    std = float(finite.std()) or 1.0
    out = data.copy()
    out[..., 0] = (flow - mean) / std
    return out, mean, std


def export_latest_pack(
    out_dir: Path,
    series: ScenarioSeries,
    splits: SplitMasks,
    windows: WindowIndices,
    cfg: dict,
    *,
    subset: str,
    scenario: str,
    run_id: str,
) -> Path:
    data = _build_features(series.data, series.timestamps, cfg)
    data, mean, std = _normalize_flow(data, splits.train_flow_only)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "his.npz",
        data=data,
        mean=np.array(mean),
        std=np.array(std),
        obs_mask=series.obs_mask.astype(np.uint8),
    )
    np.save(out_dir / "idx_train.npy", windows.train)
    np.save(out_dir / "idx_val.npy", windows.val)
    np.save(out_dir / "idx_test.npy", windows.test)
    pd.DataFrame({"node_id": series.node_ids, "col_idx": np.arange(len(series.node_ids))}).to_csv(
        out_dir / "reliable_nodes.csv", index=False
    )
    split_meta = {
        "run_id": run_id,
        "subset": subset,
        "scenario": scenario,
        "lookback": cfg["lookback"],
        "horizon": cfg["horizon"],
        "stride": cfg["stride"],
        "train_frac": cfg["train_frac"],
        "n_nodes": int(len(series.node_ids)),
        "n_timesteps": int(len(series.timestamps)),
        "timeline_years": series.segment_years,
        "segment_starts": series.segment_starts,
        "sample_counts": windows.stats,
        "normalization": {"mean": mean, "std": std, "source": "train_early_80pct"},
        "timestamps": {
            "train_end": str(series.timestamps[splits.train_end_idx]),
            "val_start": str(series.timestamps[splits.val_start_idx]),
            "test_start": str(series.timestamps[splits.test_start_idx]),
            "test_end": str(series.timestamps[splits.test_end_idx]),
        },
    }
    (out_dir / "split_meta.json").write_text(json.dumps(split_meta, indent=2) + "\n", encoding="utf-8")
    return out_dir


def update_latest_symlink(scenario_parent: Path, run_dir: Path) -> None:
    latest = scenario_parent / "LATEST"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name, target_is_directory=True)


def write_subset_adj(out_root: Path, subset: str, meta_csv: Path, adj_npy: Path, reliable_ids: np.ndarray) -> Path:
    meta = load_meta(meta_csv)
    id_to_id2 = dict(zip(meta["ID"].astype(np.int64), meta["ID2"].astype(np.intp)))
    id2 = np.array([id_to_id2[int(n)] for n in reliable_ids], dtype=np.intp)
    adj = np.load(adj_npy)
    cropped = adj[id2][:, id2].astype(np.float32)
    out = out_root / subset / f"{subset}_rn_adj.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, cropped)
    return out


def prepare_scenario(
    *,
    cache_dir: Path,
    out_root: Path,
    assets_dir: Path,
    subset: str,
    scenario: str,
    cfg: dict,
) -> dict[str, Any]:
    spec = SUBSET_SCOPES[subset]
    split_spec = cfg["splits"][scenario]
    reliable_ids = load_reliable_ids(assets_dir, subset)
    series = load_scenario_series(cache_dir, spec["cache_scope"], split_spec["timeline_years"], reliable_ids)
    splits = build_split_masks(
        series.timestamps,
        split_spec["early"],
        split_spec["eval"],
        train_frac=float(cfg["train_frac"]),
        exclude_years=tuple(cfg.get("exclude_years", [2017])),
    )
    windows = build_window_indices(
        series,
        splits,
        lookback=int(cfg["lookback"]),
        horizon=int(cfg["horizon"]),
        stride=int(cfg["stride"]),
        min_valid_frac=float(cfg.get("min_valid_frac_in_window", 0.8)),
    )
    validation = validate_export(series, splits, windows, lookback=int(cfg["lookback"]), horizon=int(cfg["horizon"]))
    if not validation.passed:
        raise RuntimeError(f"validation failed for {subset}/{scenario}: {validation.checks}")
    if any(v <= 0 for v in windows.stats.values()):
        raise RuntimeError(f"empty split for {subset}/{scenario}: {windows.stats}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"prep_{subset}_{scenario}_{ts}"
    scenario_parent = out_root / subset / scenario
    run_dir = scenario_parent / run_id
    export_latest_pack(run_dir, series, splits, windows, cfg, subset=subset, scenario=scenario, run_id=run_id)
    update_latest_symlink(scenario_parent, run_dir)
    return {"run_id": run_id, "run_dir": str(run_dir), "sample_counts": windows.stats}


def required_years(cfg: dict, scenarios: list[str]) -> set[int]:
    years: set[int] = set()
    for sc in scenarios:
        years.update(cfg["splits"][sc]["timeline_years"])
    return years


def required_scopes(subsets: list[str]) -> dict[str, list[str]]:
    """Return cache scopes needed and which subsets use them."""
    scope_to_subsets: dict[str, list[str]] = {}
    for subset in subsets:
        scope = SUBSET_SCOPES[subset]["cache_scope"]
        scope_to_subsets.setdefault(scope, []).append(subset)
    return scope_to_subsets
