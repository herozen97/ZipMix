"""I/O helpers: HDF read, resample, view cache npz."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

MISSING_CODE = 999

SUBSET_SCOPES = {
    "sd": {"districts": [11], "cache_scope": "ca_full"},
    "gba": {"districts": [4], "cache_scope": "d4_full"},
    "gla": {"districts": [7, 8, 12], "cache_scope": "ca_full"},
}


def load_meta(meta_csv: Path) -> pd.DataFrame:
    return pd.read_csv(meta_csv)


def subset_node_columns(meta: pd.DataFrame, districts: list[int]) -> tuple[np.ndarray, np.ndarray]:
    sub = meta.loc[meta["District"].isin(districts)].sort_values("ID2")
    return sub["ID"].values.astype(np.int64), sub["ID2"].values.astype(np.int64)


def ca_full_columns(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sub = meta.sort_values("ID2")
    return sub["ID"].values.astype(np.int64), sub["ID2"].values.astype(np.int64)


def decode_axis1(axis1: np.ndarray) -> pd.DatetimeIndex:
    if np.issubdtype(axis1.dtype, np.integer):
        return pd.to_datetime(axis1, unit="ns")
    return pd.to_datetime(axis1)


def apply_missing_code(arr: np.ndarray, missing_code: float = MISSING_CODE) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    out[out == missing_code] = np.nan
    return out


def read_raw_year_h5(
    h5_path: Path,
    col_indices: np.ndarray,
    *,
    chunk_size: int = 2000,
) -> tuple[pd.DataFrame, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        values = f["t/block0_values"]
        axis0 = f["t/axis0"][:]
        axis1 = f["t/axis1"][:]
        n_rows = values.shape[0]
        timestamps = decode_axis1(axis1)
        all_sensor_ids = np.array(
            [int(x.decode() if isinstance(x, (bytes, bytearray)) else x) for x in axis0],
            dtype=np.int64,
        )
        col_indices = np.asarray(col_indices, dtype=np.int64)
        node_ids = all_sensor_ids[col_indices]
        chunks: list[np.ndarray] = []
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            chunk = np.array(values[start:end, col_indices], dtype=np.float32)
            chunks.append(apply_missing_code(chunk))
        data = np.vstack(chunks)
    return pd.DataFrame(data, index=timestamps, columns=node_ids), node_ids


def resample_15min_v1(df_5min: pd.DataFrame) -> pd.DataFrame:
    values = df_5min.to_numpy(dtype=np.float32, copy=False)
    n_rows, n_cols = values.shape
    n_buckets = n_rows // 3
    if n_buckets == 0:
        return df_5min.iloc[0:0].copy()
    trimmed = values[: n_buckets * 3]
    buckets = trimmed.reshape(n_buckets, 3, n_cols)
    any_nan = np.isnan(buckets).any(axis=1)
    out = buckets.mean(axis=1)
    out[any_nan] = np.nan
    index = df_5min.index[: n_buckets * 3 : 3]
    return pd.DataFrame(out, index=index, columns=df_5min.columns)


def write_view_npz(path: Path, data: np.ndarray, timestamps: pd.DatetimeIndex, node_ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = ~np.isnan(data)
    np.savez_compressed(
        path,
        data=data.astype(np.float32),
        mask=mask,
        timestamps=timestamps.values.astype("datetime64[ns]").astype(np.int64),
        node_ids=node_ids.astype(np.int64),
        meta=json.dumps({"freq": "15min", "view": "raw_valid"}),
    )


def load_view_npz(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    arr = np.load(path)
    data = arr["data"].astype(np.float32)
    timestamps = pd.to_datetime(arr["timestamps"].astype("datetime64[ns]"))
    node_ids = arr["node_ids"].astype(np.int64)
    mask = arr.get("mask", ~np.isnan(data))
    return data, mask, timestamps, node_ids


def cache_view_path(cache_dir: Path, year: int, scope: str) -> Path:
    return cache_dir / "views" / f"raw_valid_{year}_{scope}.npz"


def resample_year_to_cache(
    *,
    year: int,
    scope: str,
    raw_h5: Path,
    meta_csv: Path,
    cache_dir: Path,
    districts: list[int] | None = None,
) -> Path:
    meta = load_meta(meta_csv)
    if scope == "ca_full":
        node_ids, col_idx = ca_full_columns(meta)
    else:
        assert districts is not None
        node_ids, col_idx = subset_node_columns(meta, districts)

    out = cache_view_path(cache_dir, year, scope)
    if out.exists():
        return out

    if not raw_h5.is_file():
        raise FileNotFoundError(f"Missing raw HDF: {raw_h5}")

    df5, read_ids = read_raw_year_h5(raw_h5, col_idx)
    if not np.array_equal(read_ids, node_ids):
        raise ValueError(f"Node ID mismatch for {year}/{scope}")

    df15 = resample_15min_v1(df5)
    write_view_npz(out, df15.values.astype(np.float32), df15.index, node_ids)
    return out


def load_reliable_ids(assets_dir: Path, subset: str) -> np.ndarray:
    path = assets_dir / f"reliable_nodes_{subset}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing reliable node list: {path}")
    ids = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    return np.asarray(ids, dtype=np.int64)


def load_view_stack(cache_dir: Path, scope: str, years: list[int]) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, list[int]]:
    chunks: list[np.ndarray] = []
    ts_parts: list[pd.DatetimeIndex] = []
    boundaries: list[int] = []
    node_ids: np.ndarray | None = None
    offset = 0
    for year in years:
        path = cache_view_path(cache_dir, year, scope)
        if not path.is_file():
            raise FileNotFoundError(f"Missing view cache: {path} (run resample first)")
        data, _mask, timestamps, nids = load_view_npz(path)
        if node_ids is None:
            node_ids = nids
        elif not np.array_equal(node_ids, nids):
            raise ValueError(f"node_ids mismatch for year {year}")
        boundaries.append(offset)
        chunks.append(data)
        ts_parts.append(timestamps)
        offset += data.shape[0]
    stacked = np.vstack(chunks)
    timestamps_all = pd.DatetimeIndex(np.concatenate([ts.values for ts in ts_parts]))
    return stacked, timestamps_all, node_ids if node_ids is not None else np.array([], dtype=np.int64), boundaries
