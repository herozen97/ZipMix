"""Geospatial helpers for node ordering and kNN graphs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _header_fields(header: list[str]) -> dict[str, int]:
    lower = {name.strip().lower(): i for i, name in enumerate(header)}
    if "lat" in lower and "lng" in lower:
        return {"lat": lower["lat"], "lng": lower["lng"], "id": lower.get("id")}
    # Legacy two-column files without a usable header: treat as Lat,Lng
    if len(header) >= 2 and "lat" not in lower:
        return {"lat": 0, "lng": 1, "id": None}
    raise ValueError(
        f"node meta CSV must have Lat,Lng columns (optional ID); got header={header!r}"
    )


def load_node_meta(meta_csv: str | Path, n_nodes: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Load coordinates and optional sensor IDs.

    Returns
    -------
    coords : (N, 2) float64 array of Lat,Lng
    ids : (N,) int64 array when an ``ID`` column is present, else None

    Row count must equal ``n_nodes`` (no silent pad/truncate).
    """
    path = Path(meta_csv)
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty node meta CSV: {path}") from exc
        fields = _header_fields(header)
        coords_rows: list[tuple[float, float]] = []
        id_rows: list[int] = []
        has_id = fields["id"] is not None
        for row in reader:
            if len(row) <= max(fields["lat"], fields["lng"]):
                continue
            coords_rows.append((float(row[fields["lat"]]), float(row[fields["lng"]])))
            if has_id:
                id_rows.append(int(float(row[fields["id"]])))
    coords = np.asarray(coords_rows, dtype=np.float64)
    n = int(n_nodes)
    if coords.shape[0] != n:
        raise ValueError(
            f"node meta row count {coords.shape[0]} != n_nodes={n} for {path}; "
            "refusing silent pad/truncate (would destroy geographic kNN)"
        )
    ids = np.asarray(id_rows, dtype=np.int64) if has_id else None
    if ids is not None and ids.shape[0] != n:
        raise ValueError(f"node meta ID count {ids.shape[0]} != n_nodes={n} for {path}")
    return coords, ids


def load_lat_lng(meta_csv: str | Path, n_nodes: int) -> np.ndarray:
    """Load Lat,Lng rows. Length must equal ``n_nodes`` (no silent pad/truncate)."""
    coords, _ = load_node_meta(meta_csv, n_nodes)
    return coords


def load_node_ids(meta_csv: str | Path, n_nodes: int) -> np.ndarray:
    """Load required ``ID`` column; raise if missing."""
    _, ids = load_node_meta(meta_csv, n_nodes)
    if ids is None:
        raise ValueError(
            f"node meta CSV lacks ID column: {meta_csv}. "
            "Expected header ID,Lat,Lng aligned with the data node axis."
        )
    return ids


def load_id_list(path: str | Path) -> np.ndarray:
    """Load one integer ID per non-empty line (reliable_nodes_*.txt)."""
    ids: list[int] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(int(s.split(",")[0]))
    return np.asarray(ids, dtype=np.int64)


def load_reliable_nodes_csv(path: str | Path) -> np.ndarray:
    """Load ``node_id`` column from a LATEST pack ``reliable_nodes.csv``."""
    ids: list[int] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "node_id" not in reader.fieldnames:
            raise ValueError(f"reliable_nodes.csv missing node_id column: {path}")
        for row in reader:
            ids.append(int(row["node_id"]))
    return np.asarray(ids, dtype=np.int64)


def assert_meta_ids_match(
    meta_csv: str | Path,
    expected_ids: np.ndarray,
    *,
    context: str = "",
) -> None:
    """Fail loudly when meta ID order disagrees with the data node axis."""
    expected = np.asarray(expected_ids, dtype=np.int64).reshape(-1)
    meta_ids = load_node_ids(meta_csv, expected.shape[0])
    if not np.array_equal(meta_ids, expected):
        mism = int(np.sum(meta_ids != expected))
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}node meta ID order does not match expected node axis "
            f"({mism}/{expected.shape[0]} mismatches) for {meta_csv}"
        )


def _hilbert_keys(coords: np.ndarray, bits: int = 10) -> np.ndarray:
    """Morton / Z-order bit interleaving (not a true Hilbert curve)."""
    lat = coords[:, 0]
    lng = coords[:, 1]

    def q(v: np.ndarray) -> np.ndarray:
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            return np.zeros(v.shape[0], dtype=np.int64)
        x = np.clip((v - lo) / (hi - lo), 0.0, 1.0 - 1e-12)
        return (x * (1 << bits)).astype(np.int64)

    xi, yi = q(lat), q(lng)
    keys = np.zeros(coords.shape[0], dtype=np.int64)
    for b in range(bits):
        keys |= ((xi >> b) & 1) << (2 * b)
        keys |= ((yi >> b) & 1) << (2 * b + 1)
    return keys


def build_axis_order(
    meta_csv: str | Path,
    n_nodes: int,
    mode: str = "random",
    seed: int = 42,
) -> np.ndarray:
    """Return permutation mapping axis position -> node id.

    Supported ``mode`` values: ``random`` / ``shuffle``, ``geo``, ``hilbert``.
    Unknown modes raise ``ValueError`` (no silent fallback).
    ``hilbert`` uses Morton/Z-order keys (see ``_hilbert_keys``).
    """
    coords = load_lat_lng(meta_csv, n_nodes)
    mode = str(mode).lower()
    if mode in ("random", "shuffle"):
        rng = np.random.default_rng(seed)
        return rng.permutation(n_nodes).astype(np.int64)
    if mode == "geo":
        keys = coords[:, 0] * 1e6 + coords[:, 1]
        return np.argsort(keys, kind="mergesort").astype(np.int64)
    if mode == "hilbert":
        keys = _hilbert_keys(coords)
        return np.argsort(keys, kind="mergesort").astype(np.int64)
    raise ValueError(
        f"unknown axis_mixer_mode={mode!r}; expected one of "
        "'random', 'shuffle', 'geo', 'hilbert'"
    )


def invert_perm(order: np.ndarray) -> np.ndarray:
    inv = np.empty_like(order)
    inv[order] = np.arange(order.shape[0], dtype=order.dtype)
    return inv


def build_knn_edge_index(coords: np.ndarray, k: int = 8) -> np.ndarray:
    n = coords.shape[0]
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    kk = min(int(k), n - 1)
    return np.argpartition(d2, kth=kk, axis=1)[:, :kk].astype(np.int64)
