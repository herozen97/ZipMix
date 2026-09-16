#!/usr/bin/env python3
"""Build a minimal npz_window pack for ZipMix (demo or custom arrays).

Usage examples:

  # Synthetic demo pack (also writes node_meta.csv with N rows)
  python scripts/build_npz_pack.py --out ./demo_pack --t 500 --n 32

  # From your own .npy arrays (T,N,C) and (T,N) mask
  python scripts/build_npz_pack.py --out ./my_pack \\
      --data-npy /path/to/data.npy --mask-npy /path/to/mask.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _split_anchors(n_anchors: int, train_frac: float = 0.7, val_frac: float = 0.15):
    n_train = int(n_anchors * train_frac)
    n_val = int(n_anchors * val_frac)
    idx = np.arange(n_anchors, dtype=np.int64)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def _write_node_meta(out: Path, n: int, seed: int = 42) -> Path:
    """Write an ID,Lat,Lng CSV with exactly ``n`` rows (synthetic grid for demos)."""
    rng = np.random.default_rng(seed)
    lat = 32.5 + rng.uniform(0.0, 0.5, size=n)
    lng = -117.2 + rng.uniform(0.0, 0.5, size=n)
    path = out / "node_meta.csv"
    with open(path, "w", encoding="utf-8") as f:
        f.write("ID,Lat,Lng\n")
        for i, (a, b) in enumerate(zip(lat, lng)):
            f.write(f"{i},{a:.6f},{b:.6f}\n")
    return path


def build_synthetic(out: Path, t: int, n: int, c: int, hist: int, pred: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(t, n)).astype(np.float32)
    tod = np.linspace(0, 1, t, dtype=np.float32)[:, None].repeat(n, axis=1)
    dow = np.zeros((t, n), dtype=np.float32)
    data = np.stack([raw, tod, dow], axis=-1)
    if c == 1:
        data = data[..., :1]
    elif c == 2:
        data = data[..., :2]
    mask = np.ones((t, n), dtype=np.uint8)

    train_raw = raw[: int(t * 0.7)]
    mean = float(train_raw.mean())
    std = float(train_raw.std()) or 1.0
    data[..., 0] = (data[..., 0] - mean) / std

    min_t = hist - 1
    max_t = t - pred - 1
    anchors = np.arange(min_t, max_t + 1, dtype=np.int64)
    tr, va, te = _split_anchors(len(anchors))

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "his.npz", data=data, obs_mask=mask, mean=mean, std=std)
    np.save(out / "idx_train.npy", anchors[tr])
    np.save(out / "idx_val.npy", anchors[va])
    np.save(out / "idx_test.npy", anchors[te])
    meta_csv = _write_node_meta(out, n, seed=seed)
    meta = {
        "source": "build_npz_pack.py",
        "n_timesteps": t,
        "n_nodes": n,
        "n_channels": c,
        "lookback": hist,
        "horizon": pred,
        "node_meta_csv": meta_csv.name,
        "sample_counts": {"n_train": len(tr), "n_val": len(va), "n_test": len(te)},
    }
    (out / "split_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote pack to {out}  anchors train/val/test = {len(tr)}/{len(va)}/{len(te)}  "
        f"meta={meta_csv.name}"
    )


def build_from_npy(
    out: Path,
    data_path: Path,
    mask_path: Path | None,
    hist: int,
    pred: int,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> None:
    data = np.load(data_path).astype(np.float32)
    if data.ndim != 3:
        raise ValueError("data.npy must have shape (T, N, C)")
    t, n, _ = data.shape
    if mask_path:
        mask = np.load(mask_path).astype(np.uint8)
    else:
        mask = np.ones((t, n), dtype=np.uint8)

    train_end = int(t * train_frac)
    train_flow = data[:train_end, :, 0]
    finite = train_flow[np.isfinite(train_flow)]
    if finite.size == 0:
        raise ValueError("no finite train values for normalization")
    mean = float(finite.mean())
    std = float(finite.std()) or 1.0
    data = data.copy()
    data[..., 0] = (data[..., 0] - mean) / std

    min_t = hist - 1
    max_t = t - pred - 1
    anchors = np.arange(min_t, max_t + 1, dtype=np.int64)
    tr, va, te = _split_anchors(len(anchors), train_frac=train_frac, val_frac=val_frac)

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "his.npz", data=data, obs_mask=mask, mean=mean, std=std)
    np.save(out / "idx_train.npy", anchors[tr])
    np.save(out / "idx_val.npy", anchors[va])
    np.save(out / "idx_test.npy", anchors[te])
    meta_csv = _write_node_meta(out, n, seed=seed)
    print(f"Wrote pack to {out}  meta={meta_csv.name} (replace with real Lat/Lng if needed)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ZipMix npz_window pack")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t", type=int, default=500, help="Synthetic timesteps")
    ap.add_argument("--n", type=int, default=32, help="Synthetic nodes")
    ap.add_argument("--c", type=int, default=3, help="Synthetic channels")
    ap.add_argument("--hist", type=int, default=12)
    ap.add_argument("--pred", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-npy", type=Path, default=None)
    ap.add_argument("--mask-npy", type=Path, default=None)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    if args.data_npy:
        build_from_npy(
            args.out,
            args.data_npy,
            args.mask_npy,
            args.hist,
            args.pred,
            args.train_frac,
            args.val_frac,
            args.seed,
        )
    else:
        build_synthetic(args.out, args.t, args.n, args.c, args.hist, args.pred, args.seed)


if __name__ == "__main__":
    main()
