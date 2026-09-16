"""Dataset factory and dataloader helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from zipmix.data.npz_window import NPZWindowDataset

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_data_root(data_cfg: dict[str, Any]) -> Path:
    root = data_cfg.get("root")
    if not root:
        raise ValueError("data.root is required for training (set in config or experiment yaml)")
    return Path(root)


def build_dataset(
    mode: str,
    data_cfg: dict[str, Any],
    task_cfg: dict[str, Any],
    *,
    max_samples: int | None = None,
):
    dtype = str(data_cfg.get("type", "npz_window")).lower()
    hist_len = int(task_cfg.get("hist_len", 12))
    pred_len = int(task_cfg.get("pred_len", 12))

    if dtype == "largest":
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from benchmarks.largest.dataset import LargeSTWindowDataset
        from benchmarks.largest.protocols import DEFAULT_PROTOCOL

        root = _resolve_data_root(data_cfg)
        return LargeSTWindowDataset(
            mode,
            data_root=root,
            subset=str(data_cfg.get("subset", "sd")),
            scenario=str(data_cfg.get("scenario", DEFAULT_PROTOCOL)),
            hist_len=hist_len,
            pred_len=pred_len,
            max_samples=max_samples,
        )

    root = _resolve_data_root(data_cfg)
    split = data_cfg.get("split_dir")
    if split:
        root = Path(root) / str(split)
    return NPZWindowDataset(
        mode,
        root=root,
        hist_len=hist_len,
        pred_len=pred_len,
        max_samples=max_samples,
    )


def build_dataloader(
    mode: str,
    data_cfg: dict[str, Any],
    task_cfg: dict[str, Any],
    batch_size: int,
    *,
    max_samples: int | None = None,
    shuffle: bool | None = None,
) -> DataLoader:
    ds = build_dataset(mode, data_cfg, task_cfg, max_samples=max_samples)
    if shuffle is None:
        shuffle = mode == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
