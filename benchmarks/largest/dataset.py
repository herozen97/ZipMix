"""LargeST benchmark adapter (optional; see README in this directory)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from benchmarks.largest.protocols import DEFAULT_PROTOCOL


class LargeSTWindowDataset(Dataset):
    def __init__(
        self,
        mode: str,
        *,
        data_root: str | Path,
        subset: str = "sd",
        scenario: str = DEFAULT_PROTOCOL,
        hist_len: int = 12,
        pred_len: int = 12,
        max_samples: int | None = None,
    ) -> None:
        assert mode in ("train", "valid", "test")
        self.mode = mode
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.data_dir = Path(data_root) / "data" / subset / scenario / "LATEST"
        pack = np.load(self.data_dir / "his.npz")
        self._full = pack["data"].astype(np.float32)
        self._mask = pack["obs_mask"].astype(np.uint8)
        self.mean = float(pack["mean"])
        self.std = float(pack["std"])
        key = {"train": "train", "valid": "val", "test": "test"}[mode]
        anchors = np.load(self.data_dir / f"idx_{key}.npy").astype(np.int64)
        if max_samples is not None and max_samples > 0:
            anchors = anchors[: int(max_samples)]
        self.anchors = anchors
        self.subset = subset

    def __len__(self) -> int:
        return int(self.anchors.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        t = int(self.anchors[index])
        history = torch.from_numpy(self._full[t - self.hist_len + 1 : t + 1].copy())
        future = torch.from_numpy(self._full[t + 1 : t + self.pred_len + 1].copy())
        obs = torch.from_numpy(
            self._mask[t + 1 : t + self.pred_len + 1].astype(np.float32, copy=True)
        )
        return {
            "inputs": history,
            "target": future,
            "obs_mask": obs,
        }

    @property
    def num_nodes(self) -> int:
        return int(self._full.shape[1])
