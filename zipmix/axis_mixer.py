"""Multi-scale 1D mixing along the reordered node axis."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from zipmix.geo import build_axis_order, invert_perm


class AxisMixer(nn.Module):
    """Non-attention multiscale conv along the node axis."""

    def __init__(self, hidden: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        pad = self.kernel_size - 1
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=self.kernel_size, padding=pad)
        self.proj = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (BT, N, H)
        x = h.transpose(1, 2)
        y = self.conv(x)[..., : h.shape[1]]
        y = y.transpose(1, 2)
        return h + self.proj(y)


def build_axis_permutation(
    meta_csv: str | Path,
    n_nodes: int,
    mode: str = "random",
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    order = build_axis_order(meta_csv, n_nodes, mode=mode, seed=seed)
    inv = invert_perm(order)
    return torch.from_numpy(order).long(), torch.from_numpy(inv).long()


def build_axis_mixer_stack(
    hidden: int,
    scales: tuple[int, ...],
    dropout: float,
) -> tuple[nn.ModuleList, nn.Sequential]:
    mixes = nn.ModuleList([AxisMixer(hidden, k, dropout) for k in scales])
    fuse = nn.Sequential(
        nn.Linear(hidden * len(scales), hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
    )
    return mixes, fuse
