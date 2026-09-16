"""Temporal bottleneck: compress history along time then expand back."""

from __future__ import annotations

import torch
from torch import nn


class TemporalBottleneck(nn.Module):
    def __init__(self, hidden: int, hist_len: int, zip_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hist_len = int(hist_len)
        self.zip_len = int(zip_len)
        self.down = nn.Linear(self.hist_len, self.zip_len)
        self.up = nn.Linear(self.zip_len, self.hist_len)
        self.refine = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (BN, C, T)
        z = self.down(h)
        z = self.refine(z)
        return self.up(z)
