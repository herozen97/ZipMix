"""Gated temporal convolution backbone."""

from __future__ import annotations

import torch
from torch import nn


class GatedTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = dilation
        self.conv = nn.Conv1d(channels, channels * 2, kernel_size=3, padding=pad, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        a, b = y.chunk(2, dim=1)
        y = torch.tanh(a) * torch.sigmoid(b)
        y = self.dropout(y)
        return self.norm(x + y)


class GatedTemporalEncoder(nn.Module):
    """Pointwise input projection + dilated gated TCN stack."""

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        depth: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.in_conv = nn.Conv1d(in_channels, hidden, kernel_size=1)
        dilations = [2 ** (i % 3) for i in range(int(depth))]
        self.blocks = nn.ModuleList([GatedTCNBlock(hidden, d, dropout) for d in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BN, C, T)
        h = self.in_conv(x)
        for blk in self.blocks:
            h = blk(h)
        return h
