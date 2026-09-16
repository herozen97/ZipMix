"""Horizon cascade decoder with stop-gradient conditioning."""

from __future__ import annotations

import torch
from torch import nn


class HorizonCascadeHead(nn.Module):
    """Shared trunk -> cascaded horizon segments (default 4+4+4 for pred_len=12)."""

    def __init__(
        self,
        hist_len: int,
        pred_len: int,
        hidden: int,
        dropout: float = 0.1,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if pred_len != 12:
            raise ValueError("HorizonCascadeHead currently supports pred_len=12")
        self.enabled = bool(enabled)
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        in_dim = hist_len * hidden
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden + 4, 4) for _ in range(3)])
        self.fallback = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, pred_len),
        )

    def forward(self, flat: torch.Tensor, batch_size: int, n_nodes: int) -> torch.Tensor:
        if not self.enabled:
            y = self.fallback(flat).reshape(batch_size, n_nodes, self.pred_len).permute(0, 2, 1)
            return y
        h = self.trunk(flat)
        chunks: list[torch.Tensor] = []
        cond: torch.Tensor | None = None
        for head in self.heads:
            if cond is None:
                pad = torch.zeros(h.shape[0], 4, device=h.device, dtype=h.dtype)
                inp = torch.cat([h, pad], dim=-1)
            else:
                inp = torch.cat([h, cond.detach()], dim=-1)
            out = head(inp)
            chunks.append(out)
            cond = out
        y = torch.cat(chunks, dim=-1)
        return y.reshape(batch_size, n_nodes, self.pred_len).permute(0, 2, 1)
