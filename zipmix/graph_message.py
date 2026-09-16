"""Short-hop graph message passing on the natural node domain."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from zipmix.geo import build_knn_edge_index, load_lat_lng


class GraphMessagePassing(nn.Module):
    def __init__(
        self,
        hidden: int,
        n_nodes: int,
        meta_csv: str | Path,
        knn_k: int = 8,
        hops: int = 2,
        dropout: float = 0.1,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.hops = max(int(hops), 1)
        coords = load_lat_lng(meta_csv, int(n_nodes))
        edges = build_knn_edge_index(coords, k=knn_k)
        self.register_buffer("knn", torch.from_numpy(edges).long(), persistent=True)
        self.msg = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.out = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, N, H)
        if not self.enabled:
            return h
        x = h
        for _ in range(self.hops):
            nb = x[:, :, self.knn, :]
            agg = nb.mean(dim=3)
            x = x + self.msg(torch.cat([x, agg], dim=-1))
        return h + self.out(x)
