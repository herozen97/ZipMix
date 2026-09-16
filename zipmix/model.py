"""ZipMix spatio-temporal forecasting model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from zipmix.axis_mixer import build_axis_mixer_stack, build_axis_permutation
from zipmix.graph_message import GraphMessagePassing
from zipmix.horizon_cascade import HorizonCascadeHead
from zipmix.temporal_bottleneck import TemporalBottleneck
from zipmix.temporal_encoder import GatedTemporalEncoder


class ZipMixModel(nn.Module):
    """ZipMix for multivariate spatio-temporal forecasting."""

    def __init__(
        self,
        *,
        hist_len: int = 12,
        pred_len: int = 12,
        n_nodes: int = 677,
        in_channels: int = 3,
        hidden_dim: int = 192,
        depth: int = 4,
        dropout: float = 0.1,
        zip_len: int = 6,
        scales: tuple[int, ...] = (3, 9, 27),
        axis_mixer_mode: str = "random",
        node_meta_csv: str | Path | None = None,
        seed: int = 42,
        use_temporal_bottleneck: bool = True,
        use_axis_mixer: bool = True,
        use_graph_message: bool = True,
        use_horizon_cascade: bool = True,
        graph_k: int = 8,
        graph_hops: int = 2,
    ) -> None:
        super().__init__()
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.n_nodes = int(n_nodes)
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.use_temporal_bottleneck = bool(use_temporal_bottleneck)
        self.use_axis_mixer = bool(use_axis_mixer)
        self.use_graph_message = bool(use_graph_message)
        self.use_horizon_cascade = bool(use_horizon_cascade)

        if not node_meta_csv:
            raise ValueError(
                "node_meta_csv is required (train.resolve_meta_csv should inject "
                "{subset}_meta.csv; refusing silent sd_meta default)"
            )
        meta = Path(node_meta_csv)
        order, inv = build_axis_permutation(meta, self.n_nodes, mode=axis_mixer_mode, seed=seed)
        self.register_buffer("axis_order", order, persistent=True)
        self.register_buffer("axis_inv", inv, persistent=True)

        self.encoder = GatedTemporalEncoder(in_channels, hidden_dim, depth, dropout)
        self.temporal_bottleneck = (
            TemporalBottleneck(hidden_dim, hist_len, zip_len, dropout=dropout)
            if self.use_temporal_bottleneck
            else None
        )
        self.graph_message = (
            GraphMessagePassing(
                hidden_dim,
                self.n_nodes,
                meta,
                knn_k=graph_k,
                hops=graph_hops,
                dropout=dropout,
                enabled=True,
            )
            if self.use_graph_message
            else None
        )
        if self.use_axis_mixer:
            self.axis_mixers, self.axis_fuse = build_axis_mixer_stack(hidden_dim, scales, dropout)
        else:
            self.axis_mixers = None
            self.axis_fuse = None
        self.horizon_cascade = HorizonCascadeHead(
            hist_len,
            pred_len,
            hidden_dim,
            dropout=dropout,
            enabled=self.use_horizon_cascade,
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ZipMixModel:
        task = cfg.get("task") or {}
        data = cfg.get("data") or {}
        model = cfg.get("model") or {}
        scales = model.get("scales") or (3, 9, 27)
        return cls(
            hist_len=int(task.get("hist_len", 12)),
            pred_len=int(task.get("pred_len", 12)),
            n_nodes=int(data.get("n_nodes", 677)),
            in_channels=int(model.get("in_channels", 3)),
            hidden_dim=int(model.get("hidden_dim", 192)),
            depth=int(model.get("depth", 4)),
            dropout=float(model.get("dropout", 0.1)),
            zip_len=int(model.get("zip_len", 6)),
            scales=tuple(int(s) for s in scales),
            axis_mixer_mode=str(model.get("axis_mixer_mode", "random")),
            node_meta_csv=model.get("node_meta_csv") or data.get("node_meta_csv"),
            seed=int(cfg.get("seed", 42)),
            use_temporal_bottleneck=bool(model.get("use_temporal_bottleneck", True)),
            use_axis_mixer=bool(model.get("use_axis_mixer", True)),
            use_graph_message=bool(model.get("use_graph_message", True)),
            use_horizon_cascade=bool(model.get("use_horizon_cascade", True)),
            graph_k=int(model.get("graph_k", 8)),
            graph_hops=int(model.get("graph_hops", 2)),
        )

    def _gather_axis(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, self.axis_order, :]

    def _scatter_axis(self, h: torch.Tensor) -> torch.Tensor:
        return h[:, :, self.axis_inv, :]

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, c = x.shape
        if c < self.in_channels:
            pad = torch.zeros(b, t, n, self.in_channels - c, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        elif c > self.in_channels:
            x = x[..., : self.in_channels]
        xt = x.permute(0, 2, 3, 1).reshape(b * n, self.in_channels, t)
        h = self.encoder(xt)
        if self.temporal_bottleneck is not None:
            h = self.temporal_bottleneck(h)
        return h.reshape(b, n, self.hidden_dim, t).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, _ = x.shape
        h_nat = self._encode(x)
        if self.graph_message is not None:
            h_nat = self.graph_message(h_nat)

        h_ord = self._gather_axis(h_nat)
        bt = b * t
        h_flat = h_ord.reshape(bt, h_ord.shape[2], self.hidden_dim)

        if self.axis_mixers is not None:
            feats = [mix(h_flat) for mix in self.axis_mixers]
            fused = self.axis_fuse(torch.cat(feats, dim=-1)) if len(feats) > 1 else feats[0]
        else:
            fused = h_flat

        h_ord = fused.reshape(b, t, -1, self.hidden_dim)
        h_nat = self._scatter_axis(h_ord)
        flat = h_nat.permute(0, 2, 1, 3).reshape(b * n, t * self.hidden_dim)
        return self.horizon_cascade(flat, b, n)
