"""Evaluation metrics for masked MAE reporting."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


REPORT_HORIZONS = (3, 6, 12)


def fill_flow_nan0(tensor: torch.Tensor, flow_channel: int = 0) -> torch.Tensor:
    out = tensor.clone()
    out[..., flow_channel] = torch.nan_to_num(out[..., flow_channel], nan=0.0)
    return out


def denorm(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return x * std + mean


@torch.no_grad()
def evaluate_mae(
    model: torch.nn.Module,
    loader,
    mean: float,
    std: float,
    device: torch.device,
    horizons: tuple[int, ...] = REPORT_HORIZONS,
) -> dict[str, float]:
    model.eval()
    tot_ae = 0.0
    tot_n = 0.0
    h_ae = {h: 0.0 for h in horizons}
    h_n = {h: 0.0 for h in horizons}
    for batch in loader:
        inp = fill_flow_nan0(batch["inputs"].to(device), 0)
        tgt = fill_flow_nan0(batch["target"].to(device), 0)
        mask = (batch["obs_mask"].to(device) > 0.5).float()
        pred = model(inp)
        pred_r = denorm(pred, mean, std)
        tgt_r = denorm(tgt[..., 0], mean, std)
        ae = (pred_r - tgt_r).abs()
        tot_ae += float((ae * mask).sum().item())
        tot_n += float(mask.sum().item())
        for h in horizons:
            idx = h - 1
            if idx >= ae.shape[1]:
                continue
            m = mask[:, idx]
            h_ae[h] += float((ae[:, idx] * m).sum().item())
            h_n[h] += float(m.sum().item())
    out = {
        "MAE_avg": tot_ae / max(tot_n, 1.0),
        "MAE_h3": h_ae.get(3, float("nan")) / max(h_n.get(3, 0.0), 1.0),
        "MAE_h6": h_ae.get(6, float("nan")) / max(h_n.get(6, 0.0), 1.0),
        "MAE_h12": h_ae.get(12, float("nan")) / max(h_n.get(12, 0.0), 1.0),
    }
    hs = [out[f"MAE_h{h}"] for h in horizons if f"MAE_h{h}" in out]
    out["MAE_protocol_avg"] = float(np.mean(hs)) if hs else out["MAE_avg"]
    return out


def masked_mae_train(
    pred: torch.Tensor,
    target: torch.Tensor,
    null_val: float = 0.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask is None:
        mask = (target != null_val).float()
    else:
        mask = (valid_mask > 0.5).float()
    mask = mask * (~torch.isnan(target)).float() * (~torch.isnan(pred)).float()
    loss = torch.abs(pred - torch.nan_to_num(target, nan=0.0)) * mask
    return loss.sum() / mask.sum().clamp(min=1.0)
