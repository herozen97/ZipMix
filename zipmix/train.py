"""Training and evaluation entry point."""

from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zipmix.data.base import build_dataloader
from zipmix.geo import (
    assert_meta_ids_match,
    load_id_list,
    load_node_meta,
    load_reliable_nodes_csv,
)
from zipmix.metrics import evaluate_mae, fill_flow_nan0, masked_mae_train
from zipmix.model import ZipMixModel


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], val)
        else:
            out[key] = val
    return out


def resolve_meta_csv(data_cfg: dict, model_cfg: dict) -> str:
    """Resolve node meta CSV for graph/axis geo.

    Explicit ``node_meta_csv`` wins; otherwise require ``{subset}_meta.csv``.
    Never silently fall back to another subset's meta.
    """
    if model_cfg.get("node_meta_csv"):
        return str(model_cfg["node_meta_csv"])
    if data_cfg.get("node_meta_csv"):
        return str(data_cfg["node_meta_csv"])
    subset = str(data_cfg.get("subset", "sd"))
    candidate = REPO_ROOT / "benchmarks" / "largest" / "assets" / f"{subset}_meta.csv"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"missing node meta for subset={subset!r}: {candidate}. "
            "Ship or point data/model.node_meta_csv at an ID,Lat,Lng CSV with exactly n_nodes rows."
        )
    return str(candidate)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Prefer reproducibility for paper-style runs; may reduce throughput.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def verify_meta_alignment(meta_csv: str, data_cfg: dict, n_nodes: int) -> None:
    """Assert meta ID order matches shipped reliable lists and/or LATEST pack."""
    _, ids = load_node_meta(meta_csv, n_nodes)
    if ids is None:
        raise ValueError(
            f"node meta must include an ID column aligned with the data node axis: {meta_csv}"
        )

    data_type = str(data_cfg.get("type", "")).lower()
    subset = str(data_cfg.get("subset", "sd"))
    shipped = (
        REPO_ROOT / "benchmarks" / "largest" / "prep" / "assets" / f"reliable_nodes_{subset}.txt"
    )
    if data_type == "largest" and shipped.is_file():
        assert_meta_ids_match(meta_csv, load_id_list(shipped), context=f"subset={subset}")

    root = data_cfg.get("root")
    scenario = data_cfg.get("scenario")
    if data_type == "largest" and root and scenario:
        pack_csv = Path(root) / "data" / subset / str(scenario) / "LATEST" / "reliable_nodes.csv"
        if pack_csv.is_file():
            assert_meta_ids_match(
                meta_csv,
                load_reliable_nodes_csv(pack_csv),
                context=f"pack={pack_csv}",
            )


def smoke_forward(cfg: dict, device: torch.device) -> dict:
    data_cfg = dict(cfg.get("data") or {})
    model_cfg = dict(cfg.get("model") or {})
    data_cfg.setdefault("n_nodes", 677)
    meta = resolve_meta_csv(data_cfg, model_cfg)
    model_cfg["node_meta_csv"] = meta
    cfg = merge_config(cfg, {"data": data_cfg, "model": model_cfg})
    verify_meta_alignment(meta, data_cfg, int(cfg["data"]["n_nodes"]))
    model = ZipMixModel.from_config(cfg).to(device)
    n = int(cfg["data"]["n_nodes"])
    hist = int((cfg.get("task") or {}).get("hist_len", 12))
    x = torch.randn(2, hist, n, int(model_cfg.get("in_channels", 3)), device=device)
    y = model(x)
    pred_len = int((cfg.get("task") or {}).get("pred_len", 12))
    assert y.shape == (2, pred_len, n), y.shape
    return {
        "ok": True,
        "output_shape": list(y.shape),
        "params": int(sum(p.numel() for p in model.parameters())),
    }


def train(cfg: dict, args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    task_cfg = cfg.get("task") or {}
    data_cfg = dict(cfg.get("data") or {})
    train_cfg = cfg.get("train") or {}
    model_cfg = dict(cfg.get("model") or {})

    meta = resolve_meta_csv(data_cfg, model_cfg)
    model_cfg["node_meta_csv"] = meta
    cfg = merge_config(cfg, {"model": model_cfg})

    max_train = args.max_train_samples
    max_eval = args.max_eval_samples
    if args.smoke_train:
        max_train = max_train or 256
        max_eval = max_eval or 128
        train_cfg = merge_config(train_cfg, {"epochs": 1})

    batch_size = int(train_cfg.get("batch_size", 32))
    train_loader = build_dataloader(
        "train", data_cfg, task_cfg, batch_size, max_samples=max_train
    )
    val_loader = build_dataloader(
        "valid", data_cfg, task_cfg, batch_size, max_samples=max_eval
    )
    test_loader = build_dataloader(
        "test", data_cfg, task_cfg, batch_size, max_samples=max_eval
    )

    data_cfg["n_nodes"] = train_loader.dataset.num_nodes
    cfg = merge_config(cfg, {"data": data_cfg})
    verify_meta_alignment(meta, data_cfg, int(data_cfg["n_nodes"]))
    model = ZipMixModel.from_config(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(train_cfg.get("epochs", 50))
    best_val = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            inp = fill_flow_nan0(batch["inputs"].to(device), 0)
            tgt = fill_flow_nan0(batch["target"].to(device), 0)
            pred = model(inp)
            loss = masked_mae_train(
                pred,
                tgt[..., 0],
                null_val=0.0,
                valid_mask=batch["obs_mask"].to(device),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        val_m = evaluate_mae(model, val_loader, train_loader.dataset.mean, train_loader.dataset.std, device)
        # Paper primary metric: masked micro-average over all 12 horizons.
        val_score = float(val_m["MAE_avg"])
        print(
            f"epoch {epoch}/{epochs} train_loss={np.mean(losses):.4f} "
            f"val_MAE_avg={val_score:.4f}",
            flush=True,
        )
        if val_score < best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_m = evaluate_mae(model, test_loader, train_loader.dataset.mean, train_loader.dataset.std, device)
    summary = {
        "seed": seed,
        "epochs": epochs,
        "val_MAE_avg": best_val,
        **{k: test_m[k] for k in test_m},
    }
    print(json.dumps(summary, indent=2), flush=True)
    if getattr(args, "summary_json", None):
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Train ZipMix")
    ap.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    ap.add_argument("--smoke", action="store_true", help="Random-tensor forward smoke test")
    ap.add_argument("--smoke-train", action="store_true", help="One-epoch training smoke test")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-eval-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="Override config seed")
    ap.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size")
    ap.add_argument("--summary-json", type=Path, default=None, help="Write run summary JSON")
    args = ap.parse_args()

    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = load_yaml(cfg_path)
    if args.seed is not None:
        cfg = merge_config(cfg, {"seed": int(args.seed)})
    if args.batch_size is not None:
        cfg = merge_config(cfg, {"train": {"batch_size": int(args.batch_size)}})
    device = torch.device("cpu")

    if args.smoke:
        rep = smoke_forward(cfg, device)
        print(json.dumps(rep, indent=2))
        return 0 if rep.get("ok") else 1

    if args.smoke_train and not (cfg.get("data") or {}).get("root"):
        rep = smoke_forward(cfg, device)
        print(json.dumps({"smoke_train_skipped_no_data": True, **rep}, indent=2))
        return 0

    train(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
