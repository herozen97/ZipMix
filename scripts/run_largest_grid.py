#!/usr/bin/env python3
"""Run LargeST 9-grid × multi-seed ZipMix training and average MAE_avg.

Example (from repo root, after setting data.root in the yaml):

  python scripts/run_largest_grid.py --config configs/experiments/largest_9grid.yaml

Defaults match the paper table: subsets sd/gba/gla, protocols eval_2019/2020/2021,
seeds 42/43/44, and per-subset batch sizes 32/16/8.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBSETS = ("sd", "gba", "gla")
DEFAULT_SCENARIOS = ("eval_2019", "eval_2020", "eval_2021")
DEFAULT_SEEDS = (42, 43, 44)
BATCH_SIZE = {"sd": 32, "gba": 16, "gla": 8}


def main() -> int:
    ap = argparse.ArgumentParser(description="ZipMix LargeST multi-seed grid runner")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs/experiments/largest_9grid.yaml")
    ap.add_argument("--subsets", default="sd,gba,gla")
    ap.add_argument("--scenarios", default="eval_2019,eval_2020,eval_2021")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON summary path")
    args = ap.parse_args()

    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    with open(cfg_path, encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    if not (base.get("data") or {}).get("root"):
        print(
            "error: set data.root in the experiment yaml to the LargeST root "
            "(directory containing data/sd/...)",
            file=sys.stderr,
        )
        return 2

    subsets = tuple(s.strip() for s in args.subsets.split(",") if s.strip()) or DEFAULT_SUBSETS
    scenarios = tuple(s.strip() for s in args.scenarios.split(",") if s.strip()) or DEFAULT_SCENARIOS
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip()) or DEFAULT_SEEDS

    cells: list[dict] = []
    for subset in subsets:
        bs = BATCH_SIZE.get(subset, 32)
        for scenario in scenarios:
            seed_metrics: list[float] = []
            seed_rows: list[dict] = []
            for seed in seeds:
                run_cfg = dict(base)
                run_cfg["seed"] = seed
                run_cfg["train"] = dict(run_cfg.get("train") or {})
                run_cfg["train"]["batch_size"] = bs
                run_cfg["data"] = dict(run_cfg.get("data") or {})
                run_cfg["data"]["subset"] = subset
                run_cfg["data"]["scenario"] = scenario
                tmp_dir = REPO_ROOT / ".cache" / "grid_runs"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp = tmp_dir / f"{subset}_{scenario}_s{seed}.yaml"
                summary_path = tmp_dir / f"{subset}_{scenario}_s{seed}.json"
                with open(tmp, "w", encoding="utf-8") as f:
                    yaml.safe_dump(run_cfg, f, sort_keys=False)
                cmd = [
                    sys.executable,
                    "-m",
                    "zipmix.train",
                    "--config",
                    str(tmp),
                    "--seed",
                    str(seed),
                    "--batch-size",
                    str(bs),
                    "--summary-json",
                    str(summary_path),
                ]
                if args.cpu:
                    cmd.append("--cpu")
                print("RUN", " ".join(cmd), flush=True)
                if args.dry_run:
                    continue
                proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
                if proc.returncode != 0:
                    return proc.returncode
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                seed_metrics.append(float(summary["MAE_avg"]))
                seed_rows.append(summary)
            cell = {
                "subset": subset,
                "scenario": scenario,
                "batch_size": bs,
                "seeds": list(seeds),
                "MAE_avg_mean": float(sum(seed_metrics) / len(seed_metrics)) if seed_metrics else None,
                "per_seed": seed_rows,
            }
            cells.append(cell)
            if cell["MAE_avg_mean"] is not None:
                print(
                    f"CELL {subset} {scenario} MAE_avg_mean={cell['MAE_avg_mean']:.4f}",
                    flush=True,
                )

    report = {"cells": cells}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
