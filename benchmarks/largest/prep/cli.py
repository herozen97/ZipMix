"""CLI for LargeST → ZipMix benchmark pack preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.largest.protocols import DEFAULT_PROTOCOLS_CSV
from benchmarks.largest.prep.io import SUBSET_SCOPES, resample_year_to_cache
from benchmarks.largest.prep.pipeline import (
    load_prep_config,
    prepare_scenario,
    required_scopes,
    required_years,
    write_subset_adj,
)

PREP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PREP_DIR / "config.yaml"
DEFAULT_ASSETS = PREP_DIR / "assets"


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_str_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def cmd_resample(args: argparse.Namespace) -> int:
    years = _parse_int_list(args.years)
    raw_dir = Path(args.raw_dir)
    meta_csv = Path(args.meta_csv)
    cache_dir = Path(args.cache_dir)
    scopes = _parse_str_list(args.scopes) if args.scopes else sorted({SUBSET_SCOPES[s]["cache_scope"] for s in SUBSET_SCOPES})

    for scope in scopes:
        districts = None
        if scope == "d4_full":
            districts = [4]
        for year in years:
            raw_h5 = raw_dir / f"ca_his_raw_{year}.h5"
            out = cache_dir / "views" / f"raw_valid_{year}_{scope}.npz"
            if out.exists() and not args.force:
                print(f"[skip] {out}")
                continue
            if args.force and out.exists():
                out.unlink()
            print(f"[resample] {raw_h5} -> {out}")
            resample_year_to_cache(
                year=year,
                scope=scope,
                raw_h5=raw_h5,
                meta_csv=meta_csv,
                cache_dir=cache_dir,
                districts=districts,
            )
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    cfg = load_prep_config(cfg_path)
    subsets = _parse_str_list(args.subsets)
    scenarios = _parse_str_list(args.scenarios)
    cache_dir = Path(args.cache_dir)
    out_root = Path(args.out_dir)
    assets_dir = Path(args.assets_dir)

    for subset in subsets:
        if subset not in SUBSET_SCOPES:
            raise SystemExit(f"unknown subset: {subset}")
        for scenario in scenarios:
            if scenario not in cfg["splits"]:
                raise SystemExit(f"unknown scenario: {scenario}")
            print(f"[build] {subset}/{scenario}")
            result = prepare_scenario(
                cache_dir=cache_dir,
                out_root=out_root,
                assets_dir=assets_dir,
                subset=subset,
                scenario=scenario,
                cfg=cfg,
            )
            print(f"  -> {result['run_dir']}")
            print(f"  samples: {result['sample_counts']}")

        if args.meta_csv and args.adj_npy:
            from benchmarks.largest.prep.io import load_reliable_ids

            reliable_ids = load_reliable_ids(assets_dir, subset)
            adj_out = write_subset_adj(out_root, subset, Path(args.meta_csv), Path(args.adj_npy), reliable_ids)
            print(f"[adj] {adj_out}")

    latest = out_root / subsets[0] / scenarios[0] / "LATEST"
    print(f"\nDone. Point data.root to: {out_root.parent if out_root.name == 'data' else out_root}")
    print(f"Verify: {latest / 'his.npz'}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Print which raw years and cache scopes are needed (no I/O)."""
    cfg = load_prep_config(Path(args.config))
    subsets = _parse_str_list(args.subsets)
    scenarios = _parse_str_list(args.scenarios)
    years = sorted(required_years(cfg, scenarios))
    scopes = required_scopes(subsets)
    print("Required raw HDF files (under --raw-dir):")
    for y in years:
        print(f"  ca_his_raw_{y}.h5")
    print("\nRequired resample scopes:")
    for scope, subs in scopes.items():
        print(f"  {scope}  (for subsets: {', '.join(subs)})")
    print("\nOfficial LargeST files also needed for adjacency / meta:")
    print("  ca_meta.csv")
    print("  ca_rn_adj.npy")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.largest.prep",
        description="Prepare LargeST temporal holdout packs for ZipMix benchmark reproduction.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_resample = sub.add_parser("resample", help="5min raw HDF -> 15min view cache (.npz)")
    p_resample.add_argument("--raw-dir", required=True, help="Directory with ca_his_raw_{year}.h5")
    p_resample.add_argument("--meta-csv", required=True, help="LargeST ca_meta.csv")
    p_resample.add_argument("--cache-dir", required=True, help="Writable cache root (views/ created here)")
    p_resample.add_argument("--years", required=True, help="Comma-separated years, e.g. 2018,2019,2020,2021")
    p_resample.add_argument(
        "--scopes",
        default="",
        help="Comma-separated cache scopes: ca_full,d4_full (default: both)",
    )
    p_resample.add_argument("--force", action="store_true", help="Overwrite existing view caches")
    p_resample.set_defaults(func=cmd_resample)

    p_build = sub.add_parser("build", help="View cache -> scenario LATEST training packs")
    p_build.add_argument("--cache-dir", required=True, help="Cache root from resample step")
    p_build.add_argument("--out-dir", required=True, help="Output root (e.g. LargeST/data)")
    p_build.add_argument("--subsets", default="sd", help="Comma-separated: sd,gba,gla")
    p_build.add_argument("--scenarios", default=DEFAULT_PROTOCOLS_CSV, help="Comma-separated protocols")
    p_build.add_argument("--config", default=str(DEFAULT_CONFIG), help="Prep protocol yaml")
    p_build.add_argument("--assets-dir", default=str(DEFAULT_ASSETS), help="reliable_nodes_*.txt directory")
    p_build.add_argument("--meta-csv", default="", help="Optional ca_meta.csv for adjacency crop")
    p_build.add_argument("--adj-npy", default="", help="Optional ca_rn_adj.npy for adjacency crop")
    p_build.set_defaults(func=cmd_build)

    p_plan = sub.add_parser("plan", help="List required downloads for given subsets/scenarios")
    p_plan.add_argument("--subsets", default="sd,gba,gla")
    p_plan.add_argument("--scenarios", default=DEFAULT_PROTOCOLS_CSV)
    p_plan.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_plan.set_defaults(func=cmd_plan)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
