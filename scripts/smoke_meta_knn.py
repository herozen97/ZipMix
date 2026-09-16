#!/usr/bin/env python3
"""Smoke: subset meta resolve + ID alignment + kNN health for sd/gba/gla (no training)."""
from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zipmix.geo import (  # noqa: E402
    assert_meta_ids_match,
    build_knn_edge_index,
    load_id_list,
    load_lat_lng,
    load_node_ids,
)

EXPECT = {
    "sd": {"n_nodes": 677, "min_distinct": 600},
    "gba": {"n_nodes": 2286, "min_distinct": 2000},
    "gla": {"n_nodes": 3699, "min_distinct": 3000},
}


def resolve_meta_csv(subset: str) -> str:
    candidate = ROOT / "benchmarks" / "largest" / "assets" / f"{subset}_meta.csv"
    if not candidate.is_file():
        raise FileNotFoundError(f"missing node meta for subset={subset!r}: {candidate}")
    return str(candidate)


def main() -> int:
    assets = ROOT / "benchmarks" / "largest" / "assets"
    prep = ROOT / "benchmarks" / "largest" / "prep" / "assets"
    for subset, spec in EXPECT.items():
        path = resolve_meta_csv(subset)
        assert path.endswith(f"{subset}_meta.csv"), path
        n = spec["n_nodes"]
        coords = load_lat_lng(path, n)
        assert coords.shape == (n, 2), coords.shape
        ids = load_node_ids(path, n)
        expected = load_id_list(prep / f"reliable_nodes_{subset}.txt")
        assert_meta_ids_match(path, expected, context=subset)
        knn = build_knn_edge_index(coords, k=8)
        ctr = collections.Counter(knn.flatten().tolist())
        distinct = len(ctr)
        max_ref = ctr.most_common(1)[0][1]
        print(
            f"{subset}: meta={Path(path).name} ids_ok={ids.shape[0]} "
            f"distinct={distinct} max_ref={max_ref}"
        )
        assert distinct >= spec["min_distinct"], (subset, distinct)
        assert max_ref < n // 2, (subset, max_ref)

    try:
        resolve_meta_csv("_no_such_subset")
    except FileNotFoundError as e:
        print("missing-subset raises:", type(e).__name__)
    else:
        raise AssertionError("expected FileNotFoundError for missing subset meta")

    try:
        load_lat_lng(assets / "sd_meta.csv", 3699)
    except ValueError as e:
        print("length-mismatch raises:", type(e).__name__)
    else:
        raise AssertionError("expected ValueError for short meta")

    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
