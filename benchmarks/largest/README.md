# LargeST benchmark data

This page explains how to obtain the **LATEST NPZ packs** used in the paper experiments.
ZipMix does not ship raw traffic data.

## What you need at runtime

Set `data.root` to the LargeST repository root. ZipMix loads:

```
{data.root}/data/{subset}/{protocol}/LATEST/
  his.npz
  idx_train.npy
  idx_val.npy
  idx_test.npy
  split_meta.json
  reliable_nodes.csv
```

| field | values |
|-------|--------|
| `subset` | `sd`, `gba`, `gla` |
| `protocol` (`data.scenario` in yaml) | `eval_2019`, `eval_2020`, `eval_2021` |

Example config: [`configs/experiments/largest_9grid.yaml`](../../configs/experiments/largest_9grid.yaml)

## Two ways to get ready-to-use packs

### Option A — Use existing LATEST packs

If you already have the processed tree:

```bash
export DATA_ROOT=/path/to/LargeST   # contains data/sd/eval_2019/LATEST/ ...
# edit largest_9grid.yaml → data.root: /path/to/LargeST
python -m zipmix.train --config configs/experiments/largest_9grid.yaml
```

Verify one cell before the full grid:

```bash
python - <<'PY'
from pathlib import Path
root = Path("/path/to/LargeST/data/sd/eval_2019/LATEST")
assert (root / "his.npz").exists()
import numpy as np
z = np.load(root / "his.npz")
print("data", z["data"].shape, "idx_train", len(np.load(root / "idx_train.npy")))
PY
```

### Option B — Build from official LargeST raw data

Official LargeST ([GitHub](https://github.com/liuxu77/LargeST)) ships **5-minute raw HDF**
(`ca_his_raw_{year}.h5`) plus `ca_meta.csv`.
ZipMix includes a **self-contained prep tool** under [`prep/`](prep/) that produces the
benchmark LATEST packs.

**Quick path** (from ZipMix repo root):

```bash
pip install -r benchmarks/largest/prep/requirements.txt

# 1) resample raw HDF -> 15min view cache
python -m benchmarks.largest.prep resample \
  --raw-dir /path/to/LargeST/download/data \
  --meta-csv /path/to/LargeST/download/data/ca_meta.csv \
  --cache-dir /path/to/largest_cache \
  --years 2018,2019,2020,2021

# 2) build protocol LATEST packs under LargeST/data/
python -m benchmarks.largest.prep build \
  --cache-dir /path/to/largest_cache \
  --out-dir /path/to/LargeST/data \
  --subsets sd,gba,gla \
  --scenarios eval_2019,eval_2020,eval_2021 \
  --meta-csv /path/to/LargeST/download/data/ca_meta.csv \
  --adj-npy /path/to/LargeST/download/data/ca_rn_adj.npy
```

Full step-by-step guide: **[prep/README.md](prep/README.md)**
List required downloads first: `python -m benchmarks.largest.prep plan`

**What ZipMix prep does *not* replace**

- Downloading raw PeMS / Kaggle archives (see LargeST repo)
- Official single-year `{subset}/{year}/` packs from `generate_data_for_training.py`
  (different split protocol)

## Temporal holdout protocols

Three **year-disjoint temporal holdout protocols** are reported in the nine-grid benchmark.
Protocol IDs name the **test calendar year** only; they are not ordinal drift-severity bins.

| protocol | source (train+val) | test period | input timeline |
|----------|-------------------|-------------|----------------|
| `eval_2019` | 2018 | 2019 | 2018–2019 |
| `eval_2020` | 2018–2019 | 2020 | 2018–2020 |
| `eval_2021` | 2018–2019 | 2021 | 2018–2019, 2021 (`2020` excluded from series) |

Each pack includes multi-year concatenation, time-ordered train/val/test, train-only
normalization, reliable-node masking, and sliding-window indices documented in
[docs/DATA.md](../../docs/DATA.md).

> **Important**: Point `data.root` at the LargeST **root** (the folder that contains
> `data/sd/...`), not at `LATEST` itself.

## Node metadata

`ID,Lat,Lng` CSVs (row count must equal subset `n_nodes`), derived from LargeST
`ca_meta.csv` and aligned with `prep/assets/reliable_nodes_{subset}.txt`:

- `benchmarks/largest/assets/sd_meta.csv` (677)
- `benchmarks/largest/assets/gba_meta.csv` (2286)
- `benchmarks/largest/assets/gla_meta.csv` (3699)

Selected automatically as `{subset}_meta.csv` from `data.subset` (or set `node_meta_csv`
explicitly). **Missing subset meta is an error** — there is no silent fallback to another
subset. Training asserts meta `ID` order against the shipped reliable-node list and, when
present, against `LATEST/reliable_nodes.csv`.

See [NOTICE](../../NOTICE) for CC BY-NC terms on these derived assets.

## Train commands

Single setting (edit `data.root` / `subset` / `scenario` first; set `train.batch_size`
to 32/16/8 for sd/gba/gla):

```bash
python -m zipmix.train --config configs/experiments/largest_9grid.yaml
# or: --seed 43 --batch-size 16
```

Suggested batch sizes:

| subset | batch_size |
|--------|------------|
| sd | 32 |
| gba | 16 |
| gla | 8 |

Full nine-grid × three seeds (`42, 43, 44`), with batch sizes applied automatically:

```bash
python scripts/run_largest_grid.py --config configs/experiments/largest_9grid.yaml
```

Checkpoint selection uses validation **MAE_avg** (12-step masked micro-average), matching
the reported primary metric.

Reported **MAE_avg** (mean over three seeds): [`results/benchmark_largest.md`](../../results/benchmark_largest.md)

## Ablation configs

To disable one component, copy the `data` section from `largest_9grid.yaml` into a run with:

- `configs/model/no_temporal_bottleneck.yaml`
- `configs/model/no_axis_mixer.yaml`
- `configs/model/no_graph_message.yaml`
- `configs/model/no_horizon_cascade.yaml`
