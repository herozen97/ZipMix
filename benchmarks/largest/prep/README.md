# LargeST data preparation

Self-contained pipeline to turn **official LargeST raw HDF** into the **LATEST NPZ packs** consumed by ZipMix (`data/{subset}/{protocol}/LATEST/`).

This module ships **only** what is needed for benchmark reproduction:

| Included | Not included |
|----------|----------------|
| 5min→15min resample (paper protocol) | Raw traffic downloads |
| Temporal holdout protocols (`eval_2019`, `eval_2020`, `eval_2021`) | Internal analysis tooling |
| Reliable-node lists per subset | LargeST year-only baseline packs |
| Train-only normalization + window indices | Model training code (see repo root) |

## Prerequisites

1. Clone [LargeST](https://github.com/liuxu77/LargeST) and download raw CA HDF files.
2. From the LargeST `download/data/` folder you need at minimum:

```
ca_his_raw_2018.h5
ca_his_raw_2019.h5
ca_his_raw_2020.h5   # for eval_2020
ca_his_raw_2021.h5   # for eval_2021
ca_meta.csv
ca_rn_adj.npy        # optional, for graph adjacency crop
```

3. Install prep dependencies (in addition to ZipMix `requirements.txt`):

```bash
pip install -r benchmarks/largest/prep/requirements.txt
```

## Step 0 — See what you need

```bash
cd /path/to/ZipMix
python -m benchmarks.largest.prep plan \
  --subsets sd,gba,gla \
  --scenarios eval_2019,eval_2020,eval_2021
```

## Step 1 — Resample raw HDF to 15-minute view cache

```bash
export RAW_DIR=/path/to/LargeST/download/data
export CACHE_DIR=/path/to/largest_cache

python -m benchmarks.largest.prep resample \
  --raw-dir "$RAW_DIR" \
  --meta-csv "$RAW_DIR/ca_meta.csv" \
  --cache-dir "$CACHE_DIR" \
  --years 2018,2019,2020,2021
```

This writes intermediate files under `$CACHE_DIR/views/raw_valid_{year}_{scope}.npz`.

**Resample rule (v1)**: each 15-minute bucket is the mean of three 5-minute bins; if **any** bin is missing, the 15-minute value is missing. No zero-fill.

**Scopes**:

| scope | used by subset |
|-------|----------------|
| `ca_full` | `sd`, `gla` |
| `d4_full` | `gba` (district 4) |

Re-running is safe (skips existing caches). Use `--force` to rebuild.

## Step 2 — Build protocol LATEST packs

```bash
export OUT_DIR=/path/to/LargeST/data   # must match ZipMix data.root layout

python -m benchmarks.largest.prep build \
  --cache-dir "$CACHE_DIR" \
  --out-dir "$OUT_DIR" \
  --subsets sd \
  --scenarios eval_2019 \
  --meta-csv "$RAW_DIR/ca_meta.csv" \
  --adj-npy "$RAW_DIR/ca_rn_adj.npy"
```

Output layout:

```
$OUT_DIR/sd/eval_2019/LATEST/
  his.npz
  idx_train.npy
  idx_val.npy
  idx_test.npy
  split_meta.json
  reliable_nodes.csv
```

Build all nine grid cells:

```bash
python -m benchmarks.largest.prep build \
  --cache-dir "$CACHE_DIR" \
  --out-dir "$OUT_DIR" \
  --subsets sd,gba,gla \
  --scenarios eval_2019,eval_2020,eval_2021 \
  --meta-csv "$RAW_DIR/ca_meta.csv" \
  --adj-npy "$RAW_DIR/ca_rn_adj.npy"
```

## Step 3 — Train ZipMix

Point `data.root` in `configs/experiments/largest_9grid.yaml` to the LargeST root (the directory that **contains** `data/sd/...`):

```bash
python -m zipmix.train --config configs/experiments/largest_9grid.yaml
```

## Protocol reference

Split boundaries and window hyperparameters live in [`config.yaml`](config.yaml).  
Protocol IDs name the **test calendar year**; they do not imply drift severity.

| protocol | early (train+val) | test (eval) | timeline years |
|----------|-------------------|-------------|----------------|
| `eval_2019` | 2018 | 2019 | 2018–2019 |
| `eval_2020` | 2018–2019 | 2020 | 2018–2020 |
| `eval_2021` | 2018–2019 | 2021 | 2018–2019, 2021 (2020 skipped in series) |

Common settings: lookback=12, horizon=12, stride=1, early 80/20 train/val, min 80% observed cells per window.

## Reliable nodes

Per-subset sensor ID lists are in [`assets/reliable_nodes_{subset}.txt`](assets/).  
These lists define the **677 / 2286 / 3699** nodes used in the paper nine-grid experiments after multi-year common-mask filtering. They are plain LargeST sensor IDs (from `ca_meta.csv`); no raw data is bundled.

To verify your build matches the reference protocol for `sd/eval_2019`:

- `his.npz` shape `(70080, 677, 3)`
- sample counts ≈ 28009 / 6985 / 35028

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Missing raw HDF` | Check `--raw-dir` and filename `ca_his_raw_{year}.h5` |
| `Missing view cache` | Run `resample` for all years listed by `plan` |
| `reliable nodes missing from cache` | Rebuild `ca_full` / `d4_full` caches with `--force` |
| Empty split / validation fail | Confirm raw files cover the protocol calendar years |
