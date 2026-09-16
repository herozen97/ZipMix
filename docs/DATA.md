# Data format for ZipMix

ZipMix training reads **pre-windowed NPZ packs**, not raw CSV/HDF/PeMS downloads directly.
You either (a) use an existing pack, or (b) preprocess your series into the layout below.

## Quick decision

| Your situation | What to do |
|----------------|------------|
| Reproduce paper numbers on LargeST | See [benchmarks/largest/README.md](../benchmarks/largest/README.md) |
| Train on your own sensors | Build an `npz_window` pack (this document) |
| Sanity check without data | `python -m zipmix.train --config configs/model/full.yaml --smoke` |

## Directory layout (`npz_window`)

Point `data.root` in the config to a directory containing:

```
{data.root}/
  his.npz
  idx_train.npy
  idx_val.npy
  idx_test.npy
  split_meta.json          # optional but recommended
  reliable_nodes.csv       # optional (LargeST benchmark only)
```

Example config:

```yaml
data:
  type: npz_window
  root: /path/to/my_split/LATEST
task:
  hist_len: 12
  pred_len: 12
model:
  node_meta_csv: /path/to/node_latlng.csv   # required (graph and axis ordering)
```

## `his.npz` schema

| key | shape | dtype | meaning |
|-----|-------|-------|---------|
| `data` | `(T, N, C)` | float32 | Multivariate series; **channel 0 = target flow** |
| `obs_mask` | `(T, N)` | uint8 | 1 = observed at this timestep & node, 0 = missing |
| `mean` | scalar | float | Train-only z-score mean for channel 0 |
| `std` | scalar | float | Train-only z-score std for channel 0 |

Typical LargeST paper settings: `C=3` (flow, time-of-day, day-of-week), `hist_len=pred_len=12`.

**Normalization**: values in `data[..., 0]` are **already z-scored** using train-only
statistics stored in `mean` / `std`. ZipMix metrics denormalize predictions back to raw
units for MAE.

## Window indexing (`idx_*.npy`)

Each file is a 1D `int64` array of **anchor timesteps** `t`.

For anchor `t`, the dataloader builds:

| field | slice | shape |
|-------|-------|-------|
| `inputs` | `data[t-hist_len+1 : t+1]` | `(hist_len, N, C)` |
| `target` | `data[t+1 : t+pred_len+1]` | `(pred_len, N, C)` |
| `obs_mask` | `obs_mask[t+1 : t+pred_len+1]` | `(pred_len, N)` |

Training loss uses channel 0 of `target` (`tgt[..., 0]`).

Constraints:

- Valid anchors satisfy `hist_len-1 ≤ t ≤ T-pred_len-1`.
- Splits are **time-ordered** (not random windows); do not reshuffle anchors across
  train/val/test after indexing.

## Node metadata (`node_meta_csv`)

**Always required** for the full model path: Axis Mixer builds a node permutation and
Graph Message Passing builds a kNN graph from the same CSV. Preferred format:

```csv
ID,Lat,Lng
1114091,32.544463,-117.032486
...
```

- One row per node, **exactly** `N` data rows (same order as the `N` axis in `data`).
- An ``ID`` column is required for LargeST runs; training asserts it matches
  `reliable_nodes_{subset}.txt` and, when present, the pack's `reliable_nodes.csv`.
- Legacy `Lat,Lng`-only files still load coordinates for custom packs, but LargeST
  / smoke paths that call alignment checks require `ID`.
- Length mismatch raises; the loader does **not** zero-pad short files.

For the LargeST benchmark, subset files ship under `benchmarks/largest/assets/` as
`{sd,gba,gla}_meta.csv` and are selected from `data.subset`. These files are geographic
metadata derived from LargeST `ca_meta.csv` (see [NOTICE](../NOTICE)).

GPU runs set `cudnn.deterministic=True` for closer reproducibility; bit-exact matches
across devices are still not guaranteed.

## Building a pack from scratch (custom data)

Minimal pipeline:

1. **Assemble** `(T, N, C)` float array and `(T, N)` mask.
2. **Normalize** channel 0 with train-only mean/std; store scalars in `his.npz`.
3. **Choose anchors** per split (chronological segments).
4. **Save**:
   ```python
   import numpy as np
   np.savez("his.npz", data=data, obs_mask=mask, mean=mean, std=std)
   np.save("idx_train.npy", train_anchors)
   np.save("idx_val.npy", val_anchors)
   np.save("idx_test.npy", test_anchors)
   ```
5. Provide a Lat/Lng CSV with exactly `N` rows and set `model.node_meta_csv`.
6. Point config `data.root` to that directory and run training.

A reference builder is [`scripts/build_npz_pack.py`](../scripts/build_npz_pack.py)
(synthetic demo writes a matching `node_meta.csv`).

## What ZipMix does *not* do (generic path)

- Download raw PeMS / sensor archives
- Automatically preprocess arbitrary CSV/HDF into window packs

For **LargeST paper reproduction**, see
[`benchmarks/largest/prep/README.md`](../benchmarks/largest/prep/README.md).
For **custom datasets**, use
[`scripts/build_npz_pack.py`](../scripts/build_npz_pack.py) or your own
pipeline following the schema above.
