# ZipMix

**ZipMix** is a spatio-temporal forecasting model for **distribution shift**: the test
distribution may differ from training in temporal patterns, scale, or regime.

- Default task: 12 history steps → 12 forecast steps
- Modular architecture with leave-one-out ablation configs
- Self-contained PyTorch implementation

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python ≥3.10 and PyTorch ≥2.0.

## Quick Start (no external data)

Verify the model builds and runs a forward pass:

```bash
python -m zipmix.train --config configs/model/full.yaml --smoke
```

## Architecture

See [docs/MODEL.md](docs/MODEL.md) for the component pipeline:

1. **Gated Temporal Encoder** — dilated gated convolutions
2. **Temporal Bottleneck** — time-axis compression (core)
3. **Graph Message Passing** — geo kNN message passing on nodes (support)
4. **Axis Mixer** — multi-scale mixing along reordered node axis (core)
5. **Horizon Cascade Head** — progressive multi-horizon decoder for `pred_len=12`
   (implementation detail; ablations show near-zero contribution)

Toggle components via `configs/model/*.yaml`.

## Training on your data

ZipMix expects a **preprocessed window pack** (not raw downloads). Full specification:
**[docs/DATA.md](docs/DATA.md)**.

Minimum layout:

```
{data.root}/
  his.npz          # arrays: data (T,N,C), obs_mask (T,N), mean, std
  idx_train.npy
  idx_val.npy
  idx_test.npy
```

Build a demo pack (also writes a matching `node_meta.csv`):

```bash
python scripts/build_npz_pack.py --out ./demo_pack --t 500 --n 32
```

Copy `configs/model/full.yaml` and set:

```yaml
data:
  type: npz_window
  root: ./demo_pack
model:
  node_meta_csv: ./demo_pack/node_meta.csv
```

Run:

```bash
python -m zipmix.train --config path/to/your_config.yaml
```

Training keeps the best validation weights in memory only; it does not write checkpoint
files by default.

## Paper benchmark reproduction

LargeST requires **preprocessed LATEST packs** (raw download → resample → temporal holdout).
See **[benchmarks/largest/README.md](benchmarks/largest/README.md)** for:

- building packs from official LargeST raw data
- holdout protocols (`eval_2019`, `eval_2020`, `eval_2021`)
- train commands and batch sizes

Reported numbers: [results/benchmark_largest.md](results/benchmark_largest.md)

Node metadata under `benchmarks/largest/assets/` and reliable-node ID lists under
`benchmarks/largest/prep/assets/` are derived from [LargeST](https://github.com/liuxu77/LargeST)
(`ca_meta.csv`). Raw traffic time series are not shipped. See [NOTICE](NOTICE) for
dataset license terms (CC BY-NC 4.0) separate from the ZipMix MIT code license.

Nine-grid reproduction over three seeds (`42, 43, 44`) with per-subset batch sizes:

```bash
python scripts/run_largest_grid.py --config configs/experiments/largest_9grid.yaml
```

## Project layout

```
zipmix/               # model + training code
configs/model/        # model / ablation configs
configs/experiments/  # benchmark experiment templates
benchmarks/largest/   # optional LargeST adapter
scripts/              # e.g. build_npz_pack.py
results/              # reported metrics tables
docs/                 # MODEL.md, DATA.md
```

## Citation

If you use ZipMix, please cite:

```bibtex
@software{huang2026zipmix,
  title  = {Temporal Compression and Shared-Weight Node Mixing for Traffic Forecasting under Distribution Shift},
  author = {Huang, Zongyuan and Wang, Weipeng and Yang, Jinming and Jin, Yaohui and Xu, Yanyan},
  year   = {2026},
  note   = {Software release}
}
```

Software metadata: [CITATION.cff](CITATION.cff).

## License

- **Software** (source code and docs): MIT — see [LICENSE](LICENSE).
- **LargeST-derived metadata / node ID lists** under `benchmarks/largest/assets/` and
  `benchmarks/largest/prep/assets/`: CC BY-NC 4.0 (LargeST dataset terms) — see
  [NOTICE](NOTICE).
