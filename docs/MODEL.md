# ZipMix Model Architecture

ZipMix is a spatio-temporal forecasting model for multivariate sensor graphs under
distribution shift.

Default task: predict 12 future steps from 12 history steps (`hist_len=12`, `pred_len=12`).
The horizon cascade head currently supports **pred_len=12** only.

## Pipeline

```
Input (B, T, N, C)
  → Gated Temporal Encoder
  → Temporal Bottleneck (optional)
  → Graph Message Passing (optional, natural node order)
  → Axis reorder
  → Axis Mixer stack (optional, multi-scale 1D conv)
  → Axis scatter back to nodes
  → Horizon Cascade Head (optional; pred_len=12)
  → Forecast (B, pred_len, N)
```

## Components

### Gated Temporal Encoder

Dilated gated TCN blocks with pointwise input projection. Encodes each node's temporal
history independently before spatial mixing.

### Temporal Bottleneck

Linear compress-expand along the time axis with lightweight conv refinement. Reduces
temporal redundancy before graph and axis operations.

Config: `use_temporal_bottleneck`, `zip_len`.

### Graph Message Passing

kNN graph built from node lat/lng metadata. Short-hop mean aggregation with residual MLP
updates on the natural node domain (before axis reorder).

Config: `use_graph_message`, `graph_k`, `graph_hops`, `node_meta_csv`.

### Axis Mixer

Nodes are permuted along a train-fixed axis (default: random seed). Multiple `AxisMixer`
layers with different kernel sizes (default 3, 9, 27) run in parallel and fuse.

Config: `use_axis_mixer`, `scales`, `axis_mixer_mode` (`random` / `shuffle` / `geo` / `hilbert`).
Mode `hilbert` uses Morton/Z-order keys (bit interleaving), not a true Hilbert curve.

### Horizon Cascade Head

Three-stage decoder for 12-step forecasts (4+4+4). Later stages condition on stop-gradient
outputs from earlier stages. When disabled, a linear fallback still requires `pred_len=12`.
This head is kept for `pred_len=12` compatibility; leave-one-out ablations show near-zero
contribution relative to Temporal Bottleneck and Axis Mixer.

Config: `use_horizon_cascade`.

## Hyperparameters

| key | default | description |
|-----|---------|-------------|
| `hidden_dim` | 192 | channel width (must be divisible by 8 for GroupNorm) |
| `depth` | 4 | TCN depth |
| `dropout` | 0.1 | dropout rate |
| `in_channels` | 3 | input feature channels |

Ablation configs live in `configs/model/`.
