# Benchmark results (LargeST)

Average MAE over the 12 forecast steps (**MAE_avg**), masked micro-average in original
flow units. Full ZipMix model. Values are the mean over three random seeds
(`42`, `43`, `44`). Reproduce with `python scripts/run_largest_grid.py`.

| subset | eval_2019 | eval_2020 | eval_2021 |
|--------|----------:|----------:|----------:|
| sd† | 16.36 | 21.56 | 20.85 |
| gba | 24.21 | 21.88 | 21.81 |
| gla | 23.73 | 22.24 | 21.80 |

† SD numbers match the prior geographic-meta configuration (unchanged after meta fix).

Validation checkpoint selection uses **MAE_avg** (same primary metric as the table).
Per-subset training batch sizes: sd=32, gba=16, gla=8.

See [`benchmarks/largest/README.md`](../benchmarks/largest/README.md) for reproduction
commands.
