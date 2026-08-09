# Diffusion nowcast scorecard (`val` split)

_13281 crops, 8-member ensembles, 25 Heun steps, guidance 1.0. Checkpoint epoch 98 (val loss 0.3095366189686152). FSS from 403 crops, PSD from 200._

`model_mean` is the ensemble mean; `model_member` is the average skill of a single member. Averaging damps peaks, so the mean scores better on MAE/RMSE and worse at high thresholds. Both are reported, see `docs/Diffusion_Run1_Results.md`.

## Pixel error (lower is better)

| metric | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| MAE (mm/h) | 0.382 | 0.323 | **0.278** | 0.333 |
| RMSE (mm/h) | 1.283 | 1.140 | **0.899** | 1.151 |
| bias (mm/h) | -0.001 | -0.019 | -0.015 | -0.015 |

## CSI by threshold (higher is better)

| mm/h | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| 0.5 | 0.307 | 0.404 | 0.456 | 0.377 |
| 1 | 0.237 | 0.321 | 0.383 | 0.298 |
| 2 | 0.153 | 0.207 | 0.256 | 0.199 |
| 4 | 0.089 | 0.105 | 0.136 | 0.114 |
| 8 | 0.048 | 0.040 | 0.053 | 0.054 |

## Detection at 1 mm/h

| metric | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| POD | 0.383 | 0.477 | 0.547 | 0.450 |
| FAR | 0.616 | 0.504 | 0.439 | 0.531 |
| freq_bias | 0.996 | 0.961 | 0.976 | 0.959 |

## Probabilistic (the ensemble's reason to exist)

- **CRPS (fair)**: 0.1735 mm/h. A deterministic forecast's CRPS equals its MAE, so compare against advection 0.323 and persistence 0.382.
- **Spread / RMSE**: 0.854 (spread 0.768, ens-mean RMSE 0.899). Near 1 is well dispersed, below 1 is over-confident.
- **Outlier rate**: 0.293 vs ideal 0.222.
- **Rank-histogram flatness** (RMSE from flat): 0.0686, 0 is perfectly flat.

## FSS (model mean vs a single member)

| threshold | 1 km | 5 km | 11 km | 21 km | 51 km | 101 km |
|---|---|---|---|---|---|---|
| mean, 0.5 mm/h | 0.632 | 0.732 | 0.791 | 0.842 | 0.904 | 0.938 |
| mean, 1 mm/h | 0.564 | 0.680 | 0.749 | 0.812 | 0.888 | 0.930 |
| mean, 2 mm/h | 0.433 | 0.564 | 0.645 | 0.722 | 0.820 | 0.874 |
| mean, 4 mm/h | 0.294 | 0.420 | 0.500 | 0.583 | 0.695 | 0.758 |
| mean, 8 mm/h | 0.164 | 0.239 | 0.295 | 0.375 | 0.471 | 0.499 |
| member, 0.5 mm/h | 0.555 | 0.660 | 0.731 | 0.800 | 0.887 | 0.938 |
| member, 1 mm/h | 0.470 | 0.588 | 0.670 | 0.750 | 0.856 | 0.918 |
| member, 2 mm/h | 0.349 | 0.475 | 0.566 | 0.658 | 0.791 | 0.878 |
| member, 4 mm/h | 0.241 | 0.367 | 0.462 | 0.567 | 0.724 | 0.830 |
| member, 8 mm/h | 0.145 | 0.237 | 0.329 | 0.461 | 0.653 | 0.753 |

## Wet-area fraction (>= 0.1 mm/h)

| field | % |
|---|---|
| obs | 23.1 |
| model_mean | 34.9 |
| model_member | 22.5 |
| advection | 21.6 |
| persistence | 22.9 |

See `diffusion_eval.png` for the histogram, PSD, CSI, reliability, rank histogram and spread panels.
