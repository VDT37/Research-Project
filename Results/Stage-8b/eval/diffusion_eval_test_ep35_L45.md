# Diffusion nowcast scorecard (`test` split)

_13281 crops, 8-member ensembles, 25 Heun steps, guidance 1.0. Checkpoint epoch 35 (val loss 0.33152407111889187). FSS from 403 crops, PSD from 200._

`model_mean` is the ensemble mean; `model_member` is the average skill of a single member. Averaging damps peaks, so the mean scores better on MAE/RMSE and worse at high thresholds. Both are reported, see `docs/Diffusion_Run1_Results.md`.

## Pixel error (lower is better)

| metric | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| MAE (mm/h) | 0.318 | 0.252 | **0.220** | 0.268 |
| RMSE (mm/h) | 1.288 | 1.121 | **0.897** | 1.177 |
| bias (mm/h) | -0.004 | -0.019 | -0.006 | -0.006 |

## CSI by threshold (higher is better)

| mm/h | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| 0.5 | 0.313 | 0.433 | 0.491 | 0.410 |
| 1 | 0.237 | 0.349 | 0.419 | 0.329 |
| 2 | 0.154 | 0.246 | 0.305 | 0.233 |
| 4 | 0.085 | 0.135 | 0.170 | 0.134 |
| 8 | 0.045 | 0.066 | 0.086 | 0.066 |

## Detection at 1 mm/h

| metric | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| POD | 0.380 | 0.505 | 0.588 | 0.492 |
| FAR | 0.615 | 0.470 | 0.407 | 0.501 |
| freq_bias | 0.986 | 0.953 | 0.992 | 0.987 |

## Probabilistic (the ensemble's reason to exist)

- **CRPS (fair)**: 0.1373 mm/h. A deterministic forecast's CRPS equals its MAE, so compare against advection 0.252 and persistence 0.318.
- **Spread / RMSE**: 0.909 (spread 0.815, ens-mean RMSE 0.897). Near 1 is well dispersed, below 1 is over-confident.
- **Outlier rate**: 0.276 vs ideal 0.222.
- **Rank-histogram flatness** (RMSE from flat): 0.0589, 0 is perfectly flat.

## FSS (model mean vs a single member)

| threshold | 1 km | 5 km | 11 km | 21 km | 51 km | 101 km |
|---|---|---|---|---|---|---|
| mean, 0.5 mm/h | 0.654 | 0.764 | 0.825 | 0.875 | 0.930 | 0.957 |
| mean, 1 mm/h | 0.582 | 0.710 | 0.783 | 0.843 | 0.912 | 0.948 |
| mean, 2 mm/h | 0.448 | 0.593 | 0.680 | 0.754 | 0.841 | 0.895 |
| mean, 4 mm/h | 0.272 | 0.417 | 0.519 | 0.615 | 0.725 | 0.786 |
| mean, 8 mm/h | 0.125 | 0.211 | 0.279 | 0.343 | 0.405 | 0.430 |
| member, 0.5 mm/h | 0.571 | 0.694 | 0.771 | 0.840 | 0.920 | 0.958 |
| member, 1 mm/h | 0.480 | 0.618 | 0.708 | 0.790 | 0.888 | 0.938 |
| member, 2 mm/h | 0.352 | 0.498 | 0.601 | 0.700 | 0.823 | 0.895 |
| member, 4 mm/h | 0.215 | 0.351 | 0.460 | 0.572 | 0.719 | 0.819 |
| member, 8 mm/h | 0.103 | 0.190 | 0.275 | 0.371 | 0.509 | 0.616 |

## Power spectrum by band (200 clean crops)

Ratio of forecast band power to observed band power, 1.0 = matched. Bands partition the resolved spectrum, so `obs share` sums to 1. The 2-8 km headline is the union of the last two columns; both estimators are given because they disagree materially (see `docs/designs/Metrics_Catalogue.md`).

| field | gt_32km | 16_32km | 8_16km | 4_8km | 2_4km | 2-8 km band power | 2-8 km mean-of-ratios |
|---|---|---|---|---|---|---|---|
| model_member | 1.028 | 0.982 | 1.024 | 1.020 | 0.928 | 0.996 | 0.935 |
| model_mean | 0.784 | 0.298 | 0.212 | 0.190 | 0.155 | 0.181 | 0.160 |
| advection | 0.917 | 0.880 | 0.842 | 0.677 | 0.403 | 0.607 | 0.458 |
| persistence | 0.928 | 0.904 | 0.906 | 0.844 | 0.853 | 0.846 | 0.850 |
| _obs share of variance_ | 89.6% | 6.1% | 2.8% | 1.1% | 0.4% | | |

## Wet-area fraction (>= 0.1 mm/h)

| field | % |
|---|---|
| obs | 22.6 |
| model_mean | 30.7 |
| model_member | 21.7 |
| advection | 21.1 |
| persistence | 22.3 |

See `diffusion_eval.png` for the histogram, PSD, CSI, reliability, rank histogram and spread panels.
