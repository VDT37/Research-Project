# Diffusion nowcast scorecard (`val` split)

_13281 crops, 8-member ensembles, 25 Heun steps, guidance 1.0. Checkpoint epoch 50 (val loss 0.3314065786726351). FSS from 403 crops, PSD from 200._

`model_mean` is the ensemble mean; `model_member` is the average skill of a single member. Averaging damps peaks, so the mean scores better on MAE/RMSE and worse at high thresholds. Both are reported, see `docs/Diffusion_Run1_Results.md`.

## Pixel error (lower is better)

| metric | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| MAE (mm/h) | 0.333 | 0.253 | **0.217** | 0.268 |
| RMSE (mm/h) | 1.203 | 0.992 | **0.796** | 1.050 |
| bias (mm/h) | -0.001 | -0.015 | +0.006 | +0.006 |

## CSI by threshold (higher is better)

| mm/h | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| 0.5 | 0.388 | 0.532 | 0.585 | 0.513 |
| 1 | 0.311 | 0.446 | 0.518 | 0.430 |
| 2 | 0.213 | 0.320 | 0.400 | 0.314 |
| 4 | 0.129 | 0.186 | 0.259 | 0.197 |
| 8 | 0.072 | 0.088 | 0.139 | 0.105 |

## Detection at 1 mm/h

| metric | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| POD | 0.474 | 0.609 | 0.706 | 0.607 |
| FAR | 0.525 | 0.374 | 0.340 | 0.404 |
| freq_bias | 0.996 | 0.973 | 1.070 | 1.019 |

## Probabilistic (the ensemble's reason to exist)

- **CRPS (fair)**: 0.1354 mm/h. A deterministic forecast's CRPS equals its MAE, so compare against advection 0.253 and persistence 0.333.
- **Spread / RMSE**: 0.920 (spread 0.732, ens-mean RMSE 0.796). Near 1 is well dispersed, below 1 is over-confident.
- **Outlier rate**: 0.258 vs ideal 0.222.
- **Rank-histogram flatness** (RMSE from flat): 0.0414, 0 is perfectly flat.

## FSS (model mean vs a single member)

| threshold | 1 km | 5 km | 11 km | 21 km | 51 km | 101 km |
|---|---|---|---|---|---|---|
| mean, 0.5 mm/h | 0.739 | 0.842 | 0.892 | 0.927 | 0.962 | 0.976 |
| mean, 1 mm/h | 0.684 | 0.806 | 0.867 | 0.912 | 0.956 | 0.975 |
| mean, 2 mm/h | 0.586 | 0.739 | 0.818 | 0.879 | 0.939 | 0.964 |
| mean, 4 mm/h | 0.452 | 0.633 | 0.732 | 0.812 | 0.888 | 0.922 |
| mean, 8 mm/h | 0.297 | 0.461 | 0.573 | 0.683 | 0.794 | 0.829 |
| member, 0.5 mm/h | 0.679 | 0.799 | 0.864 | 0.913 | 0.963 | 0.983 |
| member, 1 mm/h | 0.603 | 0.746 | 0.825 | 0.888 | 0.951 | 0.977 |
| member, 2 mm/h | 0.491 | 0.660 | 0.759 | 0.839 | 0.922 | 0.959 |
| member, 4 mm/h | 0.359 | 0.545 | 0.665 | 0.770 | 0.884 | 0.934 |
| member, 8 mm/h | 0.237 | 0.406 | 0.537 | 0.668 | 0.814 | 0.864 |

## Power spectrum by band (200 clean crops)

Ratio of forecast band power to observed band power, 1.0 = matched. Bands partition the resolved spectrum, so `obs share` sums to 1. The 2-8 km headline is the union of the last two columns; both estimators are given because they disagree materially (see `docs/designs/Metrics_Catalogue.md`).

| field | gt_32km | 16_32km | 8_16km | 4_8km | 2_4km | 2-8 km band power | 2-8 km mean-of-ratios |
|---|---|---|---|---|---|---|---|
| model_member | 1.071 | 1.076 | 1.065 | 1.005 | 0.861 | 0.965 | 0.882 |
| model_mean | 0.943 | 0.535 | 0.358 | 0.275 | 0.236 | 0.265 | 0.243 |
| advection | 0.960 | 1.036 | 0.954 | 0.775 | 0.444 | 0.684 | 0.511 |
| persistence | 0.952 | 1.002 | 0.991 | 0.969 | 0.934 | 0.959 | 0.938 |
| _obs share of variance_ | 88.8% | 6.0% | 3.2% | 1.4% | 0.5% | | |

## Wet-area fraction (>= 0.1 mm/h)

| field | % |
|---|---|
| obs | 23.0 |
| model_mean | 29.9 |
| model_member | 22.8 |
| advection | 22.1 |
| persistence | 22.9 |

See `diffusion_eval.png` for the histogram, PSD, CSI, reliability, rank histogram and spread panels.
