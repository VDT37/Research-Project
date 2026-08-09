# Diffusion nowcast scorecard (`val` split)

_13281 crops, 8-member ensembles, 25 Heun steps, guidance 1.0. Checkpoint epoch 50 (val loss 0.3314065786726351). FSS from 403 crops, PSD from 200._

`model_mean` is the ensemble mean; `model_member` is the average skill of a single member. Averaging damps peaks, so the mean scores better on MAE/RMSE and worse at high thresholds. Both are reported, see `docs/Diffusion_Run1_Results.md`.

## Pixel error (lower is better)

| metric | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| MAE (mm/h) | 0.281 | 0.193 | **0.167** | 0.211 |
| RMSE (mm/h) | 1.103 | 0.831 | **0.689** | 0.905 |
| bias (mm/h) | -0.001 | -0.015 | +0.004 | +0.004 |

## CSI by threshold (higher is better)

| mm/h | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| 0.5 | 0.476 | 0.641 | 0.681 | 0.614 |
| 1 | 0.396 | 0.559 | 0.617 | 0.534 |
| 2 | 0.290 | 0.433 | 0.506 | 0.414 |
| 4 | 0.187 | 0.284 | 0.362 | 0.280 |
| 8 | 0.107 | 0.168 | 0.225 | 0.166 |

## Detection at 1 mm/h

| metric | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| POD | 0.567 | 0.708 | 0.785 | 0.702 |
| FAR | 0.432 | 0.273 | 0.258 | 0.310 |
| freq_bias | 0.998 | 0.974 | 1.057 | 1.017 |

## Probabilistic (the ensemble's reason to exist)

- **CRPS (fair)**: 0.1066 mm/h. A deterministic forecast's CRPS equals its MAE, so compare against advection 0.193 and persistence 0.281.
- **Spread / RMSE**: 0.911 (spread 0.628, ens-mean RMSE 0.689). Near 1 is well dispersed, below 1 is over-confident.
- **Outlier rate**: 0.243 vs ideal 0.222.
- **Rank-histogram flatness** (RMSE from flat): 0.0280, 0 is perfectly flat.

## FSS (model mean vs a single member)

| threshold | 1 km | 5 km | 11 km | 21 km | 51 km | 101 km |
|---|---|---|---|---|---|---|
| mean, 0.5 mm/h | 0.808 | 0.909 | 0.946 | 0.968 | 0.985 | 0.991 |
| mean, 1 mm/h | 0.763 | 0.885 | 0.932 | 0.961 | 0.983 | 0.990 |
| mean, 2 mm/h | 0.683 | 0.842 | 0.905 | 0.944 | 0.976 | 0.987 |
| mean, 4 mm/h | 0.564 | 0.766 | 0.852 | 0.908 | 0.952 | 0.968 |
| mean, 8 mm/h | 0.417 | 0.637 | 0.748 | 0.827 | 0.897 | 0.924 |
| member, 0.5 mm/h | 0.759 | 0.882 | 0.933 | 0.963 | 0.987 | 0.994 |
| member, 1 mm/h | 0.696 | 0.846 | 0.910 | 0.951 | 0.982 | 0.992 |
| member, 2 mm/h | 0.596 | 0.785 | 0.870 | 0.926 | 0.970 | 0.985 |
| member, 4 mm/h | 0.468 | 0.695 | 0.809 | 0.887 | 0.951 | 0.973 |
| member, 8 mm/h | 0.336 | 0.571 | 0.709 | 0.814 | 0.906 | 0.938 |

## Power spectrum by band (200 clean crops)

Ratio of forecast band power to observed band power, 1.0 = matched. Bands partition the resolved spectrum, so `obs share` sums to 1. The 2-8 km headline is the union of the last two columns; both estimators are given because they disagree materially (see `docs/designs/Metrics_Catalogue.md`).

| field | gt_32km | 16_32km | 8_16km | 4_8km | 2_4km | 2-8 km band power | 2-8 km mean-of-ratios |
|---|---|---|---|---|---|---|---|
| model_member | 1.040 | 1.025 | 1.016 | 0.970 | 0.887 | 0.947 | 0.895 |
| model_mean | 0.979 | 0.664 | 0.474 | 0.315 | 0.248 | 0.297 | 0.261 |
| advection | 0.955 | 0.923 | 0.876 | 0.729 | 0.425 | 0.647 | 0.487 |
| persistence | 0.986 | 0.977 | 0.991 | 0.985 | 0.959 | 0.978 | 0.962 |
| _obs share of variance_ | 88.3% | 6.4% | 3.3% | 1.4% | 0.5% | | |

## Wet-area fraction (>= 0.1 mm/h)

| field | % |
|---|---|
| obs | 23.0 |
| model_mean | 26.7 |
| model_member | 22.7 |
| advection | 22.3 |
| persistence | 22.9 |

See `diffusion_eval.png` for the histogram, PSD, CSI, reliability, rank histogram and spread panels.
