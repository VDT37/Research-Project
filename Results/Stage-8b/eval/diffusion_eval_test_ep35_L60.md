# Diffusion nowcast scorecard (`test` split)

_13281 crops, 8-member ensembles, 25 Heun steps, guidance 1.0. Checkpoint epoch 35 (val loss 0.33152407111889187). FSS from 403 crops, PSD from 200._

`model_mean` is the ensemble mean; `model_member` is the average skill of a single member. Averaging damps peaks, so the mean scores better on MAE/RMSE and worse at high thresholds. Both are reported, see `docs/Diffusion_Run1_Results.md`.

## Pixel error (lower is better)

| metric | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| MAE (mm/h) | 0.337 | 0.278 | **0.245** | 0.294 |
| RMSE (mm/h) | 1.327 | 1.183 | **0.944** | 1.225 |
| bias (mm/h) | -0.004 | -0.022 | -0.007 | -0.007 |

## CSI by threshold (higher is better)

| mm/h | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| 0.5 | 0.280 | 0.379 | 0.438 | 0.358 |
| 1 | 0.208 | 0.296 | 0.364 | 0.280 |
| 2 | 0.132 | 0.199 | 0.250 | 0.191 |
| 4 | 0.071 | 0.102 | 0.129 | 0.106 |
| 8 | 0.035 | 0.045 | 0.060 | 0.049 |

## Detection at 1 mm/h

| metric | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| POD | 0.341 | 0.443 | 0.524 | 0.434 |
| FAR | 0.653 | 0.528 | 0.457 | 0.558 |
| freq_bias | 0.984 | 0.939 | 0.965 | 0.982 |

## Probabilistic (the ensemble's reason to exist)

- **CRPS (fair)**: 0.1515 mm/h. A deterministic forecast's CRPS equals its MAE, so compare against advection 0.278 and persistence 0.337.
- **Spread / RMSE**: 0.885 (spread 0.835, ens-mean RMSE 0.944). Near 1 is well dispersed, below 1 is over-confident.
- **Outlier rate**: 0.290 vs ideal 0.222.
- **Rank-histogram flatness** (RMSE from flat): 0.0678, 0 is perfectly flat.

## FSS (model mean vs a single member)

| threshold | 1 km | 5 km | 11 km | 21 km | 51 km | 101 km |
|---|---|---|---|---|---|---|
| mean, 0.5 mm/h | 0.604 | 0.712 | 0.774 | 0.830 | 0.897 | 0.933 |
| mean, 1 mm/h | 0.522 | 0.643 | 0.717 | 0.785 | 0.869 | 0.919 |
| mean, 2 mm/h | 0.376 | 0.504 | 0.588 | 0.668 | 0.776 | 0.845 |
| mean, 4 mm/h | 0.206 | 0.319 | 0.404 | 0.492 | 0.605 | 0.674 |
| mean, 8 mm/h | 0.070 | 0.108 | 0.148 | 0.189 | 0.229 | 0.253 |
| member, 0.5 mm/h | 0.517 | 0.630 | 0.706 | 0.780 | 0.878 | 0.932 |
| member, 1 mm/h | 0.422 | 0.546 | 0.632 | 0.719 | 0.837 | 0.904 |
| member, 2 mm/h | 0.298 | 0.422 | 0.515 | 0.612 | 0.755 | 0.849 |
| member, 4 mm/h | 0.164 | 0.267 | 0.355 | 0.452 | 0.599 | 0.719 |
| member, 8 mm/h | 0.072 | 0.127 | 0.185 | 0.251 | 0.351 | 0.471 |

## Power spectrum by band (200 clean crops)

Ratio of forecast band power to observed band power, 1.0 = matched. Bands partition the resolved spectrum, so `obs share` sums to 1. The 2-8 km headline is the union of the last two columns; both estimators are given because they disagree materially (see `docs/designs/Metrics_Catalogue.md`).

| field | gt_32km | 16_32km | 8_16km | 4_8km | 2_4km | 2-8 km band power | 2-8 km mean-of-ratios |
|---|---|---|---|---|---|---|---|
| model_member | 1.004 | 1.055 | 1.066 | 1.128 | 1.066 | 1.112 | 1.070 |
| model_mean | 0.699 | 0.250 | 0.189 | 0.191 | 0.168 | 0.185 | 0.171 |
| advection | 0.891 | 0.904 | 0.833 | 0.718 | 0.437 | 0.646 | 0.496 |
| persistence | 0.908 | 0.905 | 0.869 | 0.859 | 0.854 | 0.858 | 0.859 |
| _obs share of variance_ | 89.7% | 6.0% | 2.9% | 1.0% | 0.4% | | |

## Wet-area fraction (>= 0.1 mm/h)

| field | % |
|---|---|
| obs | 22.6 |
| model_mean | 33.0 |
| model_member | 21.7 |
| advection | 20.8 |
| persistence | 22.3 |

See `diffusion_eval.png` for the histogram, PSD, CSI, reliability, rank histogram and spread panels.
