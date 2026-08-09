# Diffusion nowcast scorecard (`val` split)

_13281 crops, 8-member ensembles, 25 Heun steps, guidance 1.0. Checkpoint epoch 50 (val loss 0.3314065786726351). FSS from 403 crops, PSD from 200._

`model_mean` is the ensemble mean; `model_member` is the average skill of a single member. Averaging damps peaks, so the mean scores better on MAE/RMSE and worse at high thresholds. Both are reported, see `docs/Diffusion_Run1_Results.md`.

## Pixel error (lower is better)

| metric | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| MAE (mm/h) | 0.382 | 0.323 | **0.277** | 0.333 |
| RMSE (mm/h) | 1.283 | 1.140 | **0.897** | 1.177 |
| bias (mm/h) | -0.001 | -0.019 | +0.002 | +0.002 |

## CSI by threshold (higher is better)

| mm/h | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| 0.5 | 0.307 | 0.404 | 0.468 | 0.396 |
| 1 | 0.237 | 0.321 | 0.399 | 0.317 |
| 2 | 0.153 | 0.207 | 0.281 | 0.217 |
| 4 | 0.089 | 0.105 | 0.162 | 0.127 |
| 8 | 0.048 | 0.040 | 0.076 | 0.063 |

## Detection at 1 mm/h

| metric | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| POD | 0.383 | 0.477 | 0.584 | 0.483 |
| FAR | 0.616 | 0.504 | 0.441 | 0.520 |
| freq_bias | 0.996 | 0.961 | 1.044 | 1.006 |

## Probabilistic (the ensemble's reason to exist)

- **CRPS (fair)**: 0.1702 mm/h. A deterministic forecast's CRPS equals its MAE, so compare against advection 0.323 and persistence 0.382.
- **Spread / RMSE**: 0.909 (spread 0.815, ens-mean RMSE 0.897). Near 1 is well dispersed, below 1 is over-confident.
- **Outlier rate**: 0.288 vs ideal 0.222.
- **Rank-histogram flatness** (RMSE from flat): 0.0614, 0 is perfectly flat.

## FSS (model mean vs a single member)

| threshold | 1 km | 5 km | 11 km | 21 km | 51 km | 101 km |
|---|---|---|---|---|---|---|
| mean, 0.5 mm/h | 0.645 | 0.745 | 0.802 | 0.852 | 0.910 | 0.942 |
| mean, 1 mm/h | 0.581 | 0.697 | 0.765 | 0.826 | 0.898 | 0.937 |
| mean, 2 mm/h | 0.462 | 0.597 | 0.680 | 0.756 | 0.851 | 0.901 |
| mean, 4 mm/h | 0.330 | 0.470 | 0.558 | 0.646 | 0.759 | 0.818 |
| mean, 8 mm/h | 0.215 | 0.314 | 0.391 | 0.486 | 0.590 | 0.619 |
| member, 0.5 mm/h | 0.577 | 0.684 | 0.754 | 0.820 | 0.902 | 0.948 |
| member, 1 mm/h | 0.492 | 0.615 | 0.698 | 0.777 | 0.877 | 0.933 |
| member, 2 mm/h | 0.375 | 0.510 | 0.604 | 0.698 | 0.826 | 0.902 |
| member, 4 mm/h | 0.261 | 0.399 | 0.501 | 0.610 | 0.763 | 0.855 |
| member, 8 mm/h | 0.168 | 0.273 | 0.376 | 0.514 | 0.706 | 0.791 |

## Power spectrum by band (200 clean crops)

Ratio of forecast band power to observed band power, 1.0 = matched. Bands partition the resolved spectrum, so `obs share` sums to 1. The 2-8 km headline is the union of the last two columns; both estimators are given because they disagree materially (see `docs/designs/Metrics_Catalogue.md`).

| field | gt_32km | 16_32km | 8_16km | 4_8km | 2_4km | 2-8 km band power | 2-8 km mean-of-ratios |
|---|---|---|---|---|---|---|---|
| model_member | 1.068 | 1.165 | 1.145 | 1.101 | 0.912 | 1.049 | 0.946 |
| model_mean | 0.826 | 0.351 | 0.268 | 0.236 | 0.194 | 0.224 | 0.201 |
| advection | 1.014 | 1.160 | 1.046 | 0.851 | 0.480 | 0.749 | 0.559 |
| persistence | 0.926 | 0.980 | 0.983 | 0.994 | 0.943 | 0.980 | 0.953 |
| _obs share of variance_ | 89.0% | 6.0% | 3.2% | 1.3% | 0.5% | | |

## Wet-area fraction (>= 0.1 mm/h)

| field | % |
|---|---|
| obs | 23.1 |
| model_mean | 34.4 |
| model_member | 22.6 |
| advection | 21.6 |
| persistence | 22.9 |

See `diffusion_eval.png` for the histogram, PSD, CSI, reliability, rank histogram and spread panels.
