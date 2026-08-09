# Diffusion nowcast scorecard (`val` split)

_13281 crops, 8-member ensembles, 25 Heun steps, guidance 1.0. Checkpoint epoch 50 (val loss 0.3314065786726351). FSS from 403 crops, PSD from 200._

`model_mean` is the ensemble mean; `model_member` is the average skill of a single member. Averaging damps peaks, so the mean scores better on MAE/RMSE and worse at high thresholds. Both are reported, see `docs/Diffusion_Run1_Results.md`.

## Pixel error (lower is better)

| metric | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| MAE (mm/h) | 0.362 | 0.293 | **0.251** | 0.305 |
| RMSE (mm/h) | 1.251 | 1.081 | **0.856** | 1.126 |
| bias (mm/h) | -0.001 | -0.016 | +0.004 | +0.004 |

## CSI by threshold (higher is better)

| mm/h | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| 0.5 | 0.341 | 0.459 | 0.519 | 0.446 |
| 1 | 0.267 | 0.374 | 0.451 | 0.364 |
| 2 | 0.175 | 0.253 | 0.331 | 0.256 |
| 4 | 0.104 | 0.137 | 0.200 | 0.155 |
| 8 | 0.058 | 0.057 | 0.098 | 0.078 |

## Detection at 1 mm/h

| metric | persistence | advection | model (mean) | model (member) |
|---|---|---|---|---|
| POD | 0.420 | 0.536 | 0.640 | 0.538 |
| FAR | 0.578 | 0.448 | 0.397 | 0.470 |
| freq_bias | 0.995 | 0.970 | 1.061 | 1.014 |

## Probabilistic (the ensemble's reason to exist)

- **CRPS (fair)**: 0.1551 mm/h. A deterministic forecast's CRPS equals its MAE, so compare against advection 0.293 and persistence 0.362.
- **Spread / RMSE**: 0.914 (spread 0.782, ens-mean RMSE 0.856). Near 1 is well dispersed, below 1 is over-confident.
- **Outlier rate**: 0.274 vs ideal 0.222.
- **Rank-histogram flatness** (RMSE from flat): 0.0522, 0 is perfectly flat.

## FSS (model mean vs a single member)

| threshold | 1 km | 5 km | 11 km | 21 km | 51 km | 101 km |
|---|---|---|---|---|---|---|
| mean, 0.5 mm/h | 0.687 | 0.788 | 0.844 | 0.889 | 0.937 | 0.960 |
| mean, 1 mm/h | 0.628 | 0.747 | 0.813 | 0.868 | 0.929 | 0.958 |
| mean, 2 mm/h | 0.519 | 0.663 | 0.745 | 0.816 | 0.896 | 0.936 |
| mean, 4 mm/h | 0.384 | 0.543 | 0.638 | 0.724 | 0.821 | 0.867 |
| mean, 8 mm/h | 0.258 | 0.392 | 0.490 | 0.596 | 0.718 | 0.757 |
| member, 0.5 mm/h | 0.623 | 0.737 | 0.807 | 0.868 | 0.935 | 0.968 |
| member, 1 mm/h | 0.542 | 0.674 | 0.758 | 0.832 | 0.918 | 0.959 |
| member, 2 mm/h | 0.424 | 0.574 | 0.673 | 0.765 | 0.877 | 0.935 |
| member, 4 mm/h | 0.304 | 0.461 | 0.573 | 0.683 | 0.822 | 0.895 |
| member, 8 mm/h | 0.192 | 0.322 | 0.436 | 0.569 | 0.727 | 0.792 |

## Power spectrum by band (200 clean crops)

Ratio of forecast band power to observed band power, 1.0 = matched. Bands partition the resolved spectrum, so `obs share` sums to 1. The 2-8 km headline is the union of the last two columns; both estimators are given because they disagree materially (see `docs/designs/Metrics_Catalogue.md`).

| field | gt_32km | 16_32km | 8_16km | 4_8km | 2_4km | 2-8 km band power | 2-8 km mean-of-ratios |
|---|---|---|---|---|---|---|---|
| model_member | 1.045 | 1.123 | 1.137 | 1.060 | 0.889 | 1.012 | 0.918 |
| model_mean | 0.868 | 0.426 | 0.313 | 0.254 | 0.208 | 0.241 | 0.216 |
| advection | 0.975 | 1.123 | 1.062 | 0.855 | 0.465 | 0.746 | 0.547 |
| persistence | 0.907 | 0.962 | 1.010 | 0.994 | 0.928 | 0.975 | 0.943 |
| _obs share of variance_ | 89.1% | 6.0% | 3.0% | 1.3% | 0.5% | | |

## Wet-area fraction (>= 0.1 mm/h)

| field | % |
|---|---|
| obs | 23.1 |
| model_mean | 32.4 |
| model_member | 22.7 |
| advection | 21.9 |
| persistence | 22.9 |

See `diffusion_eval.png` for the histogram, PSD, CSI, reliability, rank histogram and spread panels.
