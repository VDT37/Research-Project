# Advection prior — validation scorecard (`val` split)

_4858 crops; distributions from 2000 sampled crops._

## Pixel error (lower = better)

| metric | persistence | advection | advection better? |
|---|---|---|---|
| MAE (mm/h) | 0.345 | 0.293 | yes |
| RMSE (mm/h) | 1.034 | 0.911 | yes |
| bias (mm/h) | +0.002 | -0.012 | |

## Threshold skill — CSI (higher = better)

| threshold (mm/h) | persistence | advection |
|---|---|---|
| 0.5 | 0.347 | 0.428 |
| 1.0 | 0.269 | 0.343 |
| 2.0 | 0.156 | 0.210 |
| 4.0 | 0.069 | 0.099 |
| 8.0 | 0.020 | 0.030 |

## Detection at 1 mm/h (POD up, FAR down, bias~1 ideal)

| metric | persistence | advection |
|---|---|---|
| POD | 0.424 | 0.505 |
| FAR | 0.577 | 0.484 |
| freq_bias | 1.004 | 0.979 |

## Distribution analysis

- Wet-area fraction: obs 22.8% · advection 21.9% · persistence 22.8%
- See `advection_eval.png` for the rain-rate histogram, power spectrum and residual distribution.
