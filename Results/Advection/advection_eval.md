# Advection prior — validation scorecard (`val` split)

_13281 crops; distributions from 2000 sampled crops._

## Pixel error (lower = better)

| metric | persistence | advection | advection better? |
|---|---|---|---|
| MAE (mm/h) | 0.382 | 0.323 | yes |
| RMSE (mm/h) | 1.283 | 1.140 | yes |
| bias (mm/h) | -0.001 | -0.019 | |

## Threshold skill — CSI (higher = better)

| threshold (mm/h) | persistence | advection |
|---|---|---|
| 0.5 | 0.307 | 0.404 |
| 1.0 | 0.237 | 0.321 |
| 2.0 | 0.153 | 0.207 |
| 4.0 | 0.089 | 0.105 |
| 8.0 | 0.048 | 0.040 |

## Detection at 1 mm/h (POD up, FAR down, bias~1 ideal)

| metric | persistence | advection |
|---|---|---|
| POD | 0.383 | 0.477 |
| FAR | 0.616 | 0.504 |
| freq_bias | 0.996 | 0.961 |

## Distribution analysis

- Wet-area fraction: obs 23.5% · advection 21.8% · persistence 22.9%
- See `advection_eval.png` for the rain-rate histogram, power spectrum and residual distribution.
