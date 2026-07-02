# Advection Prior Stage — Design Choices, Results & Evaluation

**Project:** Nowcasting using physics-informed diffusion models
**Stage:** Physical "first guess" (advection prior) + residual target — everything before the diffusion model.
**Run reported:** 6-month build on the Exeter GPU server (`mcrugcomp02`), created 2026-06-30.

---

## 1. The idea in one line

We split the forecast into two parts:

> **future rain (y)  =  advection prior (A)  +  residual (r)**

- **A** = a classical, *non-learned* "rain moves with the wind" forecast (pySTEPS). It is interpretable physics and handles the easy part.
- **r = y − A** = what advection *can't* do: growth, decay, new storms, fine detail. **This is the only thing the diffusion model has to learn.**

This is the residual idea from CorrDiff / DiffCast. Learning a small, well-behaved residual is easier than learning the whole rain field from scratch.

---

## 2. Reproducibility log (this run)


| Item | Setting |
|---|---|
| **Data** | Met Office UK 1 km rain-rate radar composite (ODIM HDF5), 15-min cadence, public AWS S3 bucket `met-office-radar-obs-data` |
| **Data period** | 2024-11-21 → 2025-04-30 (≈6 months). Test year 2026 held out (not in this run). |
| **Preprocessing** | mask `nodata` → NaN (never counted as dry); `undetect` → 0 (genuine dry); cap at 128 mm/h; log-transform to **dBR** for motion/advection |
| **Split** | Validation = **first 2 days of each month** (first 2 *available* if 1st/2nd missing); Train = all other days; Test = 2026. → **train 59,335 / val 4,858 crops** |
| **Domain** | 256×256 km crops @ 1 km, advected on a 384×384 context (64-px margin) so inflow is handled; keep crops ≥90% in-range and ≥5% wet |
| **Conditioning / target** | 4 past frames @ 15 min (last hour) → predict the **single frame at +60 min** |
| **Advection** | pySTEPS dense **Lucas–Kanade** optical flow + backward **semi-Lagrangian** extrapolation (Germann & Zawadzki); 4 steps, keep the +60 min frame |
| **Residual** | `r = dB(y) − A_dBR`, computed in dBR (log) space |
| **Metrics** | MAE, MSE/RMSE, bias; CSI/POD/FAR/frequency-bias at 0.5/1/2/4/8 mm/h; distribution analysis; vs **persistence** baseline |
| **Compute** | Exeter `mcrugcomp02`, 64-core EPYC, parallel; download-once-to-/scratch then advect across all cores; idempotent/resumable |
| **Result** | **64,193 crops**, advection-only MAE **0.30 mm/h**, CSI@1 mm/h **0.33** |

Everything is config-driven in one block, so any of these can be changed in one place and re-run.

---

## 3. Design choices (what, why, and the alternatives)

Each choice is recorded with the alternative we did **not** take, so the reasoning is on the record.

1. **Pull data with `boto3` (download), not `s3fs` (stream).**
   We read the *whole* of every radar file, so streaming gives nothing and is slow (each read is a network round-trip, repeated every epoch). We download once to fast local `/scratch` and reuse. *Alternative:* `s3fs` streaming — rejected as too slow for repeated training.

2. **Mask `nodata` before any maths.**
   Out-of-range pixels are unknown, not dry. If we treated them as 0 the means/skill scores would be wrong. *Alternative:* fill with 0 — rejected (biases everything).

3. **256×256 crops, not the whole UK.**
   The full map (2175×1725) is far too big for a diffusion model. 256 km tiles keep it tractable. *Alternative:* full domain (too heavy) or coarsening to 2 km (loses the convective detail we care about).

4. **Advect on a 384×384 context, then centre-crop to 256.**
   Rain blows *into* the 256 box from outside; without a margin the upwind edge would be wrong. 64 px covers ~15 m/s storms over an hour. *Alternative:* advect the 256 box directly — rejected (edge errors).

5. **4 input frames → single +60 min target.**
   One hour of history is enough (DGMR found 4 frames sufficient). We start with one lead time to keep the first experiment simple. *Alternative:* predict the whole +15…+60 sequence — deferred to later.

6. **Log-transform to dBR for motion + residual.**
   Rain is extremely skewed (mostly zero, rare big values). Optical flow and diffusion both behave much better on the log scale, and the residual is closer to a nice bell shape. *Alternative:* raw mm/h — rejected (a few extreme pixels dominate).

7. **Lucas–Kanade optical flow + semi-Lagrangian advection (pySTEPS).**
   LK is fast and the recommended default; the semi-Lagrangian "trace-back, interpolate once" scheme barely blurs the field (low numerical diffusion). *Alternatives:* VET (more accurate, much slower) and DARTS (spectral) — kept as a later sensitivity check.

8. **Learn the residual `r`, not the whole field.**
   A small, near-zero-mean target trains faster and sharper than the full field. *Alternative:* model the full field directly (NowcastNet/DGMR style) — rejected for compute and the "too much freedom → blurry/displaced" problem in the notes.

9. **Validation = first 2 days of each month; Test = the next year (2026).**
   Leakage-safe and matches DGMR's philosophy; judging on the *target* date stops a training sample from peeking at a validation day. *Alternative:* random split — rejected (leakage between nearby times).

10. **Filter out near-empty crops (≥5% wet).**
    ~89% of the UK map is dry; without this the cache fills with empty patches. This only *excludes* near-empty crops — it does **not** re-weight the distribution (true importance sampling stays an ablation). *Alternative:* keep everything (wasteful) or full importance sampling (changes the distribution — later).

---

## 4. Results and what they mean

### 4.1 Advection-only baseline (6 months, all crops)

| Metric | Value |
|---|---|
| MAE | **0.30 mm/h** |
| CSI @ 0.5 mm/h | 0.415 |
| CSI @ 1 mm/h | 0.327 |
| CSI @ 2 mm/h | 0.216 |
| CSI @ 4 mm/h | 0.108 |
| CSI @ 8 mm/h | 0.036 |

**Reading it:** CSI (0 = useless, 1 = perfect) falls as rain gets heavier. That's expected and important — advection places light, organised rain reasonably well, but **cannot put a heavy convective cell in the exact right spot an hour ahead, and cannot grow or decay it.** That high-threshold gap (CSI 0.04 at 8 mm/h) is precisely the headroom the diffusion model is meant to fill.

### 4.2 The residual (the diffusion model's target)

From `check_priors.py` over 800 sampled crops:

| Quantity | Value | Meaning |
|---|---|---|
| residual mean | **+0.18 dBR** | ≈ 0 → advection is a roughly **unbiased** first guess (just slightly under, because it can't grow rain) |
| residual std | **5.17 dBR** | a real, sizeable spread → there **is** growth/decay/detail for the model to learn (not zero, so advection alone isn't enough) |
| target wet fraction | 27.2% | the rain filter worked — these are genuinely rainy crops |
| target max rain (avg) | 21.7 mm/h | a healthy range of intensities |

This is the key justification for the whole project in two numbers: residual mean ≈ 0 (good prior) but residual std ≈ 5 dBR (plenty left to learn).

### 4.3 The picture (`prior_check.png`)

Four example cases, columns = last input · target (+60) · advection prior · residual:

- The **advection prior is clearly the input *moved***, not held still — the motion estimation works.
- The **prior is smoother than the target**, most visibly in the convective/cellular case — advection loses small-scale detail and can't create new cells.
- The **residual is coherent and structured** (red = growth where reality > advection, blue = decay) — strongest exactly where advection struggles. It is *not* noise and *not* a copy of the target — it is learnable structure.
- Note: some panels show radar "spoke/wedge" artifacts that get advected along too — a data-quality caveat to keep in mind.

---

## 5. Quantitative + distribution evaluation (supervisor's request)

> *"Evaluate the advection prior quantitatively, not only through mean motion speed. Use MAE/MSE and rainfall-threshold metrics… compare against persistence and advection baselines on the same validation set."*

Implemented in **`evaluate_advection.py`**. Run it on the server:

```bash
conda activate nowcast
python evaluate_advection.py            # validation split
```
It produces `advection_eval.md` (paste-ready tables), `advection_eval.json`, and `advection_eval.png`. It reports, **on the validation set**, for **advection vs persistence**:

- **Pixel error:** MAE, **MSE/RMSE**, bias.
- **Threshold skill:** POD (hit rate), FAR (false-alarm ratio), CSI, frequency bias at 0.5/1/2/4/8 mm/h.
- **Persistence baseline** = hold the last frame still for 60 min (the simplest possible nowcast). Advection should beat it — that gap is the value the wind/motion adds.

### Results — validation set (13,281 crops; full range 2024-11-21 → 2025-12-31)

**Advection beats persistence on almost every measure**, confirming the motion estimation adds real skill:

| Pixel error (lower = better) | persistence | advection |
|---|---|---|
| MAE (mm/h) | 0.382 | **0.323**  (−15%) |
| RMSE (mm/h) | 1.283 | **1.140**  (−11%) |
| bias (mm/h) | −0.001 | −0.019 |

| CSI (higher = better) | persistence | advection |
|---|---|---|
| ≥0.5 mm/h | 0.307 | **0.404** |
| ≥1 mm/h | 0.237 | **0.321**  (+35%) |
| ≥2 mm/h | 0.153 | **0.207** |
| ≥4 mm/h | 0.089 | **0.105** |
| ≥8 mm/h | **0.048** | 0.040 |

| Detection @ 1 mm/h | persistence | advection | ideal |
|---|---|---|---|
| POD (hit rate) ↑ | 0.383 | **0.477** | 1 |
| FAR (false-alarm ratio) ↓ | 0.616 | **0.504** | 0 |
| frequency bias | 0.996 | 0.961 | 1 |

At +60 min, advection catches **~48% of rain pixels** (vs 38% for persistence), with **fewer false alarms** and **~15% lower MAE** — the stronger baseline the diffusion model must beat.

**One important exception — heavy rain (≥8 mm/h): persistence now edges advection (0.048 vs 0.040).** This is a finding to *highlight*, not a flaw. Heavy rain lives in small, short-lived convective cells; advection **smooths** them (the semi-Lagrangian interpolation), so their peaks slip *below* 8 mm/h and go undetected, while persistence keeps each cell at full, un-smoothed intensity in its original spot. You can see the cause directly in the histogram/PSD below — advection's heavy tail and small-scale power fall off. **This is precisely the gap the diffusion model exists to close: putting the heavy-rain intensity back.** (In the winter-only 6-month subset advection still won at 8 mm/h; adding summer 2025's convection is what exposed this — a more complete, honest picture.)

> **On sample size (answering a fair question):** every number in the tables above is computed over **all 13,281 validation crops** — not a sample. Only the four **distribution plots** below sub-sample **2,000** crops, because pooling every pixel of 13k crops (~870 million) into a histogram is unnecessary — 2,000 crops (~130 million pixels) already give smooth, stable curves. That is why "2,000" is the same for every run size: it is a fixed *plotting* sample (`--sample`), not a limit on the metrics.

### What is "distribution analysis"? (in simple terms)

Skill scores ask *"is the rain in the right place?"*. **Distribution analysis asks a different question: *"does the forecast have the same statistical shape as reality?"*** — even if every pixel isn't perfect. Three checks (all in `advection_eval.png`):

1. **Rain-rate histogram** — does the forecast produce the same mix of light / moderate / heavy rain as the observations? Advection **loses the heaviest values** (it smooths), so its histogram sits below observations at the high end.
2. **Power spectrum (PSD)** — how much detail lives at each spatial scale (big systems vs small cells). Advection matches observations at large scales and **drops at the smallest scales** (the smoothing again). The classic pySTEPS check.
3. **Residual distribution** — the shape of `r` (a sharp spike at 0 with growth/decay tails). This tells us how to set up and normalise the diffusion model.

Plus the **wet-area fraction** (how much of the map is raining) for observed vs advection vs persistence.

### Distribution analysis — results (`advection_eval.png`)

| Wet-area fraction (≥0.1 mm/h) | obs | advection | persistence |
|---|---|---|---|
| | 23.5% | 21.8% | 22.9% |

The four panels tell one consistent story:

1. **Rain-rate histogram.** Observation and persistence overlap almost perfectly. **Advection matches them up to ~5–10 mm/h, then falls below at the heavy end** — it under-produces the most intense rain. That is the semi-Lagrangian smoothing (and it is what costs advection the ≥8 mm/h CSI above).
2. **Power spectrum (PSD).** Observation and persistence overlap; **advection tracks them at large/medium scales but loses power below ~10 km** — the same smoothing, in spectral form.
3. **Residual `r = y − A` (dBR).** A tall spike at 0 (the mostly-dry map, where advection was already right) with **symmetric tails** for growth (positive) and decay (negative). The target is **sparse and near-zero-mean** — ideal for a diffusion model, and it confirms the dBR/normalisation plan.
4. **Wet-area.** Advection rains over slightly *less* area than reality (21.8% vs 23.5%) — it drops some light/edge rain; persistence keeps almost the exact observed amount (it *is* a real frame).

**The key insight — and the whole reason for the diffusion model:** persistence has the *right distribution and spectrum* (it's a real radar frame) but in the *wrong place*, which is why it loses on CSI/POD at most thresholds. Advection fixes the **placement** (higher CSI/POD, lower FAR) but pays a price in **sharpness** — and at the heaviest rain that price is now large enough that advection actually *loses* to persistence. **The diffusion model's job is to put that lost sharpness and those heavy-rain extremes *back*, on top of advection's better placement** — getting distribution *and* location right at once. It also shows why we report both kinds of metric: they measure different things, and a method can win one while losing the other.

---

## 6. Comparison of the three runs (small → reliable → scaled)

This is exactly the "start small, then scale" path the supervisor asked for.

| Run | Where | Window | Crops | MAE (mm/h) | CSI@1 | Purpose / what we learned |
|---|---|---|---|---|---|---|
| **1. Prototype** | Google Colab | ~2 weeks (Nov 2024)* | ~9,100 | — † | — † | Proved the pipeline end-to-end; found Colab's storage was unreliable → moved to the server |
| **2. Smoke test** | mcrugcomp02 | 3 days | 3,172 | 0.392 | 0.385 | Verified the server pipeline + metrics on a tiny, fast slice before committing |
| **3. Scaled run** | mcrugcomp02 | 6 months | **64,193** | **0.301** | **0.327** | The reliable baseline over many weather types |

\* exact Colab window as recorded; adjust if yours differed.
† the Colab baseline didn't compute (a variable-scope bug, since fixed) — the numbers above come from the server runs.

**Why the numbers differ between the 3-day and 6-month runs (this is a feature, not a bug):**
- **MAE dropped (0.39 → 0.30):** the 3-day window was a single wet, frontal spell; the 6 months include many drier winter days, so the *average* absolute error is lower.
- **CSI dropped slightly (0.385 → 0.327 @ 1 mm/h):** 6 months mix in harder cases (showers, convective initiation) that advection handles less well, whereas the 3-day frontal spell was "easy" (organised rain that just drifts).
- **Takeaway:** a 3-day score can flatter or punish the method depending on that week's weather. **The 6-month number is the honest, representative baseline** — which is the whole reason for scaling up.

---

## 7. How to reproduce

```bash
# on mcrugcomp02, in the nowcast conda env
python build_advection_prior.py --check                       # S3 reachable?
python build_advection_prior.py --start 2024-11-21 --end 2025-04-30   # build (tmux)
python clean_priors.py                                        # remove any corrupt crops
python build_advection_prior.py --baseline-only               # baseline + manifest
python check_priors.py                                        # integrity + montage
python evaluate_advection.py                                  # MAE/MSE + thresholds + distributions vs persistence
```

---

## 8. Next steps

**This week**
1. **Scale to all available samples** — extend the window to the full train range (2024-11-21 → 2025-12-31) and re-run; re-evaluate on the same validation set.
2. Record the final advection + persistence baselines (these are fixed reference numbers for the rest of the project).

**Then, latent diffusion model (LDM) hyperparameters.**: a **coarse grid first**, comparing every run against **persistence and the advection baseline on the same validation set**, then move to smarter search (random / Bayesian / Optuna). Initial coarse grid (to refine):

| Hyperparameter | Starting points to scan |
|---|---|
| Learning rate | 1e-4, 2e-4, 5e-4 |
| Batch size | 16, 32, 64 (as GPU memory allows on the L4) |
| Latent dimension (VAE) | 3, 4, 8 channels |
| Diffusion steps | 1000 train; 25/50/100 sampling (DDIM) |
| Noise schedule | linear vs cosine |
| UNet depth / width | 2–3 down/up levels; base width 64/128 |
| Dropout | 0.0, 0.1 |
| Weight decay | 0, 1e-4 |
| Conditioning method | concatenate [past frames | advection prior A] in latent space (vs cross-attention) |

Diffusion models are expensive, so we **tune few parameters at a time** (start with learning rate + batch size + conditioning), keep everything else fixed, and always report against the two baselines on the validation set.
