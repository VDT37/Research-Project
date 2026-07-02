# Metrics & analysis catalogue

Every metric and distribution analysis used in this dissertation, **what it means**
and **why we use it** — in three parts: (1) the advection prior, (2) the VAE,
(3) the diffusion model. No result values here; this is the "why" reference.

A guiding principle throughout (per the supervisor): **every model is scored on the
same validation set, always next to two baselines — persistence and advection** —
and we never rely on a single number, because each metric only captures one property
of a forecast.

---

## Part 1 — Advection prior (the physical baseline)

### Pixel-wise error (how close, on average)
- **MAE (Mean Absolute Error)** — average of |forecast − observed| in mm/h.
  *Why:* the simplest, most interpretable "typical error", robust to a few extreme
  pixels. Standard in pySTEPS and DGMR.
- **MSE / RMSE (root mean squared error)** — average of the *squared* error (RMSE is
  its square root, back in mm/h). *Why:* squaring punishes big misses much harder
  than MAE, so RMSE is sensitive to getting **heavy rain** wrong. We report both
  because they disagree in an informative way.
- **Bias (mean error)** — average of (forecast − observed). *Why:* reveals
  **systematic** over- or under-forecasting (a model can have low MAE but a
  persistent wet/dry lean).

### Threshold / categorical skill (right rain, right place?)
These binarise the field at a rain threshold (0.5, 1, 2, 4, 8 mm/h) and count
**hits (H)**, **misses (M)**, **false alarms (F)**.
- **CSI (Critical Success Index)** = H / (H + M + F). *Why:* **the** standard
  nowcasting location-accuracy score. Computing it per threshold shows how skill
  falls as rain gets heavier — the core weakness of advection.
- **POD (Probability of Detection / hit rate)** = H / (H + M). *Why:* of the rain
  that really happened, how much did we catch.
- **FAR (False Alarm Ratio)** = F / (H + F). *Why:* of the rain we forecast, how
  much was wrong. POD and FAR together explain *why* a CSI is high or low.
- **Frequency bias** = (H + F) / (H + M) = forecast rain-area ÷ observed rain-area.
  *Why:* are we raining over too much area (>1) or too little (<1)?

### Spatial-scale skill
- **FSS (Fractions Skill Score)** across neighbourhood sizes. *Why:* pixel CSI
  unfairly zeroes a forecast that is right but shifted a few pixels. FSS asks whether
  the **right fraction of rain** is in the right **neighbourhood**, and reveals the
  **smallest scale at which the forecast is useful** for each intensity. Used by both
  the pySTEPS paper and DGMR (its "pooled" scores). (See `Advection_Analysis_Extended.md`.)

### Distribution analysis (right statistical *shape*?)
Skill scores can't tell you if the forecast has realistic *texture*. These do:
- **Rain-rate histogram (marginal distribution)** — the mix of light/moderate/heavy
  rain. *Why:* checks whether the forecast **preserves the heavy-rain tail** or
  smooths it away (advection does the latter).
- **Power spectrum (PSD)** — how much variance sits at each spatial scale. *Why:* a
  direct measure of **blurriness**; the classic pySTEPS "did the scheme keep the
  small scales?" check.
- **Residual distribution** — the shape of `r = y − A`. *Why:* it *is* the diffusion
  model's target; its spread/shape tells us how to normalise and set up that model.
- **Wet-area fraction** — % of the map raining. *Why:* a forecast can have the right
  average intensity but the wrong *amount* of rain area.

### Reference forecast
- **Persistence baseline** — hold the last frame still for 60 min. *Why:* the
  simplest possible nowcast; the **floor** any method (including advection) must beat.

---

## Part 2 — VAE (the compressor for latent diffusion)

The VAE is judged on one thing: **can it shrink a rain field and rebuild it without
losing what matters?** A weak VAE silently caps the final nowcast.

- **Reconstruction loss (masked L1)** — average |reconstruction − input| over valid
  pixels. *Why:* the primary fidelity measure; **L1 (not L2)** keeps sharp features
  because L2 rewards blurring toward a safe average.
- **Validation reconstruction loss** — the same on held-out data. *Why:* used to
  **select the best checkpoint** and to catch overfitting.
- **KL divergence** — how far the latent space is from a standard bell curve.
  *Why:* monitored (not hard-minimised) to keep the latent space **regular and
  well-scaled** so the diffusion model can operate in it. (We keep its weight tiny —
  fidelity first.)
- **Distribution-preservation check (histogram + PSD of reconstruction vs observed)**
  — *Why:* **the crucial test.** A low reconstruction number can still hide blurred
  extremes; overlaying the reconstruction's rain-rate histogram and power spectrum on
  the observations proves the VAE keeps the heavy rain and the fine detail. If these
  don't overlap, the VAE is not good enough.
- **Latent statistics (std → latent scale)** — *Why:* ensures the latents are scaled
  to ~unit variance so the diffusion model trains stably.

**What we may add:** **SSIM** (structural similarity — a perceptual-ish structure
score) and reconstruction **CSI/PSD at thresholds**, to confirm heavy-rain cells
specifically survive compression. (We avoid LPIPS-style perceptual losses — there is
no radar-pretrained perceptual network, a point PreDiff also makes.)

---

## Part 3 — Diffusion model (the final probabilistic nowcast)

The final nowcast is `ŷ = A + r̂`, and it is a **probabilistic ensemble** (many
samples). We evaluate it two ways.

### Deterministic (the ensemble mean) — reuse Part 1, for direct comparison
- **MAE, RMSE, CSI (all thresholds), POD, FAR, FSS** — computed exactly as for
  advection, on the same validation crops. *Why:* to show, apples-to-apples, that the
  model **beats persistence and advection**, especially the **heavy-rain CSI (≥8
  mm/h)** where advection currently loses to persistence.

### Sharpness / structure — did it fix advection's weakness?
- **Power spectrum (PSD)** and **rain-rate histogram** vs observations. *Why:* the
  central hypothesis is that diffusion **restores the small-scale power and heavy-rain
  tail** that advection smoothed away. These plots test it directly.

### Probabilistic skill — the LDM's real selling point
Because the model outputs an **ensemble**, we can measure forecast *uncertainty*:
- **CRPS (Continuous Ranked Probability Score)** — how well the whole predicted
  distribution matches the truth (a probabilistic generalisation of MAE). *Why:* the
  headline probabilistic accuracy score; used by DGMR.
- **Reliability diagram** — forecast probability vs observed frequency. On the
  diagonal = **calibrated**. *Why:* are the model's probabilities **trustworthy**
  (when it says 30% chance of rain, does it rain 30% of the time)?
- **Rank histogram (Talagrand)** — where the truth falls among the sorted ensemble
  members. Flat = well-dispersed. *Why:* is the ensemble **spread** right, or is it
  over-confident (U-shaped) / under-confident (dome)?
- **Spread–skill ratio** and **outlier percentage** — *Why:* further checks that the
  ensemble's spread matches its actual error (a reliable ensemble has spread ≈ error).

### Baselines & protocol
Every diffusion run is reported **next to persistence and advection on the same
validation set**, and — following the supervisor — we tune **a few hyperparameters at
a time** and log each run's settings + metrics so results stay reproducible and
comparable. (Search plan and hyperparameters: `docs/LDM_Design_and_Hyperparameters.md`.)

---

### One-line summary of the philosophy
*Pixel scores (MAE/RMSE) say how close; categorical scores (CSI/POD/FAR) say if the
rain is in the right place; FSS says at what scale that's true; distribution scores
(histogram/PSD) say if it looks realistic; and — for the diffusion model —
probabilistic scores (CRPS, reliability, rank histogram) say if the uncertainty is
honest. No single one is enough, so we report the suite.*
