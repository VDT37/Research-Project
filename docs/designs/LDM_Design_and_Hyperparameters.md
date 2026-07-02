# Latent Diffusion Model — Design & Hyperparameter Plan

**Project:** Nowcasting using physics-informed diffusion models.
**This stage:** learn the **residual** `r = y − A` that the advection prior can't produce, with a conditional **latent diffusion model (LDM)**. Final nowcast = `A + r̂`.
**Status:** advection prior built + verified (64,193 crops, baseline MAE 0.30 mm/h, CSI@1 = 0.33). This document plans the model _before_ writing it, per the supervisor's "reproducible, log everything, start small" guidance.

---

## 1. The pipeline

```
 past 4 frames ─┐
                ├─VAE encoder─► latents ─┐
 advection A ───┘                        ├─ concat ─► EDM denoiser (UNet) ─► residual latent r̂
                          noisy target ──┘                                         │
                                                                z_y = z_A + r̂  ─► VAE decoder ─► ŷ (dBR) ─► mm/h
```

1. **VAE** compresses each 256×256 dBR field to a small latent (keeps it within the L4's 24 GB).
2. **EDM diffusion in latent space** generates the **residual latent**, conditioned on the encoded past frames + advection prior `A` (concatenated as channels).
3. **Reconstruct** `ẑ_y = z_A + r̂` → decode → ŷ in dBR → invert to mm/h.
4. **Calibrate** (optional, pySTEPS-style) and **evaluate** vs persistence + advection on the validation set.

---

## 2. Decided design choices (with alternatives)

| #   | Choice                  | Decided                                                       | Why                                                                                                                         | Alternative (not taken)                                                                                |
| --- | ----------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 1   | **VAE source**          | Train our **own small VAE** on dBR radar crops                | Off-the-shelf VAEs reconstruct radar poorly (Score-based note); we pick ours by reconstruction loss                         | Pretrained SD-VAE (likely poor); no-VAE pixel diffusion (heavier)                                      |
| 2   | **Diffusion framework** | **EDM** (Karras et al. 2022)                                  | Better samples, far fewer sampling steps, clean noise scaling; CorrDiff uses it                                             | DDPM (simpler but more steps)                                                                          |
| 3   | **Conditioning**        | **Concatenate** encoded conditioning as extra latent channels | Standard, cheap, enough for a strong physical prior like `A`                                                                | Cross-attention / SPADE (heavier, more to tune)                                                        |
| 4   | **What we predict**     | The **residual** (latent `r = z_y − z_A`)                     | The thesis: advection does the bulk, the model only learns growth/decay/detail (small, ~zero-mean → trains faster, sharper) | Predict full `z_y` conditioned on `z_A` (kept as a fallback if latent-residual reconstruction is poor) |

## 2b. Sub-choices I'm defaulting (tunable)

| Item               | Default                                                                                | Why / alternative                                                                                                                         |
| ------------------ | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Compression factor | **×4** (256→64)                                                                        | Score-based paper got good reconstruction + ~5× speedup at ×4; ×8 (→32) risks losing convective detail. (In the tuning grid.)             |
| Latent channels    | **4**                                                                                  | SD-style; 3 (smaller) / 8 (richer) are in the grid                                                                                        |
| VAE losses         | **L1 reconstruction + small KL**                                                       | No perceptual loss — no pretrained perceptual net for radar (PreDiff dropped it too). Optional light adversarial term later for sharpness |
| Backbone           | **2D UNet**, 3 down/up levels, base width 64–128, self-attention at the coarsest level | Single target frame → 2D is enough; 3D/Earthformer is overkill here                                                                       |
| Sampler            | **EDM Heun**, ~18–30 steps                                                             | Karras 2nd-order; far fewer steps than DDPM's hundreds                                                                                    |
| Weight averaging   | **EMA** (decay ~0.999)                                                                 | Standard for stable diffusion samples                                                                                                     |
| Precision          | **bf16/fp16 mixed**                                                                    | Fits the L4, faster                                                                                                                       |

---

## 3. The two models

### 3.1 VAE (trained first, then frozen)

- **Input:** one 256×256 dBR field (normalised — see §4). **Output:** reconstruction.
- **Encoder:** conv stem → 2 downsampling residual blocks (256→128→64) → conv to `4×64×64` latent (mean+logvar).
- **Decoder:** mirror, 64→128→256.
- **Loss:** `L1(recon, input)` on valid pixels + `β·KL` (β small, e.g. 1e-4 to 1e-6 → near-AE, keeps latents usable for the residual sum).
- **Selection:** lowest reconstruction loss on the **val** split; **also check it preserves the rain-rate histogram and PSD** (a VAE that blurs would defeat the point). This reuses the distribution analysis from `evaluate_advection.py`.

### 3.2 EDM latent diffusion (conditional)

- **Target:** residual latent `z_r = z_y − z_A`.
- **Condition (concatenated channels):** `z_A` and the four encoded past frames `z_{t-45..t0}` → 5 latents.
- **UNet input channels:** `C` (noisy `z_r`) + `5·C` (condition).
- **EDM:** Karras preconditioning (`c_skip/c_out/c_in/c_noise`), `σ_data` = std of the normalised latents, training noise `ln σ ~ N(P_mean=−1.2, P_std=1.2)`, sampling `σ∈[0.002, 80]`, `ρ=7`.
- **At inference:** sample `r̂` → `ẑ_y = z_A + r̂` → decode → ŷ (dBR) → invert to mm/h → **ensemble** by sampling N times.

---

## 4. Data handling & normalisation

- **Source:** the cached prior crops (`x_mmh`, `A_mmh`/`A_dbr`, `y_mmh`, `r_dbr`, `valid`).
- **Work in dBR**, normalised to ~zero-mean/unit-var (Score-based note: "scale to mean 0, variance 1" — critical for diffusion). **Compute dBR mean/std over the train split once** and store them (reproducibility).
- **Masking:** VAE reconstruction loss is computed on `valid` pixels only; `nodata` filled with the dry value before encoding.
- **Split:** the existing leakage-safe split (val = first 2 days/month; test = 2026). VAE and diffusion both train on **train**, are selected on **val**, never touch **test**.

---

## 5. Hyperparameters & tuning plan (supervisor's request)

Coarse grid first on the **few** most impactful parameters; everything else fixed at the defaults above. Then move to smarter search (random → Bayesian / Optuna with Hyperband early-stopping). Diffusion is expensive, so **tune a few at a time** and **compare every run against persistence + advection on the same val set**.

| Hyperparameter           | Coarse grid                                           | Tune in phase      |
| ------------------------ | ----------------------------------------------------- | ------------------ |
| Learning rate            | 1e-4, 2e-4, 5e-4                                      | **1 (first)**      |
| Batch size               | 16, 32, 64 (L4 memory permitting)                     | **1**              |
| Conditioning detail      | A-only vs A + past frames                             | **1**              |
| Latent dim / compression | 3 / 4 / 8 ch; ×4 vs ×8                                | 2                  |
| Sampling steps           | 18, 25, 50 (Heun)                                     | 2 (inference only) |
| Noise schedule           | EDM σ (default); linear vs cosine for a DDPM ablation | 2                  |
| UNet depth / width       | 2–3 levels; base 64 vs 128                            | 3                  |
| Dropout                  | 0.0, 0.1                                              | 3                  |
| Weight decay             | 0, 1e-4                                               | 3                  |

**Optimiser:** AdamW. **Schedule:** warm-up → cosine. **EMA** on. **Seeds** fixed and logged.

**Search method:** Phase 1 = manual coarse grid (lr × batch × conditioning). Phase 2+ = Optuna (TPE) with Hyperband early-stopping on the val metric, capped to a small budget given training cost.

---

## 6. Experimental protocol & reproducibility (per run)

For **every** run, log a small config + results record (one JSON/row), exactly as the supervisor asked:

- **data period** (start/end, split), **preprocessing** (dBR, normalisation stats), **counts** (train/val crops);
- **VAE settings** (compression, channels, β, recon loss) and its val reconstruction;
- **diffusion settings** (EDM params, UNet config, lr, batch, steps, dropout, wd, EMA, seed);
- **metrics on val** (below) **alongside persistence + advection** for the same crops;
- wall-clock + GPU, and the git commit.

This becomes a "runs table" so any result is reproducible and comparable.

## 7. Metrics (full suite, vs baselines)

- **Deterministic / ensemble-mean:** MAE, MSE/RMSE, CSI/POD/FAR/bias at 0.5/1/2/4/8 mm/h (the `evaluate_advection.py` machinery, extended to the model).
- **Sharpness / structure:** radially-averaged **PSD** vs observations (does the LDM restore the small-scale power advection lost?).
- **Probabilistic (the LDM's selling point):** **reliability diagram**, **rank histogram**, CRPS — over an ensemble of samples.
- Every number reported next to **persistence** and **advection** on the same val set.

---

## 8. Implementation order (milestones)

1. **Normalisation stats** over train (dBR mean/std) → saved.
2. **Dataloader** over the prior cache (returns normalised dBR past frames, `A`, `y`, mask).
3. **VAE**: train on train, select on val, verify it preserves histogram + PSD.
4. **Freeze VAE**; encode latents (precompute or on-the-fly).
5. **EDM latent diffusion** conditioned on `[past, A]`, predicting the residual latent.
6. **Sampling + reconstruction** (`A + r̂` → mm/h) + ensemble.
7. **Evaluation** vs persistence + advection (deterministic + probabilistic).
8. **Calibration** (pySTEPS-style probability matching / masking) — optional refinement.

---

## 9. Open questions to confirm before coding

- **Compression ×4 (→64×64) vs ×8 (→32×32)** for the first VAE? (Default ×4.)
- **Latent-residual** (`z_y − z_A`) vs **predict `z_y` conditioned on `z_A`**? (Default latent-residual; fall back if reconstruction of the sum is poor.)
- Train the VAE on the **6-month** cache now, or wait for the **full** train range to finish building? (We can start on 6 months immediately and re-use/fine-tune later.)
