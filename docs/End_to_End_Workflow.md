# End-to-end workflow: from radar frames to a diffusion nowcast

This document resolves a specific confusion: whether we are "replacing" the real
radar observations with the advection output, and therefore training and
evaluating on advection fields instead of ground truth. We are not. This explains
exactly what is stored, what each model sees, and what we optimise and evaluate
against, with a worked example and the effect of each design choice.

## 1. What a cached "prior crop" actually is

The name "prior crop" is misleading. It refers to the directory `prior/`, not to
the file contents. Each cached `.npz` is a bundle of six arrays for one
(location, target-time) sample, all 256x256 unless noted:

| key | shape | meaning | source |
|-----|-------|---------|--------|
| `x_mmh` | (4, 256, 256) | the four input frames, mm/h | real radar observations |
| `y_mmh` | (256, 256) | the target frame at t0+60, mm/h | real radar observation (ground truth) |
| `A_mmh` | (256, 256) | advection forecast of the target, mm/h | pysteps, derived from `x` |
| `A_dbr` | (256, 256) | same advection forecast in dBR | pysteps |
| `r_dbr` | (256, 256) | residual `dB(y) - A_dbr` | derived from `y` and `A` |
| `valid` | (256, 256) | mask, True where radar is in range | from `y` |

The key point: the ground truth `y_mmh` is stored in every crop. The advection
`A` is an additional field, a physics first guess, not a replacement for `y`.
Nothing in the pipeline ever discards `y`.

## 2. Notation and the residual decomposition

Let `x = (x_1, ..., x_4)` be the four input observations, `y` the observation 60
minutes after the last input, and `A = Adv(x)` the semi-Lagrangian advection
forecast of `y`. Advection is a fixed, non-learned operator: it estimates a motion
field from `x` and transports the last frame along it. It is a deterministic
function of the inputs, with no free parameters fit to data.

Work in the dBR domain, `dB(R) = 10 log10(R)` with a dry floor at -15 dBR. Define
the residual in that domain:

```
r = dB(y) - A_dbr           (this is r_dbr in the cache)
dB(y) = A_dbr + r           (exact, by definition)
```

So `A` carries the part of the future that "rain moves with the wind" already
explains, and `r` carries everything advection cannot produce: growth, decay,
initiation, and small-scale detail. `r` is what the diffusion model will learn.
`y` remains the quantity of interest throughout.

## 3. Worked example (one crop)

Take the crop with target time T = 2025-04-19 04:15 (t0 = 03:15):

1. Inputs `x`: real radar at 02:30, 02:45, 03:00, 03:15.
2. Target `y`: real radar at 04:15. This is ground truth, never modified.
3. Advection `A`: pysteps estimates motion from the 02:30 to 03:15 sequence and
   transports the 03:15 field forward four 15-minute steps to 04:15. `A` is a
   forecast of `y`, and it is imperfect (that is the whole premise).
4. Residual `r = dB(y) - A_dbr`: at each pixel, how far advection was from reality
   in the log domain. Positive where real rain grew or advection under-shot,
   negative where it decayed or advection over-shot.

Both `y` (sharp, real) and `A` (smoother, forecast) are ordinary rainfall fields.
They live in the same value range and the same physical units. This matters for
the VAE.

## 4. Stage 2, the VAE: a codec for rainfall fields

The VAE is trained on `y_mmh` and `A_mmh` from every crop (two fields per crop,
converted to dBR and normalised). Its objective is reconstruction fidelity, masked
L1 plus a small KL term. It is not told which field is "real" and which is
"advection", and it does not need to be. The VAE is a compressor for rainfall
fields in general: image in, latent out, image back.

Why train on both `y` and `A`, and why this is not circular:

- The diffusion stage must encode both `y` and `A` (see Stage 3). A codec is only
  trusted on inputs from its training distribution, so we train it on both so that
  both `E(y)` and `E(A)` reconstruct faithfully.
- The VAE never "learns advection". It learns to represent rainfall fields
  compactly. The accuracy of `A` is irrelevant to the VAE: whether `A` is a good or
  bad forecast of `y`, the VAE only has to compress and rebuild it as an image. The
  correction of `A` toward `y` is the diffusion model's job, not the VAE's.
- `A` is smoother than `y`, so it is an easier, in-distribution input for the codec.
  Including it does not degrade the VAE's ability to reconstruct sharp `y`, because
  `y` is half the training data. We verify this explicitly: after training, we
  compare the rain-rate histogram and PSD of `D(E(y))` against `y`. If those match,
  the VAE preserves the real distribution, including the heavy-rain tail.

Design alternative: train the VAE on `y` only. Then `A` (smoother, in-distribution)
would likely still reconstruct well, but we would not have guaranteed it. Training
on both is the safe choice and is cheap.

Important, the VAE is not trained on the residual `r`. In dBR, `r = dB(y) - A_dbr`
is a signed field (it can be negative), which is out of the distribution of
non-negative rainfall fields the VAE learns. The residual is therefore handled in
latent space, not encoded directly. This is the subject of Stage 3.

## 5. Stage 3, the diffusion model: learning the residual in latent space

Freeze the VAE. Encode everything with its encoder `E` (latents are 4x64x64):

```
z_x1..z_x4 = E(x_1..x_4)     latents of the four input frames
z_A        = E(A)            latent of the advection prior
z_y        = E(y)            latent of the ground truth
```

The training target is the latent residual, and the conditioning is the encoded
context concatenated along the channel axis:

```
target:      delta = z_y - z_A
condition:   c = concat(z_x1, z_x2, z_x3, z_x4, z_A)      (5 * 4 = 20 channels)
model:       an EDM denoiser predicts delta from a noised delta, sigma, and c
```

The denoiser learns the conditional distribution of the residual given the past
frames and the prior. It is trained with `y` as ground truth, through
`delta = z_y - z_A`, so it is learning to correct advection toward reality.

Inference for a new input sequence:

```
1. compute A = Adv(x)                    prescribed physics
2. z_A = E(A),  z_xk = E(x_k)            encode context
3. sample delta_hat ~ model( . | c)      draw a residual (many samples = ensemble)
4. z_y_hat = z_A + delta_hat             reconstruct the target latent
5. y_dbr_hat = D(z_y_hat)                decode
6. y_hat = inverse_dB(y_dbr_hat)         back to mm/h
```

The final nowcast is `y_hat`, a full rainfall field, produced as prior plus learned
residual. Sampling step 3 several times yields an ensemble, which is where the
probabilistic metrics (CRPS, reliability, rank histogram) come from.

Technical subtlety worth stating: step 4 assumes the decoder behaves approximately
linearly, so that `D(z_A + delta_hat)` reconstructs `y` when `delta_hat` is close to
`z_y - z_A`. With the very small KL weight we use (a near-deterministic
autoencoder) and small residuals this holds approximately, but it is an empirical
assumption we will validate. The alternative formulation predicts `z_y` directly
from the condition `c` (physics enters purely through conditioning, decode `z_y`
with no additive step). That removes the linearity assumption but makes the
residual framing implicit rather than explicit. This choice is logged as open in
`LDM_Design_and_Hyperparameters.md`.

## 6. Inference and evaluation: what we compare against

Every metric compares a forecast against `y_mmh`, the real observation stored in
the crop. This is true at both stages:

- Advection baseline: metrics compare `A_mmh` against `y_mmh`.
- Persistence baseline: metrics compare `x_mmh[-1]` (the last input) against `y_mmh`.
- Diffusion model: metrics compare `y_hat` (which is `A + r_hat`) against `y_mmh`.

We never evaluate against `A`. `A` is an intermediate scaffold. The residual `r` is
a training target, not an evaluation target. The question the evaluation answers is
always "how close is the forecast to the real future radar", and the diffusion model
wins only if `A + r_hat` is closer to `y` than `A` alone.

## 7. Answering the three worries directly

1. "We train the VAE on advection prior crops instead of image crops." The crops
   contain the real image `y`, and the VAE trains on `y` and `A`. It is trained on
   real image crops (plus advection fields), not on advection instead of images.

2. "We evaluate against advection prior crops instead of the original." We evaluate
   `y_hat` against `y_mmh`, the stored ground truth. `A` is never the target of any
   metric.

3. "Advection is inaccurate, so training the VAE on it is questionable." The VAE's
   job is representation fidelity, which is independent of whether `A` forecasts `y`
   well. Advection accuracy is measured separately (Section 6) and is corrected by
   the diffusion model, not by the VAE.

## 8. How each design choice propagates end to end

| Design choice | Immediate effect | Downstream consequence |
|---------------|------------------|------------------------|
| Cache `x`, `y`, `A`, `r` together | one self-contained sample per file | VAE and diffusion read from one source; `y` is always available for evaluation |
| Residual in dBR, `r = dB(y) - A_dbr` | target is near zero-mean, lighter-tailed | easier, more stable diffusion target; requires the inverse-dB step at output |
| Advection is prescribed, not learned | `A` is a fixed function of `x` | interpretable physics; the model only learns `r`, reducing what must be fit |
| VAE trained on `y` and `A` | both encode and decode faithfully | valid to compute `z_A` and `z_y`, and to reconstruct `z_A + delta_hat` |
| Small KL weight in the VAE | near-deterministic latents, high fidelity | supports the additive latent step in Section 5; less "generative" latent space, which is fine because the diffusion model provides the generative part |
| Latent residual `delta = z_y - z_A` | diffusion models a small correction | strong physics prior, but relies on approximate decoder linearity (validated) |
| Conditioning by channel concat of `z_x`, `z_A` | context enters the denoiser directly | cheap and sufficient for a strong prior; cross-attention is the heavier alternative |
| Rain-filtered crops (>= 5% wet) | training data is not mostly dry | VAE and diffusion see informative fields; evaluation base rate differs from full-frame studies, noted when comparing to literature |
| Single +60 min target | one lead time | simplest first system; extend to a sequence later without changing the VAE |

## 9. Summary

The observations `y` are never replaced. A cached crop is a bundle that holds the
inputs `x`, the ground truth `y`, the physics first guess `A`, and the residual `r`.
The VAE is a codec trained to reconstruct rainfall fields (`y` and `A`), independent
of advection accuracy. The diffusion model learns the residual `r` in latent space,
conditioned on `x` and `A`, with `y` as ground truth. The final nowcast is
`A + r_hat`, and it is evaluated against the real `y`, next to the persistence and
advection baselines. The role of `A` is to remove the easy, large-scale part of the
problem so the diffusion model can focus on the hard part, not to stand in for the
observations.
