# The VAE, explained from scratch

A complete, beginner-friendly walkthrough of the VAE we train in `train_vae.py`:
what it is, why we need it, every layer, every loss term, every hyperparameter,
the training procedure, and how we check it worked. No prior knowledge assumed.

---

## 0. The one-paragraph version

We want a **diffusion model** to invent realistic rainfall detail. Diffusion is
expensive: it passes over the image **dozens of times**. Doing that on a full
256×256 picture is slow. So first we train a **VAE** — a neural network that
**shrinks** each 256×256 rain picture into a small **64×64 "code"** and can
**rebuild** the picture from that code with little loss. The diffusion model then
works on the small code instead of the big picture (≈16× cheaper), and at the end
the VAE's decoder turns the code back into a full-size rain field. This whole idea
is called a **Latent Diffusion Model (LDM)** — "latent" just means "the small code".

---

## 1. What is an autoencoder?

An **autoencoder** is two networks bolted together:

```
   image  ──►  ENCODER  ──►  small code (the "latent")  ──►  DECODER  ──►  image again
   256×256                        4×64×64                                  256×256
```

- The **encoder** squeezes the image down through a narrow "bottleneck".
- The **decoder** tries to rebuild the original image from that bottleneck.
- We train them together by punishing the difference between the input and the
  rebuilt output (the **reconstruction loss**).

Because the bottleneck is small, the network is forced to keep only the
**important** information and throw away noise. Think of it as a **smart `.zip`
for rain images** — but a lossy one that learns what matters for *our* data.

## 2. What makes it *variational* (the "V" in VAE)?

A plain autoencoder maps each image to **one** point in code-space. A **VAE** maps
each image to a **small cloud** (a probability distribution) instead — described
by a **mean** (`mu`) and a **spread** (`logvar`, the log of the variance). To get
the actual code we **sample** a point from that cloud.

Why bother? Two reasons:
1. It makes the code-space **smooth and organised**: similar images land in nearby
   clouds, and the space has no "holes". That's important if you later want to
   *generate* new samples by moving around in that space (which is exactly what the
   diffusion model does).
2. It lets us nudge the whole code-space toward a **standard bell curve** (mean 0,
   variance 1) — a tidy, predictable space that diffusion models love.

### The reparameterisation trick (one subtle but important detail)
We need to **sample** a code from the cloud, but you can't send training signal
(gradients) *through* a random draw. The trick: write the sample as

> `z = mu + sigma · ε`,  where `ε` is fresh standard-normal noise, `sigma = exp(½·logvar)`

Now the randomness lives only in `ε` (which we don't need gradients for), and the
learnable parts (`mu`, `sigma`) are connected by plain arithmetic, so training
works normally. In the code this is the line:
```python
z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
```

## 3. Why we need a VAE for *this* project

Our nowcasting model is a **diffusion model that generates the residual** (the
growth/decay/detail advection misses). Diffusion is iterative and costly. Running
it on 256×256 fields would not fit comfortably on the L4 GPU (24 GB) and would be
slow. So we:

1. Train the VAE once to compress 256×256 → **4×64×64**.
2. **Freeze** it.
3. Run the diffusion model in that small **latent space**.
4. Decode the result back to a full 256×256 rain field.

The compression: 256×256 = 65,536 numbers → 4×64×64 = 16,384 numbers (**4× fewer
values**). More importantly, the *spatial* size drops from 256×256 to 64×64 —
**16× fewer pixels** — and since diffusion cost scales with spatial area, that's
roughly a **16× speed-up per diffusion step**. That is the whole point of "latent".

> Why train our own instead of downloading one? Our notes (Score-based Diffusion)
> warn that off-the-shelf image VAEs (trained on photos) reconstruct **radar**
> poorly. Rain fields don't look like cats. So we train a small VAE on our own dBR
> crops and **prove** it reconstructs them well before trusting it (§9).

---

## 4. Our architecture, layer by layer

The VAE is a small **convolutional** network (it slides small filters over the
image, which is the standard tool for pictures). Here is exactly what
`train_vae.py` builds, with the image size and channel count at each step.

### Encoder (image → code)
| Step | Layer | Output size | What it does |
|---|---|---|---|
| in | — | 1 × 256 × 256 | one rain channel |
| 1 | Conv 3×3 (1→64) + ResBlock | 64 × 256 × 256 | learn 64 feature maps |
| 2 | Conv 3×3 stride-2 (64→128) + ResBlock | 128 × 128 × 128 | **downsample** ×2, more channels |
| 3 | Conv 3×3 stride-2 (128→128) + ResBlock | 128 × 64 × 64 | **downsample** ×2 again |
| 4 | GroupNorm + SiLU + Conv 1×1 (128→8) | 8 × 64 × 64 | produce **mean(4)** and **logvar(4)** |

So the latent is **4 channels × 64 × 64**. (Two stride-2 layers = ÷4 in each
spatial dimension = 256→64.)

### Decoder (code → image), a mirror image
| Step | Layer | Output size |
|---|---|---|
| in | code | 4 × 64 × 64 |
| 1 | Conv 3×3 (4→128) + ResBlock | 128 × 64 × 64 |
| 2 | ConvTranspose 4×4 stride-2 (128→128) + ResBlock | 128 × 128 × 128 (**upsample**) |
| 3 | ConvTranspose 4×4 stride-2 (128→64) + ResBlock | 64 × 256 × 256 (**upsample**) |
| 4 | GroupNorm + SiLU + Conv 3×3 (64→1) | 1 × 256 × 256 (the reconstruction) |

### The small building blocks (what the jargon means)
- **Conv (convolution):** a small learnable filter slid over the image to detect
  patterns (edges, blobs, bands). "3×3" = a 3-pixel-wide filter.
- **stride-2 conv / ConvTranspose:** convolution that also **halves** the size
  (downsample, encoder) or **doubles** it (upsample, decoder).
- **Channels (the 64, 128…):** how many different feature maps a layer learns.
  More channels = more capacity to represent detail (and more compute/memory).
- **ResBlock (residual block):** two convs whose output is **added** to their input
  (`x + f(x)`). This "skip" makes deep networks train stably and stops detail from
  being lost. (From ResNet — your notes mention it.)
- **GroupNorm:** keeps the numbers inside the network at a sensible scale (stops
  them exploding/vanishing), which makes training stable.
- **SiLU:** the **activation function** — the bit of non-linearity that lets the
  network represent complex shapes (a smooth cousin of ReLU).

---

## 5. The data the VAE sees

### Which fields
We train on the **target `y`** and the **advection prior `A`** of every cached
crop (set in `RadarFields(..., fields=("y_mmh","A_mmh"))`). The diffusion stage
will need to encode **both**, so the VAE must reconstruct both well. (Both are
ordinary rain fields, so this is one consistent distribution.)

### dBR transform (why we don't use raw mm/h)
Raw rain rate is horribly **skewed**: most pixels are 0, a few are huge. Neural
networks hate that. We take the **logarithm** — the **dBR** transform
`10·log10(rain)` — which spreads the values out into a friendlier range. Dry
pixels get a floor value of **−15 dBR** (≈ 0.03 mm/h). Functions `to_dbr` /
`from_dbr` do this and its inverse.

### Normalisation (mean 0, variance 1)
Diffusion and VAEs train best when inputs are centred around 0 with spread ≈ 1
(your notes call this out explicitly). So we compute the **mean and standard
deviation of the dBR fields over the training set** (`compute_norm`) and transform
every field as `(field − mean) / std`. We **save these two numbers** in the
checkpoint so the exact same normalisation is used everywhere later.

### The valid mask (handling missing radar)
Some pixels are `nodata` (outside radar range). We never want the VAE punished for
"reconstructing" pixels that were never real. So each item also returns a
**mask** (1 = real pixel, 0 = nodata), and the reconstruction loss is computed
**only over real pixels**.

---

## 6. The loss function (what we actually minimise)

```
total loss  =  reconstruction loss  +  β · KL
```

### Reconstruction loss — "rebuild the picture accurately"
We use **masked L1**: the average **absolute** difference `|reconstruction − input|`
over valid pixels.
- **Why L1 (absolute) and not L2 (squared)?** L2 punishes big mistakes
  disproportionately, so the safest way to reduce it is to **blur** (predict a
  bland average). L1 is gentler on the rare extremes, so it keeps **sharper**
  detail — important because rain has heavy tails and sharp cells. (PreDiff uses
  L1/L2 and drops perceptual loss; we follow that — there is no pretrained
  "perceptual" network for radar.)

### KL loss — "keep the code-space tidy"
The **KL divergence** measures how far each latent cloud `N(mu, sigma)` is from a
standard bell curve `N(0, 1)`. Minimising it gently pulls the whole code-space
toward that clean, standard shape. Formula in the code:
```python
kl = -0.5 · mean(1 + logvar − mu² − exp(logvar))
```

### β — the dial between the two
`β` is how strongly we enforce the tidy-space goal:
- **β too large** → the network is forced so hard toward `N(0,1)` that it **ignores
  the code** and reconstructs a blurry average. This failure is called **posterior
  collapse**.
- **β too small (→0)** → essentially a plain autoencoder: great reconstruction, but
  the code-space is less regular.
- **Our choice: β = 1e-6 (tiny).** For a latent **diffusion** model we want
  **faithful reconstruction** above all; the diffusion model itself provides the
  "generative" power. A tiny KL just keeps the latent scale sane. (This matches how
  Stable-Diffusion-style VAEs are trained — a very small KL weight.)

---

## 7. Every hyperparameter, and what it does

| Flag | Meaning | Default | If you increase it |
|---|---|---|---|
| `--epochs` | how many full passes over the data | 40 | better fit, then risk of overfitting; watch **val** recon |
| `--batch` | images per training step | 32 | smoother gradients, more GPU memory; too big can hurt generalisation |
| `--lr` | **learning rate** — step size of each update | 1e-4 | faster but unstable if too high; too low = very slow |
| `--beta` | KL weight (see §6) | 1e-6 | tidier latents but blurrier reconstruction (posterior collapse) |
| `--width` | base number of channels (capacity) | 64 | sharper reconstructions, more compute/memory (must be ÷8 for GroupNorm) |
| `--zc` | latent channels (richness of the code) | 4 | better reconstruction but **less** compression (heavier diffusion later) |
| `--weight-decay` | gentle pull of weights toward 0 (regularisation) | 0 | more regularisation; for a VAE we usually want fidelity, so keep ~0 |
| `--workers` | parallel data-loading processes | 8 | faster loading (uses CPU cores) |
| `--limit` | cap #crops for a quick smoke test | none | use a small number to check the script runs end-to-end |

**Two structural knobs that change the compression** (also in the LDM tuning grid):
`--zc` (latent channels) and the number of downsamples (fixed at 2 here → ×4
spatial). ×8 compression (32×32) would be faster diffusion but risks losing
convective detail — that's an experiment, not the default.

---

## 8. The training procedure, step by step

This is exactly what the loop in `main()` does, in plain English:

1. **List the data.** Gather all `train/` and `val/` crops. (`--limit` for a quick test.)
2. **Compute normalisation** (dBR mean/std) on a sample of train, and remember it.
3. **Build the loaders** that stream batches of (normalised field, mask) — shuffled
   for train, in-order for val, using several CPU workers.
4. **Create the model + optimiser.** Optimiser = **AdamW** (see below), learning
   rate `--lr`.
5. **For each epoch:**
   - **Train:** for every batch → run the VAE (`recon, mu, logvar`) → compute
     `recon_loss + β·KL` → `loss.backward()` (work out how to nudge every weight) →
     `optimizer.step()` (apply the nudge). Repeat over all batches.
   - **Validate:** run over the val set **without** updating weights; average the
     reconstruction error. This is our honest measure of quality.
   - **Checkpoint the best:** if val reconstruction improved, **save the model** (so
     we always keep the best, not just the last).
   - Every 5 epochs, save a **reconstruction figure** (input vs rebuilt) to eyeball.
6. **After training:** reload the best checkpoint and **verify** it preserves the
   rain distribution + power spectrum (§9).

### AdamW, learning rate, weight decay
- **AdamW** is the optimiser — the rule for turning "this weight should change" into
  an actual update. It adapts the step size per-weight (fast, robust) and applies
  weight-decay correctly. It's the default choice for this kind of model.
- **Learning rate** is the master step-size. 1e-4 is a safe, common value; it's the
  **first** thing we'll tune in the diffusion stage.
- **Weight decay** gently shrinks weights to prevent overfitting; we keep it ~0 for
  the VAE because we care about faithful reconstruction.

### Mixed precision (bf16)
On the L4 we run in **bfloat16** (`torch.autocast`), a 16-bit number format. It uses
**half the memory** and is **faster**, with negligible accuracy loss for this task.
The code falls back to normal 32-bit on CPU.

### Seeds
We fix the random seeds (`torch.manual_seed(0)`, etc.) so a run is **reproducible** —
the same data and settings give the same result (your supervisor asked for this).

---

## 9. How we know the VAE is good (the crucial check)

A VAE can have a low reconstruction number but still quietly **blur away the heavy
rain** — which would silently cap how good the final nowcast can be. So we don't
just trust the loss; we **verify the distribution** (`verify()` → `vae_verify.png`):

- **Rain-rate histogram:** reconstruction vs observations. **Good** = the two curves
  overlap, *including the heavy-rain tail*. **Bad** = the reconstruction curve drops
  below at high rain → the VAE is losing extremes.
- **Power spectrum (PSD):** how much fine detail survives. **Good** = curves overlap
  out to small scales. **Bad** = reconstruction falls below at short wavelengths →
  the VAE is smoothing.

If either looks bad, the fixes are: **lower β**, **increase `--width` or `--zc`**,
or train longer. We only move to the diffusion stage once the VAE clearly preserves
both. (This reuses the same distribution analysis idea from the advection
evaluation — consistency across the project.)

### Other things to watch
- **Train recon keeps dropping but val recon rises** → overfitting; stop earlier
  (we keep the best-val checkpoint anyway).
- **Reconstructions look blurry/constant from the start** → β likely too high
  (posterior collapse) → lower it.

---

## 10. The latent scaling factor (one last detail)

Diffusion models assume their input has roughly **unit variance**. After training we
measure the **standard deviation of the latents** and save
`latent_scale = 1 / std`. In the diffusion stage every latent is multiplied by this
number so the diffusion model sees a nicely-scaled space (and we divide it back out
before decoding). This is standard LDM practice (Stable Diffusion uses a fixed
scale of ~0.18). It's saved inside `vae_best.pt`.

---

## 11. How the VAE feeds the next (diffusion) stage

Once trained and **frozen**, the VAE is the bridge to the diffusion model:

```
 past 4 frames ─► encoder ─► z_past ─┐
 advection A   ─► encoder ─► z_A ─────┼─ condition ─► EDM diffusion ─► residual latent r̂
 target y      ─► encoder ─► z_y      ┘                                     │
                                            ẑ_y = z_A + r̂ ─► DECODER ─► ŷ (dBR) ─► mm/h
```

The diffusion model learns the **residual in latent space** (`z_y − z_A`),
conditioned on the encoded past frames and prior. At inference we sample a residual,
add it to `z_A`, and **decode** to get the final nowcast. (Full plan in
`docs/LDM_Design_and_Hyperparameters.md`.)

---

## 12. Glossary (quick reference)

- **Latent / code:** the small compressed representation (here 4×64×64).
- **Encoder / Decoder:** networks that compress / rebuild.
- **Reconstruction loss:** how different the rebuilt image is from the input (we use L1).
- **KL divergence:** how far the latent space is from a standard bell curve.
- **β (beta):** the weight on the KL term; small = favour reconstruction.
- **Reparameterisation trick:** the `mu + sigma·ε` rewrite that lets us train through sampling.
- **Epoch:** one full pass over the training data.
- **Batch:** a small group of images processed together in one step.
- **Learning rate:** how big each weight update is.
- **AdamW:** the optimiser (update rule).
- **Convolution / channels / stride:** the image-processing building blocks.
- **Posterior collapse:** failure where the VAE ignores its code and outputs blur.
- **dBR:** decibel rain rate, the log transform `10·log10(rain)`.
- **PSD:** power spectral density — how much detail exists at each spatial scale.

---

## 13. How to run it

```bash
# one-time: install PyTorch for the L4 GPU into the nowcast env
conda activate nowcast
pip install torch --index-url https://download.pytorch.org/whl/cu124

# quick smoke test (a few minutes) — confirm it runs end-to-end
python train_vae.py --limit 4000 --epochs 5

# full training once the whole prior cache is built (run in tmux)
python train_vae.py --epochs 40
```
Watch the per-epoch `val recon` fall and level off. Then open `vae_verify.png`:
if the reconstruction curves sit on top of the observed curves, the VAE is good and
we move to the diffusion model. Outputs are in `~/dissertation_outputs/vae/`.
