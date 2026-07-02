# Advection prior — extended analysis: literature comparison, FSS, residuals

Three additions on top of the main report, each with **what it is, what we did, and
what to look for**:
1. how my pySTEPS advection scores compare to the **literature** (DGMR + the pySTEPS paper),
2. the **Fractions Skill Score (FSS)** — skill across spatial scales (`fss_analysis.py`),
3. a **residual analysis** across rain-intensity regimes (`residual_analysis.py`).

---

## 1. Comparison with the literature (DGMR + pySTEPS papers)

### Why these two papers
- **DGMR** (Ravuri et al., *Nature* 2021) trained on the **same Met Office UK radar
  composite** as me, evaluates on **256 km × 256 km crops** at **T+60 min** — so its
  PySTEPS baseline is the closest published like-for-like to my setup.
- **The pySTEPS paper** (Pulkkinen et al. 2019) is the source of the exact method I
  use (dense Lucas–Kanade + semi-Lagrangian). It reports CSI, MAE **and FSS across
  scales**, which is why my metric suite mirrors it.

### The numbers (CSI at T+60 min, per grid cell)
From DGMR's Figure 1 case study (24 June 2019, convective cells over eastern
Scotland, 256 km crops):

| CSI @ T+60 min | ≥2 mm/h | ≥8 mm/h |
|---|---|---|
| **PySTEPS** (DGMR's baseline) | 0.19 | 0.02 |
| **DGMR** (full generative model) | 0.50 | 0.04 |
| **My advection prior** (aggregate, 13,281 val crops) | **0.21** | **0.04** |

### What this tells us
- **My pySTEPS advection is behaving correctly.** My CSI@2 (0.21) sits right on
  DGMR's PySTEPS case-study value (0.19), and CSI@8 (0.04) matches too. That is the
  key sanity check: my implementation reproduces a *faithful* pySTEPS advection, not
  a broken or unusually weak one.
- **Why mine is marginally higher:** DGMR's 0.19 is a *single hard convective event*;
  mine is an *aggregate* over 13k mixed crops that also include easier frontal rain,
  which pulls the average up. Different data (2019 vs 2025–26) and my rain-filtering
  also shift the base rate. So treat this as "same ballpark", not an exact match.
- **The gap is the point.** DGMR beats its own PySTEPS baseline massively at 2 mm/h
  (0.50 vs 0.19). That gap — what a *generative* model adds over advection — is
  exactly the headroom my **diffusion model** is aiming to capture over my advection
  prior. My job is to open a similar gap on my own validation set.

### Honest caveats (say these out loud)
- DGMR's numbers above are a **case study**; its aggregate 2019 CSI is only in a
  figure (not extractable as exact numbers).
- DGMR evaluates on **full frames** (mostly dry); I evaluate on **rain-filtered
  crops** (≥5% wet), a higher base rate — so absolute CSI is not perfectly
  comparable, only indicative.
- The pySTEPS paper reports CSI at a **0.1 mm/h** threshold and FSS across scales on
  Swiss/US/Finnish radar — different data, so I use it to justify the **method and
  metric choices**, not for head-to-head CSI numbers.

---

## 2. Fractions Skill Score (FSS) — skill across spatial scales

### What it is (simple version)
Pixel CSI is harsh: a forecast that puts a storm **two pixels off** scores as a
total miss, even though it's basically right. **FSS fixes this by asking a gentler
question at several zoom levels:** *"inside a neighbourhood of size N, does the
forecast have about the same fraction of rain as reality?"* We compute it for
neighbourhoods of **1, 5, 11, 21, 51, 101 km**.

- FSS = **1** → perfect at that scale; FSS = **0** → no skill.
- FSS **rises** as the neighbourhood grows (easier to match on a coarse view).
- The scale where a curve crosses **~0.5** is the **smallest scale at which the
  forecast is useful** for that rain intensity.

It's the standard neighbourhood score (Roberts & Lean 2008), and both the pySTEPS
paper (its Fig. 22) and DGMR (its "pooled" scores) use exactly this idea.

### What we did
`fss_analysis.py` computes FSS over the validation set for **advection vs
persistence**, at every threshold (0.5–8 mm/h) × every scale, and plots FSS-vs-scale.

### What to look for in `fss_analysis.png`
- **Advection curves should sit above persistence** — better placement means skill
  appears at *smaller* neighbourhoods.
- **Read off the useful scale:** e.g. if the ≥1 mm/h curve crosses 0.5 at ~11 km,
  the forecast is trustworthy for light rain once you look at ~11 km blocks, not
  single pixels.
- **Heavier thresholds cross later** (need bigger neighbourhoods) — heavy convection
  is only skilful at coarse scales. This quantifies "how far can I trust the
  location, and for how heavy a rain".
- Later, the **diffusion model should push these curves left** (skill at finer
  scales) — that's the visual target.

---

## 3. Residual analysis across rain regimes

### Why
The diffusion model learns the residual `r = y − A`. Before designing it, we want to
*see* what that residual looks like in different weather — is it tiny for light rain
and large for convection? That tells us where the model's work actually is.

### What we did
`residual_analysis.py` buckets validation crops by their **heaviest rain** (light /
moderate / heavy / extreme), shows input → target → advection → residual for a few
of each, and reports the residual's **mean and spread per bucket**.

### What to look for
- In `residual_stats.md`: **residual std should grow with intensity.** Light cases →
  `r ≈ 0` (advection already nails them); heavy/extreme cases → large, structured
  residual (advection can't grow/decay or sharpen convection). **That is where the
  diffusion model earns its keep.**
- In `residuals_montage.png`: for the heavy/extreme rows, the advection panel looks
  **smoother** than the target and the residual is **strong and cellular** (lots of
  red/blue); for light rows the residual is nearly blank. This visually confirms the
  intensity-dependent story from the CSI@8 result in the main report.

---

## 4. How to run all three (on the server)

```bash
conda activate nowcast
python evaluate_advection.py         # (already done) MAE/MSE/CSI/POD/FAR + distributions
python fss_analysis.py               # FSS across scales  -> fss_analysis.png, fss.md
python residual_analysis.py          # regime residuals   -> residuals_montage.png, residual_stats.md
```
Then `scp` the three PNGs to your laptop to view. Paste me `fss.md` and
`residual_stats.md` and I'll fold the numbers + interpretation into the main report.

See `docs/Metrics_Catalogue.md` for what every metric means and why we use it.
