#!/usr/bin/env python3
"""
check_priors.py — health check + visualisation for the advection-prior cache.

Run on the server after build_advection_prior.py finishes:
    conda activate nowcast
    python check_priors.py

It prints a summary (counts, integrity, residual statistics, baseline) and saves
a montage PNG to ~/dissertation_outputs/prior_check.png that you scp to your
laptop to look at. No GUI needed (uses the Agg backend).
"""
import os, glob, json, random, getpass
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")                 # headless: save figures, never open a window
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

USER  = getpass.getuser()
PRIOR = f"/scratch/{USER}/dissertation/prior"
OUT   = os.path.expanduser("~/dissertation_outputs")
os.makedirs(OUT, exist_ok=True)
EXPECT = {"x_mmh", "A_dbr", "A_mmh", "y_mmh", "r_dbr", "valid", "split", "target"}

# 1) manifest (the run's own summary + baseline) -----------------------------
mpath = os.path.join(OUT, "manifest.json")
if os.path.exists(mpath):
    man = json.load(open(mpath))
    print("=== manifest.json ===")
    print(f"  created  : {man.get('created')}")
    print(f"  window   : {man.get('start')} .. {man.get('end')}")
    print(f"  n_crops  : {man.get('n_crops')}")
    print(f"  by_split : {man.get('by_split')}")
    base = man.get("baseline_advection_only")
    if base:
        print(f"  BASELINE  MAE = {base['MAE_mmh']:.4f} mm/h")
        for t, v in base["CSI"].items():
            print(f"            CSI @ {t} mm/h = {v:.3f}")
else:
    print("(!) no manifest.json — the run may not have reached the end")

# 2) counts on disk ----------------------------------------------------------
files = sorted(glob.glob(os.path.join(PRIOR, "*", "*", "*.npz")))
print(f"\n=== counts on disk ===\n  total crops      : {len(files)}")
print(f"  by split         : {dict(Counter(f.split(os.sep)[-3] for f in files))}")
print(f"  leftover .part   : {len(glob.glob(os.path.join(PRIOR, '*', '*', '*.part')))} (want 0)")
if not files:
    raise SystemExit("no prior crops found — nothing to check")

# 3) integrity + content on a random sample ---------------------------------
rng = random.Random(0)
sample = rng.sample(files, min(800, len(files)))
bad = []
for f in sample:
    try:
        z = np.load(f, allow_pickle=True)
        if not EXPECT <= set(z.files):
            bad.append(f)
    except Exception:
        bad.append(f)
print(f"\n=== integrity (sampled {len(sample)}) ===\n  unreadable/malformed: {len(bad)}")

z0 = np.load(sample[0], allow_pickle=True)
print("  one record:")
for k in ("x_mmh", "A_mmh", "y_mmh", "r_dbr", "valid"):
    print(f"    {k:7s} shape={tuple(z0[k].shape)} dtype={z0[k].dtype}")
print(f"    split={str(z0['split'])}  target={str(z0['target'])}")

# 4) residual statistics — the scientific sanity check ----------------------
r_mean, r_std, y_max, wet = [], [], [], []
for f in sample:
    z = np.load(f, allow_pickle=True)
    v = z["valid"]
    r = z["r_dbr"].astype("float32")[v]
    y = z["y_mmh"].astype("float32")
    r_mean.append(float(r.mean())); r_std.append(float(r.std()))
    y_max.append(float(np.nanmax(y)))
    wet.append(float((np.nan_to_num(y) >= 0.1)[v].mean()))
print("\n=== residual  r = y - A  (dBR), over sample ===")
print(f"  mean of per-crop means : {np.mean(r_mean):+.3f}   (near 0  -> advection ~unbiased)")
print(f"  mean of per-crop stds  : {np.mean(r_std):.3f}    (the spread the LDM must learn)")
print(f"  target max rain (avg)  : {np.mean(y_max):.1f} mm/h")
print(f"  target wet fraction    : {np.mean(wet)*100:.1f}%")

# 5) montage of the 4 rainiest sampled cases --------------------------------
cases = sorted(sample, key=lambda f: -np.nanmean(np.nan_to_num(
    np.load(f, allow_pickle=True)["y_mmh"].astype("float32"))))[:4]
fig, axes = plt.subplots(len(cases), 4, figsize=(14, 3.4 * len(cases)))
if len(cases) == 1:
    axes = axes[None, :]
titles = ["last input  t0", "target  t0+60", "advection prior A", "residual r (dBR)"]
for i, f in enumerate(cases):
    z = np.load(f, allow_pickle=True)
    x = z["x_mmh"].astype("float32"); A = z["A_mmh"].astype("float32")
    y = z["y_mmh"].astype("float32"); r = z["r_dbr"].astype("float32")
    norm = LogNorm(vmin=0.1, vmax=max(2.0, np.nanmax(y)))
    m = np.nanmax(np.abs(r)) or 1.0
    panels = [(np.nan_to_num(x[-1]), dict(norm=norm, cmap="viridis")),
              (np.nan_to_num(y),     dict(norm=norm, cmap="viridis")),
              (np.nan_to_num(A),     dict(norm=norm, cmap="viridis")),
              (r,                    dict(cmap="RdBu_r", vmin=-m, vmax=m))]
    for j, (img, kw) in enumerate(panels):
        axes[i, j].imshow(img, **kw)
        axes[i, j].set_title(titles[j] if i == 0 else "", fontsize=11)
        axes[i, j].axis("off")
plt.tight_layout()
png = os.path.join(OUT, "prior_check.png")
plt.savefig(png, dpi=110, bbox_inches="tight")
print(f"\nsaved montage -> {png}")
print(f"  view it locally:  scp {USER}@<server>:{png} .")
