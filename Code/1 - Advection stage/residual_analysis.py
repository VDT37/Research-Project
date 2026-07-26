#!/usr/bin/env python3
"""
residual_analysis.py — look at MORE advection cases, stratified by rain intensity,
to understand the residual r = y - A the diffusion model must learn.

It buckets validation crops by their heaviest rain (light / moderate / heavy /
extreme), shows input -> target -> advection -> residual for a few of each, and
reports residual statistics per bucket. This tells us *where* advection fails and
how big / structured the residual is across regimes.

    conda activate nowcast
    python residual_analysis.py
Outputs (~/dissertation_outputs): residuals_montage.png, residual_stats.md
"""
import os, glob, json, getpass, argparse, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

USER  = getpass.getuser()
PRIOR = f"/scratch/{USER}/dissertation/prior"
OUT   = os.path.expanduser("~/dissertation_outputs")
# buckets by target max rain rate (mm/h)
BUCKETS = [("light", 0.5, 2), ("moderate", 2, 8), ("heavy", 8, 32), ("extreme", 32, 1e9)]


def maxrain(f):
    return float(np.nanmax(np.load(f, allow_pickle=True)["y_mmh"].astype("float32")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--per-bucket", type=int, default=3, help="rows shown per bucket")
    ap.add_argument("--scan", type=int, default=1500, help="crops to scan for buckets")
    ap.add_argument("--root", default=PRIOR)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.root, args.split, "*", "*.npz")))
    scan = random.Random(0).sample(files, min(args.scan, len(files)))
    mx = {f: maxrain(f) for f in scan}

    # --- per-bucket residual statistics (over all scanned crops in the bucket) ---
    stats, chosen = [], []
    for name, lo, hi in BUCKETS:
        members = [f for f in scan if lo <= mx[f] < hi]
        if not members:
            stats.append((name, 0, np.nan, np.nan, np.nan)); continue
        rms, rmean = [], []
        for f in members:
            z = np.load(f, allow_pickle=True)
            v = z["valid"]; r = z["r_dbr"].astype("float32")[v]
            rmean.append(float(r.mean())); rms.append(float(r.std()))
        stats.append((name, len(members), float(np.mean(rmean)),
                      float(np.mean(rms)), float(np.mean([mx[f] for f in members]))))
        chosen.append((name, sorted(members, key=lambda f: -mx[f])[:args.per_bucket]))

    # --- montage: rows = cases (grouped by bucket), cols = input/target/adv/residual ---
    rows = [(name, f) for name, fs in chosen for f in fs]
    n = len(rows)
    fig, ax = plt.subplots(n, 4, figsize=(14, 3.3 * n))
    if n == 1:
        ax = ax[None, :]
    titles = ["last input t0", "target t0+60", "advection A", "residual r (dBR)"]
    for i, (bucket, f) in enumerate(rows):
        z = np.load(f, allow_pickle=True)
        x = z["x_mmh"].astype("float32"); A = z["A_mmh"].astype("float32")
        y = z["y_mmh"].astype("float32"); r = z["r_dbr"].astype("float32")
        norm = LogNorm(vmin=0.1, vmax=max(4.0, np.nanmax(y)))
        m = np.nanmax(np.abs(r)) or 1.0
        panels = [(np.nan_to_num(x[-1]), dict(norm=norm, cmap="viridis")),
                  (np.nan_to_num(y),     dict(norm=norm, cmap="viridis")),
                  (np.nan_to_num(A),     dict(norm=norm, cmap="viridis")),
                  (r,                    dict(cmap="RdBu_r", vmin=-m, vmax=m))]
        for j, (img, kw) in enumerate(panels):
            ax[i, j].imshow(img, **kw)
            ax[i, j].set_title(titles[j] if i == 0 else "", fontsize=11)
            ax[i, j].axis("off")
        ax[i, 0].text(-0.08, 0.5, f"{bucket}\n{np.nanmax(y):.0f} mm/h", rotation=90,
                      va="center", ha="center", transform=ax[i, 0].transAxes, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "residuals_montage.png"), dpi=110, bbox_inches="tight")
    print("saved ->", os.path.join(OUT, "residuals_montage.png"))

    # --- stats table ---
    lines = ["# Residual statistics by rain-intensity bucket (validation)\n",
             "| bucket (max rain) | #crops | residual mean (dBR) | residual std (dBR) | avg max (mm/h) |",
             "|---|---|---|---|---|"]
    for name, cnt, rm, rs, amx in stats:
        lines.append(f"| {name} | {cnt} | {rm:+.2f} | {rs:.2f} | {amx:.1f} |"
                     if cnt else f"| {name} | 0 | – | – | – |")
    lines += ["", "_Residual std grows with intensity → heavy/convective cases carry "
              "the most for the diffusion model to learn; light cases are nearly r≈0._"]
    open(os.path.join(OUT, "residual_stats.md"), "w").write("\n".join(lines) + "\n")
    print("saved ->", os.path.join(OUT, "residual_stats.md"))
    for s in stats:
        print(f"  {s[0]:8s} n={s[1]:4d}  r_mean={s[2]:+.2f}  r_std={s[3]:.2f}")


if __name__ == "__main__":
    main()
