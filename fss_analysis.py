#!/usr/bin/env python3
"""
fss_analysis.py — Fractions Skill Score (FSS) of the advection prior across
spatial scales, on the validation split, vs the persistence baseline.

FSS asks: "at a given rain threshold, does the forecast get roughly the RIGHT
AMOUNT of rain in the right NEIGHBOURHOOD?" — relaxing the exact-pixel demand of
CSI. We sweep neighbourhood sizes (1, 5, 11, 21, 51, 101 km) so we can see the
*smallest scale at which the forecast is skilful* for each intensity. (Roberts &
Lean 2008; used in the pySTEPS paper and DGMR's pooled scores.)

FSS(scale, thr) = 1 - Σ(M_f - M_o)² / Σ(M_f² + M_o²)
  where M = fraction of pixels ≥ thr inside each `scale`×`scale` window.
FSS = 1 is perfect; FSS = 0 is no skill. The usual "useful skill" reference is
FSS = 0.5 + f0/2 (f0 = fraction of wet pixels); FSS = 0.5 is a simple guide line.

    conda activate nowcast
    python fss_analysis.py                 # ~1-2 min on the val set (sampled)
Outputs (~/dissertation_outputs): fss_analysis.png, fss.json, fss.md
"""
import os, glob, json, getpass, argparse, random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

USER  = getpass.getuser()
PRIOR = f"/scratch/{USER}/dissertation/prior"
OUT   = os.path.expanduser("~/dissertation_outputs")
THRESHOLDS = [0.5, 1.0, 2.0, 4.0, 8.0]         # mm/h
SCALES     = [1, 5, 11, 21, 51, 101]           # neighbourhood size in km (= px @ 1km)


def crop_fss(path):
    """Return {(method, thr, scale): (numerator, denominator)} for one crop."""
    z = np.load(path, allow_pickle=True)
    y = np.nan_to_num(z["y_mmh"].astype("float32"))
    A = np.nan_to_num(z["A_mmh"].astype("float32"))
    P = np.nan_to_num(z["x_mmh"].astype("float32")[-1])
    out = {}
    for name, Fld in (("advection", A), ("persistence", P)):
        for t in THRESHOLDS:
            Io = (y >= t).astype("float32")
            If = (Fld >= t).astype("float32")
            for s in SCALES:
                if s > 1:
                    Mo = uniform_filter(Io, s, mode="constant")
                    Mf = uniform_filter(If, s, mode="constant")
                else:
                    Mo, Mf = Io, If
                out[(name, t, s)] = (float(((Mf - Mo) ** 2).sum()),
                                     float((Mf ** 2 + Mo ** 2).sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--sample", type=int, default=3000, help="crops (0 = all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    ap.add_argument("--root", default=PRIOR)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.root, args.split, "*", "*.npz")))
    if args.sample and args.sample < len(files):
        files = random.Random(0).sample(files, args.sample)
    print(f"FSS over {len(files)} '{args.split}' crops ...")

    acc = {k: [0.0, 0.0] for k in
           [(m, t, s) for m in ("advection", "persistence")
            for t in THRESHOLDS for s in SCALES]}
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=mp.get_context("fork")) as ex:
        for res in ex.map(crop_fss, files, chunksize=32):
            for k, (n, d) in res.items():
                acc[k][0] += n; acc[k][1] += d

    fss = {}
    for (m, t, s), (n, d) in acc.items():
        fss[(m, t, s)] = (1 - n / d) if d > 0 else float("nan")

    # plot: FSS vs scale, one line per threshold, advection vs persistence
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for j, m in enumerate(("advection", "persistence")):
        for t in THRESHOLDS:
            ax[j].plot(SCALES, [fss[(m, t, s)] for s in SCALES],
                       marker="o", label=f"≥{t} mm/h")
        ax[j].axhline(0.5, color="grey", ls="--", lw=1)
        ax[j].set(title=f"FSS — {m}", xlabel="neighbourhood size (km)", ylim=(0, 1))
        ax[j].set_xscale("log"); ax[j].grid(alpha=0.3)
    ax[0].set_ylabel("FSS  (1 = perfect, 0 = no skill)")
    ax[0].legend(title="rain threshold")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fss_analysis.png"), dpi=120)
    print("saved ->", os.path.join(OUT, "fss_analysis.png"))

    # json + markdown table (advection)
    json.dump({f"{m}|{t}|{s}": fss[(m, t, s)]
               for (m, t, s) in fss}, open(os.path.join(OUT, "fss.json"), "w"), indent=2)
    lines = ["# FSS — advection (validation split)\n",
             "| threshold \\ scale (km) | " + " | ".join(str(s) for s in SCALES) + " |",
             "|" + "---|" * (len(SCALES) + 1)]
    for t in THRESHOLDS:
        row = " | ".join(f"{fss[('advection', t, s)]:.3f}" for s in SCALES)
        lines.append(f"| ≥{t} mm/h | {row} |")
    lines += ["", "_FSS rises with neighbourhood size; the scale where a curve "
              "crosses ~0.5 is the smallest skilful scale for that intensity._"]
    open(os.path.join(OUT, "fss.md"), "w").write("\n".join(lines) + "\n")
    print("saved ->", os.path.join(OUT, "fss.md"))
    print("\nadvection FSS @ 1 mm/h:",
          {s: round(fss[('advection', 1.0, s)], 3) for s in SCALES})


if __name__ == "__main__":
    main()
