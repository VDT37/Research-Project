#!/usr/bin/env python3
"""
evaluate_advection.py — full quantitative + distribution evaluation of the
advection prior, on the held-out validation split.

Answers the supervisor's asks:
  * MAE / MSE (RMSE) and rainfall-threshold metrics (POD, FAR, CSI, freq-bias),
  * compared against a PERSISTENCE baseline (hold the last frame still),
  * plus DISTRIBUTION ANALYSIS: rain-rate histogram, spatial power spectrum
    (PSD), and the residual distribution.

Persistence forecast = the last input frame (t0) held still for 60 min; it is the
simplest possible nowcast. Advection should beat it — that gap is the value the
motion estimation adds. (PySTEPS advection *is* our prior, so "advection baseline"
= this prior.)

    conda activate nowcast
    python evaluate_advection.py                 # validation split
    python evaluate_advection.py --split train   # or train
Outputs (to ~/dissertation_outputs):
    advection_eval.json            machine-readable scorecard
    advection_eval.md              paste-ready tables
    advection_eval.png             histogram + PSD + residual + CSI panels
"""
import os, glob, json, getpass, argparse, random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

USER  = getpass.getuser()
PRIOR = f"/scratch/{USER}/dissertation/prior"
OUT   = os.path.expanduser("~/dissertation_outputs")
THRESHOLDS = [0.5, 1.0, 2.0, 4.0, 8.0]            # mm/h
WET = 0.1                                          # mm/h -> "raining"


# ---- per-file scalar metrics (advection AND persistence) -------------------
def _metrics_file(path):
    try:
        z = np.load(path, allow_pickle=True)
        y = z["y_mmh"].astype("float32")
        A = z["A_mmh"].astype("float32")
        P = z["x_mmh"].astype("float32")[-1]       # persistence = last input (t0)
        v = z["valid"] & np.isfinite(A) & np.isfinite(P)
        out = {}
        for name, F in (("advection", A), ("persistence", P)):
            d = (F - y)[v]
            s = {"sum_abs": float(np.abs(d).sum()), "sum_sq": float((d * d).sum()),
                 "sum_err": float(d.sum()), "n": int(v.sum()), "cont": {}}
            for t in THRESHOLDS:
                o, p = (y >= t), (F >= t)
                s["cont"][t] = (int(np.sum(v & o & p)),
                                int(np.sum(v & o & ~p)),
                                int(np.sum(v & ~o & p)))
            out[name] = s
        return out
    except Exception:
        return None


def scalar_metrics(files, workers):
    methods = ("advection", "persistence")
    agg = {m: {"sum_abs": 0.0, "sum_sq": 0.0, "sum_err": 0.0, "n": 0,
               "cont": {t: [0, 0, 0] for t in THRESHOLDS}} for m in methods}
    skipped = 0
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=mp.get_context("fork")) as ex:
        for res in ex.map(_metrics_file, files, chunksize=64):
            if res is None:
                skipped += 1; continue
            for m in methods:
                for k in ("sum_abs", "sum_sq", "sum_err", "n"):
                    agg[m][k] += res[m][k]
                for t in THRESHOLDS:
                    H, M, F = res[m]["cont"][t]
                    agg[m]["cont"][t][0] += H
                    agg[m]["cont"][t][1] += M
                    agg[m]["cont"][t][2] += F
    out = {"n_skipped": skipped}
    for m in methods:
        n = max(agg[m]["n"], 1)
        scores = {"MAE_mmh": agg[m]["sum_abs"] / n,
                  "RMSE_mmh": (agg[m]["sum_sq"] / n) ** 0.5,
                  "bias_mmh": agg[m]["sum_err"] / n, "by_threshold": {}}
        for t in THRESHOLDS:
            H, M, F = agg[m]["cont"][t]
            scores["by_threshold"][t] = {
                "POD":  H / (H + M) if (H + M) else float("nan"),
                "FAR":  F / (H + F) if (H + F) else float("nan"),
                "CSI":  H / (H + M + F) if (H + M + F) else float("nan"),
                "freq_bias": (H + F) / (H + M) if (H + M) else float("nan")}
        out[m] = scores
    return out


# ---- distribution analysis (on a random sample) ---------------------------
def radial_psd(field):
    """Radially-averaged 2D power spectrum (power per spatial wavenumber)."""
    f = np.nan_to_num(field).astype("float64")
    f = f - f.mean()
    P = np.fft.fftshift(np.abs(np.fft.fft2(f)) ** 2)
    H, W = f.shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    r = np.hypot(Y - cy, X - cx).astype(int)
    tot = np.bincount(r.ravel(), P.ravel())
    cnt = np.bincount(r.ravel())
    return tot / np.maximum(cnt, 1)


def distributions(files, sample_n, seed=0):
    rng = random.Random(seed)
    sample = rng.sample(files, min(sample_n, len(files)))

    rbins = np.logspace(np.log10(WET), np.log10(128), 41)      # rain-rate bins
    dbins = np.linspace(-30, 30, 61)                            # residual (dBR) bins
    hist = {m: np.zeros(len(rbins) - 1) for m in ("obs", "advection", "persistence")}
    rhist = np.zeros(len(dbins) - 1)
    psd = {m: None for m in ("obs", "advection", "persistence")}
    npsd = 0
    wet_area = {m: [] for m in ("obs", "advection", "persistence")}

    for f in sample:
        z = np.load(f, allow_pickle=True)
        y = z["y_mmh"].astype("float32")
        A = z["A_mmh"].astype("float32")
        P = z["x_mmh"].astype("float32")[-1]
        v = z["valid"] & np.isfinite(A) & np.isfinite(P)
        fields = {"obs": y, "advection": A, "persistence": P}
        for m, F in fields.items():
            vals = F[v]
            hist[m] += np.histogram(vals[vals >= WET], bins=rbins)[0]
            wet_area[m].append(float((vals >= WET).mean()))
        rhist += np.histogram(z["r_dbr"].astype("float32")[v], bins=dbins)[0]
        if v.mean() > 0.99:                                    # clean crops only for PSD
            for m, F in fields.items():
                p = radial_psd(F)
                psd[m] = p if psd[m] is None else psd[m] + p
            npsd += 1

    for m in psd:
        if psd[m] is not None:
            psd[m] = psd[m] / max(npsd, 1)
    return {"rbins": rbins, "dbins": dbins, "hist": hist, "rhist": rhist,
            "psd": psd, "npsd": npsd,
            "wet_area": {m: float(np.mean(w)) for m, w in wet_area.items()},
            "n_sample": len(sample)}


# ---- plotting --------------------------------------------------------------
def make_plots(dist, png):
    rb, db = dist["rbins"], dist["dbins"]
    rc = np.sqrt(rb[:-1] * rb[1:])                      # bin centres (log)
    dc = 0.5 * (db[:-1] + db[1:])
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (1) rain-rate distribution
    for m, c in (("obs", "k"), ("advection", "C0"), ("persistence", "C1")):
        h = dist["hist"][m]; h = h / max(h.sum(), 1)
        ax[0, 0].loglog(rc, h, label=m, color=c, lw=2 if m == "obs" else 1.5)
    ax[0, 0].set(title="Rain-rate distribution (wet pixels)",
                 xlabel="rain rate (mm/h)", ylabel="frequency")
    ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

    # (2) power spectrum
    for m, c in (("obs", "k"), ("advection", "C0"), ("persistence", "C1")):
        p = dist["psd"][m]
        if p is None:
            continue
        k = np.arange(1, len(p))
        wl = 256.0 / k                                  # wavelength in km (256 km domain)
        ax[0, 1].loglog(wl, p[1:], label=m, color=c, lw=2 if m == "obs" else 1.5)
    ax[0, 1].set(title="Spatial power spectrum (PSD)",
                 xlabel="wavelength (km)", ylabel="power")
    ax[0, 1].invert_xaxis(); ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3)

    # (3) residual distribution
    h = dist["rhist"]; h = h / max(h.sum(), 1)
    ax[1, 0].plot(dc, h, color="C3", lw=2)
    ax[1, 0].axvline(0, color="k", ls="--", lw=1)
    ax[1, 0].set(title="Residual r = y - A  (dBR) distribution",
                 xlabel="residual (dBR)", ylabel="frequency")
    ax[1, 0].grid(alpha=0.3)

    # (4) wet-area fractions
    ms = ["obs", "advection", "persistence"]
    ax[1, 1].bar(ms, [dist["wet_area"][m] * 100 for m in ms],
                 color=["k", "C0", "C1"])
    ax[1, 1].set(title="Wet-area fraction (>= 0.1 mm/h)", ylabel="%")
    ax[1, 1].grid(alpha=0.3, axis="y")

    plt.tight_layout(); plt.savefig(png, dpi=120, bbox_inches="tight")
    print("saved plots ->", png)


# ---- markdown table --------------------------------------------------------
def write_markdown(sc, dist, split, n_files, md):
    adv, per = sc["advection"], sc["persistence"]
    lines = [f"# Advection prior — validation scorecard (`{split}` split)\n",
             f"_{n_files} crops; distributions from {dist['n_sample']} sampled crops._\n",
             "## Pixel error (lower = better)\n",
             "| metric | persistence | advection | advection better? |",
             "|---|---|---|---|",
             f"| MAE (mm/h) | {per['MAE_mmh']:.3f} | {adv['MAE_mmh']:.3f} | "
             f"{'yes' if adv['MAE_mmh'] < per['MAE_mmh'] else 'no'} |",
             f"| RMSE (mm/h) | {per['RMSE_mmh']:.3f} | {adv['RMSE_mmh']:.3f} | "
             f"{'yes' if adv['RMSE_mmh'] < per['RMSE_mmh'] else 'no'} |",
             f"| bias (mm/h) | {per['bias_mmh']:+.3f} | {adv['bias_mmh']:+.3f} | |\n",
             "## Threshold skill — CSI (higher = better)\n",
             "| threshold (mm/h) | persistence | advection |",
             "|---|---|---|"]
    for t in THRESHOLDS:
        lines.append(f"| {t} | {per['by_threshold'][t]['CSI']:.3f} | "
                     f"{adv['by_threshold'][t]['CSI']:.3f} |")
    lines += ["\n## Detection at 1 mm/h (POD up, FAR down, bias~1 ideal)\n",
              "| metric | persistence | advection |", "|---|---|---|"]
    for k in ("POD", "FAR", "freq_bias"):
        lines.append(f"| {k} | {per['by_threshold'][1.0][k]:.3f} | "
                     f"{adv['by_threshold'][1.0][k]:.3f} |")
    lines += ["\n## Distribution analysis\n",
              f"- Wet-area fraction: obs {dist['wet_area']['obs']*100:.1f}% · "
              f"advection {dist['wet_area']['advection']*100:.1f}% · "
              f"persistence {dist['wet_area']['persistence']*100:.1f}%",
              "- See `advection_eval.png` for the rain-rate histogram, power "
              "spectrum and residual distribution."]
    open(md, "w").write("\n".join(lines) + "\n")
    print("saved table ->", md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "train", "test"])
    ap.add_argument("--sample", type=int, default=2000, help="crops for distributions")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    ap.add_argument("--root", default=f"/scratch/{USER}/dissertation/prior")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.root, args.split, "*", "*.npz")))
    if not files:
        raise SystemExit(f"no crops for split '{args.split}' under {args.root}")
    print(f"evaluating {len(files)} '{args.split}' crops ...")

    sc = scalar_metrics(files, args.workers)
    dist = distributions(files, args.sample)

    adv, per = sc["advection"], sc["persistence"]
    print(f"\n=== pixel error ({args.split}) ===")
    print(f"  MAE  mm/h : persistence {per['MAE_mmh']:.3f} | advection {adv['MAE_mmh']:.3f}")
    print(f"  RMSE mm/h : persistence {per['RMSE_mmh']:.3f} | advection {adv['RMSE_mmh']:.3f}")
    print(f"  bias mm/h : persistence {per['bias_mmh']:+.3f} | advection {adv['bias_mmh']:+.3f}")
    print(f"\n=== CSI (advection vs persistence) ===")
    for t in THRESHOLDS:
        print(f"  @ {t:>4} mm/h : persistence {per['by_threshold'][t]['CSI']:.3f}"
              f" | advection {adv['by_threshold'][t]['CSI']:.3f}")
    print(f"\nwet-area %: obs {dist['wet_area']['obs']*100:.1f} | "
          f"adv {dist['wet_area']['advection']*100:.1f} | "
          f"pers {dist['wet_area']['persistence']*100:.1f}")

    json.dump({"split": args.split, "n_crops": len(files), "scalar": sc,
               "wet_area": dist["wet_area"], "n_sample_dist": dist["n_sample"]},
              open(os.path.join(OUT, "advection_eval.json"), "w"), indent=2, default=float)
    make_plots(dist, os.path.join(OUT, "advection_eval.png"))
    write_markdown(sc, dist, args.split, len(files), os.path.join(OUT, "advection_eval.md"))
    print(f"\nJSON -> {os.path.join(OUT, 'advection_eval.json')}")


if __name__ == "__main__":
    main()
