#!/usr/bin/env python3
"""
evaluate_diffusion.py - full scorecard for the latent diffusion nowcast, scored
against persistence and the advection prior on identical crops.

Metric definitions and thresholds are deliberately identical to
evaluate_advection.py and fss_analysis.py, so numbers are directly comparable
across the dissertation. Everything is streamed: ensembles are sampled, scored
and discarded crop-batch by crop-batch, so peak memory does not grow with the
split size.

FOUR forecasts are scored, which is the point of this script:
  model_mean    the ensemble mean (the usual deterministic proxy)
  model_member  metrics computed per member, then averaged over members
  advection     A_mmh (the physics prior)
  persistence   x_mmh[-1] (last input held still)

Reporting both model_mean and model_member matters. Averaging an ensemble damps
peaks, so ensemble-mean CSI at high thresholds mostly measures how often the MEAN
exceeds the threshold, which understates a generative model. Individual members do
produce heavy rain. See docs/Diffusion_Run1_Results.md section 4.

Metrics produced:
  deterministic   MAE, RMSE, bias; POD/FAR/CSI/frequency-bias at 0.5/1/2/4/8 mm/h
  spatial         FSS over neighbourhoods 1..101 km (on a subsample)
  probabilistic   fair CRPS, reliability (exact member-fraction bins), rank
                  histogram, spread-skill ratio, outlier rate
  distribution    rain-rate histogram, radially-averaged PSD, wet-area fraction

    conda activate nowcast
    export DISS_SCRATCH=/work/scratch-pw4/$USER/dissertation
    # smoke run first, a few hundred crops
    python "Code/3 - Diffusion stage/evaluate_diffusion.py" --limit 256
    # full validation split
    python "Code/3 - Diffusion stage/evaluate_diffusion.py"

Cost warning: each crop needs members x (2*steps - 1) UNet passes. With the
defaults (8 members, 25 steps) that is ~392 forward passes per crop, so a full
split takes hours on an A100. Use --limit for a first pass, and --members /
--steps to trade cost against ensemble quality.

CorrDiff arm: pass --mu-dir (and optionally --reg-sha) when scoring a checkpoint
trained by train_corrdiff.py. The frozen regression mean is read from
the pack_mu.py memmap, sliced with the same strided index as the latents, and
passed to sample_ensemble, which moves the anchor to z_A + mu_r and widens the
conditioning when the checkpoint was trained with hr_mean_cond on. Pairing is
checked both ways and a mismatch is fatal.

Every scorecard carries its git hash, host, GPU, argv and --batch, and appends a
row to runs.jsonl in --out. The raw contingency counts, the FSS numerator and
denominator, and the CRPS / spread / rank pixel counts are all stored, because
four per-lead scorecards have to be pooled exactly by pool_scorecards.py and an
average of four ratios is not a pooled ratio.

Outputs -> --out (default ~/dissertation_outputs/diffusion/eval/)
    diffusion_eval.json    machine-readable scorecard (the runs-table record)
    diffusion_eval.md      paste-ready tables for the report
    diffusion_eval.png     histogram / PSD / CSI / reliability / rank / spread
    runs.jsonl             append-only one row per invocation
"""
import os, sys, json, time, socket, getpass, argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

from sample_diffusion import (load_denoiser, load_codec, sample_ensemble,
                              read_truth, open_split, check_corrdiff_pairing,
                              resolve_lead_idx, LATENTS, DIFF, VAE_CKPT)
from train_ldm import git_hash
from train_vae_v2 import radial_psd, atomic_json

USER = getpass.getuser()
THRESHOLDS = [0.5, 1.0, 2.0, 4.0, 8.0]          # mm/h, same as evaluate_advection
SCALES     = [1, 5, 11, 21, 51, 101]            # km, same as fss_analysis
WET        = 0.1                                # mm/h -> "raining"
METHODS    = ("model_mean", "model_member", "advection", "persistence")

# Wavelength bands for the spectral decomposition, as half-open intervals so they
# PARTITION the resolved spectrum. Overlapping edges are not cosmetic: an earlier
# band table in this project double-counted them and its variance shares summed to
# 105.3%. The 2-8 km headline band is exactly the union of the last two rows.
PSD_BANDS = (("gt_32km", 32.0, None), ("16_32km", 16.0, 32.0),
             ("8_16km", 8.0, 16.0), ("4_8km", 4.0, 8.0), ("2_4km", 2.0, 4.0))


def psd_band_metrics(psd_model, psd_obs, domain_km=256.0):
    """Band ratios from two radially averaged PSD arrays.

    The 2-8 km band MASK is train_vae_v2.domain_metrics' verbatim (k = 1.., wl =
    256/k, 2 <= wl <= 8, i.e. radial wavenumbers 32..128), so every PSD number in
    this project sits on the same axis. The REDUCTION over that mask is NOT
    inherited: train_vae_v2 line 336 is a mean of per-wavenumber ratios, and the
    two estimators disagree materially (single60 member: 0.763 band power against
    0.644 mean-of-ratios, and the apparent advantage over advection collapses from
    about +15% to about +3%). Both are emitted, with the band POWER ratio as the
    headline because it is the physically meaningful quantity, the fraction of
    band variance the forecast reproduces (CorrDiff_Design.md 3.4 and 10.7)."""
    if psd_model is None or psd_obs is None:
        return None
    m = np.asarray(psd_model, dtype="float64")[1:]     # drop the DC bin
    o = np.asarray(psd_obs, dtype="float64")[1:]
    n = min(len(m), len(o))
    m, o = m[:n], o[:n]
    wl = domain_km / np.arange(1, n + 1)
    band = (wl >= 2.0) & (wl <= 8.0)
    out = {"psd_band_power": float(m[band].sum() / max(o[band].sum(), 1e-12)),
           "psd_mean_ratio": float(np.mean(m[band] / np.maximum(o[band], 1e-12))),
           "n_wavenumbers_2_8km": int(band.sum()), "bands": {}}
    total_o = float(o[wl >= 2.0].sum())
    for name, lo, hi in PSD_BANDS:
        b = (wl >= lo) if lo == 2.0 else (wl > lo)
        if hi is not None:
            b = b & (wl <= hi)
        po, pm = float(o[b].sum()), float(m[b].sum())
        out["bands"][name] = {"ratio": pm / max(po, 1e-12),
                              "obs_power": po, "model_power": pm,
                              "obs_share": po / max(total_o, 1e-12),
                              "n_wavenumbers": int(b.sum())}
    return out


def run_stamp(args, extra=None):
    """The provenance block every scorecard carries. The training scripts already
    write runs.jsonl and stamp a git hash; the scoring scripts did not, and this
    arm is about to produce a large number of scorecards, so the gap closes first
    (CorrDiff_Design.md 3.6). --batch is here because evaluate_diffusion seeds per
    crop-batch as seed + 7919*s0: a scorecard that does not record its batch size
    cannot be paired with another one."""
    s = {"git": git_hash(), "host": socket.gethostname(),
         "gpu": (torch.cuda.get_device_name(0)
                 if torch.cuda.is_available() else "cpu"),
         "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv,
         "batch": args.batch, "tag": args.tag}
    if extra:
        s.update(extra)
    return s


RUN_KEYS = ("ts", "git", "host", "gpu", "split", "lead", "n_crops",
            "n_crops_available", "subsample", "members", "steps", "guidance",
            "churn", "batch", "seed", "ckpt", "ckpt_epoch", "vae_sha256",
            "sigma_data", "wall_min", "tag", "field", "reg_ckpt", "reg_sha256",
            "mu_dir")


def append_runs_row(r, methods, out_dir):
    """One compact append-only row per scorecard. Built by intersection so the
    same helper serves the probabilistic and the deterministic scorer."""
    row = {k: r[k] for k in RUN_KEYS if k in r}
    row["MAE"] = {m: r["deterministic"][m]["MAE_mmh"] for m in methods}
    row["CSI8"] = {m: r["deterministic"][m]["by_threshold"][8.0]["CSI"]
                   for m in methods}
    if "probabilistic" in r:
        row["CRPS_fair_mmh"] = r["probabilistic"]["CRPS_fair_mmh"]
    with open(os.path.join(out_dir, "runs.jsonl"), "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


# ----------------------------------------------------------------------------
# Accumulators
# ----------------------------------------------------------------------------
def new_det():
    return {"sum_abs": 0.0, "sum_sq": 0.0, "sum_err": 0.0, "n": 0,
            "cont": {t: [0, 0, 0] for t in THRESHOLDS}}


def acc_det(acc, F, y, v, weight_n=1):
    """Accumulate pixel error + contingency counts for one field (or one member).
    weight_n counts the pixels once per member so member-averaged means are right."""
    d = (F - y)[v]
    acc["sum_abs"] += float(np.abs(d).sum())
    acc["sum_sq"]  += float((d * d).sum())
    acc["sum_err"] += float(d.sum())
    acc["n"]       += int(v.sum()) * weight_n
    for t in THRESHOLDS:
        o, p = (y >= t), (F >= t)
        acc["cont"][t][0] += int(np.sum(v & o & p))
        acc["cont"][t][1] += int(np.sum(v & o & ~p))
        acc["cont"][t][2] += int(np.sum(v & ~o & p))


def finish_det(acc):
    n = max(acc["n"], 1)
    out = {"MAE_mmh": acc["sum_abs"] / n,
           "RMSE_mmh": (acc["sum_sq"] / n) ** 0.5,
           "bias_mmh": acc["sum_err"] / n,
           "n_pixels": acc["n"], "by_threshold": {}}
    for t in THRESHOLDS:
        H, M, F = acc["cont"][t]
        out["by_threshold"][t] = {
            "POD": H / (H + M) if (H + M) else float("nan"),
            "FAR": F / (H + F) if (H + F) else float("nan"),
            "CSI": H / (H + M + F) if (H + M + F) else float("nan"),
            "freq_bias": (H + F) / (H + M) if (H + M) else float("nan"),
            # Keep the raw contingency counts. Ratios cannot be pooled across the
            # four per-lead JSONs (averaging four CSIs is not a CSI), and
            # re-running one lead costs 3.26 A100-hours (design 3.6, 3.9).
            "counts": [H, M, F]}
    return out


def crps_fair(members, obs, valid):
    """Fair (unbiased) ensemble CRPS, averaged over valid pixels.

        CRPS = mean_m |x_m - y|  -  1/(2 M (M-1)) sum_i sum_j |x_i - x_j|

    The 1/(M(M-1)) coefficient is the fair estimator (Ferro 2014). The biased NRG
    form uses 1/M^2, which flatters small ensembles by under-penalising spread.
    The in-training diagnostic used the biased form, so numbers here differ
    slightly from the training log by design.
    """
    M = members.shape[0]
    X = members[:, valid]                                   # (M, P)
    yv = obs[valid][None]
    t1 = np.abs(X - yv).mean(axis=0)
    if M > 1:
        # sum_{i,j} |x_i - x_j| via sorting: O(M log M + M) per pixel instead of M^2
        Xs = np.sort(X, axis=0)
        w = (2 * np.arange(1, M + 1) - M - 1).astype("float64")[:, None]
        pair = 2.0 * (w * Xs).sum(axis=0)                    # = sum_i sum_j |x_i-x_j|
        t2 = pair / (2.0 * M * (M - 1))
    else:
        t2 = 0.0
    return float((t1 - t2).sum()), int(valid.sum())


# ----------------------------------------------------------------------------
# Main evaluation stream
# ----------------------------------------------------------------------------
def evaluate(args, device):
    mm, files, n_all, meta = open_split(args.latents_dir, args.split, args.lead)
    # --limit must SUBSAMPLE the split, never truncate it. Row order is pack order,
    # which pack_latents.py builds as sorted(glob(prior_ml/{split}/YYYYMMDD/*.npz)),
    # so it is chronological: taking the first N crops returns the earliest val days
    # only (winter and early spring) and biases every regime-sensitive metric, CSI@8
    # worst of all. A deterministic stride spans the whole record, needs no seed, and
    # mirrors what train_ldm.load_diag_crops already does for its fixed crops.
    if args.limit and args.limit < n_all:
        idx = np.unique(np.linspace(0, n_all - 1, args.limit).round().astype(int))
    else:
        idx = np.arange(n_all)
    n = len(idx)
    # Sanity line: a truncating --limit would show first and last on the same day.
    print(f"  crops {n}/{n_all} ({'stride' if n < n_all else 'full split'}) | "
          f"first {os.path.basename(os.path.dirname(files[idx[0]]))} | "
          f"last {os.path.basename(os.path.dirname(files[idx[-1]]))}", flush=True)
    den, ck, cond_mode, ck_leads, hr_mean_cond = load_denoiser(args.ckpt, device)
    # CorrDiff arm: the frozen mean is read from the pack_mu.py memmap and sliced
    # with the SAME strided index as the latents, so alignment is by construction.
    check_corrdiff_pairing(ck, args.mu_dir)
    mu_mm = mu_meta = None
    if args.mu_dir:
        # Imported here rather than at module scope, so the generic evaluator does
        # not depend on the CorrDiff trainer unless a CorrDiff run is being scored.
        from train_corrdiff import open_mu_split
        mu_mm, mu_meta = open_mu_split(args.mu_dir, args.latents_dir, args.split,
                                       args.lead, n_rows=n_all,
                                       reg_sha=args.reg_sha)
    lead_idx = resolve_lead_idx(ck_leads, args.lead)
    vae = load_codec(args.vae, meta, device, strict_sha=not args.allow_vae_mismatch)
    latent_scale = float(meta["latent_scale"])
    mean, std = float(meta["norm"]["mean"]), float(meta["norm"]["std"])
    M = args.members

    det = {m: new_det() for m in METHODS}
    fss = {(m, t, s): [0.0, 0.0] for m in ("model_mean", "model_m0", "advection",
                                           "persistence")
           for t in THRESHOLDS for s in SCALES}
    crps_sum = crps_n = 0.0
    # reliability: forecast probability is k/M for k members exceeding, so the bins
    # are exact (no binning error). counts[t][k] = (n_forecast, n_obs_yes)
    rel = {t: np.zeros((M + 1, 2)) for t in THRESHOLDS}
    rank_hist = np.zeros(M + 1)
    spread_sq_sum = 0.0            # sum of ensemble variance over pixels
    err_sq_sum = 0.0               # sum of squared ens-mean error (same pixels)
    spread_n = 0
    outliers = 0
    rbins = np.logspace(np.log10(WET), np.log10(128), 41)
    hist = {k: np.zeros(len(rbins) - 1) for k in
            ("obs", "model_mean", "model_member", "advection", "persistence")}
    wet_area = {k: [0.0, 0] for k in hist}
    psd = {k: None for k in hist}
    n_psd = 0
    fss_done = 0
    fss_every = max(1, n // max(args.fss_sample, 1))
    # PSD used to accept the first psd_sample crops that pass the validity filter,
    # which on a full split meant the earliest ~200 crops, under half of one val day.
    # Space them over the whole record instead. The >= n_psd * psd_stride test is
    # self-correcting: if the validity filter rejects a crop the next qualifying one
    # is taken, so the sample still fills to psd_sample where the data allows.
    psd_stride = n / max(args.psd_sample, 1)

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    done = 0
    for s0 in range(0, n, args.batch):
        sel = idx[s0:s0 + args.batch]
        rows = torch.from_numpy(np.asarray(mm[sel], dtype="float32"))
        mu_rows = (torch.from_numpy(np.asarray(mu_mm[sel], dtype="float32"))
                   if mu_mm is not None else None)
        y, A, P, V = read_truth([files[i] for i in sel])
        members = sample_ensemble(den, vae, rows, cond_mode, latent_scale, mean, std,
                                  members=M, steps=args.steps, guidance=args.guidance,
                                  churn=args.churn, seed=args.seed + 7919 * s0,
                                  device=device, lead_idx=lead_idx,
                                  mu=mu_rows, hr_mean_cond=hr_mean_cond)
        ens = members.mean(axis=1)

        for b in range(len(sel)):
            pos = s0 + b                      # position within idx, not the raw row
            v = V[b]
            if not v.any():
                continue
            acc_det(det["model_mean"], ens[b], y[b], v)
            acc_det(det["advection"], A[b], y[b], v)
            acc_det(det["persistence"], P[b], y[b], v)
            for m in range(M):                     # member-level skill
                acc_det(det["model_member"], members[b, m], y[b], v, weight_n=1)

            # ---- probabilistic ----
            cs, cn = crps_fair(members[b], y[b], v)
            crps_sum += cs; crps_n += cn
            for t in THRESHOLDS:
                k = (members[b] >= t).sum(axis=0)[v]          # 0..M members exceeding
                obs_yes = (y[b] >= t)[v]
                np.add.at(rel[t][:, 0], k, 1.0)
                np.add.at(rel[t][:, 1], k, obs_yes.astype("float64"))
            # rank histogram on wet pixels only: an all-dry pixel is a huge tie and
            # would swamp the histogram. Ties broken randomly (standard practice).
            wm = v & (y[b] >= WET)
            if wm.any():
                Xw = members[b][:, wm]                        # (M, P)
                yw = y[b][wm]
                below = (Xw < yw[None]).sum(axis=0)
                ties = (Xw == yw[None]).sum(axis=0)
                r = below + (rng.random(ties.shape) * (ties + 1)).astype(int)
                np.add.at(rank_hist, np.clip(r, 0, M), 1.0)
                outliers += int(np.sum((yw < Xw.min(axis=0)) | (yw > Xw.max(axis=0))))
            spread_sq_sum += float(members[b].var(axis=0, ddof=1)[v].sum())
            err_sq_sum    += float(((ens[b] - y[b]) ** 2)[v].sum())
            spread_n      += int(v.sum())

            # ---- distributions ----
            for key, F in (("obs", y[b]), ("model_mean", ens[b]),
                           ("model_member", members[b, 0]),
                           ("advection", A[b]), ("persistence", P[b])):
                vals = F[v]
                hist[key] += np.histogram(vals[vals >= WET], bins=rbins)[0]
                wet_area[key][0] += float((vals >= WET).mean()); wet_area[key][1] += 1
            if (v.mean() > 0.99 and n_psd < args.psd_sample
                    and pos >= n_psd * psd_stride):
                for key, F in (("obs", y[b]), ("model_mean", ens[b]),
                               ("model_member", members[b, 0]),
                               ("advection", A[b]), ("persistence", P[b])):
                    p = radial_psd(F)
                    psd[key] = p if psd[key] is None else psd[key] + p
                n_psd += 1

            # ---- FSS on a subsample (uniform_filter is the expensive part) ----
            if pos % fss_every == 0:
                for name, F in (("model_mean", ens[b]), ("model_m0", members[b, 0]),
                                ("advection", A[b]), ("persistence", P[b])):
                    for t in THRESHOLDS:
                        Io = (np.nan_to_num(y[b]) >= t).astype("float32")
                        If = (np.nan_to_num(F) >= t).astype("float32")
                        for sc in SCALES:
                            if sc > 1:
                                Mo = uniform_filter(Io, sc, mode="constant")
                                Mf = uniform_filter(If, sc, mode="constant")
                            else:
                                Mo, Mf = Io, If
                            fss[(name, t, sc)][0] += float(((Mf - Mo) ** 2).sum())
                            fss[(name, t, sc)][1] += float((Mf ** 2 + Mo ** 2).sum())
                fss_done += 1
            done += 1
        el = time.time() - t0
        print(f"  {done}/{n} crops | {done/el:.2f} crops/s | "
              f"eta {(n-done)/max(done/el,1e-9)/60:.0f} min", flush=True)

    # ---- finalise ----
    res = {m: finish_det(det[m]) for m in METHODS}
    fss_out = {}
    for (name, t, sc), (num, den_) in fss.items():
        fss_out[f"{name}|{t}|{sc}"] = (1 - num / den_) if den_ > 0 else float("nan")
        # FSS pools as 1 - sum(num)/sum(den) across leads, which needs both terms;
        # the ratio alone is not poolable (design 3.6, 3.9).
        fss_out[f"{name}|{t}|{sc}|num"] = num
        fss_out[f"{name}|{t}|{sc}|den"] = den_
    prob = {
        "CRPS_fair_mmh": crps_sum / max(crps_n, 1),
        "CRPS_n_pixels": crps_n,
        "spread_n_pixels": spread_n,
        "rank_n": float(rank_hist.sum()),
        "outlier_count": int(outliers),
        "CRPS_estimator": "fair (Ferro 2014), 1/(2M(M-1)) spread term",
        "spread_rmse_ratio": (spread_sq_sum / max(spread_n, 1)) ** 0.5 /
                             max((err_sq_sum / max(spread_n, 1)) ** 0.5, 1e-9),
        "spread_mmh": (spread_sq_sum / max(spread_n, 1)) ** 0.5,
        "rmse_ens_mean_mmh": (err_sq_sum / max(spread_n, 1)) ** 0.5,
        "outlier_rate": outliers / max(rank_hist.sum(), 1),
        "outlier_rate_ideal": 2.0 / (M + 1),
        "rank_histogram": (rank_hist / max(rank_hist.sum(), 1)).tolist(),
        "rank_flatness_rmse": float(np.sqrt(np.mean(
            (rank_hist / max(rank_hist.sum(), 1) - 1.0 / (M + 1)) ** 2))),
        "reliability": {str(t): {"prob": (np.arange(M + 1) / M).tolist(),
                                "n": rel[t][:, 0].tolist(),
                                "obs_freq": (rel[t][:, 1] /
                                             np.maximum(rel[t][:, 0], 1)).tolist()}
                        for t in THRESHOLDS},
    }
    psd_avg = {k: (None if v is None else (v / max(n_psd, 1))) for k, v in psd.items()}
    dist = {"rbins": rbins.tolist(),
            "hist": {k: v.tolist() for k, v in hist.items()},
            "psd": {k: (None if v is None else v.tolist()) for k, v in psd_avg.items()},
            "n_psd": n_psd,
            # Band ratios computed here rather than offline, so the ml_v2
            # comparator and the CorrDiff arm sit on the same axis by
            # construction rather than by whoever recomputes them (design 10.7).
            "psd_bands": {k: psd_band_metrics(psd_avg[k], psd_avg["obs"])
                          for k in psd_avg if k != "obs"},
            "wet_area": {k: (v[0] / max(v[1], 1)) for k, v in wet_area.items()},
            # the denominator, so wet area pools exactly rather than as an
            # average of per-lead averages
            "wet_area_n": int(wet_area["obs"][1])}
    return {"split": args.split, "lead": args.lead, "n_crops": n,
            "n_crops_available": n_all,
            "subsample": "stride" if n < n_all else "full",
            "members": M, "steps": args.steps, "guidance": args.guidance,
            "churn": args.churn, "seed": args.seed,
            "ckpt": os.path.abspath(args.ckpt),
            "ckpt_epoch": ck.get("epoch"), "ckpt_val_loss": ck.get("val_loss"),
            "vae_sha256": meta.get("vae_sha256"),
            "sigma_data": meta.get("sigma_data"),
            "sigma_data_model": ck.get("sigma_data", ck["config"].get("sigma_data")),
            "mu_dir": (os.path.abspath(args.mu_dir) if args.mu_dir else None),
            "reg_sha256": (mu_meta or {}).get("reg_sha256"),
            "hr_mean_cond": ck["config"].get("hr_mean_cond", "off"),
            "n_fss_crops": fss_done,
            **run_stamp(args),
            "deterministic": res, "fss": fss_out, "probabilistic": prob,
            "distribution": dist}


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def make_plots(r, png):
    d, dist, prob = r["deterministic"], r["distribution"], r["probabilistic"]
    rb = np.array(dist["rbins"]); rc = np.sqrt(rb[:-1] * rb[1:])
    styles = {"obs": ("k", 2.2), "model_member": ("C2", 1.6),
              "model_mean": ("C0", 1.6), "advection": ("C1", 1.4),
              "persistence": ("C3", 1.2)}
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    for k, (c, lw) in styles.items():
        h = np.array(dist["hist"][k]); h = h / max(h.sum(), 1)
        ax[0, 0].loglog(rc, h, color=c, lw=lw, label=k)
    ax[0, 0].set(title="Rain-rate distribution (wet pixels)",
                 xlabel="mm/h", ylabel="frequency")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

    for k, (c, lw) in styles.items():
        p = dist["psd"][k]
        if p is None:
            continue
        p = np.array(p); kk = np.arange(1, len(p)); wl = 256.0 / kk
        ax[0, 1].loglog(wl, p[1:], color=c, lw=lw, label=k)
    ax[0, 1].set(title=f"Power spectrum ({dist['n_psd']} clean crops)",
                 xlabel="wavelength (km)", ylabel="power")
    ax[0, 1].invert_xaxis(); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

    for m, (c, lw) in styles.items():
        if m == "obs":
            continue
        ax[0, 2].plot(THRESHOLDS, [d[m]["by_threshold"][t]["CSI"] for t in THRESHOLDS],
                      "o-", color=c, lw=lw, label=m)
    ax[0, 2].set(title="CSI vs threshold", xlabel="mm/h", ylabel="CSI",
                 xscale="log", ylim=(0, 1))
    ax[0, 2].legend(fontsize=8); ax[0, 2].grid(alpha=0.3)

    for t in (1.0, 8.0):
        rl = prob["reliability"][str(t)]
        ax[1, 0].plot(rl["prob"], rl["obs_freq"], "o-", label=f">= {t:g} mm/h")
    ax[1, 0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax[1, 0].set(title="Reliability (ensemble probability)",
                 xlabel="forecast probability", ylabel="observed frequency")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    rh = np.array(prob["rank_histogram"])
    ax[1, 1].bar(np.arange(len(rh)), rh, color="C0")
    ax[1, 1].axhline(1.0 / len(rh), color="k", ls="--", lw=1, label="flat = reliable")
    ax[1, 1].set(title="Rank histogram (wet pixels)", xlabel="rank of obs",
                 ylabel="frequency")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3, axis="y")

    names = ["model_mean", "model_member", "advection", "persistence"]
    ax[1, 2].bar(names, [d[m]["MAE_mmh"] for m in names],
                 color=[styles[m][0] for m in names])
    ax[1, 2].set(title=f"MAE (mm/h) | CRPS {prob['CRPS_fair_mmh']:.3f} | "
                       f"spread/RMSE {prob['spread_rmse_ratio']:.2f}",
                 ylabel="mm/h")
    ax[1, 2].tick_params(axis="x", rotation=20)
    ax[1, 2].grid(alpha=0.3, axis="y")

    plt.tight_layout(); plt.savefig(png, dpi=120, bbox_inches="tight"); plt.close()
    print("plots ->", png, flush=True)


def write_markdown(r, md):
    d, prob, dist = r["deterministic"], r["probabilistic"], r["distribution"]
    mm, mb, ad, pe = (d["model_mean"], d["model_member"],
                      d["advection"], d["persistence"])
    L = [f"# Diffusion nowcast scorecard (`{r['split']}` split)\n",
         f"_{r['n_crops']} crops, {r['members']}-member ensembles, "
         f"{r['steps']} Heun steps, guidance {r['guidance']}. "
         f"Checkpoint epoch {r['ckpt_epoch']} (val loss {r['ckpt_val_loss']}). "
         f"FSS from {r['n_fss_crops']} crops, PSD from {dist['n_psd']}._\n",
         "`model_mean` is the ensemble mean; `model_member` is the average skill of a "
         "single member. Averaging damps peaks, so the mean scores better on MAE/RMSE "
         "and worse at high thresholds. Both are reported, see "
         "`docs/Diffusion_Run1_Results.md`.\n",
         "## Pixel error (lower is better)\n",
         "| metric | persistence | advection | model (mean) | model (member) |",
         "|---|---|---|---|---|",
         f"| MAE (mm/h) | {pe['MAE_mmh']:.3f} | {ad['MAE_mmh']:.3f} | "
         f"**{mm['MAE_mmh']:.3f}** | {mb['MAE_mmh']:.3f} |",
         f"| RMSE (mm/h) | {pe['RMSE_mmh']:.3f} | {ad['RMSE_mmh']:.3f} | "
         f"**{mm['RMSE_mmh']:.3f}** | {mb['RMSE_mmh']:.3f} |",
         f"| bias (mm/h) | {pe['bias_mmh']:+.3f} | {ad['bias_mmh']:+.3f} | "
         f"{mm['bias_mmh']:+.3f} | {mb['bias_mmh']:+.3f} |\n",
         "## CSI by threshold (higher is better)\n",
         "| mm/h | persistence | advection | model (mean) | model (member) |",
         "|---|---|---|---|---|"]
    for t in THRESHOLDS:
        L.append(f"| {t:g} | {pe['by_threshold'][t]['CSI']:.3f} | "
                 f"{ad['by_threshold'][t]['CSI']:.3f} | "
                 f"{mm['by_threshold'][t]['CSI']:.3f} | "
                 f"{mb['by_threshold'][t]['CSI']:.3f} |")
    L += ["\n## Detection at 1 mm/h\n",
          "| metric | persistence | advection | model (mean) | model (member) |",
          "|---|---|---|---|---|"]
    for k in ("POD", "FAR", "freq_bias"):
        L.append(f"| {k} | {pe['by_threshold'][1.0][k]:.3f} | "
                 f"{ad['by_threshold'][1.0][k]:.3f} | "
                 f"{mm['by_threshold'][1.0][k]:.3f} | "
                 f"{mb['by_threshold'][1.0][k]:.3f} |")
    L += ["\n## Probabilistic (the ensemble's reason to exist)\n",
          f"- **CRPS (fair)**: {prob['CRPS_fair_mmh']:.4f} mm/h. A deterministic "
          f"forecast's CRPS equals its MAE, so compare against advection "
          f"{ad['MAE_mmh']:.3f} and persistence {pe['MAE_mmh']:.3f}.",
          f"- **Spread / RMSE**: {prob['spread_rmse_ratio']:.3f} "
          f"(spread {prob['spread_mmh']:.3f}, ens-mean RMSE "
          f"{prob['rmse_ens_mean_mmh']:.3f}). Near 1 is well dispersed, "
          "below 1 is over-confident.",
          f"- **Outlier rate**: {prob['outlier_rate']:.3f} vs ideal "
          f"{prob['outlier_rate_ideal']:.3f}.",
          f"- **Rank-histogram flatness** (RMSE from flat): "
          f"{prob['rank_flatness_rmse']:.4f}, 0 is perfectly flat.",
          "\n## FSS (model mean vs a single member)\n",
          "| threshold | " + " | ".join(f"{s} km" for s in SCALES) + " |",
          "|---" * (len(SCALES) + 1) + "|"]
    for t in THRESHOLDS:
        row = " | ".join(f"{r['fss'][f'model_mean|{t}|{s}']:.3f}" for s in SCALES)
        L.append(f"| mean, {t:g} mm/h | {row} |")
    for t in THRESHOLDS:
        row = " | ".join(f"{r['fss'][f'model_m0|{t}|{s}']:.3f}" for s in SCALES)
        L.append(f"| member, {t:g} mm/h | {row} |")
    bands = dist.get("psd_bands") or {}
    if any(bands.values()):
        names = [k for k in ("model_member", "model_mean", "advection",
                             "persistence") if bands.get(k)]
        ref = bands[names[0]]["bands"]
        L += [f"\n## Power spectrum by band ({dist['n_psd']} clean crops)\n",
              "Ratio of forecast band power to observed band power, 1.0 = matched. "
              "Bands partition the resolved spectrum, so `obs share` sums to 1. The "
              "2-8 km headline is the union of the last two columns; both estimators "
              "are given because they disagree materially (see "
              "`docs/designs/Metrics_Catalogue.md`).\n",
              "| field | " + " | ".join(b for b in ref) +
              " | 2-8 km band power | 2-8 km mean-of-ratios |",
              "|---" * (len(ref) + 3) + "|"]
        for nm in names:
            b = bands[nm]
            L.append(f"| {nm} | " +
                     " | ".join(f"{b['bands'][k]['ratio']:.3f}" for k in ref) +
                     f" | {b['psd_band_power']:.3f} | {b['psd_mean_ratio']:.3f} |")
        L.append("| _obs share of variance_ | " +
                 " | ".join(f"{ref[k]['obs_share']*100:.1f}%" for k in ref) +
                 " | | |")
    L += ["\n## Wet-area fraction (>= 0.1 mm/h)\n",
          "| field | % |", "|---|---|"]
    for k, v in dist["wet_area"].items():
        L.append(f"| {k} | {v*100:.1f} |")
    L += ["\nSee `diffusion_eval.png` for the histogram, PSD, CSI, reliability, "
          "rank histogram and spread panels."]
    open(md, "w").write("\n".join(L) + "\n")
    print("tables ->", md, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(DIFF, "diff_best.pt"),
                    help="ALWAYS name an explicit ckpt_epNNN.pt. The default "
                         "diff_best.pt ranks on val EDM loss alone, which is "
                         "anti-correlated with small-scale power in every run of "
                         "this project, so it is systematically the smoothest "
                         "checkpoint the run produced.")
    ap.add_argument("--vae", default=VAE_CKPT)
    ap.add_argument("--latents-dir", default=LATENTS)
    ap.add_argument("--mu-dir", default=None,
                    help="directory of {split}_mu{_LNN}.npy packs (pack_mu.py). "
                         "REQUIRED for a CorrDiff checkpoint, refused for a plain one.")
    ap.add_argument("--reg-sha", default=None,
                    help="assert the mu pack's reg_sha256; a mismatch is fatal")
    ap.add_argument("--split", default="val", choices=["val", "train", "test"])
    ap.add_argument("--lead", type=int, default=None)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--churn", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=16, help="crops per sampling pass")
    ap.add_argument("--limit", type=int, default=None, help="cap crops (smoke run)")
    ap.add_argument("--fss-sample", type=int, default=400,
                    help="crops used for FSS (uniform_filter is the slow part)")
    ap.add_argument("--psd-sample", type=int, default=200,
                    help="clean crops used for the power spectrum")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(DIFF, "eval"))
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    ap.add_argument("--allow-vae-mismatch", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found; sampling on CPU is impractically slow.", flush=True)

    t0 = time.time()
    r = evaluate(args, device)
    r["wall_min"] = round((time.time() - t0) / 60, 1)

    d, prob = r["deterministic"], r["probabilistic"]
    print(f"\n=== {r['split']} split, {r['n_crops']} crops, {r['members']} members ===")
    for m in METHODS:
        print(f"  {m:14s} MAE {d[m]['MAE_mmh']:.3f} | RMSE {d[m]['RMSE_mmh']:.3f} | "
              f"CSI@1 {d[m]['by_threshold'][1.0]['CSI']:.3f} | "
              f"CSI@8 {d[m]['by_threshold'][8.0]['CSI']:.3f}")
    print(f"  CRPS (fair)    {prob['CRPS_fair_mmh']:.4f}  "
          f"(advection MAE {d['advection']['MAE_mmh']:.3f})")
    print(f"  spread/RMSE    {prob['spread_rmse_ratio']:.3f} | "
          f"outliers {prob['outlier_rate']:.3f} (ideal {prob['outlier_rate_ideal']:.3f})")

    pb = r["distribution"]["psd_bands"].get("model_member")
    if pb:
        pa = r["distribution"]["psd_bands"].get("advection")
        print(f"  PSD 2-8 km     band power {pb['psd_band_power']:.3f} "
              f"(mean-of-ratios {pb['psd_mean_ratio']:.3f}) | advection "
              f"{pa['psd_band_power']:.3f} / {pa['psd_mean_ratio']:.3f}, "
              f"member field, {r['distribution']['n_psd']} crops")

    tag = args.tag or ""
    atomic_json(r, os.path.join(args.out, f"diffusion_eval{tag}.json"))
    make_plots(r, os.path.join(args.out, f"diffusion_eval{tag}.png"))
    write_markdown(r, os.path.join(args.out, f"diffusion_eval{tag}.md"))
    append_runs_row(r, METHODS, args.out)
    print(f"\ndone in {r['wall_min']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
