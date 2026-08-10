#!/usr/bin/env python3
"""
hpo_baselines.py - paired persistence, advection and climatology scores on the
EXACT diagnostic crops a hyperparameter trial will be scored on.

The supervisor's brief requires every run to be compared against persistence and
the pysteps/advection baselines on the same validation set. Two thirds of that
already exists: train_ldm.py and train_corrdiff.py score the advection prior
inside sampled_diagnostics on the identical crops and write mae_adv, crps_adv,
csi_1_adv and csi_8_adv into train_log.json, and that advection field IS the
pysteps baseline (dense Lucas-Kanade optical flow plus Germann-Zawadzki
extrapolation, computed once in build_advection_prior.py).

Persistence is the missing third, and it deliberately is NOT added by editing the
trainers. train_ldm.py is the file that produced single60, ml_v1 and ml_v2, and
docs/PLAN_to_Aug28.md records that it was left byte-identical apart from four
docstring lines precisely so those three runs stay provably comparable. Adding a
column to its diagnostics would break that guarantee for the sake of a number
that can be recovered exactly from outside, which is what this script does.

"Exactly" is the whole point, so the crop selection is reproduced rather than
approximated. train_ldm.py picks its diagnostic crops as:

    va_ds     = LatentRows(val shards, limit=max(400, train_limit // 10))
                where LatentRows strides sel = linspace(0, N-1, limit) over the
                CONCATENATED index, so every lead is represented
    diag_idx  = linspace(0, len(va_ds) - 1, K).astype(int),  K = min(sample_crops,
                len(va_ds))
    file      = {split}_latents{suffix}_index.json[shard][local_row]

so the crop set is a deterministic function of (latents_dir, leads, train --limit,
--sample-crops) and nothing else. This script takes those four, rebuilds the same
index arithmetic from the .npy headers and the index JSONs, and scores the
baselines on the resulting crops with the same estimators the trainer uses:
per-crop-mean MAE averaged over crops, POOLED contingency counts for CSI, and
the radial PSD band ratio accumulated over the crops that pass the same
validity gate.

One consequence must be stated in the write-up rather than buried: because
`va_ds`'s length depends on the training --limit, each fidelity rung has its own
crop set. Comparisons between trials are therefore paired WITHIN a rung, which
is what the search needs, and are not paired ACROSS rungs, which is why the
Spearman rank correlation between rungs (hpo_report.py) is reported as a
diagnostic rather than assumed.

Runs on CPU with numpy only. It needs read access to the latent pack index JSONs
and to the npz crop cache they point at, not to the latents themselves.

    conda activate nowcast
    export DISS_SCRATCH=/work/scratch-nopw2/$USER/dissertation

    # one entry per rung of the ldm ladder, written to the study directory
    python hpo_baselines.py --latents-dir $DISS_SCRATCH/latents_ml_ep17 \
        --leads 15,30,45,60 --arm ldm --out $HPO/ldm_coarse/baselines.json

    # a single explicit fidelity instead of the whole ladder
    python hpo_baselines.py --latents-dir $DISS_SCRATCH/latents_ml_ep17 \
        --leads 15,30,45,60 --limit 60000 --sample-crops 96 --rung-name r0 \
        --out $HPO/ldm_coarse/baselines.json

Output: a JSON keyed by rung name. hpo_search.py --baselines merges it into every
trial's metric namespace, so `mae_pers` and `crps_pers` become usable in an
objective or a gate expression.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hpo_spaces as S                                            # noqa: E402

CSI_T = [1.0, 8.0]                       # must match train_ldm.CSI_T


# ----------------------------------------------------------------------------
# radial_psd: a verbatim copy of train_vae_v2.radial_psd.
#
# Copied rather than imported because train_vae_v2 imports torch at module level
# and this script is meant to run on the CPU box, where torch is not installed.
# The copy is checked against the original at startup whenever the original is
# importable, so the two can never drift silently.
# ----------------------------------------------------------------------------
def radial_psd(field):
    f = np.nan_to_num(field).astype("float64")
    f = f - f.mean()
    P = np.fft.fftshift(np.abs(np.fft.fft2(f)) ** 2)
    Hh, Ww = f.shape
    Y, X = np.ogrid[:Hh, :Ww]
    r = np.hypot(Y - Hh // 2, X - Ww // 2).astype(int)
    return np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


def _verify_psd_copy():
    """If train_vae_v2 can be imported, assert this copy still agrees with it."""
    here = HERE
    for d in sorted(glob.glob(os.path.join(os.path.dirname(here), "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "train_vae_v2.py")):
            sys.path.insert(0, d)
            break
    try:
        from train_vae_v2 import radial_psd as ref
    except Exception:
        return "not-checked (train_vae_v2 not importable here)"
    rng = np.random.default_rng(0)
    a = rng.normal(size=(256, 256))
    if not np.allclose(radial_psd(a), ref(a), rtol=1e-12, atol=0):
        raise SystemExit("ERROR: the local radial_psd copy has drifted from "
                         "train_vae_v2.radial_psd. Fix the copy before scoring "
                         "anything: a PSD ratio computed under two definitions "
                         "is not comparable.")
    return "verified against train_vae_v2.radial_psd"


# ----------------------------------------------------------------------------
# Crop selection, reproducing train_ldm.LatentRows + load_diag_crops exactly
# ----------------------------------------------------------------------------
def shard_suffix(lead=None):
    return "" if lead is None else f"_L{lead:02d}"


def build_index(latents_dir, split, leads):
    """Return (shard_ids, local_rows, files_per_shard) for the concatenated pack,
    before any --limit is applied. Mirrors LatentRows.__init__."""
    suffixes = [shard_suffix(L) for L in (leads if leads else [None])]
    npys, idxs = [], []
    for suf in suffixes:
        npy = os.path.join(latents_dir, f"{split}_latents{suf}.npy")
        idx = os.path.join(latents_dir, f"{split}_latents{suf}_index.json")
        for p in (npy, idx):
            if not os.path.exists(p):
                raise SystemExit(f"ERROR: {p} not found. Check --latents-dir and "
                                 "--leads against the pack that the trials will use.")
        npys.append(npy)
        idxs.append(json.load(open(idx)))
    sizes = [np.load(p, mmap_mode="r").shape[0] for p in npys]
    shard = np.concatenate([np.full(n, s, dtype=np.int16)
                            for s, n in enumerate(sizes)])
    local = np.concatenate([np.arange(n, dtype=np.int64) for n in sizes])
    return shard, local, idxs, sizes


def diag_files(latents_dir, leads, train_limit, sample_crops, split="val"):
    """The ordered (shard, local_row, npz_path) the trainer's diagnostics open."""
    shard, local, idxs, sizes = build_index(latents_dir, split, leads)
    # train_ldm.main(): va_ds = LatentRows(..., limit=max(400, args.limit // 10))
    val_limit = max(400, train_limit // 10) if train_limit else None
    if val_limit and val_limit < len(shard):
        sel = np.linspace(0, len(shard) - 1, val_limit).astype(np.int64)
        shard, local = shard[sel], local[sel]
    K = min(sample_crops, len(shard))
    diag_idx = np.linspace(0, len(shard) - 1, K).astype(int)
    files = []
    for i in diag_idx:
        s = int(shard[int(i)])
        files.append((s, int(local[int(i)]), idxs[s][int(local[int(i)])]))
    return files, len(shard), sizes


def verify_against_trainer(latents_dir, leads, train_limit, sample_crops,
                           split="val"):
    """Assert that the crop selection above is byte-identical to the one
    train_ldm.py will actually make.

    The index arithmetic is reproduced here rather than imported, because
    importing it would drag in torch and this script is meant to run on the CPU
    box. Reproduced arithmetic drifts, so on any machine where train_ldm IS
    importable (every GPU machine) the reproduction is checked against the real
    thing rather than trusted. A mismatch means every paired comparison in the
    study would be against the wrong crops, so it is a hard failure, not a
    warning."""
    try:
        from train_ldm import LatentRows, shard_suffix as tshard
    except Exception as e:
        return (f"NOT CHECKED ({type(e).__name__}) - train_ldm is not importable "
                "here, which is expected on a CPU-only box. Run this once on a "
                "GPU machine to confirm the crop selection.")
    paths = [os.path.join(latents_dir, f"{split}_latents{tshard(L)}.npy")
             for L in (leads if leads else [None])]
    ds = LatentRows(paths, limit=max(400, train_limit // 10) if train_limit else None)
    K = min(sample_crops, len(ds))
    ref_idx = np.linspace(0, len(ds) - 1, K).astype(int)
    ref = [(int(ds.shard[int(i)]), int(ds.local[int(i)])) for i in ref_idx]
    mine, n_val, _ = diag_files(latents_dir, leads, train_limit, sample_crops, split)
    got = [(s, l) for s, l, _f in mine]
    if len(ds) != n_val or ref != got:
        bad = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y), None)
        raise SystemExit(
            "ERROR: the crop selection in hpo_baselines.py has drifted from "
            "train_ldm.LatentRows/load_diag_crops.\n"
            f"  val rows: trainer {len(ds)}, here {n_val}\n"
            f"  crops: trainer {len(ref)}, here {len(got)}, first mismatch at {bad}\n"
            "Every paired comparison in the study depends on these being the same "
            "crops. Fix the reproduction before scoring anything.")
    return f"VERIFIED against train_ldm.LatentRows ({len(ref)} crops, {len(ds)} val rows)"


# ----------------------------------------------------------------------------
# Scoring, using the same estimators as train_ldm.sampled_diagnostics
# ----------------------------------------------------------------------------
def score(files, psd_crops, log):
    """Score persistence, advection and a sample climatology on these crops.

    Estimators are chosen to match the trainer exactly:
      MAE   mean over valid pixels within a crop, then averaged over crops
      CRPS  for a deterministic forecast the CRPS reduces to the MAE, which is
            how the trainer scores its advection control, so the same identity
            is used here
      CSI   contingency counts POOLED over all crops, then CSI = H/(H+M+F)
      PSD   radial spectra summed over the crops passing valid.mean() > 0.99,
            capped at psd_crops in file order, then the mean of the
            per-wavenumber ratio across the 2 to 8 km band
    """
    methods = ("persistence", "advection", "climatology")
    mae = {m: 0.0 for m in methods}
    cont = {m: {t: np.zeros(3) for t in CSI_T} for m in methods}
    psd_sum = {m: None for m in methods}
    psd_obs, n_psd, n_ok = None, 0, 0
    wet_obs = wet_n = 0.0
    y_sum = y_n = 0.0
    missing = []

    # Two passes: the climatology constant needs the mean observed rate over the
    # same crops, so it cannot be scored in the same sweep it is estimated from
    # without leaking. Pass one estimates it, pass two scores everything.
    for _s, _l, f in files:
        if not os.path.exists(f):
            missing.append(f)
            continue
        z = np.load(f, allow_pickle=True)
        y = np.nan_to_num(z["y_mmh"].astype("float32"))
        V = z["valid"].astype(bool)
        y_sum += float(y[V].sum())
        y_n += float(V.sum())
    if missing:
        log(f"WARNING: {len(missing)} of {len(files)} diagnostic crops are missing "
            f"from the npz cache (scratch wiped?). First: {missing[0]}")
    if y_n == 0:
        raise SystemExit("ERROR: no readable diagnostic crops. Check that the npz "
                         "prior cache the index JSONs point at is still on this "
                         "machine; pack_latents.py wrote absolute paths.")
    clim = y_sum / y_n

    t0 = time.time()
    for _s, _l, f in files:
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        y = np.nan_to_num(z["y_mmh"].astype("float32"))
        A = np.nan_to_num(z["A_mmh"].astype("float32"))
        P = np.nan_to_num(z["x_mmh"][-1].astype("float32"))     # last input frame
        C = np.full_like(y, clim)
        V = z["valid"].astype(bool)
        n_ok += 1
        wet_obs += float((y[V] >= 0.1).sum())
        wet_n += float(V.sum())
        for name, F in (("persistence", P), ("advection", A), ("climatology", C)):
            mae[name] += float(np.abs(F - y)[V].mean())
            for t in CSI_T:
                o, p = y >= t, F >= t
                cont[name][t] += [np.sum(V & o & p), np.sum(V & o & ~p),
                                  np.sum(V & ~o & p)]
        if V.mean() > 0.99 and n_psd < psd_crops:
            po = radial_psd(y)
            psd_obs = po if psd_obs is None else psd_obs + po
            for name, F in (("persistence", P), ("advection", A),
                            ("climatology", C)):
                pf = radial_psd(F)
                psd_sum[name] = pf if psd_sum[name] is None else psd_sum[name] + pf
            n_psd += 1

    out = {"n_crops": n_ok, "n_crops_requested": len(files), "n_psd": n_psd,
           "n_missing": len(missing), "climatology_mmh": round(clim, 5),
           "wet_fraction_obs": round(wet_obs / max(wet_n, 1), 5),
           "seconds": round(time.time() - t0, 1)}
    short = {"persistence": "pers", "advection": "advcheck", "climatology": "clim"}
    for name in methods:
        k = short[name]
        out[f"mae_{k}"] = mae[name] / max(n_ok, 1)
        out[f"crps_{k}"] = out[f"mae_{k}"]        # deterministic CRPS == MAE
        for t in CSI_T:
            Hh, Mi, Fa = cont[name][t]
            tot = Hh + Mi + Fa
            out[f"csi_{t:g}_{k}"] = float(Hh / tot) if tot > 0 else float("nan")
        if n_psd and psd_sum[name] is not None:
            kk = np.arange(1, len(psd_obs))
            wl = 256.0 / kk
            band = (wl >= 2.0) & (wl <= 8.0)
            out[f"psd_ratio_2_8km_{k}"] = float(np.mean(
                psd_sum[name][1:][band] / np.maximum(psd_obs[1:][band], 1e-12)))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Paired persistence/advection/climatology baselines on the "
                    "exact diagnostic crops an HPO trial will be scored on.")
    ap.add_argument("--latents-dir", required=True,
                    help="the pack the trials will train on, e.g. latents_ml_ep17")
    ap.add_argument("--leads", default=None, help="e.g. 15,30,45,60 (omit for +60 packs)")
    ap.add_argument("--split", default="val", choices=["val", "train", "test"])
    ap.add_argument("--arm", default=None, choices=sorted(S.ARMS),
                    help="score every rung of this arm's ladder")
    ap.add_argument("--limit", type=int, default=None,
                    help="a single training --limit instead of --arm's whole ladder")
    ap.add_argument("--sample-crops", type=int, default=96)
    ap.add_argument("--psd-crops", type=int, default=None,
                    help="default: the same as --sample-crops")
    ap.add_argument("--rung-name", default="r0", help="key for a single --limit run")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the check that this script's crop selection matches "
                         "train_ldm.LatentRows (only useful if torch is present but "
                         "the pack is not the one the trials will use)")
    ap.add_argument("--out", required=True, help="baselines.json path")
    args = ap.parse_args()

    def log(*a):
        print(*a, flush=True)

    log(f"radial_psd: {_verify_psd_copy()}")
    leads = [int(x) for x in args.leads.split(",")] if args.leads else None

    if args.arm:
        rungs = S.RUNGS[args.arm]
        if S.ARMS[args.arm]["kind"] == "infer":
            raise SystemExit("ERROR: the inference arm scores its baselines inside "
                             "evaluate_diffusion.py, which already reports "
                             "persistence and advection columns. No separate "
                             "baseline pass is needed or wanted there.")
    else:
        if not args.limit:
            raise SystemExit("ERROR: pass either --arm (whole ladder) or --limit.")
        rungs = [{"name": args.rung_name, "rows": args.limit,
                  "sample_crops": args.sample_crops,
                  "psd_crops": args.psd_crops or args.sample_crops}]

    result = {}
    if os.path.exists(args.out):
        try:
            result = json.load(open(args.out))
            log(f"merging into the existing {args.out} "
                f"({len(result)} rung(s) already present)")
        except Exception:
            result = {}

    for r in rungs:
        sc = args.sample_crops if not args.arm else r.get("sample_crops", args.sample_crops)
        pc = args.psd_crops or (r.get("psd_crops") if args.arm else None) or sc
        files, n_val, sizes = diag_files(args.latents_dir, leads, r["rows"], sc,
                                         split=args.split)
        log(f"\nrung {r['name']}: train --limit {r['rows']} -> val rows {n_val} "
            f"(shard sizes {sizes}) -> {len(files)} diagnostic crops")
        if not args.no_verify:
            log("  crop selection: " + verify_against_trainer(
                args.latents_dir, leads, r["rows"], sc, split=args.split))
        per_shard = {}
        for s, _l, _f in files:
            per_shard[s] = per_shard.get(s, 0) + 1
        log(f"  crops per lead shard: {per_shard}")
        out = score(files, pc, log)
        out.update({"rung": r["name"], "train_limit": r["rows"],
                    "sample_crops": sc, "psd_crops": pc, "split": args.split,
                    "leads": leads, "latents_dir": args.latents_dir,
                    "n_val_rows": int(n_val), "crops_per_shard": per_shard,
                    "first_crop": files[0][2], "last_crop": files[-1][2]})
        result[r["name"]] = out
        log(f"  persistence  MAE {out['mae_pers']:.4f}  CSI@1 {out['csi_1_pers']:.4f}  "
            f"CSI@8 {out['csi_8_pers']:.4f}  PSD {out.get('psd_ratio_2_8km_pers', float('nan')):.4f}")
        log(f"  advection    MAE {out['mae_advcheck']:.4f}  CSI@1 {out['csi_1_advcheck']:.4f}  "
            f"CSI@8 {out['csi_8_advcheck']:.4f}  PSD {out.get('psd_ratio_2_8km_advcheck', float('nan')):.4f}")
        log(f"  climatology  MAE {out['mae_clim']:.4f}  (constant "
            f"{out['climatology_mmh']:.4f} mm/h)")
        log("  the advection row is an INTEGRITY CHECK: the trainer computes its "
            "own mae_adv on these same crops, and the two must agree to within "
            "float noise. If they do not, the crop sets have diverged and no "
            "paired comparison in this study is valid.")

    tmp = args.out + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    os.replace(tmp, args.out)
    log(f"\nwrote {args.out} ({len(result)} rung(s))")
    log("pass it to hpo_search.py with --baselines so mae_pers and crps_pers "
        "become available to the objective and gate expressions.")


if __name__ == "__main__":
    main()
