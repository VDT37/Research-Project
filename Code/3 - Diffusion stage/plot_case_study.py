#!/usr/bin/env python3
"""
plot_case_study.py - place observation, advection and model output for one
timestamp onto the UK grid, as a geographically correct mosaic.

WHAT THIS DOES AND DOES NOT DO. The models work on 256 km crops in latent space,
and only crops that passed build_advection_prior's quality filter (>=90% in
radar range and >=5% wet) were ever encoded into the latent packs. This script
mosaics exactly those accepted tiles and leaves the rest blank. That is not a
limitation worth apologising for: the rejected tiles are out of radar range or
dry, so they would be empty anyway. What this is NOT is a full-composite
nowcast; producing one would mean re-running the advection prior, the VAE
encoder and the sampler over every tile including the rejected ones, which is a
different pipeline.

The tiles are located from the crop filenames, which encode the top-left corner
of the 384x384 context window as _rNNNN_cNNNN. The scored 256x256 crop sits
inside that with a 64-pixel margin, and STRIDE equals CROP so the accepted tiles
tile the plane without overlap. See plot_uk_domain.py for the same arithmetic.

COST. One model-epoch at one timestamp is however many tiles that timestamp
contributed, typically 10 to 40, sampled at 8 members and 25 steps. At the
measured 1.13 crops/s that is roughly 10 to 40 seconds of GPU, so a ten-epoch
sweep is a few minutes. This is what makes the per-epoch figure affordable.

    conda activate nowcast
    export DISS_SCRATCH=/work/scratch-nopw2/$USER/dissertation

    # one model, one timestamp: obs | advection | ens mean | two members
    python plot_case_study.py --latents-dir $DISS_SCRATCH/latents_ml_ep17 \
        --split test --lead 60 --frame 202606151200 \
        --ckpt $ML/ckpt_ep025.pt --vae $VAE17 --out ~/dissertation_outputs/figures

    # the epoch sweep: one row per checkpoint, showing the over-smoothing
    python plot_case_study.py --latents-dir $DISS_SCRATCH/latents_ml_ep17 \
        --split test --lead 60 --frame 202606151200 \
        --ckpt-glob "$ML/ckpt_ep0*.pt" --vae $VAE17 \
        --out ~/dissertation_outputs/figures --tag mlv2

    # CorrDiff needs its frozen-mean pack, exactly as the evaluator does
    python plot_case_study.py --latents-dir $DISS_SCRATCH/latents_ml_ep17 \
        --mu-dir $DISS_SCRATCH/latents_ml_ep17_mu_delta \
        --split test --lead 60 --frame 202606151200 \
        --ckpt $CD/ckpt_ep025.pt --vae $VAE17 --out ~/dissertation_outputs/figures

Outputs into --out:
    case_{tag}_{frame}_L{lead}.png        obs / advection / mean / members
    case_{tag}_{frame}_L{lead}_epochs.png one row per checkpoint (--ckpt-glob)
    case_{tag}_{frame}_L{lead}.json       which tiles were used, and per-panel
                                          MAE and wet-area against the observation
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sample_diffusion import (load_denoiser, load_codec, sample_ensemble,     # noqa: E402
                              read_truth, open_split, resolve_lead_idx,
                              check_corrdiff_pairing)

CROP, MARGIN, CONTEXT = 256, 64, 384
NAME_RE = re.compile(r"^(\d{12})_r(\d{4})_c(\d{4})(?:_L(\d{2}))?\.npz$")


def tiles_for_frame(npz_files, frame):
    """Row indices in the pack whose crop belongs to this timestamp, with the
    (row, col) each one occupies in the composite."""
    sel, places = [], []
    for i, f in enumerate(npz_files):
        m = NAME_RE.match(os.path.basename(f))
        if not m or m.group(1) != frame:
            continue
        sel.append(i)
        places.append((int(m.group(2)) + MARGIN, int(m.group(3)) + MARGIN))
    return sel, places


def mosaic(fields, places, shape):
    """Lay 256x256 crops onto a full-composite canvas. STRIDE equals CROP, so
    accepted tiles never overlap and no blending is needed."""
    out = np.full(shape, np.nan, dtype="float32")
    for f, (r, c) in zip(fields, places):
        out[r:r + CROP, c:c + CROP] = f
    return out


def bbox(places, shape, pad=CROP // 2):
    rs = [r for r, _c in places]
    cs = [c for _r, c in places]
    r0 = max(0, min(rs) - pad)
    r1 = min(shape[0], max(rs) + CROP + pad)
    c0 = max(0, min(cs) - pad)
    c1 = min(shape[1], max(cs) + CROP + pad)
    return r0, r1, c0, c1


def scores(panel, obs, valid):
    m = valid & np.isfinite(panel) & np.isfinite(obs)
    if not m.any():
        return {"mae": float("nan"), "wet_area": float("nan")}
    return {"mae": float(np.abs(panel[m] - obs[m]).mean()),
            "wet_area": float((panel[m] >= 0.1).mean())}


def main():
    ap = argparse.ArgumentParser(
        description="Geographically placed case study for one timestamp.")
    ap.add_argument("--latents-dir", required=True)
    ap.add_argument("--split", default="test", choices=["val", "train", "test"])
    ap.add_argument("--lead", type=int, required=True)
    ap.add_argument("--frame", required=True, help="YYYYMMDDHHMM (the TARGET time)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--ckpt-glob", default=None,
                    help="sample every matching checkpoint and stack them as rows")
    ap.add_argument("--vae", required=True)
    ap.add_argument("--mu-dir", default=None, help="CorrDiff frozen-mean pack")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--churn", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid-shape", default=None,
                    help="H,W of the composite. Default: inferred from the tiles, "
                         "which is enough for a cropped view but not for placing "
                         "them on a full-UK canvas.")
    ap.add_argument("--tight", action="store_true",
                    help="crop the figure to the tiles used, instead of the whole grid")
    ap.add_argument("--vmax", type=float, default=8.0)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--allow-vae-mismatch", action="store_true")
    args = ap.parse_args()

    if not args.ckpt and not args.ckpt_glob:
        raise SystemExit("ERROR: pass --ckpt or --ckpt-glob.")
    ckpts = ([args.ckpt] if args.ckpt
             else sorted(glob.glob(os.path.expanduser(args.ckpt_glob))))
    if not ckpts:
        raise SystemExit(f"ERROR: --ckpt-glob matched nothing: {args.ckpt_glob}")
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device; this will be very slow.")

    mm, npz_files, n_rows, meta = open_split(args.latents_dir, args.split, args.lead)
    sel, places = tiles_for_frame(npz_files, args.frame)
    if not sel:
        stamps = sorted({os.path.basename(f)[:12] for f in npz_files})
        near = [s for s in stamps if s[:8] == args.frame[:8]][:8]
        raise SystemExit(
            f"ERROR: no accepted crops at {args.frame} in split '{args.split}' "
            f"lead +{args.lead}.\n  {len(stamps)} timestamps are present; same-day "
            f"examples: {near or 'none'}\n  Use plot_uk_domain.py --list-events to "
            "pick a timestamp that actually contributed crops.")
    print(f"{args.frame}: {len(sel)} accepted tiles at +{args.lead} min "
          f"({args.split} split)")

    if args.grid_shape:
        H, W = (int(v) for v in args.grid_shape.split(","))
    else:
        H = max(r for r, _c in places) + CROP + MARGIN
        W = max(c for _r, c in places) + CROP + MARGIN
        print(f"  grid shape inferred as {H} x {W}; pass --grid-shape H,W from "
              "uk_domain.json for a true full-UK canvas")
    shape = (H, W)

    sel = np.asarray(sel)
    rows = torch.from_numpy(np.asarray(mm[sel], dtype="float32"))
    files = [npz_files[i] for i in sel]
    y, A, P, V = read_truth(files)

    vae = load_codec(args.vae, meta, device,
                     strict_sha=not args.allow_vae_mismatch)
    latent_scale = float(meta["latent_scale"])
    mean, std = float(meta["norm"]["mean"]), float(meta["norm"]["std"])

    # The CorrDiff frozen-mean pack is opened through the same reader the
    # trainer and the evaluator use, so its reg_sha256 guard fires here too.
    mu = None
    if args.mu_dir:
        from train_corrdiff import open_mu_split
        mu_mm = open_mu_split(args.mu_dir, args.latents_dir, args.split,
                              args.lead, n_rows=n_rows)[0]
        mu = torch.from_numpy(np.asarray(mu_mm[sel], dtype="float32"))
        print(f"  mu pack: {args.mu_dir}")

    obs_m = mosaic(list(y), places, shape)
    adv_m = mosaic(list(A), places, shape)
    per_m = mosaic(list(P), places, shape)
    val_m = mosaic([v.astype("float32") for v in V], places, shape) > 0.5

    panels, report = [], {"frame": args.frame, "split": args.split,
                          "lead": args.lead, "n_tiles": len(sel),
                          "tiles": [f"r{r - MARGIN:04d}_c{c - MARGIN:04d}"
                                    for r, c in places],
                          "grid": {"H": H, "W": W}, "checkpoints": []}
    panels.append(("observation", obs_m))
    panels.append(("advection (pysteps)", adv_m))
    panels.append(("persistence", per_m))

    rowsets = []
    for ck in ckpts:
        den, ck_meta, cond_mode, ck_leads, hr_mean_cond = load_denoiser(ck, device)
        check_corrdiff_pairing(ck_meta, args.mu_dir)
        lead_idx = resolve_lead_idx(ck_leads, args.lead)
        ens = sample_ensemble(
            den, vae, rows, cond_mode, latent_scale, mean, std,
            members=args.members, steps=args.steps, guidance=args.guidance,
            churn=args.churn, seed=args.seed, device=device,
            lead_idx=lead_idx, mu=mu, hr_mean_cond=hr_mean_cond)
        ens = np.asarray(ens)                       # (K, members, 256, 256)
        mean_m = mosaic(list(ens.mean(axis=1)), places, shape)
        mem_m = mosaic(list(ens[:, 0]), places, shape)
        ep = ck_meta.get("epoch")
        rowsets.append((os.path.basename(ck), ep, mean_m, mem_m))
        report["checkpoints"].append({
            "ckpt": ck, "epoch": ep,
            "ens_mean": scores(mean_m, obs_m, val_m),
            "member_0": scores(mem_m, obs_m, val_m)})
        print(f"  {os.path.basename(ck)} (ep{ep}): "
              f"mean MAE {report['checkpoints'][-1]['ens_mean']['mae']:.4f} | "
              f"member MAE {report['checkpoints'][-1]['member_0']['mae']:.4f}")
        del den
        if device == "cuda":
            torch.cuda.empty_cache()

    report["baselines"] = {"advection": scores(adv_m, obs_m, val_m),
                           "persistence": scores(per_m, obs_m, val_m),
                           "observation_wet_area": scores(obs_m, obs_m, val_m)["wet_area"]}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable ({e}); scores written, no figure.")
        plt = None

    r0, r1, c0, c1 = bbox(places, shape) if args.tight else (0, H, 0, W)

    def show(ax, field, title):
        m = np.where(val_m, field, np.nan)[r0:r1, c0:c1]
        im = ax.imshow(m, origin="upper", cmap="viridis", vmin=0, vmax=args.vmax,
                       interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        return im

    if plt is not None:
        # ---- single-row figure, using the last checkpoint -------------------
        name, ep, mean_m, mem_m = rowsets[-1]
        cols = [("observation", obs_m), ("advection", adv_m),
                (f"ensemble mean (ep{ep})", mean_m), (f"member 1 (ep{ep})", mem_m)]
        fig, axes = plt.subplots(1, len(cols), figsize=(4.0 * len(cols), 4.4))
        for ax, (t, f) in zip(np.atleast_1d(axes), cols):
            im = show(ax, f, t)
        fig.colorbar(im, ax=list(np.atleast_1d(axes)), shrink=0.75, pad=0.01,
                     label="rain rate (mm/h)")
        fig.suptitle(f"{args.frame}  |  +{args.lead} min  |  {args.split} split  |  "
                     f"{len(sel)} tiles", fontsize=11)
        p1 = os.path.join(args.out,
                          f"case_{args.tag}_{args.frame}_L{args.lead}.png")
        fig.savefig(p1, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {p1}")

        # ---- epoch sweep ----------------------------------------------------
        if len(rowsets) > 1:
            nrow = len(rowsets)
            fig, axes = plt.subplots(nrow, 4, figsize=(14, 3.5 * nrow),
                                     squeeze=False)
            for i, (name, ep, mean_m, mem_m) in enumerate(rowsets):
                show(axes[i][0], obs_m, "observation" if i == 0 else "")
                show(axes[i][1], adv_m, "advection" if i == 0 else "")
                show(axes[i][2], mean_m, "ensemble mean" if i == 0 else "")
                im = show(axes[i][3], mem_m, "member 1" if i == 0 else "")
                axes[i][0].set_ylabel(f"epoch {ep}", fontsize=10)
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.01,
                         label="rain rate (mm/h)")
            fig.suptitle(f"{args.tag}: sampled field against epoch  |  "
                         f"{args.frame}  +{args.lead} min  |  {len(sel)} tiles",
                         fontsize=12)
            p2 = os.path.join(
                args.out, f"case_{args.tag}_{args.frame}_L{args.lead}_epochs.png")
            fig.savefig(p2, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {p2}")

    p3 = os.path.join(args.out, f"case_{args.tag}_{args.frame}_L{args.lead}.json")
    tmp = p3 + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    os.replace(tmp, p3)
    print(f"wrote {p3}")
    print("\nThese panels are a single case and are illustrative, not evidence. "
          "Quote the full-split scorecards for any number.")


if __name__ == "__main__":
    main()
