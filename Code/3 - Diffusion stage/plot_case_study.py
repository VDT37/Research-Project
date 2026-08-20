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

    # TWO ARMS IN ONE FIGURE, on the UK basemap, panels labelled by model name.
    # --mu-dir is handed only to checkpoints that declare hr_mean_cond, so one
    # flag serves a mixed set of LDM and CorrDiff arms.
    python plot_case_study.py --latents-dir $DISS_SCRATCH/latents_ml_ep17         --split test --lead 60 --frame 202606151200         --arm "ml_v2=$ML/ckpt_ep025.pt"         --arm "CorrDiff=$CD/ckpt_ep025.pt"         --mu-dir $DISS_SCRATCH/latents_ml_ep17_mu_delta         --vae $VAE17 --grid-shape 2175,1725         --domain-json ~/dissertation_outputs/figures/uk_domain.json         --tag arms --out ~/dissertation_outputs/figures

    # the epoch sweep: every checkpoint becomes its own labelled arm
    python plot_case_study.py --latents-dir $DISS_SCRATCH/latents_ml_ep17         --split test --lead 60 --frame 202606151200         --ckpt-glob "$ML/ckpt_ep0*.pt" --vae $VAE17 --grid-shape 2175,1725         --domain-json ~/dissertation_outputs/figures/uk_domain.json         --ncol 4 --tag mlv2_epochs --out ~/dissertation_outputs/figures

MUST be submitted through gpu.sbatch. The login VM has no CUDA, so running it
bare falls back to CPU and takes tens of minutes instead of about ten seconds.

Outputs into --out:
    case_{tag}_{frame}_L{lead}.png    observation and advection as reference
                                      panels, then an ensemble-mean and a
                                      single-member panel per named arm
    case_{tag}_{frame}_L{lead}.json   per-arm MAE and wet area against the
                                      observation, plus the baselines
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
                    help="sample every matching checkpoint, one panel pair each")
    ap.add_argument("--arm", action="append", default=None, metavar="LABEL=CKPT",
                    help="a named model to include, e.g. --arm \"ml_v2=$ML/ckpt_ep025.pt\" "
                         "--arm \"CorrDiff=$CD/ckpt_ep025.pt\". Repeat to put several "
                         "models in ONE figure. Panels are labelled with the name.")
    ap.add_argument("--domain-json", default=None,
                    help="figures/uk_domain.json, which supplies the projection "
                         "and grid origin so panels are drawn on a UK basemap "
                         "with coastlines and the pysteps intensity scale")
    ap.add_argument("--ncol", type=int, default=4, help="panels per row")
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

    if not (args.ckpt or args.ckpt_glob or args.arm):
        raise SystemExit("ERROR: pass --arm, --ckpt or --ckpt-glob.")
    arms = []
    for spec in (args.arm or []):
        if "=" not in spec:
            raise SystemExit(f"ERROR: --arm must be LABEL=CKPT, got {spec!r}")
        lab, ck = spec.split("=", 1)
        ck = os.path.expanduser(ck)
        if not os.path.exists(ck):
            raise SystemExit(f"ERROR: --arm {lab!r} checkpoint not found: {ck}")
        arms.append((lab, ck))
    if args.ckpt_glob:
        hits = sorted(glob.glob(os.path.expanduser(args.ckpt_glob)))
        if not hits:
            raise SystemExit(f"ERROR: --ckpt-glob matched nothing: {args.ckpt_glob}")
        arms += [(os.path.basename(h).replace(".pt", ""), h) for h in hits]
    if args.ckpt:
        arms.append((args.tag, os.path.expanduser(args.ckpt)))
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

    report = {"frame": args.frame, "split": args.split, "lead": args.lead,
              "n_tiles": len(sel),
              "tiles": [f"r{r - MARGIN:04d}_c{c - MARGIN:04d}" for r, c in places],
              "grid": {"H": H, "W": W}, "arms": []}

    # ---- sample every arm ---------------------------------------------------
    # mu is only handed to a checkpoint that actually declares hr_mean_cond, so
    # one --mu-dir can serve a mixed set of ml_v2 and CorrDiff arms without the
    # LDM arms being fed a conditioning stack they were not trained with.
    rowsets = []
    for label, ck in arms:
        den, ck_meta, cond_mode, ck_leads, hr_mean_cond = load_denoiser(ck, device)
        if hr_mean_cond:
            check_corrdiff_pairing(ck_meta, args.mu_dir)
            if mu is None:
                raise SystemExit(
                    f"ERROR: {label} ({os.path.basename(ck)}) was trained with "
                    "hr_mean_cond on and needs --mu-dir. Pass the frozen-mean "
                    "pack that matches its regression checkpoint.")
        lead_idx = resolve_lead_idx(ck_leads, args.lead)
        ens = sample_ensemble(
            den, vae, rows, cond_mode, latent_scale, mean, std,
            members=args.members, steps=args.steps, guidance=args.guidance,
            churn=args.churn, seed=args.seed, device=device,
            lead_idx=lead_idx, mu=(mu if hr_mean_cond else None),
            hr_mean_cond=hr_mean_cond)
        ens = np.asarray(ens)
        mean_m = mosaic(list(ens.mean(axis=1)), places, shape)
        mem_m = mosaic(list(ens[:, 0]), places, shape)
        ep = ck_meta.get("epoch")
        rowsets.append({"label": label, "epoch": ep, "mean": mean_m,
                        "member": mem_m, "corrdiff": bool(hr_mean_cond)})
        sc_mean = scores(mean_m, obs_m, val_m)
        sc_mem = scores(mem_m, obs_m, val_m)
        report["arms"].append({"label": label, "ckpt": ck, "epoch": ep,
                               "hr_mean_cond": bool(hr_mean_cond),
                               "ens_mean": sc_mean, "member_0": sc_mem})
        print(f"  {label:<22} ep{ep:<4} mean MAE {sc_mean['mae']:.4f} | "
              f"member MAE {sc_mem['mae']:.4f}")
        del den
        if device == "cuda":
            torch.cuda.empty_cache()

    report["baselines"] = {
        "advection": scores(adv_m, obs_m, val_m),
        "persistence": scores(per_m, obs_m, val_m),
        "observation_wet_area": scores(obs_m, obs_m, val_m)["wet_area"]}

    # ---- figure -------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable ({e}); scores written, no figure.")
        plt = None

    if plt is not None:
        # Geometry for the basemap. Taken from uk_domain.json so no HDF5 needs
        # opening; without it the panels fall back to a plain array plot.
        crs = cmap = norm = None
        extent = None
        if args.domain_json and os.path.exists(args.domain_json):
            dj = json.load(open(args.domain_json))
            geo = dj.get("geo", {})
            org = dj.get("projected_origin")
            if org and geo.get("projdef"):
                try:
                    from pysteps.visualization.utils import proj4_to_cartopy
                    from pysteps.visualization.precipfields import get_colormap
                    import cartopy.feature as cfeature
                    crs = proj4_to_cartopy(geo["projdef"])
                    cmap, norm, _clevs, _cstr = get_colormap(
                        "intensity", "mm/h", "pysteps")
                    x0, y0 = float(org[0]), float(org[1])
                    extent = (x0, x0 + W * float(geo["xscale"]),
                              y0, y0 + H * float(geo["yscale"]))
                except Exception as e:
                    print(f"note: basemap unavailable ({type(e).__name__}: {e}); "
                          "drawing plain panels instead.")
                    crs = None
        if crs is None and not args.domain_json:
            print("note: no --domain-json given, so panels are plain array plots. "
                  "Pass figures/uk_domain.json for a proper UK basemap.")

        panels = [("observation", obs_m, None), ("advection (pysteps)", adv_m, None)]
        for rs in rowsets:
            tail = " [CorrDiff]" if rs["corrdiff"] else ""
            panels.append((f"{rs['label']}{tail}\nensemble mean (ep{rs['epoch']})",
                           rs["mean"], rs["label"]))
            panels.append((f"{rs['label']}{tail}\nsingle member (ep{rs['epoch']})",
                           rs["member"], rs["label"]))

        ncol = min(args.ncol, len(panels))
        nrow = int(np.ceil(len(panels) / ncol))
        fig = plt.figure(figsize=(4.2 * ncol, 5.0 * nrow))
        last_im = None
        for i, (title, field, _lab) in enumerate(panels):
            shown = np.where(val_m, field, np.nan)
            if crs is not None:
                ax = fig.add_subplot(nrow, ncol, i + 1, projection=crs)
                last_im = ax.imshow(shown, origin="upper", extent=extent,
                                    cmap=cmap, norm=norm, transform=crs,
                                    interpolation="nearest", zorder=3)
                ax.add_feature(cfeature.OCEAN, facecolor="#a8b8c8", zorder=0)
                ax.add_feature(cfeature.LAND, facecolor="#efe9dc", zorder=1)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5,
                               edgecolor="0.25", zorder=4)
                ax.add_feature(cfeature.BORDERS, linewidth=0.35,
                               edgecolor="0.45", zorder=4)
                if args.tight:
                    # A map extent, not an array crop, so the projection and the
                    # aspect ratio stay correct.
                    r0, r1, c0, c1 = bbox(places, shape, pad=CROP // 4)
                    ax.set_extent(
                        [extent[0] + c0 * float(geo["xscale"]),
                         extent[0] + c1 * float(geo["xscale"]),
                         extent[2] + (H - r1) * float(geo["yscale"]),
                         extent[2] + (H - r0) * float(geo["yscale"])], crs=crs)
            else:
                ax = fig.add_subplot(nrow, ncol, i + 1)
                r0, r1, c0, c1 = (bbox(places, shape) if args.tight
                                  else (0, H, 0, W))
                last_im = ax.imshow(shown[r0:r1, c0:c1], origin="upper",
                                    cmap="viridis", vmin=0, vmax=args.vmax,
                                    interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=9.5)
        if last_im is not None:
            cb = fig.colorbar(last_im, ax=fig.axes, shrink=0.62, pad=0.015,
                              extend="max")
            cb.set_label("precipitation intensity (mm/h)")
        fig.suptitle(
            f"{args.frame}  |  +{args.lead} min  |  {args.split} split  |  "
            f"{len(sel)} of 42 tiles  |  {args.members} members, {args.steps} steps",
            fontsize=12, y=0.995)
        p1 = os.path.join(args.out,
                          f"case_{args.tag}_{args.frame}_L{args.lead}.png")
        fig.savefig(p1, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {p1}")
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
