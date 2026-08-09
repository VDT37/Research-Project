#!/usr/bin/env python3
"""
train_regression.py - CorrDiff stage one: the frozen conditional mean of the
latent residual (CorrDiff_Design.md sections 3.1 and 4).

The project's diffusion model learns delta = z_y - z_A, the latent residual left
by the non-learned pysteps advection prior A. This script trains a deterministic
network R_phi to predict the CONDITIONAL MEAN of that residual,

    mu_r = R_phi(z_x1..z_x4, z_A, L) ~= E[delta | c, L]

by minimising plain unweighted MSE over latent cells, which is exactly what makes
the output the conditional mean and nothing else (Mardani et al. 2023, Methods
5.2.1). R_phi is then hard-frozen and the diffusion model is retrained on what it
leaves behind, r' = delta - mu_r. Two variants:

    --target delta --cond-mode full     variant (b), the arm: the learned mean
                                        sits ON TOP of the physical prior, so
                                        y = A + mu_r + r'
    --target z_y   --cond-mode x-only   variant (a), the Chase diagnostic: a
                                        learned mean INSTEAD of A, scored but
                                        never fed to a diffusion run

The number the whole arm turns on is the explained-variance fraction

    EV = 1 - mse / E[target^2]

measured against the target's true SECOND MOMENT, not its variance. That choice
is not cosmetic: out_conv is zero-initialised, so an untrained network outputs
exactly zero and sits at mse = E[target^2]; against the second moment it therefore
scores exactly EV = 0.000 and a perfect predictor scores 1.000, which is the
property every threshold in design section 9.3 assumes. Against the variance it
would start at -mean^2/var and eat part of ABORT-A's margin.

EV is a LATENT-SPACE quantity. It does not bound anything in mm/h, because the
decoder is nonlinear and D(E[z]) is not E[D(z)]. The pixel-space consequence is
measured, not argued: evaluate_deterministic.py decodes A + mu_r and scores it in
mm/h against y, next to advection, persistence and both codec bounds.

Tripwires printed by this script (design 9.3), as prints, not as auto-kills:
  ABORT-A  step 5000 of epoch 1 (~16 min): windowed train EV must exceed 0.010
  ABORT-B  end of epoch 1 (~0.5 h): val EV must exceed 0.02 AND the held-out
           EV of the closed-form ridge gate (--ridge-gate, latent_ridge_gate.py)
  GATE-C   end of the run: pooled val EV >= 0.10 authorises the 26 h retrain

    conda activate nowcast
    export DISS_SCRATCH=/work/scratch-pw4/$USER/dissertation
    # Phase-0 overfit sanity (must reach train EV > 0.98 on 512 rows)
    python train_regression.py --limit 512 --epochs 200 --batch 32 --warmup 20 \
        --lr 3e-4 --lr-schedule constant --diag-every 0 --out ~/dissertation_outputs/regression_overfit
    # smoke run
    python train_regression.py --leads 15,30,45,60 --limit 4000 --epochs 2 \
        --warmup 20 --diag-every 1 --diag-crops 16 --psd-crops 16
    # the real run, variant (b)
    python train_regression.py --leads 15,30,45,60 --epochs 8 --target delta \
        --cond-mode full --resume auto --out ~/dissertation_outputs/regression_delta_ep17

Note on --limit: the EV denominator comes from the pack meta and therefore
describes the WHOLE pack, while a --limit run sees a strided subset, so EV on a
smoke run is approximate. Full runs are exact.

Outputs -> --out (default ~/dissertation_outputs/regression/):
    reg_last.pt, reg_best.pt (EMA, selected on val MSE), ckpt_epNNN.pt,
    train_log.json, config.json, curves.png, diag_epNNN.png, runs.jsonl, DONE
"""
import os, sys, json, time, math, random, socket, getpass, argparse, contextlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# train_ldm.py resolves train_vae_v2 across the Code/<stage>/ layout on import, so
# importing it first fixes both. Everything below is imported, never copied: the
# regression and the diffusion stage must share one UNet, one loader and one set
# of conventions or the two-stage comparison is not controlled.
from train_ldm import (ZC, COND_CH, CSI_T, UNet, LatentRows, shard_suffix,
                       load_pack_meta, load_diag_crops, git_hash, sha256_file,
                       ema_init, ema_update, ema_weights)
from evaluate_diffusion import psd_band_metrics
from train_vae_v2 import VAE, from_dbr, radial_psd, atomic_save, atomic_json

USER    = getpass.getuser()
SCRATCH = os.environ.get("DISS_SCRATCH", f"/work/scratch-nopw2/{USER}/dissertation")
LATENTS = os.path.join(SCRATCH, "latents")
OUT     = os.path.expanduser("~/dissertation_outputs/regression")
VAE_CKPT = os.path.expanduser("~/dissertation_outputs/vae_v2/vae_best.pt")

ABORT_A_STEP = 5000        # design 9.3, ~16 min in at 326 img/s and batch 64
ABORT_A_EV   = 0.010
ABORT_B_EV   = 0.020
GATE_C_EV    = 0.100

# The regression's own conditioning table. "x-only" (past frames, no z_A) exists
# only for variant (a): no diffusion run uses it, so train_ldm.COND_CH is left
# exactly as the file that produced ml_v2 had it rather than widened with a path
# the diffusion trainer would never exercise.
REG_COND_CH = {**COND_CH, "x-only": 4 * ZC}


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class RegressionNet(nn.Module):
    """The same UNet as the diffusion stage with the noise input pinned at zero.

    CorrDiff uses architecturally identical networks for both stages and disables
    the time embedding in the regression net, because no probability-flow ODE is
    involved (Methods 5.3.2). c_noise = 0 is a constant, so NoiseEmb produces the
    constant vector [1..1, 0..0] for every sample and emb_mlp maps it to one fixed
    vector, which the lead embedding is then added to: the noise pathway is inert
    but PRESENT, which is the point, because the FiLM machinery that carries the
    lead conditioning stays intact. Stripping the embedding out of UNet would fork
    the class for no measurable gain (0.4M of 65.7M parameters).

    Keeping the identical 65.7M-parameter architecture is also what makes a weak
    result interpretable: a reviewer cannot then attribute a low EV to a network
    that was too small."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, cond, lead_idx=None):
        zeros = torch.zeros(cond.shape[0], device=cond.device)
        return self.unet(cond, zeros, lead_idx)


def slice_cond(rows, cond_mode):
    """Conditioning stack. Pure slicing of the existing 24-channel pack row: no
    repack is needed for either variant (design 6.1)."""
    if cond_mode == "full":
        return rows[:, :5 * ZC]                     # z_x1..z_x4, z_A  (20 ch)
    if cond_mode == "x-only":
        return rows[:, :4 * ZC]                     # z_x1..z_x4       (16 ch)
    return rows[:, 4 * ZC:5 * ZC]                   # z_A              (4 ch)


def slice_target(rows, target):
    if target == "delta":
        return rows[:, 5 * ZC:6 * ZC] - rows[:, 4 * ZC:5 * ZC]
    return rows[:, 5 * ZC:6 * ZC]                   # z_y


def load_regression(ckpt_path, device):
    """Rebuild the frozen conditional mean from reg_best.pt / ckpt_epNNN.pt.

    Two consumers (pack_mu.py and evaluate_deterministic.py) have to do this with
    no architecture flags on their command line, so the checkpoint describes
    itself, exactly as sample_diffusion.load_denoiser relies on ck["config"] for
    the diffusion side. This is the ONLY place a regression checkpoint is turned
    back into a network (design 3.1)."""
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"ERROR: regression checkpoint not found: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck.get("config", {})
    cond_mode = ck.get("cond_mode", cfg.get("cond_mode"))
    target = ck.get("target", cfg.get("target"))
    ck_leads = ck.get("leads", cfg.get("leads"))
    if cond_mode not in REG_COND_CH or target not in ("delta", "z_y"):
        raise SystemExit(f"ERROR: {ckpt_path} has cond_mode={cond_mode!r} "
                         f"target={target!r}; it was not written by "
                         "train_regression.py")
    mults = tuple(int(m) for m in cfg["mults"].split(","))
    attn = tuple(int(r) for r in cfg["attn"].split(",") if r.strip())
    # No "+ ZC": there is no noised latent to concatenate, the input is the
    # conditioning alone.
    unet = UNet(in_ch=REG_COND_CH[cond_mode], out_ch=ZC, width=cfg["width"],
                mults=mults, dropout=cfg.get("dropout", 0.0), attn_res=attn,
                n_leads=len(ck_leads) if ck_leads else 0)
    net = RegressionNet(unet).to(device)
    net.load_state_dict(ck["model"])
    net.eval().requires_grad_(False)
    n = sum(p.numel() for p in net.parameters())
    print(f"regression: {ckpt_path}", flush=True)
    print(f"  epoch {ck.get('epoch')} | val_mse {ck.get('val_mse', float('nan')):.5f} "
          f"| {n/1e6:.1f}M params | target={target} cond={cond_mode} "
          f"({REG_COND_CH[cond_mode]}ch)"
          f"{' | leads ' + str(ck_leads) if ck_leads else ''}", flush=True)
    return net, ck, cond_mode, target, ck_leads


# ----------------------------------------------------------------------------
# Target moments (the EV denominator)
# ----------------------------------------------------------------------------
def pool(moments):
    """(sum, sumsq, count) list -> (std, second moment, mean, count)."""
    S = sum(m["sum"] for m in moments)
    SS = sum(m["sumsq"] for m in moments)
    N = sum(m["count"] for m in moments)
    mu = S / N
    return float(math.sqrt(max(SS / N - mu ** 2, 0.0))), float(SS / N), float(mu), N


def measure_zy_moments(path, chunk=4096):
    """Raw moments of z_y for one shard, streamed from channels 20:24.

    pack_latents.py stores only a per-shard zy_std (line 295), and a pooled
    four-shard SECOND MOMENT cannot be recomputed from four standard deviations,
    so variant (a) has no defined EV until this exists. The alternative fix is to
    add zy_moments to the pack metas and rewrite the four metas (the .npy files do
    not change). Either is acceptable; what is NOT acceptable is silently falling
    back to delta's normaliser, because all three thresholds in design 9.3 are
    stated against the target's own second moment (design 3.1)."""
    mm = np.load(path, mmap_mode="r")
    S = SS = 0.0
    N = 0
    for i in range(0, mm.shape[0], chunk):
        a = np.asarray(mm[i:i + chunk, 5 * ZC:6 * ZC], dtype="float64")
        S += float(a.sum())
        SS += float((a * a).sum())
        N += int(a.size)
    return {"sum": S, "sumsq": SS, "count": N}


def target_moments(metas, paths, target, cache_path=None, tag=""):
    """Per-shard raw moments of the regression target."""
    if target == "delta":
        return [m["delta_moments"] for m in metas]
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except (json.JSONDecodeError, ValueError):
            cache = {}
    out, dirty = [], False
    for meta, p in zip(metas, paths):
        if meta.get("zy_moments"):                     # metadata-rewrite fix
            out.append(meta["zy_moments"])
            continue
        key = f"{os.path.abspath(p)}|{meta.get('n_files')}"
        if key in cache:
            out.append(cache[key])
            continue
        t0 = time.time()
        print(f"  [{tag}] measuring z_y moments for {os.path.basename(p)} "
              f"({meta.get('n_files')} rows, one sequential pass) ...", flush=True)
        mom = measure_zy_moments(p)
        print(f"    done in {time.time() - t0:.0f}s "
              f"(E[z_y^2] = {mom['sumsq'] / mom['count']:.4f})", flush=True)
        cache[key] = mom
        out.append(mom)
        dirty = True
    if dirty and cache_path:
        atomic_json(cache, cache_path)
    return out


# ----------------------------------------------------------------------------
# Decoded diagnostics: A + mu_r in mm/h, always next to advection
# ----------------------------------------------------------------------------
@torch.no_grad()
def decoded_diagnostics(net, vae, diag, cond_diag, zA_diag, target, latent_scale,
                        mean, std, device, args, png_path, lead_diag=None):
    """Decode the deterministic mean on the fixed val crops and score it against
    the advection baseline on those same crops. There is no sampling here, so
    this is one regression forward plus one decode per crop."""
    K = cond_diag.shape[0]
    step = max(1, args.diag_batch)
    fields = []
    for s in range(0, K, step):
        c = cond_diag[s:s + step].to(device)
        li = (lead_diag[s:s + step].to(device) if lead_diag is not None else None)
        mu_r = net(c, li)
        # variant (b): the mean is an increment on the advection anchor.
        # variant (a): the mean IS z_y, so there is nothing to anchor it to.
        z = (zA_diag[s:s + step].to(device) + mu_r) if target == "delta" else mu_r
        dec = vae.decode(z / latent_scale).float().cpu()
        fields.append(dec[:, 0].numpy() * std + mean)
    pred = from_dbr(np.concatenate(fields))                       # (K, 256, 256)

    cont = {t: np.zeros(3) for t in CSI_T}
    cont_a = {t: np.zeros(3) for t in CSI_T}
    mae_m = mae_a = 0.0
    psd_o = psd_m = psd_a = None
    n_psd = 0
    for k in range(K):
        y, A, V = diag[k]["y"], diag[k]["A"], diag[k]["valid"]
        mae_m += float(np.abs(pred[k] - y)[V].mean())
        mae_a += float(np.abs(A - y)[V].mean())
        for t in CSI_T:
            o = y >= t
            for c, p in ((cont, pred[k] >= t), (cont_a, A >= t)):
                c[t] += [np.sum(V & o & p), np.sum(V & o & ~p), np.sum(V & ~o & p)]
        if V.mean() > 0.99 and n_psd < args.psd_crops:
            po, pm, pa = radial_psd(y), radial_psd(pred[k]), radial_psd(A)
            psd_o = po if psd_o is None else psd_o + po
            psd_m = pm if psd_m is None else psd_m + pm
            psd_a = pa if psd_a is None else psd_a + pa
            n_psd += 1

    out = {"mae": mae_m / K, "mae_adv": mae_a / K, "n_crops": K, "n_psd": n_psd}
    for t in CSI_T:
        for key, c in ((f"csi_{t:g}", cont), (f"csi_{t:g}_adv", cont_a)):
            H, M, F = c[t]
            out[key] = float(H / (H + M + F)) if (H + M + F) > 0 else float("nan")
    if n_psd:
        bm = psd_band_metrics(psd_m, psd_o)
        ba = psd_band_metrics(psd_a, psd_o)
        out["psd_band_power"] = bm["psd_band_power"]
        out["psd_ratio_2_8km"] = bm["psd_mean_ratio"]
        out["psd_band_power_adv"] = ba["psd_band_power"]
        out["psd_ratio_2_8km_adv"] = ba["psd_mean_ratio"]
    else:
        for k in ("psd_band_power", "psd_ratio_2_8km",
                  "psd_band_power_adv", "psd_ratio_2_8km_adv"):
            out[k] = float("nan")

    nrow = min(4, K)
    ttl = "decoded A + mu_r" if target == "delta" else "decoded mu_y"
    fig, ax = plt.subplots(nrow, 4, figsize=(13, 3.2 * nrow), squeeze=False)
    for r in range(nrow):
        d = pred[r] - diag[r]["y"]
        panels = [(diag[r]["y"], "obs y", dict(vmin=0, vmax=8, cmap="viridis")),
                  (diag[r]["A"], "advection A", dict(vmin=0, vmax=8, cmap="viridis")),
                  (pred[r], ttl, dict(vmin=0, vmax=8, cmap="viridis")),
                  (d, "forecast - obs", dict(vmin=-4, vmax=4, cmap="RdBu_r"))]
        for cix, (img, t, kw) in enumerate(panels):
            ax[r, cix].imshow(img, **kw)
            ax[r, cix].set_title(f"{t}  max {np.nanmax(img):.1f}", fontsize=8)
            ax[r, cix].axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close()
    return out


def plot_curves(log, png):
    if not log:
        return
    ep = [r["epoch"] for r in log]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax[0, 0].plot(ep, [r["train"]["mse"] for r in log], label="train MSE")
    ax[0, 0].plot(ep, [r["val"]["mse"] for r in log], label="val MSE (EMA)")
    ax[0, 0].axhline(log[0]["sq_mean_train"], color="grey", ls="--", lw=1,
                     label="E[target^2] (EV = 0)")
    ax[0, 0].set(title="Regression MSE (latent cells)", xlabel="epoch")
    ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)
    ax[0, 1].plot(ep, [r["train"]["ev"] for r in log], label="train EV")
    ax[0, 1].plot(ep, [r["val"]["ev"] for r in log], label="val EV (EMA)")
    for lv, c, lab in ((ABORT_B_EV, "C3", "ABORT-B 0.02"),
                       (GATE_C_EV, "C2", "GATE-C 0.10")):
        ax[0, 1].axhline(lv, color=c, ls="--", lw=1, label=lab)
    ax[0, 1].set(title="Explained variance (latent space)", xlabel="epoch")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)
    ax[0, 2].plot(ep, [r["train"]["lr"] for r in log], label="lr")
    ax[0, 2].plot(ep, [r["train"]["grad_norm"] for r in log], label="grad norm")
    ax[0, 2].set(title="Optimisation", xlabel="epoch")
    ax[0, 2].legend(); ax[0, 2].grid(alpha=0.3)
    de = [(r["epoch"], r["decoded"]) for r in log if r.get("decoded")]
    if de:
        eps = [e for e, _ in de]
        ax[1, 0].plot(eps, [d["mae"] for _, d in de], "o-", label="decoded mean")
        ax[1, 0].plot(eps, [d["mae_adv"] for _, d in de], "--", label="advection")
        ax[1, 0].set(title="Decoded MAE (mm/h)", xlabel="epoch")
        ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)
        for t, style in ((1.0, "o-"), (8.0, "s-")):
            ax[1, 1].plot(eps, [d[f"csi_{t:g}"] for _, d in de], style,
                          label=f"model CSI@{t:g}")
            ax[1, 1].plot(eps, [d[f"csi_{t:g}_adv"] for _, d in de], "--",
                          label=f"advection CSI@{t:g}")
        ax[1, 1].set(title="Decoded CSI", xlabel="epoch", ylim=(0, 1))
        ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)
        ax[1, 2].plot(eps, [d["psd_band_power"] for _, d in de], "o-",
                      label="model 2-8 km band power")
        ax[1, 2].plot(eps, [d["psd_band_power_adv"] for _, d in de], "--",
                      label="advection")
        ax[1, 2].axhline(1.0, color="grey", ls="--", lw=1)
        ax[1, 2].set(title="Decoded PSD 2-8 km (band power)", xlabel="epoch")
        ax[1, 2].legend(fontsize=8); ax[1, 2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(png, dpi=110, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8,
                    help="8 sits above CorrDiff's own 1:14 (paper) and 1:10 "
                         "(released configs) regression:diffusion image ratios, "
                         "which imply 3.6 and 5.0 against this project's 50 "
                         "diffusion epochs, and 8 x 1884.5 s = 4.2 h fits inside a "
                         "single Orchid job with margin")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000, help="lr warmup steps")
    ap.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--width", type=int, default=128, help="UNet base channels (div by 8)")
    ap.add_argument("--mults", default="1,2,4")
    ap.add_argument("--attn", default="16", help="resolutions with self-attention")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--target", choices=["delta", "z_y"], default="delta",
                    help="delta = variant (b), predicts z_y - z_A from [z_x, z_A]; "
                         "z_y = variant (a) diagnostic, predicts z_y from [z_x] only")
    ap.add_argument("--cond-mode", choices=["full", "x-only", "a-only"], default="full",
                    help="full = [z_x1..z_x4, z_A] (20ch); x-only = [z_x1..z_x4] "
                         "(16ch), forced when --target z_y; a-only = [z_A] (4ch), "
                         "the conditioning ablation")
    ap.add_argument("--diag-every", type=int, default=2,
                    help="epochs between decoded diagnostics (0 = off)")
    ap.add_argument("--diag-crops", type=int, default=64)
    ap.add_argument("--psd-crops", type=int, default=64)
    ap.add_argument("--diag-batch", type=int, default=32, help="crops per decode pass")
    ap.add_argument("--no-keep-sampled", dest="keep_sampled", action="store_false",
                    help="do not archive ckpt_epNNN.pt at diagnostic epochs")
    ap.add_argument("--patience", type=int, default=0,
                    help="0 = never early stop (8 epochs is short already)")
    ap.add_argument("--limit", type=int, default=None, help="cap rows (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--leads", default=None, help="e.g. 15,30,45,60")
    ap.add_argument("--latents-dir", default=LATENTS)
    ap.add_argument("--vae", default=VAE_CKPT, help="codec ckpt (diagnostics decode)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--resume", default=None,
                    help="path to reg_last.pt, or 'auto' to pick it up from --out")
    ap.add_argument("--ignore-done", action="store_true")
    ap.add_argument("--ridge-gate", default=None,
                    help="latent_ridge_gate.py JSON. Only used to print the ABORT-B "
                         "comparison automatically; the gate is a print, not a kill.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    done_p = os.path.join(args.out, "DONE")
    if os.path.exists(done_p) and not args.ignore_done:
        print(f"DONE marker exists ({done_p}); nothing to do. "
              "Use --ignore-done to force.", flush=True)
        return

    if args.target == "z_y" and args.cond_mode != "x-only":
        # z_A is the thing variant (a) is defined to do without. Conditioning on it
        # while predicting z_y would make the task trivially "copy z_A and correct
        # it", which is variant (b) wearing variant (a)'s name.
        print(f"NOTE: --target z_y forces --cond-mode x-only "
              f"(was {args.cond_mode}).", flush=True)
        args.cond_mode = "x-only"

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: no GPU found, running on CPU (very slow)", flush=True)

    leads = ([int(x) for x in args.leads.split(",") if x.strip()]
             if args.leads else None)
    shard_leads = leads if leads else [None]

    # ---- resume: read the checkpoint config before anything is built ---------
    resume_path = args.resume
    if resume_path == "auto":
        cand = os.path.join(args.out, "reg_last.pt")
        resume_path = cand if os.path.exists(cand) else None
        print(f"resume auto: {'found ' + cand if resume_path else 'fresh start'}",
              flush=True)
    resume_ck = torch.load(resume_path, map_location="cpu") if resume_path else None
    if resume_ck is not None:
        rc = resume_ck["config"]
        for k in ("width", "mults", "attn", "dropout", "cond_mode", "target",
                  "ema_decay"):
            if rc[k] != getattr(args, k):
                print(f"resume: overriding --{k.replace('_', '-')} "
                      f"{getattr(args, k)} -> {rc[k]} (from checkpoint)", flush=True)
            setattr(args, k, rc[k])
        if rc.get("leads") != leads:
            raise SystemExit(
                f"ERROR: checkpoint was trained with leads {rc.get('leads')} but "
                f"--leads gives {leads}. Resuming would change the lead embedding.")

    # ---- data ---------------------------------------------------------------
    tr_metas = [load_pack_meta(args.latents_dir, "train", L) for L in shard_leads]
    va_metas = [load_pack_meta(args.latents_dir, "val", L) for L in shard_leads]
    tr_meta = tr_metas[0]
    if tr_meta.get("zc") != ZC:
        raise SystemExit(f"ERROR: pack zc={tr_meta.get('zc')} but ZC={ZC}.")
    shas = {m.get("vae_sha256") for m in tr_metas + va_metas}
    if len(shas) != 1:
        raise SystemExit("ERROR: the latent packs were encoded with different VAE "
                         "checkpoints; re-run pack_latents.py.")
    latent_scale = float(tr_meta["latent_scale"])
    mean, std = float(tr_meta["norm"]["mean"]), float(tr_meta["norm"]["std"])

    tr_paths = [os.path.join(args.latents_dir, f"train_latents{shard_suffix(L)}.npy")
                for L in shard_leads]
    va_paths = [os.path.join(args.latents_dir, f"val_latents{shard_suffix(L)}.npy")
                for L in shard_leads]

    zy_cache = os.path.join(args.out, "zy_moments.json")
    tr_mom = target_moments(tr_metas, tr_paths, args.target, zy_cache, "train")
    va_mom = target_moments(va_metas, va_paths, args.target, zy_cache, "val")
    sigma_data, sq_train, tgt_mean, _ = pool(tr_mom)
    _, sq_val, _, _ = pool(va_mom)
    sq_val_shard = [m["sumsq"] / m["count"] for m in va_mom]
    per = ", ".join(f"+{L}min sd {math.sqrt(max(m['sumsq']/m['count'] - (m['sum']/m['count'])**2, 0)):.4f}"
                    f" E[t^2] {m['sumsq']/m['count']:.4f}"
                    for L, m in zip(shard_leads, tr_mom))
    print(f"target={args.target} | pooled std {sigma_data:.4f} | mean {tgt_mean:+.4f} "
          f"| E[target^2] train {sq_train:.4f} val {sq_val:.4f}\n  per shard: [{per}]",
          flush=True)

    tr_ds = LatentRows(tr_paths, limit=args.limit)
    va_ds = LatentRows(va_paths,
                       limit=max(400, args.limit // 10) if args.limit else None)
    dl_kw = dict(num_workers=args.workers, pin_memory=True,
                 persistent_workers=args.workers > 0,
                 prefetch_factor=4 if args.workers > 0 else None)
    dl_tr = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=True, **dl_kw)
    dl_va = DataLoader(va_ds, batch_size=args.batch, shuffle=False, **dl_kw)
    nsteps = len(dl_tr)
    if nsteps == 0:
        raise SystemExit("ERROR: no training batches (check --limit / --batch).")

    # ---- model --------------------------------------------------------------
    mults = tuple(int(m) for m in args.mults.split(","))
    attn_res = tuple(int(r) for r in args.attn.split(",") if r.strip())
    cond_ch = REG_COND_CH[args.cond_mode]
    n_leads = len(leads) if leads else 0
    unet = UNet(in_ch=cond_ch, out_ch=ZC, width=args.width, mults=mults,
                dropout=args.dropout, attn_res=attn_res, n_leads=n_leads)
    net = RegressionNet(unet).to(device)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=args.weight_decay)
    ema = ema_init(net)

    use_amp = device == "cuda"
    def amp():
        return (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp
                else contextlib.nullcontext())

    total_steps = args.epochs * nsteps
    def lr_at(step):
        base = args.lr * min(step / max(args.warmup, 1), 1.0)
        if args.lr_schedule == "constant" or step < args.warmup:
            return base
        t = min((step - args.warmup) / max(total_steps - args.warmup, 1), 1.0)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))

    # ---- diagnostics setup ---------------------------------------------------
    vae = None
    diag = cond_diag = zA_diag = lead_diag = None
    if args.diag_every > 0:
        if not os.path.exists(args.vae):
            print(f"WARNING: --vae {args.vae} not found; decoded diagnostics "
                  "disabled.", flush=True)
            args.diag_every = 0
        else:
            vsha = sha256_file(args.vae)
            if tr_meta.get("vae_sha256") and vsha != tr_meta["vae_sha256"]:
                raise SystemExit(
                    f"ERROR: --vae {args.vae} (sha {vsha[:12]}) is not the "
                    f"checkpoint that produced the latent pack "
                    f"(sha {tr_meta['vae_sha256'][:12]}).")
            vck = torch.load(args.vae, map_location=device)
            vae = VAE(w=vck["config"]["width"], zc=vck["config"]["zc"]).to(device)
            vae.load_state_dict(vck["model"])
            vae.eval()
            diag, rows_diag, diag_idx, lead_diag = load_diag_crops(
                args.latents_dir, va_ds, args.diag_crops, leads)
            if diag is None:
                args.diag_every = 0
            else:
                cond_diag = slice_cond(rows_diag, args.cond_mode)
                zA_diag = rows_diag[:, 4 * ZC:5 * ZC]
                if not leads:
                    lead_diag = None
                print(f"diagnostics: {len(diag)} fixed val crops "
                      f"(rows {diag_idx[0]}..{diag_idx[-1]})", flush=True)

    # ---- resume / bookkeeping ------------------------------------------------
    start_ep, best, best_ep, global_step = 1, float("inf"), 0, 0
    log = []
    if resume_ck is not None:
        net.load_state_dict(resume_ck["model"])
        opt.load_state_dict(resume_ck["opt"])
        ema = {k: v.float().to(device) for k, v in resume_ck["ema"].items()}
        start_ep = resume_ck["epoch"] + 1
        best, best_ep = resume_ck.get("best", float("inf")), resume_ck.get("best_ep", 0)
        global_step = resume_ck.get("global_step", (start_ep - 1) * nsteps)
        lp = os.path.join(args.out, "train_log.json")
        if os.path.exists(lp):
            log = json.load(open(lp))
        print(f"resumed at epoch {start_ep} (best val MSE {best:.5f} @ ep{best_ep})",
              flush=True)

    best_disk = None
    bp0 = os.path.join(args.out, "reg_best.pt")
    if os.path.exists(bp0):
        try:
            best_disk = torch.load(bp0, map_location="cpu").get("val_mse")
        except Exception:
            best_disk = None

    config = {"width": args.width, "mults": args.mults, "attn": args.attn,
              "dropout": args.dropout, "cond_mode": args.cond_mode,
              "target": args.target, "loss": "mse", "leads": leads,
              "lr": args.lr, "warmup": args.warmup, "lr_schedule": args.lr_schedule,
              "batch": args.batch, "weight_decay": args.weight_decay,
              "ema_decay": args.ema_decay, "seed": args.seed,
              "epochs": args.epochs, "limit": args.limit,
              "sigma_data": sigma_data, "target_sq_mean_train": sq_train,
              "target_sq_mean_val": sq_val, "target_mean": tgt_mean,
              "latent_scale": latent_scale, "norm": {"mean": mean, "std": std},
              "latents_dir": os.path.abspath(args.latents_dir),
              "vae_sha256": tr_meta.get("vae_sha256"), "git": git_hash(),
              "host": socket.gethostname(), "argv": sys.argv,
              "n_train": len(tr_ds), "n_val": len(va_ds), "n_params": n_par}
    atomic_json(config, os.path.join(args.out, "config.json"))
    with open(os.path.join(args.out, "runs.jsonl"), "a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "event": "start", "resumed_from_epoch": start_ep - 1,
                             "config": config}) + "\n")
    print(f"device={device} | R_phi {n_par/1e6:.1f}M params | "
          f"cond={args.cond_mode} ({cond_ch}ch) -> {ZC}ch | "
          f"latent_scale={latent_scale:.3f}", flush=True)
    print(f"train rows {len(tr_ds)} | val rows {len(va_ds)} | {nsteps} steps/epoch "
          f"(batch {args.batch})", flush=True)

    ridge = None
    if args.ridge_gate and os.path.exists(args.ridge_gate):
        try:
            ridge = json.load(open(args.ridge_gate))
        except (json.JSONDecodeError, ValueError):
            print(f"WARNING: {args.ridge_gate} is not readable JSON; the ABORT-B "
                  "ridge comparison will be skipped.", flush=True)

    # ---- the zero-init data-path check (design 7.2) --------------------------
    # out_conv is zero-initialised, so an untrained network outputs exactly zero
    # and its MSE must equal the target's second moment. A broken slice shows up
    # here as a first loss that is not E[target^2], before a GPU-hour is spent.
    k0 = min(8, len(va_ds))
    if k0 == 0:
        raise SystemExit("ERROR: the validation set is empty; check --limit and "
                         "the val packs.")
    with torch.no_grad():
        # Built straight from the dataset, not from dl_va: taking one batch off a
        # persistent-worker DataLoader and abandoning the iterator is a needless
        # way to interact with worker processes before training has started.
        rows0 = torch.stack([va_ds[i][0] for i in range(k0)]).to(device)
        lead0 = torch.tensor([va_ds[i][1] for i in range(k0)],
                             dtype=torch.long, device=device)
        t0_ = slice_target(rows0, args.target).float()
        batch_sq = float((t0_ ** 2).mean())
        pred0 = net(slice_cond(rows0, args.cond_mode),
                    lead0 if leads else None).float()
        mse0 = float(((pred0 - t0_) ** 2).mean())
    print(f"data-path check: batch E[target^2] {batch_sq:.4f} vs pack meta "
          f"{sq_val:.4f} (ratio {batch_sq / max(sq_val, 1e-12):.3f}) | "
          f"untrained MSE {mse0:.4f} -> EV {1 - mse0 / max(batch_sq, 1e-12):+.4f} "
          f"(must be ~0.000 on a fresh start)", flush=True)
    if resume_ck is None and abs(mse0 - batch_sq) > 1e-3 * max(batch_sq, 1e-9):
        print("  WARNING: the untrained network is not outputting exactly zero. "
              "Either out_conv is no longer zero-initialised or the target slice "
              "is not what the network is being scored against.", flush=True)

    log_every = min(200, max(10, nsteps // 5))
    n_shard = len(shard_leads)

    def ckpt_last():
        return {"model": net.state_dict(),
                "ema": {k: v.cpu() for k, v in ema.items()},
                "opt": opt.state_dict(), "epoch": ep, "global_step": global_step,
                "best": best, "best_ep": best_ep, "config": config}

    def ckpt_frozen(val_mse):
        """What pack_mu.py and evaluate_deterministic.py rebuild R_phi from. It
        has to describe itself completely: neither consumer carries architecture
        flags on its command line (design 3.1)."""
        return {"model": {k: v.cpu() for k, v in ema.items()},
                "config": config, "epoch": ep, "val_mse": val_mse,
                "target": args.target, "cond_mode": args.cond_mode,
                "latent_scale": latent_scale, "norm": {"mean": mean, "std": std},
                "leads": leads, "n_leads": n_leads, "sigma_data": sigma_data,
                # named delta_sq_mean for the design's contract; it is the EV
                # denominator for whichever target this run used
                "delta_sq_mean": sq_train, "target_sq_mean": sq_train,
                "vae_sha256": tr_meta.get("vae_sha256")}

    # ---- training loop -------------------------------------------------------
    t_all = time.time()
    reason = "completed"
    ep = start_ep - 1
    if start_ep > args.epochs:
        print(f"resume: checkpoint is already at epoch {start_ep - 1} >= "
              f"--epochs {args.epochs}; nothing to train, writing DONE.", flush=True)
        reason = "already-complete"
    for ep in range(start_ep, args.epochs + 1):
        net.train()
        run_mse = run_gn = 0.0
        win_mse = win_n = 0.0
        seen, t0 = 0, time.time()
        abort_a_done = False
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for i, (rows, lead_b) in enumerate(dl_tr, 1):
            rows = rows.to(device, non_blocking=True)
            lead_i = lead_b.to(device, non_blocking=True).long() if leads else None
            lr = lr_at(global_step)
            for gparam in opt.param_groups:
                gparam["lr"] = lr
            with amp():
                pred = net(slice_cond(rows, args.cond_mode), lead_i)
                tgt = slice_target(rows, args.target)
                # Plain UNWEIGHTED MSE. Weighting by rain rate is not available in
                # latent space (there is no per-cell rain rate) and, more
                # importantly, only the unweighted minimiser is the conditional
                # MEAN, which is the property the whole two-stage decomposition
                # and var(delta - mu_r) <= var(delta) rest on (design 4.2).
                loss = ((pred.float() - tgt.float()) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ema_update(ema, net, args.ema_decay)

            lv = loss.item()
            run_mse += lv; run_gn += float(gn)
            win_mse += lv; win_n += 1
            seen += rows.size(0); global_step += 1
            if i % log_every == 0:
                ips = seen / (time.time() - t0)
                wm = win_mse / max(win_n, 1)
                print(f"  ep{ep} step {i}/{nsteps} | mse {run_mse/i:.4f} "
                      f"(win {wm:.4f}) | EV {1 - run_mse/i/sq_train:.3f} "
                      f"(win {1 - wm/sq_train:.3f}) | lr {lr:.2e} "
                      f"| gn {run_gn/i:.2f} | {ips:.0f} img/s", flush=True)
                if ep == 1 and i >= ABORT_A_STEP and not abort_a_done:
                    ev_win = 1 - wm / sq_train
                    verdict = "PASS" if ev_win > ABORT_A_EV else "FAIL"
                    print(f"  ABORT-A ({verdict}) at step {i}: windowed train EV "
                          f"{ev_win:.4f} against threshold {ABORT_A_EV:.3f}. "
                          + ("Continue." if verdict == "PASS" else
                             "KILL THE JOB. After 320k samples a model that has "
                             "not recovered 1% of the target variance is not "
                             "learning slowly, it is not learning: check the "
                             "target/conditioning slices, the lead index, and "
                             "whether lr ever left warmup."), flush=True)
                    abort_a_done = True
                win_mse = win_n = 0.0
        dt = time.time() - t0
        tr_mse = run_mse / nsteps

        # ---- validation (EMA weights) + decoded diagnostics ------------------
        net.eval()
        se = np.zeros(n_shard)
        cnt = np.zeros(n_shard)
        decoded = None
        with ema_weights(net, ema):
            with torch.no_grad():
                for rows, sidx in dl_va:
                    rows = rows.to(device, non_blocking=True)
                    lead_i = sidx.to(device).long() if leads else None
                    with amp():
                        pred = net(slice_cond(rows, args.cond_mode), lead_i)
                        tgt = slice_target(rows, args.target)
                    per_row = ((pred.float() - tgt.float()) ** 2).mean(dim=(1, 2, 3))
                    per_row = per_row.cpu().numpy()
                    s_np = sidx.numpy()
                    for s in range(n_shard):
                        m = s_np == s
                        if m.any():
                            se[s] += float(per_row[m].sum())
                            cnt[s] += int(m.sum())
            va_mse = float(se.sum() / max(cnt.sum(), 1))
            if args.diag_every > 0 and (ep % args.diag_every == 0 or ep == args.epochs):
                decoded = decoded_diagnostics(
                    net, vae, diag, cond_diag, zA_diag, args.target, latent_scale,
                    mean, std, device, args,
                    os.path.join(args.out, f"diag_ep{ep:03d}.png"),
                    lead_diag=lead_diag)
        if not math.isfinite(va_mse):
            raise SystemExit(f"ERROR: validation MSE is not finite at epoch {ep}.")

        per_lead = {}
        for s, L in enumerate(shard_leads):
            if cnt[s]:
                m = float(se[s] / cnt[s])
                per_lead[str(L)] = {"mse": m, "ev": 1 - m / sq_val_shard[s],
                                    "n_rows": int(cnt[s])}
        gpu = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        rec = {"epoch": ep,
               "train": {"mse": tr_mse, "ev": 1 - tr_mse / sq_train,
                         "grad_norm": run_gn / nsteps, "lr": lr},
               "val": {"mse": va_mse, "ev": 1 - va_mse / sq_val,
                       "per_lead": per_lead},
               "sq_mean_train": sq_train, "sq_mean_val": sq_val,
               "decoded": decoded,
               "sys": {"epoch_sec": round(dt, 1), "gpu_gb": round(gpu, 2),
                       "imgs_per_s": round(seen / dt, 1)}}
        log.append(rec)
        msg = (f"epoch {ep:3d}/{args.epochs} | train mse {tr_mse:.5f} "
               f"(EV {rec['train']['ev']:.4f}) | val mse {va_mse:.5f} "
               f"(EV {rec['val']['ev']:.4f}) | {dt:.0f}s "
               f"| {rec['sys']['imgs_per_s']:.0f} img/s | GPU {gpu:.1f} GB")
        if per_lead:
            msg += "\n  val EV per lead: " + ", ".join(
                f"+{k}min {v['ev']:.4f}" for k, v in per_lead.items())
        if decoded:
            msg += (f"\n  decoded: MAE {decoded['mae']:.3f} "
                    f"(adv {decoded['mae_adv']:.3f}) "
                    f"| CSI@1 {decoded['csi_1']:.3f} (adv {decoded['csi_1_adv']:.3f}) "
                    f"| CSI@8 {decoded['csi_8']:.3f} (adv {decoded['csi_8_adv']:.3f}) "
                    f"| PSD 2-8km band power {decoded['psd_band_power']:.2f} "
                    f"(adv {decoded['psd_band_power_adv']:.2f})")
        print(msg, flush=True)

        if ep == 1:
            ev_v = rec["val"]["ev"]
            ok = ev_v > ABORT_B_EV
            line = (f"  ABORT-B ({'PASS' if ok else 'FAIL'}) after epoch 1: val EV "
                    f"{ev_v:.4f} against threshold {ABORT_B_EV:.3f}")
            if ridge:
                rg = ridge.get("by_lead", {})
                parts = []
                for k, v in per_lead.items():
                    r_ev = ((rg.get(k) or {}).get(args.cond_mode)
                            or (rg.get(k) or {}).get("full") or {}).get("ev_eval_sm")
                    if r_ev is not None:
                        parts.append(f"+{k}min UNet {v['ev']:.4f} vs ridge "
                                     f"{r_ev:.4f} "
                                     f"({'PASS' if v['ev'] > r_ev else 'FAIL'})")
                if parts:
                    line += "\n    vs ridge gate: " + "; ".join(parts)
                    line += ("\n    A 65.7M-parameter nonlinear UNet that after a "
                             "full epoch cannot beat a closed-form linear filter "
                             "has a TRAINING problem, not a data problem.")
            print(line, flush=True)

        if va_mse < best:
            best, best_ep = va_mse, ep
            if best_disk is None or va_mse < best_disk:
                atomic_save(ckpt_frozen(best), os.path.join(args.out, "reg_best.pt"))
                best_disk = va_mse
                print(f"  new best (val MSE {best:.5f}, EV "
                      f"{1 - best / sq_val:.4f}) -> reg_best.pt", flush=True)
            else:
                print(f"  val improved ({va_mse:.5f}) but reg_best.pt already holds "
                      f"{best_disk:.5f} (run killed mid-save); kept", flush=True)
        # Archive the diagnostic epochs. Unlike the diffusion stage, val MSE IS the
        # right selection criterion for a deterministic regression, so reg_best.pt
        # is what gets frozen. The archives exist because if the decoded diagnostic
        # and the val MSE ever disagree, that disagreement is a result about the
        # codec and needs the checkpoints to investigate (design 4.3).
        if decoded and args.keep_sampled:
            ck = ckpt_frozen(va_mse)
            ck["decoded"] = decoded
            atomic_save(ck, os.path.join(args.out, f"ckpt_ep{ep:03d}.pt"))
        atomic_save(ckpt_last(), os.path.join(args.out, "reg_last.pt"))
        atomic_json(log, os.path.join(args.out, "train_log.json"))
        plot_curves(log, os.path.join(args.out, "curves.png"))

        if args.patience and ep - best_ep >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs "
                  f"(best ep{best_ep}, val MSE {best:.5f})", flush=True)
            reason = "early-stop"
            break

    # ---- finish: GATE-C, DONE marker, runs table -----------------------------
    wall_min = (time.time() - t_all) / 60
    best_ev = (1 - best / sq_val) if math.isfinite(best) else float("nan")
    gate_c = best_ev >= GATE_C_EV
    sd_prime = sigma_data * math.sqrt(max(1 - best_ev, 0.0))
    print(f"\nGATE-C ({'PASS' if gate_c else 'FAIL'}): pooled val EV {best_ev:.4f} "
          f"against threshold {GATE_C_EV:.2f}. sigma_data would fall "
          f"{sigma_data:.4f} -> {sd_prime:.4f} "
          f"({100 * (1 - sd_prime / max(sigma_data, 1e-9)):.1f}%).", flush=True)
    if gate_c and best_ev < 0.25:
        print("  Between 0.10 and 0.25: run the retrain, but pre-register the "
              "expectation that the effect is small. A 5% change in target scale "
              "is inside the run-to-run variation already observed between ml_v1 "
              "and ml_v2.", flush=True)
    if not gate_c:
        print("  This is a RESULT, not a bug. Do not add capacity or weight the "
              "loss to chase it: report EV per lead next to the ridge gate's "
              "linear lower bound, and the conclusion that the residual against a "
              "pysteps prior is close to unpredictable in the mean at these leads "
              "(design 9.4, 9.5).", flush=True)
    summary = {"reason": reason, "epochs_run": ep, "best_val_mse": best,
               "best_val_ev": best_ev, "best_ep": best_ep,
               "gate_c_pass": bool(gate_c), "gate_c_threshold": GATE_C_EV,
               "sigma_data": sigma_data, "sigma_data_resid_implied": sd_prime,
               "target": args.target, "cond_mode": args.cond_mode,
               "wall_min": round(wall_min, 1),
               "last_decoded": next((r["decoded"] for r in reversed(log)
                                     if r.get("decoded")), None),
               "val_ev_per_lead": (log[-1]["val"]["per_lead"] if log else {})}
    atomic_json({**summary, "config": config}, done_p)
    with open(os.path.join(args.out, "runs.jsonl"), "a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "event": "finish", **summary, "config": config}) + "\n")
    print(f"\n{reason}: total {wall_min:.1f} min | best val MSE {best:.5f} "
          f"(ep{best_ep}) | DONE written -> the sbatch chain will stop", flush=True)


if __name__ == "__main__":
    main()
