#!/usr/bin/env python3
"""
train_ldm.py - EDM latent diffusion for the residual delta = z_y - z_A.

Implements LDM.md sections 3-5: Karras (2022) preconditioning and loss on the
scaled latents produced by pack_latents.py, conditioned by channel-concatenation
of the encoded past frames and the encoded advection prior. The final nowcast
(sampled here only as a periodic diagnostic; full evaluation is a later stage)
is y_hat = D((z_A + delta_hat) / latent_scale) -> dBR -> mm/h.

Key facts the code relies on (see LDM.md):
  * sigma_data = std(z_y - z_A), measured by pack_latents.py and read from the
    pack meta. It is NOT 1: the target is a residual.
  * Conditioning is clean (never noised): c = [z_x1..z_x4, z_A] (20 ch) or z_A
    only (--cond-mode a-only). Conditioning dropout (--cond-drop 0.1) trains
    the unconditional branch so classifier-free guidance stays available at
    inference (w=1 recovers the plain conditional model).
  * The weighted EDM loss is ~1.0 at initialisation by construction (the
    denoiser output layer is zero-initialised); values falling below ~0.9 mean
    the network is learning. Validation uses FIXED noise (seeded per epoch with
    the same seed) so the val loss is comparable across epochs.
  * Best-model selection and all sampling use the EMA weights.

Monitoring per epoch: train/val weighted + raw loss, lr, grad norm, throughput,
GPU memory; every --sample-every epochs an ensemble is sampled on a fixed val
subset and scored (ensemble-mean MAE / CSI@1 / CSI@8, CRPS, member PSD ratio)
NEXT TO the advection baseline on the same crops, plus a sample montage PNG.

Resumable and idempotent: diff_last.pt is saved every epoch through the atomic
temp-file + os.replace pattern; a DONE marker stops the JASMIN self-resubmit
chain (README_jasmin.md).

    conda activate nowcast
    # smoke test (~minutes)
    python train_ldm.py --limit 4000 --epochs 2 --warmup 20 --sample-every 1
    # full run
    python train_ldm.py --epochs 100
    # resume (or let the sbatch chain do it)
    python train_ldm.py --epochs 100 --resume auto

Outputs -> ~/dissertation_outputs/diffusion/ (override with --out):
    diff_last.pt, diff_best.pt, train_log.json, config.json, curves.png,
    samples_epXX.png, runs.jsonl, DONE
"""
import os, sys, json, time, math, glob, random, getpass, hashlib, argparse, contextlib, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the frozen codec + shared helpers from the VAE stage. train_vae_v2.py sits
# next to this file when the scripts are deployed flat, or in a sibling Code/ stage
# folder in the repo layout (Code/2 - VAE Stage/). Make it importable either way,
# without moving files or touching train_vae_v2.py / pack_latents.py.
def _ensure_vae_importable():
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, "train_vae_v2.py")):
        return                                       # flat layout: already importable
    for d in sorted(glob.glob(os.path.join(os.path.dirname(here), "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "train_vae_v2.py")):
            sys.path.insert(0, d)
            return
_ensure_vae_importable()
from train_vae_v2 import VAE, AttnBlock, from_dbr, radial_psd, atomic_save, atomic_json

USER    = getpass.getuser()
SCRATCH = os.environ.get("DISS_SCRATCH", f"/work/scratch-nopw2/{USER}/dissertation")
LATENTS = os.path.join(SCRATCH, "latents")
OUT     = os.path.expanduser("~/dissertation_outputs/diffusion")
VAE_CKPT = os.path.expanduser("~/dissertation_outputs/vae_v2/vae_best.pt")

ZC = 4                     # latent channels per field (asserted against the pack meta)
COND_CH = {"full": 5 * ZC, "a-only": ZC}
CSI_T = [1.0, 8.0]         # diagnostic thresholds (full suite comes at evaluation)


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class LatentRows(Dataset):
    """Rows of one or more latent packs -> ((24, 64, 64) float32, lead_index).

    Channel layout (pack_latents.py): 0:16 z_x1..z_x4, 16:20 z_A, 20:24 z_y.
    A single-lead (+60) pack is simply the one-shard case, where lead_index is
    always 0 and the model is built with no lead embedding, so the legacy path
    behaves exactly as before."""

    def __init__(self, paths, limit=None):
        if isinstance(paths, str):
            paths = [paths]
        self.paths = list(paths)
        sizes = [np.load(p, mmap_mode="r").shape[0] for p in self.paths]
        # Flat index over the concatenated shards. int16 shard id + int64 row id
        # costs ~10 bytes/row (6 MB for the 621k-row multi-lead train set).
        self.shard = np.concatenate(
            [np.full(n, s, dtype=np.int16) for s, n in enumerate(sizes)])
        self.local = np.concatenate(
            [np.arange(n, dtype=np.int64) for n in sizes])
        if limit and limit < len(self.shard):
            # Stride across the WHOLE concatenated index, so a --limit smoke run
            # still sees every lead rather than only the first shard.
            sel = np.linspace(0, len(self.shard) - 1, limit).astype(np.int64)
            self.shard, self.local = self.shard[sel], self.local[sel]
        self.sizes = sizes
        self._mm = None                          # lazy per-worker open

    def __len__(self):
        return len(self.shard)

    def __getitem__(self, i):
        if self._mm is None:
            self._mm = [np.load(p, mmap_mode="r") for p in self.paths]
        s = int(self.shard[i])
        row = np.asarray(self._mm[s][int(self.local[i])], dtype="float32")
        return torch.from_numpy(row), s


def shard_suffix(lead=None):
    """'' for a single-lead (+60) pack, '_L45' for a multi-lead shard. Must match
    pack_latents.shard_names."""
    return "" if lead is None else f"_L{lead:02d}"


def load_pack_meta(latents_dir, split, lead=None):
    p = os.path.join(latents_dir, f"{split}_latents{shard_suffix(lead)}_meta.json")
    if not os.path.exists(p):
        raise SystemExit(f"ERROR: {p} not found. Run pack_latents.py first "
                         "(and check DISS_SCRATCH / --latents-dir).")
    return json.load(open(p))


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class NoiseEmb(nn.Module):
    """Karras-style random Fourier features of c_noise. The random frequencies
    are a registered buffer, so they are saved in the checkpoint and identical
    after resume."""

    def __init__(self, dim=256, scale=16.0):
        super().__init__()
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, t):                                    # t: (B,)
        y = t[:, None].float() * self.freqs[None, :] * 2.0 * math.pi
        return torch.cat([y.cos(), y.sin()], dim=1)          # (B, dim)


class ResBlockT(nn.Module):
    """ADM-style residual block with FiLM conditioning on the noise embedding."""

    def __init__(self, cin, cout, emb_dim, dropout=0.0):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return self.skip(x) + h


class Cell(nn.Module):
    """ResBlockT with optional self-attention (one skip-entry per Cell)."""

    def __init__(self, cin, cout, emb_dim, dropout, attn):
        super().__init__()
        self.res = ResBlockT(cin, cout, emb_dim, dropout)
        self.attn = AttnBlock(cout) if attn else None

    def forward(self, x, emb):
        h = self.res(x, emb)
        if self.attn is not None:
            h = self.attn(h)
        return h


class Upsample(nn.Module):
    """Nearest-neighbour upsample + conv (no checkerboard), as in the VAE."""

    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """2D UNet on 64x64 latents. in_ch = zc (noised delta) + cond channels."""

    def __init__(self, in_ch, out_ch=ZC, width=128, mults=(1, 2, 4),
                 dropout=0.0, attn_res=(16,), emb_dim=512, n_leads=0):
        super().__init__()
        assert width % 8 == 0, "--width must be divisible by 8 (GroupNorm groups)"
        chans = [width * m for m in mults]
        self.noise_emb = NoiseEmb(dim=256)
        self.emb_mlp = nn.Sequential(nn.Linear(256, emb_dim), nn.SiLU(),
                                     nn.Linear(emb_dim, emb_dim))
        # Lead-time conditioning: a learned embedding added to the noise embedding,
        # so it FiLM-modulates every ResBlock (the standard ADM class-conditioning
        # route). Zero-initialised, so a fresh model is numerically identical to the
        # unconditioned one and the "loss starts at 1.0" property still holds.
        # n_leads=0 disables it entirely for single-lead runs.
        self.lead_emb = nn.Embedding(n_leads, emb_dim) if n_leads else None
        if self.lead_emb is not None:
            nn.init.zeros_(self.lead_emb.weight)
        self.stem = nn.Conv2d(in_ch, width, 3, padding=1)

        # ---- encoder ----
        self.down = nn.ModuleList()
        skip_ch = [width]
        c_prev, res = width, 64
        for lvl, c in enumerate(chans):
            for _ in range(2):
                self.down.append(Cell(c_prev, c, emb_dim, dropout, res in attn_res))
                c_prev = c
                skip_ch.append(c)
            if lvl < len(chans) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                skip_ch.append(c)
                res //= 2

        # ---- bottleneck ----
        self.mid1 = ResBlockT(c_prev, c_prev, emb_dim, dropout)
        self.mid_attn = AttnBlock(c_prev)
        self.mid2 = ResBlockT(c_prev, c_prev, emb_dim, dropout)

        # ---- decoder (mirrors the skip stack exactly) ----
        self.up = nn.ModuleList()
        for lvl in reversed(range(len(chans))):
            c = chans[lvl]
            for _ in range(3):
                self.up.append(Cell(c_prev + skip_ch.pop(), c, emb_dim, dropout,
                                    res in attn_res))
                c_prev = c
            if lvl > 0:
                self.up.append(Upsample(c))
                res *= 2
        assert not skip_ch, "skip bookkeeping mismatch"

        self.out_norm = nn.GroupNorm(8, width)
        self.out_conv = nn.Conv2d(width, out_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)          # D(x) = c_skip*x at init
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, c_noise, lead_idx=None):
        emb = self.emb_mlp(self.noise_emb(c_noise))
        if self.lead_emb is not None:
            if lead_idx is None:
                raise ValueError("this model was built with lead conditioning; "
                                 "lead_idx must be provided")
            emb = emb + self.lead_emb(lead_idx)
        h = self.stem(x)
        skips = [h]
        for m in self.down:
            if isinstance(m, Cell):
                h = m(h, emb)
            else:                                     # stride-2 conv downsample
                h = m(h)
            skips.append(h)
        h = self.mid1(h, emb)
        h = self.mid_attn(h)
        h = self.mid2(h, emb)
        for m in self.up:
            if isinstance(m, Cell):
                h = m(torch.cat([h, skips.pop()], dim=1), emb)
            else:                                     # upsample
                h = m(h)
        assert not skips, "skip bookkeeping mismatch in forward"
        return self.out_conv(F.silu(self.out_norm(h)))


class EDMDenoiser(nn.Module):
    """Karras preconditioning wrapper: D(x; sigma, c). sigma_data is the std of
    the training target delta (a residual, so < 1), from the pack meta."""

    def __init__(self, unet, sigma_data, out_ch=ZC):
        super().__init__()
        self.unet = unet
        self.out_ch = out_ch
        self.register_buffer("sigma_data", torch.tensor(float(sigma_data)))

    def forward(self, x, sigma, cond, lead_idx=None):
        # x: (B, zc, 64, 64) noised delta; sigma: (B,1,1,1); cond: clean latents;
        # lead_idx: (B,) long, index into the lead list (None for single-lead)
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = 0.25 * sigma.clamp(min=1e-8).log().view(-1)
        F_out = self.unet(torch.cat([c_in * x, cond], dim=1), c_noise, lead_idx)
        return c_skip * x + c_out * F_out


# ----------------------------------------------------------------------------
# EMA
# ----------------------------------------------------------------------------
def ema_init(model):
    return {k: v.detach().clone().float() for k, v in model.state_dict().items()}


@torch.no_grad()
def ema_update(ema, model, decay):
    for k, v in model.state_dict().items():
        if v.dtype.is_floating_point:
            ema[k].mul_(decay).add_(v.detach().float(), alpha=1.0 - decay)
        else:
            ema[k].copy_(v)


@contextlib.contextmanager
def ema_weights(model, ema):
    """Temporarily load the EMA weights (validation + sampling use these)."""
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict({k: ema[k].to(v.dtype) for k, v in model.state_dict().items()})
    try:
        yield
    finally:
        model.load_state_dict(backup)


# ----------------------------------------------------------------------------
# EDM sampling (Heun, Karras Algorithm 1; optional churn and CF guidance)
# ----------------------------------------------------------------------------
def edm_sigma_schedule(steps, sigma_min, sigma_max, rho, device):
    assert steps >= 2
    i = torch.arange(steps, dtype=torch.float32, device=device)
    sig = (sigma_max ** (1 / rho)
           + i / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([sig, torch.zeros(1, device=device)])


@torch.no_grad()
def edm_sample(denoiser, cond, steps=25, sigma_min=0.002, sigma_max=80.0, rho=7.0,
               guidance=1.0, churn=0.0, generator=None, lead_idx=None):
    B, dev = cond.shape[0], cond.device
    sig = edm_sigma_schedule(steps, sigma_min, sigma_max, rho, dev)
    x = torch.randn(B, denoiser.out_ch, cond.shape[2], cond.shape[3],
                    device=dev, generator=generator) * sig[0]

    def D(xt, s):
        sb = s.expand(B).view(B, 1, 1, 1)
        if guidance == 1.0:
            return denoiser(xt, sb, cond, lead_idx)
        # Guidance strengthens the PAST-FRAMES/advection conditioning only. The
        # lead index is a task selector (which lead time is being forecast), not
        # something to guide on, so it is kept in both branches.
        d_c = denoiser(xt, sb, cond, lead_idx)
        d_u = denoiser(xt, sb, torch.zeros_like(cond), lead_idx)
        return d_u + guidance * (d_c - d_u)

    for i in range(steps):
        s_i, s_n = sig[i], sig[i + 1]
        x_i = x
        if churn > 0:
            gamma = min(churn / steps, 2 ** 0.5 - 1)
            s_hat = s_i * (1 + gamma)
            x_i = x + (s_hat ** 2 - s_i ** 2).sqrt() * torch.randn(
                x.shape, device=dev, generator=generator)
            s_i = s_hat
        d = (x_i - D(x_i, s_i)) / s_i
        x_next = x_i + (s_n - s_i) * d
        if s_n > 0:
            d2 = (x_next - D(x_next, s_n)) / s_n
            x_next = x_i + (s_n - s_i) * 0.5 * (d + d2)
        x = x_next
    return x


# ----------------------------------------------------------------------------
# Loss helpers
# ----------------------------------------------------------------------------
def edm_loss_terms(denoiser, rows, cond_mode, p_mean, p_std, cond_drop,
                   device, generator=None, lead_idx=None):
    """One EDM training/validation forward. Returns (weighted loss, raw mse)."""
    z_A = rows[:, 4 * ZC:5 * ZC]
    z_y = rows[:, 5 * ZC:6 * ZC]
    delta = z_y - z_A
    cond = rows[:, :5 * ZC] if cond_mode == "full" else z_A
    B = rows.shape[0]

    if generator is None:
        n = torch.randn(B, device=device)
        eps = torch.randn_like(delta)
        u = torch.rand(B, device=device)
    else:                                    # fixed-noise validation path
        n = torch.randn(B, device=device, generator=generator)
        eps = torch.randn(delta.shape, device=device, generator=generator)
        u = torch.ones(B, device=device)     # never drop cond in validation
    sigma = (p_mean + p_std * n).exp().view(B, 1, 1, 1)
    if cond_drop > 0:
        keep = (u >= cond_drop).float().view(B, 1, 1, 1)
        cond = cond * keep

    x = delta + sigma * eps
    D = denoiser(x, sigma, cond, lead_idx)
    sd = denoiser.sigma_data
    w = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2
    err2 = (D.float() - delta.float()) ** 2
    return (w.float() * err2).mean(), err2.mean()


# ----------------------------------------------------------------------------
# Sampled diagnostics on a fixed val subset, always next to advection
# ----------------------------------------------------------------------------
def crps_ensemble(members, obs, valid):
    """Empirical CRPS (NRG estimator): mean_m|X_m - y| - 1/(2 M^2) sum_ij |X_i - X_j|,
    averaged over valid pixels. members: (M,H,W); obs: (H,W); valid: bool (H,W)."""
    X = members[:, valid]                                        # (M, P)
    y = obs[valid][None]                                         # (1, P)
    t1 = np.abs(X - y).mean()
    t2 = np.abs(X[:, None, :] - X[None, :, :]).mean() / 2.0
    return float(t1 - t2)


@torch.no_grad()
def sampled_diagnostics(denoiser, vae, diag, cond_diag, zA_diag, latent_scale,
                        mean, std, device, args, png_path, lead_diag=None):
    """Sample an M-member ensemble on the fixed diagnostic crops, decode to
    mm/h, and score next to the advection baseline on the same crops."""
    K, M = cond_diag.shape[0], args.sample_members
    g = torch.Generator(device=device).manual_seed(4242)  # same noise every epoch
    cond = cond_diag.repeat_interleave(M, dim=0).to(device)
    zA = zA_diag.repeat_interleave(M, dim=0).to(device)
    lead = (lead_diag.repeat_interleave(M, dim=0).to(device)
            if lead_diag is not None else None)
    # Sample in chunks so --sample-crops can be raised without OOM: the UNet sees
    # K*M latents at once otherwise. The default chunk (128) equals the old
    # K=16, M=8 batch, so a 16-crop run draws noise in exactly the same order and
    # reproduces the previous numbers bit for bit.
    lat = []
    step = max(1, args.sample_batch)
    for s in range(0, cond.shape[0], step):
        lat.append(edm_sample(denoiser, cond[s:s + step], steps=args.sample_steps,
                              guidance=args.guidance, churn=args.churn, generator=g,
                              lead_idx=None if lead is None else lead[s:s + step]))
    delta_hat = torch.cat(lat)
    mu_hat = (zA + delta_hat) / latent_scale
    outs = []
    for s in range(0, mu_hat.shape[0], 32):
        outs.append(vae.decode(mu_hat[s:s + 32]).float().cpu())
    y_dbr = torch.cat(outs)[:, 0].numpy() * std + mean
    members = from_dbr(y_dbr).reshape(K, M, 256, 256)            # mm/h

    ens = members.mean(axis=1)
    cont = {t: np.zeros(3) for t in CSI_T}
    cont_a = {t: np.zeros(3) for t in CSI_T}
    mae_m = mae_a = 0.0
    crps_m = crps_a = 0.0
    psd_o = psd_s = None
    n_psd = 0
    for k in range(K):
        y, A, V = diag[k]["y"], diag[k]["A"], diag[k]["valid"]
        mae_m += float(np.abs(ens[k] - y)[V].mean())
        mae_a += float(np.abs(A - y)[V].mean())
        for t in CSI_T:
            o = y >= t
            for c, p in ((cont, ens[k] >= t), (cont_a, A >= t)):
                c[t] += [np.sum(V & o & p), np.sum(V & o & ~p), np.sum(V & ~o & p)]
        crps_m += crps_ensemble(members[k], y, V)
        crps_a += float(np.abs(A - y)[V].mean())    # CRPS of a point forecast = MAE
        # PSD was hard-capped at 8 crops here, which is what made the preliminary
        # +60 diagnostic report 0.955 against a true 0.644 on the full split
        # (Diffusion_Run1_Results.md section 7). It is now a flag, because the
        # codec comparison turns on this number.
        if V.mean() > 0.99 and n_psd < args.psd_crops:
            po = radial_psd(y)
            ps = np.mean([radial_psd(members[k, m]) for m in range(M)], axis=0)
            psd_o = po if psd_o is None else psd_o + po
            psd_s = ps if psd_s is None else psd_s + ps
            n_psd += 1

    out = {"mae": mae_m / K, "mae_adv": mae_a / K,
           "crps": crps_m / K, "crps_adv": crps_a / K, "n_crops": K, "n_members": M}
    for t in CSI_T:
        for key, c in ((f"csi_{t:g}", cont), (f"csi_{t:g}_adv", cont_a)):
            Hh, Mi, Fa = c[t]
            out[key] = float(Hh / (Hh + Mi + Fa)) if (Hh + Mi + Fa) > 0 else float("nan")
    if n_psd:
        kk = np.arange(1, len(psd_o))
        wl = 256.0 / kk
        band = (wl >= 2.0) & (wl <= 8.0)
        out["psd_ratio_2_8km"] = float(np.mean(
            psd_s[1:][band] / np.maximum(psd_o[1:][band], 1e-12)))
    else:
        out["psd_ratio_2_8km"] = float("nan")

    # montage: obs | advection | ensemble mean | two members, for 4 crops
    nrow = min(4, K)
    fig, ax = plt.subplots(nrow, 5, figsize=(16, 3.2 * nrow), squeeze=False)
    for r in range(nrow):
        panels = [(diag[r]["y"], "obs y"), (diag[r]["A"], "advection A"),
                  (ens[r], "ens mean"), (members[r, 0], "member 1"),
                  (members[r, 1 % M], "member 2")]
        for cix, (img, ttl) in enumerate(panels):
            ax[r, cix].imshow(img, vmin=0, vmax=8, cmap="viridis")
            ax[r, cix].set_title(f"{ttl}  max {np.nanmax(img):.1f}", fontsize=8)
            ax[r, cix].axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=100, bbox_inches="tight")
    plt.close()
    return out


def load_diag_crops(latents_dir, va_ds, n_crops, leads=None):
    """Fixed, evenly-spaced val crops with their pixel-space ground truth. If the
    npz cache is unavailable (wiped scratch), diagnostics are disabled.

    With multiple lead shards the crops are strided across the whole concatenated
    dataset, so the montage and the scores span every lead rather than one."""
    suffixes = [shard_suffix(L) for L in (leads if leads else [None])]
    index = []
    for suf in suffixes:
        idx_p = os.path.join(latents_dir, f"val_latents{suf}_index.json")
        if not os.path.exists(idx_p):
            print(f"WARNING: {idx_p} missing; sampled diagnostics disabled.", flush=True)
            return None, None, None, None
        index.append(json.load(open(idx_p)))
    K = min(n_crops, len(va_ds))
    diag_idx = np.linspace(0, len(va_ds) - 1, K).astype(int)
    diag, rows, lead_ids = [], [], []
    for i in diag_idx:
        row, s = va_ds[int(i)]                      # s = shard (lead) index
        f = index[s][int(va_ds.local[int(i)])]      # provenance for THIS shard
        if not os.path.exists(f):
            print(f"WARNING: {f} missing (scratch wiped?); sampled diagnostics "
                  "disabled.", flush=True)
            return None, None, None, None
        z = np.load(f, allow_pickle=True)
        diag.append({"y": np.nan_to_num(z["y_mmh"].astype("float32")),
                     "A": np.nan_to_num(z["A_mmh"].astype("float32")),
                     "valid": z["valid"].astype(bool)})
        rows.append(row)
        lead_ids.append(s)
    return diag, torch.stack(rows), diag_idx, torch.tensor(lead_ids, dtype=torch.long)


# ----------------------------------------------------------------------------
# Curves
# ----------------------------------------------------------------------------
def plot_curves(log, png):
    if not log:
        return
    ep = [r["epoch"] for r in log]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax[0, 0].plot(ep, [r["train"]["loss_w"] for r in log], label="train (weighted)")
    ax[0, 0].plot(ep, [r["val"]["loss_w"] for r in log], label="val (weighted, EMA)")
    ax[0, 0].axhline(1.0, color="grey", ls="--", lw=1, label="init level")
    ax[0, 0].set(title="EDM loss", xlabel="epoch")
    ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)
    ax[0, 1].plot(ep, [r["train"]["loss_raw"] for r in log], label="train raw MSE")
    ax[0, 1].plot(ep, [r["val"]["loss_raw"] for r in log], label="val raw MSE")
    ax[0, 1].set(title="Unweighted MSE", xlabel="epoch")
    ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3)
    ax[0, 2].plot(ep, [r["train"]["lr"] for r in log], label="lr")
    ax[0, 2].plot(ep, [r["train"]["grad_norm"] for r in log], label="grad norm")
    ax[0, 2].set(title="Optimisation", xlabel="epoch")
    ax[0, 2].legend(); ax[0, 2].grid(alpha=0.3)
    se = [(r["epoch"], r["sampled"]) for r in log if r.get("sampled")]
    if se:
        eps = [e for e, _ in se]
        ax[1, 0].plot(eps, [s["mae"] for _, s in se], "o-", label="model (ens mean)")
        ax[1, 0].plot(eps, [s["mae_adv"] for _, s in se], "--", label="advection")
        ax[1, 0].set(title="Sampled MAE (mm/h)", xlabel="epoch")
        ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)
        for t, style in ((1.0, "o-"), (8.0, "s-")):
            ax[1, 1].plot(eps, [s[f"csi_{t:g}"] for _, s in se], style, label=f"model CSI@{t:g}")
            ax[1, 1].plot(eps, [s[f"csi_{t:g}_adv"] for _, s in se], "--",
                          label=f"advection CSI@{t:g}")
        ax[1, 1].set(title="Sampled CSI", xlabel="epoch", ylim=(0, 1))
        ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)
        ax[1, 2].plot(eps, [s["crps"] for _, s in se], "o-", label="model CRPS")
        ax[1, 2].plot(eps, [s["crps_adv"] for _, s in se], "--", label="advection (MAE)")
        ax[1, 2].plot(eps, [s["psd_ratio_2_8km"] for _, s in se], "s-",
                      label="PSD ratio 2-8 km")
        ax[1, 2].axhline(1.0, color="grey", ls="--", lw=1)
        ax[1, 2].set(title="Sampled CRPS / PSD ratio", xlabel="epoch")
        ax[1, 2].legend(fontsize=8); ax[1, 2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(png, dpi=110, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000, help="lr warmup steps")
    ap.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--width", type=int, default=128, help="UNet base channels (div by 8)")
    ap.add_argument("--mults", default="1,2,4", help="channel multipliers per level")
    ap.add_argument("--attn", default="16", help="resolutions with self-attention (csv)")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--cond-mode", choices=["full", "a-only"], default="full",
                    help="full = past frames + A (20ch); a-only = z_A (4ch), the "
                         "Phase-1 conditioning ablation")
    ap.add_argument("--cond-drop", type=float, default=0.1,
                    help="conditioning dropout prob (enables CF guidance; 0 = off)")
    ap.add_argument("--p-mean", type=float, default=-1.2)
    ap.add_argument("--p-std", type=float, default=1.2)
    ap.add_argument("--sigma-data", type=float, default=None,
                    help="override; default read from the train pack meta")
    ap.add_argument("--sample-every", type=int, default=5,
                    help="epochs between sampled diagnostics (0 = off)")
    ap.add_argument("--sample-steps", type=int, default=25)
    ap.add_argument("--sample-members", type=int, default=8)
    ap.add_argument("--sample-crops", type=int, default=16)
    ap.add_argument("--sample-batch", type=int, default=128,
                    help="latents per sampling pass in the diagnostic (crops x members). "
                         "128 reproduces the old unchunked K=16, M=8 behaviour exactly.")
    ap.add_argument("--no-keep-sampled", dest="keep_sampled", action="store_false",
                    help="do not archive EMA weights as ckpt_epNNN.pt at each sampled "
                         "epoch. Default is to keep them (~263 MB each); diff_best.pt "
                         "ranks on val loss alone, which is anti-correlated with PSD.")
    ap.add_argument("--psd-crops", type=int, default=64,
                    help="crops contributing to the diagnostic PSD ratio (was hard-coded "
                         "at 8, which is too few to be stable). Capped by --sample-crops.")
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="CF guidance weight for diagnostics (1 = plain conditional)")
    ap.add_argument("--churn", type=float, default=0.0, help="EDM S_churn for diagnostics")
    ap.add_argument("--patience", type=int, default=15,
                    help="early stop after N epochs without val improvement")
    ap.add_argument("--limit", type=int, default=None, help="cap rows (smoke test)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--leads", default=None,
                    help="comma-separated lead times (e.g. 15,30,45,60) to train ONE "
                         "lead-conditioned model over the per-lead shards written by "
                         "pack_latents.py --lead. Omit for a single-lead pack.")
    ap.add_argument("--latents-dir", default=LATENTS)
    ap.add_argument("--vae", default=VAE_CKPT, help="codec ckpt (diagnostics decode)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--resume", default=None,
                    help="path to diff_last.pt, or 'auto' to pick it up from --out")
    ap.add_argument("--ignore-done", action="store_true",
                    help="train even if the DONE marker exists")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    done_p = os.path.join(args.out, "DONE")
    if os.path.exists(done_p) and not args.ignore_done:
        print(f"DONE marker exists ({done_p}); nothing to do. "
              "Use --ignore-done to force.", flush=True)
        return

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: no GPU found, running on CPU (very slow)", flush=True)

    # ---- data ---------------------------------------------------------------
    leads = ([int(x) for x in args.leads.split(",") if x.strip()]
             if args.leads else None)
    shard_leads = leads if leads else [None]        # [None] = legacy single pack
    tr_metas = [load_pack_meta(args.latents_dir, "train", L) for L in shard_leads]
    va_metas = [load_pack_meta(args.latents_dir, "val", L) for L in shard_leads]
    tr_meta, va_meta = tr_metas[0], va_metas[0]
    if tr_meta.get("zc") != ZC:
        raise SystemExit(f"ERROR: pack zc={tr_meta.get('zc')} but trainer ZC={ZC}.")
    shas = {m.get("vae_sha256") for m in tr_metas + va_metas}
    if len(shas) != 1:
        raise SystemExit("ERROR: the latent packs were encoded with different VAE "
                         "checkpoints; re-run pack_latents.py so every shard and "
                         "split shares one codec.")
    latent_scale = float(tr_meta["latent_scale"])
    mean, std = float(tr_meta["norm"]["mean"]), float(tr_meta["norm"]["std"])
    if args.sigma_data:
        sigma_data = args.sigma_data
    elif len(shard_leads) == 1:
        sigma_data = float(tr_meta["sigma_data"])
    else:
        # A lead-conditioned model needs ONE sigma_data. The residual grows with
        # lead time, so pool it exactly from the per-shard raw moments rather than
        # averaging the per-shard standard deviations.
        S = sum(m["delta_moments"]["sum"] for m in tr_metas)
        SS = sum(m["delta_moments"]["sumsq"] for m in tr_metas)
        N = sum(m["delta_moments"]["count"] for m in tr_metas)
        sigma_data = float(math.sqrt(max(SS / N - (S / N) ** 2, 0.0)))
        per = ", ".join(f"+{L}min {float(m['sigma_data']):.4f}"
                        for L, m in zip(shard_leads, tr_metas))
        print(f"pooled sigma_data {sigma_data:.4f} from [{per}]", flush=True)
    if not (0.02 <= sigma_data <= 2.0):
        raise SystemExit(f"ERROR: sigma_data={sigma_data:.4f} looks wrong "
                         "(expected ~0.1-1.0); check the latent pack.")

    tr_paths = [os.path.join(args.latents_dir, f"train_latents{shard_suffix(L)}.npy")
                for L in shard_leads]
    va_paths = [os.path.join(args.latents_dir, f"val_latents{shard_suffix(L)}.npy")
                for L in shard_leads]
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
    resume_path = args.resume
    if resume_path == "auto":
        cand = os.path.join(args.out, "diff_last.pt")
        resume_path = cand if os.path.exists(cand) else None
        print(f"resume auto: {'found ' + cand if resume_path else 'fresh start'}", flush=True)
    resume_ck = torch.load(resume_path, map_location="cpu") if resume_path else None
    if resume_ck is not None:
        rc = resume_ck["config"]
        # Restore everything that must not silently drift mid-run: architecture
        # plus the training-regime constants. lr / batch / warmup / lr-schedule /
        # weight-decay stay CLI-adjustable on purpose (e.g. resuming with a
        # lower --lr after a divergence).
        for k in ("width", "mults", "attn", "cond_mode", "dropout",
                  "cond_drop", "ema_decay", "p_mean", "p_std"):
            if rc[k] != getattr(args, k):
                print(f"resume: overriding --{k.replace('_', '-')} "
                      f"{getattr(args, k)} -> {rc[k]} (from checkpoint)", flush=True)
            setattr(args, k, rc[k])
        sigma_data = rc["sigma_data"]
        # The lead list fixes the embedding size, so a mismatch would silently
        # build a different model or map leads to the wrong embedding rows.
        if rc.get("leads") != leads:
            raise SystemExit(
                f"ERROR: checkpoint was trained with leads {rc.get('leads')} but "
                f"--leads gives {leads}. Resuming would change the lead embedding. "
                "Pass the original --leads, or start a fresh run with a new --out.")

    mults = tuple(int(m) for m in args.mults.split(","))
    attn_res = tuple(int(r) for r in args.attn.split(",") if r.strip())
    cond_ch = COND_CH[args.cond_mode]
    n_leads = len(leads) if leads else 0
    unet = UNet(in_ch=ZC + cond_ch, out_ch=ZC, width=args.width, mults=mults,
                dropout=args.dropout, attn_res=attn_res, n_leads=n_leads)
    denoiser = EDMDenoiser(unet, sigma_data).to(device)
    n_par = sum(p.numel() for p in denoiser.parameters())
    opt = torch.optim.AdamW(denoiser.parameters(), lr=args.lr,
                            betas=(0.9, 0.999), weight_decay=args.weight_decay)
    ema = ema_init(denoiser)

    use_amp = device == "cuda"
    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()

    total_steps = args.epochs * nsteps
    def lr_at(step):
        base = args.lr * min(step / max(args.warmup, 1), 1.0)
        if args.lr_schedule == "constant" or step < args.warmup:
            return base
        t = min((step - args.warmup) / max(total_steps - args.warmup, 1), 1.0)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))

    # ---- diagnostics setup (frozen VAE decoder + fixed val crops) -----------
    vae = None
    diag = cond_diag = zA_diag = None
    if args.sample_every > 0:
        if not os.path.exists(args.vae):
            print(f"WARNING: --vae {args.vae} not found; sampled diagnostics "
                  "disabled.", flush=True)
            args.sample_every = 0
        else:
            vsha = sha256_file(args.vae)
            if tr_meta.get("vae_sha256") and vsha != tr_meta["vae_sha256"]:
                raise SystemExit(
                    f"ERROR: --vae {args.vae} (sha {vsha[:12]}) is not the "
                    f"checkpoint that produced the latent pack "
                    f"(sha {tr_meta['vae_sha256'][:12]}). The VAE was probably "
                    "retrained after packing. Re-run pack_latents.py (it repacks "
                    "automatically on a changed checkpoint) or point --vae at "
                    "the original checkpoint.")
            vck = torch.load(args.vae, map_location=device)
            vae = VAE(w=vck["config"]["width"], zc=vck["config"]["zc"]).to(device)
            vae.load_state_dict(vck["model"])
            vae.eval()
            diag, rows_diag, diag_idx, lead_diag = load_diag_crops(
                args.latents_dir, va_ds, args.sample_crops, leads)
            if diag is None:
                args.sample_every = 0
            else:
                cond_diag = rows_diag[:, :5 * ZC] if args.cond_mode == "full" \
                    else rows_diag[:, 4 * ZC:5 * ZC]
                zA_diag = rows_diag[:, 4 * ZC:5 * ZC]
                if not leads:
                    lead_diag = None
                print(f"diagnostics: {len(diag)} fixed val crops "
                      f"(rows {diag_idx[0]}..{diag_idx[-1]}), "
                      f"{args.sample_members} members, {args.sample_steps} steps", flush=True)

    # ---- resume / bookkeeping ------------------------------------------------
    start_ep, best, best_ep, global_step = 1, float("inf"), 0, 0
    log = []
    if resume_ck is not None:
        denoiser.load_state_dict(resume_ck["model"])
        opt.load_state_dict(resume_ck["opt"])
        ema = {k: v.float().to(device) for k, v in resume_ck["ema"].items()}
        start_ep = resume_ck["epoch"] + 1
        best, best_ep = resume_ck.get("best", float("inf")), resume_ck.get("best_ep", 0)
        global_step = resume_ck.get("global_step", (start_ep - 1) * nsteps)
        lp = os.path.join(args.out, "train_log.json")
        if os.path.exists(lp):
            log = json.load(open(lp))
        print(f"resumed at epoch {start_ep} (best {best:.4f} @ ep{best_ep})", flush=True)

    # Never overwrite a strictly better diff_best.pt left by a run that was
    # killed between the best and last checkpoint writes: read the val loss it
    # actually holds, once, and compare against it before every overwrite.
    best_disk = None
    bp0 = os.path.join(args.out, "diff_best.pt")
    if os.path.exists(bp0):
        try:
            best_disk = torch.load(bp0, map_location="cpu").get("val_loss")
        except Exception:
            best_disk = None

    config = {"width": args.width, "mults": args.mults, "attn": args.attn,
              "leads": leads,
              "dropout": args.dropout, "cond_mode": args.cond_mode,
              "cond_drop": args.cond_drop, "lr": args.lr, "warmup": args.warmup,
              "lr_schedule": args.lr_schedule, "batch": args.batch,
              "weight_decay": args.weight_decay, "ema_decay": args.ema_decay,
              "p_mean": args.p_mean, "p_std": args.p_std,
              "sigma_data": sigma_data, "latent_scale": latent_scale,
              "norm": {"mean": mean, "std": std}, "seed": args.seed,
              "epochs": args.epochs, "limit": args.limit,
              "vae_sha256": tr_meta.get("vae_sha256"), "git": git_hash(),
              "n_train": len(tr_ds), "n_val": len(va_ds), "n_params": n_par}
    atomic_json(config, os.path.join(args.out, "config.json"))
    # append-only history: config.json holds the current run, runs.jsonl keeps
    # every start/finish with its config, so a resume with changed flags can
    # never silently rewrite what earlier epochs trained with
    with open(os.path.join(args.out, "runs.jsonl"), "a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "event": "start", "resumed_from_epoch": start_ep - 1,
                             "config": config}) + "\n")
    print(f"device={device} | denoiser {n_par/1e6:.1f}M params | "
          f"cond={args.cond_mode} ({cond_ch}ch) | sigma_data={sigma_data:.4f} | "
          f"latent_scale={latent_scale:.3f}", flush=True)
    print(f"train rows {len(tr_ds)} | val rows {len(va_ds)} | {nsteps} steps/epoch "
          f"(batch {args.batch}) | weighted loss starts near 1.0 by construction", flush=True)
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} | {p.total_memory/1e9:.0f} GB total", flush=True)

    log_every = min(200, max(10, nsteps // 5))

    def ckpt_last():
        return {"model": denoiser.state_dict(),
                "ema": {k: v.cpu() for k, v in ema.items()},
                "opt": opt.state_dict(), "epoch": ep, "global_step": global_step,
                "best": best, "best_ep": best_ep, "config": config}

    def ckpt_best():
        return {"model": {k: v.cpu() for k, v in ema.items()},   # EMA weights
                "config": config, "epoch": ep, "val_loss": best,
                "sigma_data": sigma_data, "latent_scale": latent_scale,
                "norm": {"mean": mean, "std": std}, "cond_mode": args.cond_mode}

    # ---- training loop -------------------------------------------------------
    t_all = time.time()
    reason = "completed"
    ep = start_ep - 1        # bound even if the loop never runs (resume past --epochs)
    if start_ep > args.epochs:
        print(f"resume: checkpoint is already at epoch {start_ep - 1} >= "
              f"--epochs {args.epochs}; nothing to train, writing DONE.", flush=True)
        reason = "already-complete"
    for ep in range(start_ep, args.epochs + 1):
        denoiser.train()
        sums = {"loss_w": 0.0, "loss_raw": 0.0, "gn": 0.0}
        seen, t0 = 0, time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for i, (rows, lead_b) in enumerate(dl_tr, 1):
            rows = rows.to(device, non_blocking=True)
            lead_b = lead_b.to(device, non_blocking=True).long() if leads else None
            lr = lr_at(global_step)
            for gparam in opt.param_groups:
                gparam["lr"] = lr
            with amp():
                loss_w, loss_raw = edm_loss_terms(
                    denoiser, rows, args.cond_mode, args.p_mean, args.p_std,
                    args.cond_drop, device, lead_idx=lead_b)
            opt.zero_grad(set_to_none=True)
            loss_w.backward()
            gn = torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
            ema_update(ema, denoiser, args.ema_decay)

            sums["loss_w"] += loss_w.item()
            sums["loss_raw"] += loss_raw.item()
            sums["gn"] += float(gn)
            seen += rows.size(0); global_step += 1
            if i % log_every == 0:
                ips = seen / (time.time() - t0)
                print(f"  ep{ep} step {i}/{nsteps} | loss_w {sums['loss_w']/i:.4f} "
                      f"raw {sums['loss_raw']/i:.4f} | lr {lr:.2e} "
                      f"| gn {sums['gn']/i:.2f} | {ips:.0f} img/s", flush=True)
        dt = time.time() - t0

        # ---- validation (EMA weights, fixed noise) + diagnostics ------------
        denoiser.eval()
        vl_w = vl_raw = 0.0
        sampled = None
        with ema_weights(denoiser, ema):
            g = torch.Generator(device=device).manual_seed(1234)
            with torch.no_grad():
                for rows, lead_b in dl_va:
                    rows = rows.to(device, non_blocking=True)
                    lead_b = lead_b.to(device, non_blocking=True).long() if leads else None
                    with amp():
                        lw, lraw = edm_loss_terms(
                            denoiser, rows, args.cond_mode, args.p_mean, args.p_std,
                            0.0, device, generator=g, lead_idx=lead_b)
                    vl_w += lw.item(); vl_raw += lraw.item()
            vl_w /= max(len(dl_va), 1); vl_raw /= max(len(dl_va), 1)
            if args.sample_every > 0 and (ep % args.sample_every == 0 or ep == args.epochs):
                sampled = sampled_diagnostics(
                    denoiser, vae, diag, cond_diag, zA_diag, latent_scale,
                    mean, std, device, args,
                    os.path.join(args.out, f"samples_ep{ep:03d}.png"),
                    lead_diag=lead_diag)
        if not math.isfinite(vl_w):
            raise SystemExit(f"ERROR: validation loss is not finite at epoch {ep}; "
                             "training diverged. Resume from diff_last.pt with a "
                             "lower --lr.")

        gpu = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        rec = {"epoch": ep,
               "train": {"loss_w": sums["loss_w"] / nsteps,
                         "loss_raw": sums["loss_raw"] / nsteps,
                         "grad_norm": sums["gn"] / nsteps, "lr": lr},
               "val": {"loss_w": vl_w, "loss_raw": vl_raw},
               "sampled": sampled,
               "sys": {"epoch_sec": round(dt, 1), "gpu_gb": round(gpu, 2),
                       "imgs_per_s": round(seen / dt, 1)}}
        log.append(rec)
        msg = (f"epoch {ep:3d}/{args.epochs} | train {rec['train']['loss_w']:.4f} "
               f"| val {vl_w:.4f} (raw {vl_raw:.4f}) | {dt:.0f}s "
               f"| {rec['sys']['imgs_per_s']:.0f} img/s | GPU {gpu:.1f} GB")
        if sampled:
            msg += (f"\n  sampled: MAE {sampled['mae']:.3f} (adv {sampled['mae_adv']:.3f}) "
                    f"| CSI@1 {sampled['csi_1']:.3f} (adv {sampled['csi_1_adv']:.3f}) "
                    f"| CSI@8 {sampled['csi_8']:.3f} (adv {sampled['csi_8_adv']:.3f}) "
                    f"| CRPS {sampled['crps']:.3f} (adv {sampled['crps_adv']:.3f}) "
                    f"| PSD 2-8km {sampled['psd_ratio_2_8km']:.2f}")
        print(msg, flush=True)

        if vl_w < best:
            best, best_ep = vl_w, ep
            if best_disk is None or vl_w < best_disk:
                atomic_save(ckpt_best(), os.path.join(args.out, "diff_best.pt"))
                best_disk = vl_w
                print(f"  new best (val loss_w {best:.4f}) -> diff_best.pt", flush=True)
            else:
                print(f"  val improved ({vl_w:.4f}) but diff_best.pt already holds "
                      f"{best_disk:.4f} (from a run killed mid-save); kept", flush=True)
        # Archive the EMA weights at every sampled epoch. diff_best.pt ranks on val
        # loss alone, but val loss and small-scale power move in opposite
        # directions, so the loss-best checkpoint is systematically the smoothest
        # one the run produced (multi-lead run 1: ep15 beat ep40 by 35% on CSI@8
        # and 24% on PSD for a 2.6% MAE cost, yet ep40 held diff_best.pt).
        # diff_last.pt is overwritten every epoch, so without this a good-PSD
        # epoch is unrecoverable. Keeping the sampled epochs lets the choice be
        # made afterwards from a real evaluation instead of a few-crop proxy.
        if sampled and args.keep_sampled:
            ck = ckpt_best()
            ck["val_loss"] = vl_w            # this epoch's, not the running best
            ck["sampled"] = sampled
            atomic_save(ck, os.path.join(args.out, f"ckpt_ep{ep:03d}.pt"))
        atomic_save(ckpt_last(), os.path.join(args.out, "diff_last.pt"))
        atomic_json(log, os.path.join(args.out, "train_log.json"))
        plot_curves(log, os.path.join(args.out, "curves.png"))

        if args.patience and ep - best_ep >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs "
                  f"(best ep{best_ep}, val loss_w {best:.4f})", flush=True)
            reason = "early-stop"
            break

    # ---- finish: DONE marker + runs table ------------------------------------
    wall_min = (time.time() - t_all) / 60
    summary = {"reason": reason, "epochs_run": ep, "best": best, "best_ep": best_ep,
               "wall_min": round(wall_min, 1),
               "last_sampled": next((r["sampled"] for r in reversed(log)
                                     if r.get("sampled")), None)}
    atomic_json({**summary, "config": config}, done_p)
    with open(os.path.join(args.out, "runs.jsonl"), "a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "event": "finish", **summary, "config": config}) + "\n")
    print(f"\n{reason}: total {wall_min:.1f} min | best val loss_w {best:.4f} "
          f"(ep{best_ep}) | DONE written -> the sbatch chain will stop", flush=True)


if __name__ == "__main__":
    main()
