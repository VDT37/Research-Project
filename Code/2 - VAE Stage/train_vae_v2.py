#!/usr/bin/env python3
"""
train_vae_v2.py - VAE v2 for 256x256 dBR radar crops -> zc x 64 x 64 latents.

Changes vs v1, motivated by docs/Analysis with VAE v1.md and grounded in the
literature (docs/VAE_architecture_change.md):
  architecture: two ResBlocks per resolution, self-attention at the 64x64
      bottleneck, nearest-neighbour-upsample + conv instead of ConvTranspose
      (removes the checkerboard PSD uptick)
  loss: intensity-weighted masked L1 (DGMR-style rain-rate weighting) + tiny KL
      + hinge PatchGAN adversarial term with warmup and a VQGAN-style adaptive
      weight (restores small-scale power and the heavy tail)
  data: reads the float16 memmaps written by pack_vae_data.py (npz fallback)
  monitoring, per epoch: plain + weighted val reconstruction, reconstruction
      CSI at 7 thresholds, heavy-tail frequency ratios (16 / 32 mm/h),
      small-scale PSD ratio (2-8 km band), latent statistics, discriminator
      logits, adaptive weight, throughput, GPU memory; live loss-curve PNG,
      periodic histogram+PSD verification PNG, reconstruction montage,
      checkpoints (last + best), early stopping, resume support.

    conda activate nowcast
    python pack_vae_data.py                                   # once
    python train_vae_v2.py --limit 4000 --epochs 2 --disc-start 60 --verify-every 1   # smoke test
    python train_vae_v2.py --epochs 50                        # full run (tmux)
    python train_vae_v2.py --verify-only                      # re-run final check
    python train_vae_v2.py --resume ~/dissertation_outputs/vae_v2/vae_last.pt

Outputs -> ~/dissertation_outputs/vae_v2/
    vae_best.pt, vae_last.pt, train_log.json, training_curves.png,
    verify_epXX.png, recon_epXX.png, vae_verify.png
"""
import os, glob, json, getpass, argparse, random, contextlib, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

USER       = getpass.getuser()
PRIOR      = f"/scratch/{USER}/dissertation/prior"
PACKED     = f"/scratch/{USER}/dissertation/packed"
OUT        = os.path.expanduser("~/dissertation_outputs/vae_v2")
DBR_THRESH = 0.1
DBR_ZERO   = -15.0
CSI_T      = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
TAIL_T     = [16.0, 32.0]


# ----------------------------------------------------------------------------
# dBR transforms (identical conventions to v1 and to the packer)
# ----------------------------------------------------------------------------
def to_dbr(R):
    R = np.asarray(R, dtype="float32")
    out = np.full_like(R, DBR_ZERO)
    wet = np.isfinite(R) & (R >= DBR_THRESH)
    out[wet] = 10.0 * np.log10(R[wet])
    return out


def from_dbr(dbr):
    R = np.power(10.0, np.asarray(dbr, dtype="float32") / 10.0)
    R[dbr <= DBR_ZERO + 1e-3] = 0.0
    return R


def radial_psd(field):
    f = np.nan_to_num(field).astype("float64")
    f = f - f.mean()
    P = np.fft.fftshift(np.abs(np.fft.fft2(f)) ** 2)
    Hh, Ww = f.shape
    Y, X = np.ogrid[:Hh, :Ww]
    r = np.hypot(Y - Hh // 2, X - Ww // 2).astype(int)
    return np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


def atomic_save(obj, path):
    """torch.save via temp-file + atomic rename, so a SIGKILL mid-write can
    never leave a truncated checkpoint (os.replace is atomic on the server)."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class PackedFields(Dataset):
    """Rows of the packed memmap -> (normalised dBR tensor (1,H,W), mask (1,H,W)).
    Row 2i = y (observation), row 2i+1 = A (advection); NaN = invalid radar."""

    def __init__(self, npy_path, mean, std, limit=None):
        self.path = npy_path
        shape = np.load(npy_path, mmap_mode="r").shape
        self.n = shape[0] if not limit else min(limit, shape[0])
        self.mean, self.std = mean, std
        self._mm = None                      # lazy per-worker open

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._mm is None:
            self._mm = np.load(self.path, mmap_mode="r")
        a = np.asarray(self._mm[i], dtype="float32")
        valid = np.isfinite(a)
        x = np.where(valid, a, DBR_ZERO)
        x = (x - self.mean) / self.std
        return torch.from_numpy(x)[None], torch.from_numpy(valid.astype("float32"))[None]


class NpzFields(Dataset):
    """Fallback: read y/A fields straight from the prior npz cache (slow path)."""

    def __init__(self, files, mean, std, fields=("y_mmh", "A_mmh")):
        self.items = [(f, k) for f in files for k in fields]   # y then A per file
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, key = self.items[i]
        R = np.load(f, allow_pickle=True)[key].astype("float32")
        valid = np.isfinite(R)
        x = (to_dbr(R) - self.mean) / self.std
        return torch.from_numpy(x)[None], torch.from_numpy(valid.astype("float32"))[None]


def compute_norm_packed(npy_path, n=800, seed=0):
    mm = np.load(npy_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    rows = rng.choice(mm.shape[0], size=min(n, mm.shape[0]), replace=False)
    vals = np.asarray(mm[np.sort(rows)], dtype="float32")
    vals = np.where(np.isfinite(vals), vals, DBR_ZERO)
    return float(vals.mean()), float(vals.std())


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, c), nn.Conv2d(c, c, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(8, c), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class AttnBlock(nn.Module):
    """Single-head self-attention over spatial positions (SDPA, memory-efficient)."""

    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.q = nn.Conv2d(c, c, 1)
        self.k = nn.Conv2d(c, c, 1)
        self.v = nn.Conv2d(c, c, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, Hh, Ww = x.shape
        h = self.norm(x)
        q = self.q(h).flatten(2).transpose(1, 2).unsqueeze(1)   # B,1,HW,C
        k = self.k(h).flatten(2).transpose(1, 2).unsqueeze(1)
        v = self.v(h).flatten(2).transpose(1, 2).unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v).squeeze(1)  # B,HW,C
        out = out.transpose(1, 2).reshape(B, C, Hh, Ww)
        return x + self.proj(out)


class Up(nn.Module):
    """Nearest-neighbour upsample then conv: no checkerboard artifacts."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class VAE(nn.Module):
    """256x256x1 <-> zc x 64 x 64 (x4 spatial compression), v2 architecture."""

    def __init__(self, w=64, zc=4):
        super().__init__()
        assert w % 8 == 0, "--width must be divisible by 8 (GroupNorm groups)"
        self.zc = zc
        self.enc = nn.Sequential(
            nn.Conv2d(1, w, 3, padding=1),
            ResBlock(w), ResBlock(w),
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1),        # 256 -> 128
            ResBlock(2 * w), ResBlock(2 * w),
            nn.Conv2d(2 * w, 2 * w, 3, stride=2, padding=1),    # 128 -> 64
            ResBlock(2 * w), ResBlock(2 * w),
            AttnBlock(2 * w), ResBlock(2 * w),
            nn.GroupNorm(8, 2 * w), nn.SiLU(),
            nn.Conv2d(2 * w, 2 * zc, 1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(zc, 2 * w, 3, padding=1),
            ResBlock(2 * w), AttnBlock(2 * w), ResBlock(2 * w),
            Up(2 * w, 2 * w),                                    # 64 -> 128
            ResBlock(2 * w), ResBlock(2 * w),
            Up(2 * w, w),                                        # 128 -> 256
            ResBlock(w), ResBlock(w),
            nn.GroupNorm(8, w), nn.SiLU(),
            nn.Conv2d(w, 1, 3, padding=1),
        )

    def encode(self, x):
        h = self.enc(x)
        return h[:, :self.zc], h[:, self.zc:]

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.dec(z), mu, logvar

    def get_last_layer(self):
        return self.dec[-1].weight


class PatchDiscriminator(nn.Module):
    """pix2pix 70x70 PatchGAN: judges realism of local patches, not the frame."""

    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------
def masked_l1(recon, x, mask):
    return ((recon - x).abs() * mask).sum() / mask.sum().clamp(min=1.0)


def weighted_l1(recon, x, mask, mean, std, alpha, rcap):
    """Masked L1 with DGMR-style intensity weighting. With the defaults
    (alpha=23, rcap=24) the weight w = 1 + alpha*min(R,rcap)/rcap ranges 1..24
    and closely matches DGMR's grid-cell regulariser weight w(y)=min(y+1,24)
    (Ravuri et al. 2021, Methods; the printed max() is a known typo). The
    division by w.sum() makes the absolute scale irrelevant, only the relative
    emphasis on heavy rain matters."""
    with torch.no_grad():
        R = torch.pow(10.0, (x * std + mean) / 10.0)          # target rain, mm/h
        w = (1.0 + alpha * torch.clamp(R, max=rcap) / rcap) * mask
    return ((recon - x).abs() * w).sum() / w.sum().clamp(min=1.0)


def kl_loss(mu, logvar):
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()


def adaptive_weight(rec_loss, g_loss, last_layer, disc_weight):
    """VQGAN adaptive weight: balances the gradient scale of the adversarial
    term against the reconstruction term at the last decoder layer."""
    g_rec = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
    g_adv = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    w = g_rec.detach().float().norm() / (g_adv.detach().float().norm() + 1e-4)
    return (w.clamp(0.0, 1e4) * disc_weight).detach()


# ----------------------------------------------------------------------------
# Evaluation: domain metrics on a fixed observation subset (even rows = y)
# ----------------------------------------------------------------------------
@torch.no_grad()
def domain_metrics(model, ds, indices, mean, std, device, batch=16, max_psd=48):
    cont = {t: np.zeros(3) for t in CSI_T}
    tail = {t: np.zeros(2) for t in TAIL_T}       # [recon count, obs count]
    psd_o = psd_r = None
    npsd = 0
    mu_all = []
    model.eval()
    for s in range(0, len(indices), batch):
        idx = indices[s:s + batch]
        xs = torch.stack([ds[i][0] for i in idx])
        ms = torch.stack([ds[i][1] for i in idx])
        recon, mu, _ = model(xs.to(device))
        mu_all.append(mu.float().cpu())
        Ro = from_dbr(xs[:, 0].numpy() * std + mean)
        Rr = from_dbr(recon[:, 0].float().cpu().numpy() * std + mean)
        V = ms[:, 0].numpy() > 0.5
        for t in CSI_T:
            o, p = (Ro >= t), (Rr >= t)
            cont[t] += [np.sum(V & o & p), np.sum(V & o & ~p), np.sum(V & ~o & p)]
        for t in TAIL_T:
            tail[t] += [np.sum(V & (Rr >= t)), np.sum(V & (Ro >= t))]
        for b in range(len(idx)):
            if npsd < max_psd and V[b].mean() > 0.99:
                po, pr = radial_psd(Ro[b]), radial_psd(Rr[b])
                psd_o = po if psd_o is None else psd_o + po
                psd_r = pr if psd_r is None else psd_r + pr
                npsd += 1
    out = {}
    for t in CSI_T:
        Hh, M, Fa = cont[t]
        out[f"csi_{t:g}"] = float(Hh / (Hh + M + Fa)) if (Hh + M + Fa) > 0 else float("nan")
    for t in TAIL_T:
        r, o = tail[t]
        out[f"tail_ratio_{t:g}"] = float(r / o) if o > 0 else float("nan")
    if npsd:
        k = np.arange(1, len(psd_o))
        wl = 256.0 / k
        band = (wl >= 2.0) & (wl <= 8.0)
        out["psd_ratio_2_8km"] = float(np.mean(psd_r[1:][band] / np.maximum(psd_o[1:][band], 1e-12)))
    else:
        out["psd_ratio_2_8km"] = float("nan")
    mu_cat = torch.cat(mu_all)
    out["mu_mean"], out["mu_std"] = float(mu_cat.mean()), float(mu_cat.std())
    out["n_psd"] = npsd
    return out


@torch.no_grad()
def verify(model, npz_files, mean, std, device, out_png, n=300):
    """Histogram + PSD of D(E(y)) vs y, from the npz cache (independent data
    path, so it also cross-checks the packed pipeline)."""
    rng = random.Random(1)
    sample = rng.sample(npz_files, min(n, len(npz_files)))
    if not sample:
        print(f"  verify skipped: no npz files available ({out_png})", flush=True)
        return
    rbins = np.logspace(np.log10(DBR_THRESH), np.log10(128), 41)
    h_obs, h_rec = np.zeros(40), np.zeros(40)
    psd_obs = psd_rec = None
    model.eval()
    for f in sample:
        R = np.load(f, allow_pickle=True)["y_mmh"].astype("float32")
        x = torch.from_numpy((to_dbr(R) - mean) / std)[None, None].to(device)
        rec = model(x)[0][0, 0].float().cpu().numpy()
        R_rec, R_obs = from_dbr(rec * std + mean), np.nan_to_num(R)
        h_obs += np.histogram(R_obs[R_obs >= DBR_THRESH], bins=rbins)[0]
        h_rec += np.histogram(R_rec[R_rec >= DBR_THRESH], bins=rbins)[0]
        po, pr = radial_psd(R_obs), radial_psd(R_rec)
        psd_obs = po if psd_obs is None else psd_obs + po
        psd_rec = pr if psd_rec is None else psd_rec + pr
    rc = np.sqrt(rbins[:-1] * rbins[1:])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].loglog(rc, h_obs / max(h_obs.sum(), 1), "k", lw=2, label="observed")
    ax[0].loglog(rc, h_rec / max(h_rec.sum(), 1), "C0", lw=1.5, label="VAE v2 reconstruction")
    ax[0].set(title="Rain-rate distribution", xlabel="mm/h", ylabel="frequency")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    k = np.arange(1, len(psd_obs))
    wl = 256.0 / k
    ax[1].loglog(wl, psd_obs[1:] / len(sample), "k", lw=2, label="observed")
    ax[1].loglog(wl, psd_rec[1:] / len(sample), "C0", lw=1.5, label="VAE v2 reconstruction")
    ax[1].set(title="Power spectrum (PSD)", xlabel="wavelength (km)", ylabel="power")
    ax[1].invert_xaxis(); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  verify -> {out_png}", flush=True)


@torch.no_grad()
def save_recons(model, ds, indices, mean, std, device, png):
    idx = indices[:4]
    xs = torch.stack([ds[i][0] for i in idx]).to(device)
    rec = model(xs)[0]
    fig, ax = plt.subplots(2, len(idx), figsize=(3.5 * len(idx), 7))
    for j in range(len(idx)):
        for row, img in ((0, xs), (1, rec)):
            a = img[j, 0].float().cpu().numpy() * std + mean
            R = from_dbr(a)
            ax[row, j].imshow(R, vmin=0, vmax=8, cmap="viridis")
            ax[row, j].set_title(f"{'input' if row == 0 else 'recon'}  max {np.max(R):.1f}",
                                 fontsize=9)
            ax[row, j].axis("off")
    plt.tight_layout(); plt.savefig(png, dpi=100, bbox_inches="tight"); plt.close()


def plot_curves(log, png):
    if not log:
        return
    ep = [r["epoch"] for r in log]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax[0, 0].plot(ep, [r["train"]["rec_w"] for r in log], label="train rec (weighted)")
    ax[0, 0].plot(ep, [r["val"]["rec_w"] for r in log], label="val rec (weighted)")
    ax[0, 0].plot(ep, [r["val"]["rec_p"] for r in log], label="val rec (plain)")
    ax[0, 0].set(title="Reconstruction", xlabel="epoch"); ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)
    ax[0, 1].plot(ep, [r["train"]["kl"] for r in log])
    ax[0, 1].set(title="KL", xlabel="epoch"); ax[0, 1].grid(alpha=0.3)
    ax[0, 2].plot(ep, [r["train"]["g_adv"] for r in log], label="G adv")
    ax[0, 2].plot(ep, [r["train"]["d_loss"] for r in log], label="D loss")
    ax[0, 2].plot(ep, [r["train"]["lam"] for r in log], label="adaptive weight")
    ax[0, 2].set(title="Adversarial", xlabel="epoch"); ax[0, 2].legend(); ax[0, 2].grid(alpha=0.3)
    for t in (1.0, 8.0, 32.0):
        ax[1, 0].plot(ep, [r["domain"][f"csi_{t:g}"] for r in log], label=f"CSI {t:g}")
    ax[1, 0].set(title="Reconstruction CSI", xlabel="epoch", ylim=(0, 1)); ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)
    for t in TAIL_T:
        ax[1, 1].plot(ep, [r["domain"][f"tail_ratio_{t:g}"] for r in log], label=f">= {t:g} mm/h")
    ax[1, 1].axhline(1.0, color="grey", ls="--", lw=1)
    ax[1, 1].set(title="Heavy-tail frequency ratio (recon/obs)", xlabel="epoch"); ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3)
    ax[1, 2].plot(ep, [r["domain"]["psd_ratio_2_8km"] for r in log])
    ax[1, 2].axhline(1.0, color="grey", ls="--", lw=1)
    ax[1, 2].set(title="PSD ratio 2-8 km (recon/obs)", xlabel="epoch"); ax[1, 2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(png, dpi=110, bbox_inches="tight"); plt.close()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32, help="reduce to 24/16 if OOM")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=1e-6, help="KL weight")
    ap.add_argument("--alpha", type=float, default=23.0,
                    help="intensity-weight gain; 23 with rcap 24 reproduces DGMR's w(y)=min(y+1,24)")
    ap.add_argument("--rcap", type=float, default=24.0, help="intensity-weight cap, mm/h")
    ap.add_argument("--width", type=int, default=64, help="base channels, divisible by 8")
    ap.add_argument("--zc", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--disc-start", type=int, default=-1,
                    help="global step at which the discriminator starts; -1 = after 2 epochs "
                         "(recon must be most of the way converged first, cf. PreDiff's 50k-step delay)")
    ap.add_argument("--disc-weight", type=float, default=0.5,
                    help="adversarial weight scale; 0 disables the GAN entirely")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--patience", type=int, default=8,
                    help="early stop after N epochs without val improvement (post-grace)")
    ap.add_argument("--verify-every", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="cap rows (smoke test)")
    ap.add_argument("--root", default=PRIOR)
    ap.add_argument("--packed-dir", default=PACKED)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--resume", default=None, help="path to vae_last.pt")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--psd-tol", type=float, default=0.15,
                    help="vae_best_psd.pt only considers epochs whose val rec_w is "
                         "within this fraction of the best, so chasing PSD can never "
                         "select a badly-reconstructing codec")
    ap.add_argument("--ignore-done", action="store_true",
                    help="train even if a DONE marker exists in --out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    done_p = os.path.join(args.out, "DONE")
    if os.path.exists(done_p) and not (args.ignore_done or args.verify_only):
        print(f"DONE marker exists ({done_p}); nothing to do. Use --ignore-done to "
              "train further (raise --epochs too), or delete it.", flush=True)
        return

    va_npz = sorted(glob.glob(os.path.join(args.root, "val", "*", "*.npz")))
    if not va_npz:
        raise SystemExit(f"ERROR: no val npz files under {os.path.join(args.root, 'val')}. "
                         "The periodic verify() needs the npz cache as an independent data "
                         "path; check --root (or re-create the prior cache) before training.")

    if args.verify_only:
        bp = os.path.join(args.out, "vae_best.pt")
        if not os.path.exists(bp):
            raise SystemExit(f"ERROR: {bp} not found. No best checkpoint exists yet "
                             "(selection only starts after the GAN grace period); train "
                             "longer, or resume from vae_last.pt.")
        ck = torch.load(bp, map_location=device)
        model = VAE(w=ck["config"]["width"], zc=ck["config"]["zc"]).to(device)
        model.load_state_dict(ck["model"])
        print(f"verify-only: vae_best.pt epoch {ck.get('epoch')} "
              f"latent_scale {ck['latent_scale']:.3f}", flush=True)
        verify(model, va_npz, ck["norm"]["mean"], ck["norm"]["std"], device,
               os.path.join(args.out, "vae_verify.png"))
        return

    # ---- data --------------------------------------------------------------
    # A pack is only trusted if its meta json exists (the packer writes meta and
    # renames the .npy into place only after a successful spot-check), and the
    # on-disk row count matches the meta. This rejects partial or stale packs.
    tr_npy = os.path.join(args.packed_dir, "train_fields.npy")
    va_npy = os.path.join(args.packed_dir, "val_fields.npy")
    tr_meta = os.path.join(args.packed_dir, "train_meta.json")
    va_meta = os.path.join(args.packed_dir, "val_meta.json")
    packed = all(os.path.exists(p) for p in (tr_npy, va_npy, tr_meta, va_meta))
    if packed:
        for npy_p, meta_p in ((tr_npy, tr_meta), (va_npy, va_meta)):
            n_meta = json.load(open(meta_p))["n_rows"]
            n_disk = np.load(npy_p, mmap_mode="r").shape[0]
            if n_meta != n_disk:
                raise SystemExit(f"ERROR: {npy_p} has {n_disk} rows but its meta says "
                                 f"{n_meta}; the pack looks partial or stale. "
                                 "Re-run pack_vae_data.py --force.")

    resume_ck = torch.load(args.resume, map_location="cpu") if args.resume else None
    if resume_ck is not None:
        mean, std = resume_ck["norm"]["mean"], resume_ck["norm"]["std"]
        print(f"resume: reusing normalisation mean={mean:.3f} std={std:.3f}", flush=True)
    elif packed:
        mean, std = compute_norm_packed(tr_npy)
    else:
        print("WARNING: packed memmaps not found, falling back to slow npz reads.")
        print("         Run pack_vae_data.py first for a ~5-10x faster epoch.", flush=True)
        tr_files = sorted(glob.glob(os.path.join(args.root, "train", "*", "*.npz")))
        rng = random.Random(0)
        vals = [to_dbr(np.load(f, allow_pickle=True)[k].astype("float32")).ravel()
                for f in rng.sample(tr_files, min(400, len(tr_files)))
                for k in ("y_mmh", "A_mmh")]
        v = np.concatenate(vals)
        mean, std = float(v.mean()), float(v.std())

    if packed:
        tr_ds = PackedFields(tr_npy, mean, std, limit=args.limit)
        va_ds = PackedFields(va_npy, mean, std,
                             limit=max(400, args.limit // 10) if args.limit else None)
    else:
        tr_files = sorted(glob.glob(os.path.join(args.root, "train", "*", "*.npz")))
        if args.limit:
            tr_files = tr_files[:args.limit // 2]
        tr_ds = NpzFields(tr_files, mean, std)
        va_files = va_npz[:max(200, args.limit // 20)] if args.limit else va_npz
        va_ds = NpzFields(va_files, mean, std)
    print(f"train rows {len(tr_ds)} | val rows {len(va_ds)} | packed={packed}", flush=True)
    print(f"dBR normalisation: mean={mean:.3f} std={std:.3f}", flush=True)

    dl_kw = dict(num_workers=args.workers, pin_memory=True,
                 persistent_workers=args.workers > 0,
                 prefetch_factor=4 if args.workers > 0 else None)
    dl_tr = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=True, **dl_kw)
    dl_va = DataLoader(va_ds, batch_size=args.batch, shuffle=False, **dl_kw)

    # fixed observation subset (even rows = y) for the domain metrics
    obs_rows = np.arange(0, len(va_ds), 2)
    dm_indices = obs_rows[np.linspace(0, len(obs_rows) - 1,
                                      min(512, len(obs_rows))).astype(int)]

    # ---- models ------------------------------------------------------------
    gan = args.disc_weight > 0
    model = VAE(w=args.width, zc=args.zc).to(device)
    disc = PatchDiscriminator().to(device) if gan else None
    opt_g = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.5, 0.9),
                              weight_decay=args.weight_decay)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.5, 0.9)) if gan else None
    n_g = sum(p.numel() for p in model.parameters())
    n_d = sum(p.numel() for p in disc.parameters()) if gan else 0
    print(f"device={device} | VAE {n_g/1e6:.2f}M params | disc {n_d/1e6:.2f}M | gan={gan}", flush=True)

    use_amp = device == "cuda"
    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} | {p.total_memory/1e9:.0f} GB total", flush=True)
    else:
        print("WARNING: no GPU found, running on CPU (very slow)", flush=True)

    nsteps = len(dl_tr)
    disc_start = args.disc_start if args.disc_start >= 0 else 2 * nsteps
    grace_ep = (math.ceil(disc_start / max(nsteps, 1)) + 1) if gan else 0
    log_every = min(200, max(10, nsteps // 5))     # short smoke runs still print
    print(f"{nsteps} steps/epoch (batch {args.batch}) | disc starts at step {disc_start} "
          f"| best-model selection from epoch {grace_ep + 1} "
          f"| step log every {log_every}", flush=True)

    # ---- resume ------------------------------------------------------------
    start_ep, best, best_ep, global_step = 1, float("inf"), 0, 0
    best_psd_score, best_psd_ep, best_psd_val = float("inf"), 0, float("nan")
    log = []
    if resume_ck is not None:
        if "opt_g" not in resume_ck:
            raise SystemExit("ERROR: --resume needs vae_last.pt (full checkpoint with "
                             "optimizer state); vae_best.pt is the compact inference "
                             "checkpoint and cannot be resumed from.")
        model.load_state_dict(resume_ck["model"])
        if gan and resume_ck.get("disc") is not None:
            disc.load_state_dict(resume_ck["disc"])
        opt_g.load_state_dict(resume_ck["opt_g"])
        if gan and resume_ck.get("opt_d") is not None:
            opt_d.load_state_dict(resume_ck["opt_d"])
        start_ep = resume_ck["epoch"] + 1
        best, best_ep = resume_ck.get("best", float("inf")), resume_ck.get("best_ep", 0)
        # .get with defaults keeps older checkpoints (written before this existed)
        # loadable; they simply start the PSD search fresh.
        best_psd_score = resume_ck.get("best_psd_score", float("inf"))
        best_psd_ep    = resume_ck.get("best_psd_ep", 0)
        best_psd_val   = resume_ck.get("best_psd_val", float("nan"))
        global_step = resume_ck.get("global_step", (start_ep - 1) * nsteps)
        lp = os.path.join(args.out, "train_log.json")
        if os.path.exists(lp):
            log = json.load(open(lp))
        print(f"resumed at epoch {start_ep} (best {best:.4f} @ ep{best_ep})", flush=True)

    # Second guard, for the window between the two saves: vae_best_psd.pt is
    # written before vae_last.pt, so a kill in between leaves a good psd
    # checkpoint on disk that the resumed (older) state does not know about.
    # Trust whichever score is better.
    bpp = os.path.join(args.out, "vae_best_psd.pt")
    if os.path.exists(bpp):
        try:
            on_disk = torch.load(bpp, map_location="cpu").get("best_psd_score")
        except Exception:
            on_disk = None
        if on_disk is not None and float(on_disk) < best_psd_score:
            best_psd_score = float(on_disk)
            print(f"vae_best_psd.pt on disk holds |psd-1| {best_psd_score:.3f}; "
                  "keeping it unless beaten", flush=True)

    def latent_scale_now():
        with torch.no_grad():
            xb = torch.stack([va_ds[i][0] for i in dm_indices[:32]]).to(device)
            return 1.0 / max(float(model.encode(xb)[0].std().item()), 1e-6)

    def ckpt_payload(lat_scale, full):
        ck = {"model": model.state_dict(),
              "config": {"width": args.width, "zc": args.zc, "arch": "v2"},
              "norm": {"mean": mean, "std": std,
                       "dbr_thresh": DBR_THRESH, "dbr_zero": DBR_ZERO},
              "latent_scale": lat_scale, "epoch": ep,
              "best": best, "best_ep": best_ep,
              # PSD-selection state must persist too: it is the ONLY thing that
              # decides whether vae_best_psd.pt gets overwritten, so losing it on
              # a chained resume would let a worse epoch replace a better one.
              "best_psd_score": best_psd_score, "best_psd_ep": best_psd_ep,
              "best_psd_val": best_psd_val}
        if full:
            ck.update({"disc": disc.state_dict() if gan else None,
                       "opt_g": opt_g.state_dict(),
                       "opt_d": opt_d.state_dict() if gan else None,
                       "global_step": global_step})
        return ck

    # ---- training loop -----------------------------------------------------
    t_all = time.time()
    ep = start_ep - 1        # bound even if the loop never runs (resume past --epochs)
    if start_ep > args.epochs:
        print(f"resume: checkpoint is already at epoch {start_ep - 1} >= --epochs "
              f"{args.epochs}; nothing to train, writing DONE.", flush=True)
    for ep in range(start_ep, args.epochs + 1):
        model.train()
        if gan:
            disc.train()
        sums = {"rec_w": 0.0, "rec_p": 0.0, "kl": 0.0, "g_adv": 0.0, "d_loss": 0.0,
                "lam": 0.0, "d_real": 0.0, "d_fake": 0.0}
        n_gan_steps, seen, t0 = 0, 0, time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for i, (x, m) in enumerate(dl_tr, 1):
            x = x.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            gan_on = gan and global_step >= disc_start

            with amp():
                recon, mu, lv = model(x)
                l_rec_w = weighted_l1(recon, x, m, mean, std, args.alpha, args.rcap)
                l_rec_p = masked_l1(recon, x, m)
                l_kl = kl_loss(mu, lv)
                if gan_on:
                    logits_fake = disc(recon)
                    l_g = -logits_fake.mean()
            if gan_on:
                lam = adaptive_weight(l_rec_w, l_g, model.get_last_layer(), args.disc_weight)
                loss = l_rec_w + args.beta * l_kl + lam * l_g
            else:
                lam = torch.zeros((), device=device)
                loss = l_rec_w + args.beta * l_kl
            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_g.step()

            if gan_on:
                with amp():
                    logits_real = disc(x)
                    logits_fake2 = disc(recon.detach())
                    l_d = (F.relu(1.0 - logits_real).mean()
                           + F.relu(1.0 + logits_fake2).mean())
                opt_d.zero_grad(set_to_none=True)
                l_d.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                opt_d.step()
                sums["g_adv"] += l_g.item(); sums["d_loss"] += l_d.item()
                sums["lam"] += lam.item()
                sums["d_real"] += logits_real.mean().item()
                sums["d_fake"] += logits_fake2.mean().item()
                n_gan_steps += 1

            sums["rec_w"] += l_rec_w.item(); sums["rec_p"] += l_rec_p.item()
            sums["kl"] += l_kl.item()
            seen += x.size(0); global_step += 1
            if i % log_every == 0:
                ips = seen / (time.time() - t0)
                extra = (f" | g_adv {sums['g_adv']/max(n_gan_steps,1):.3f}"
                         f" d {sums['d_loss']/max(n_gan_steps,1):.3f}"
                         f" lam {sums['lam']/max(n_gan_steps,1):.3f}") if n_gan_steps else ""
                print(f"  ep{ep} step {i}/{nsteps} | rec_w {sums['rec_w']/i:.4f}"
                      f" rec {sums['rec_p']/i:.4f}{extra} | {ips:.0f} img/s", flush=True)
        dt = time.time() - t0

        # ---- validation + domain metrics -----------------------------------
        model.eval()
        vl_p = vl_w = 0.0
        with torch.no_grad():
            for x, m in dl_va:
                x, m = x.to(device), m.to(device)
                with amp():
                    recon, mu, lv = model(x)
                vl_p += masked_l1(recon, x, m).item()
                vl_w += weighted_l1(recon, x, m, mean, std, args.alpha, args.rcap).item()
        vl_p /= max(len(dl_va), 1); vl_w /= max(len(dl_va), 1)
        if not (math.isfinite(vl_p) and math.isfinite(vl_w)):
            raise SystemExit(f"ERROR: validation loss is not finite at epoch {ep} "
                             f"(rec {vl_p}, rec_w {vl_w}): training diverged, most "
                             "likely the adversarial term. Resume from the previous "
                             "vae_last.pt with a lower --disc-weight (e.g. 0.25).")
        dm = domain_metrics(model, va_ds, dm_indices, mean, std, device)
        gpu = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        gd = max(n_gan_steps, 1)

        rec = {"epoch": ep,
               "train": {"rec_w": sums["rec_w"] / nsteps, "rec_p": sums["rec_p"] / nsteps,
                         "kl": sums["kl"] / nsteps, "g_adv": sums["g_adv"] / gd,
                         "d_loss": sums["d_loss"] / gd, "lam": sums["lam"] / gd,
                         "d_real": sums["d_real"] / gd, "d_fake": sums["d_fake"] / gd,
                         "gan_steps": n_gan_steps},
               "val": {"rec_p": vl_p, "rec_w": vl_w},
               "domain": dm,
               "sys": {"epoch_sec": round(dt, 1), "gpu_gb": round(gpu, 2),
                       "imgs_per_s": round(seen / dt, 1)}}
        log.append(rec)
        print(f"epoch {ep:3d}/{args.epochs} | val rec {vl_p:.4f} (w {vl_w:.4f}) "
              f"| csi8 {dm['csi_8']:.3f} csi32 {dm['csi_32']:.3f} "
              f"| tail16 {dm['tail_ratio_16']:.2f} tail32 {dm['tail_ratio_32']:.2f} "
              f"| psd2-8km {dm['psd_ratio_2_8km']:.2f} "
              f"| {dt:.0f}s | {rec['sys']['imgs_per_s']:.0f} img/s | GPU {gpu:.1f} GB", flush=True)

        # ---- checkpoints, curves, periodic verify --------------------------
        lat = latent_scale_now()
        if ep > grace_ep and vl_w < best:
            best, best_ep = vl_w, ep
            atomic_save(ckpt_payload(lat, full=False), os.path.join(args.out, "vae_best.pt"))
            print(f"  new best (val rec_w {best:.4f}) -> vae_best.pt", flush=True)
        # Second selection criterion: small-scale power. vae_best.pt ranks on
        # reconstruction loss ALONE, which is why epoch 9 (PSD 0.739) was kept over
        # epoch 10 (PSD 0.884) in the first v2 run. The diffusion stage cares about
        # PSD, and vae_last.pt is overwritten every epoch, so a good-PSD epoch is
        # otherwise unrecoverable. Score |psd - 1| (over-sharp is as wrong as blurred)
        # and require the reconstruction to stay within psd-tol of the best seen, so
        # this never selects a checkpoint that reconstructs badly.
        psd_now = dm.get("psd_ratio_2_8km", float("nan"))
        if ep > grace_ep and math.isfinite(psd_now) and vl_w <= best * (1.0 + args.psd_tol):
            score = abs(psd_now - 1.0)
            if score < best_psd_score:
                best_psd_score, best_psd_ep, best_psd_val = score, ep, psd_now
                atomic_save(ckpt_payload(lat, full=False),
                            os.path.join(args.out, "vae_best_psd.pt"))
                print(f"  new best PSD ({psd_now:.3f}, |psd-1| {score:.3f}, "
                      f"val rec_w {vl_w:.4f}) -> vae_best_psd.pt", flush=True)
        atomic_save(ckpt_payload(lat, full=True), os.path.join(args.out, "vae_last.pt"))
        atomic_json(log, os.path.join(args.out, "train_log.json"))
        plot_curves(log, os.path.join(args.out, "training_curves.png"))
        if (args.verify_every > 0 and ep % args.verify_every == 0) or ep == args.epochs:
            verify(model, va_npz, mean, std, device,
                   os.path.join(args.out, f"verify_ep{ep:02d}.png"), n=200)
            save_recons(model, va_ds, dm_indices, mean, std, device,
                        os.path.join(args.out, f"recon_ep{ep:02d}.png"))
        # early stop counts from the end of the grace period, so it also fires
        # if no post-grace epoch ever improves on the pre-GAN baseline
        if args.patience and ep > grace_ep and ep - max(best_ep, grace_ep) >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs "
                  f"(best ep{best_ep or 'none'}, val rec_w {best:.4f})", flush=True)
            break

    # ---- completion marker: stops the sbatch self-resubmit chain -----------
    # Written on normal completion or early stop, so a chained job that starts
    # afterwards exits immediately instead of retraining. Delete it (or pass
    # --ignore-done) to extend a finished run with more epochs.
    atomic_json({"reason": "early-stop" if ep < args.epochs else "completed",
                 "epochs_run": ep, "best_val_rec_w": best, "best_ep": best_ep,
                 "best_psd": best_psd_val, "best_psd_ep": best_psd_ep,
                 "wall_min": round((time.time() - t_all) / 60, 1)},
                os.path.join(args.out, "DONE"))

    # ---- final: reload best and run the full verification ------------------
    bp = os.path.join(args.out, "vae_best.pt")
    if os.path.exists(bp):
        ck = torch.load(bp, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"\ntotal time {(time.time()-t_all)/60:.1f} min | "
              f"best val rec_w {best:.4f} (ep{best_ep}) | "
              f"latent_scale {ck['latent_scale']:.3f}", flush=True)
        if best_psd_ep:
            print(f"best PSD {best_psd_val:.3f} at ep{best_psd_ep} -> vae_best_psd.pt "
                  "(use this one for the diffusion stage if PSD is the priority)",
                  flush=True)
        verify(model, va_npz, mean, std, device, os.path.join(args.out, "vae_verify.png"))
    else:
        print("\nno vae_best.pt was saved (run ended before the grace period); "
              "use vae_last.pt or rerun with more epochs", flush=True)


if __name__ == "__main__":
    main()
