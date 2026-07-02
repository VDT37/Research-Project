#!/usr/bin/env python3
"""
train_vae.py — train a small VAE that compresses 256x256 dBR radar crops into a
4x64x64 latent (x4 spatial compression), for the latent-diffusion stage.

It trains on the advection-prior cache (the target `y` and the prior `A` of each
crop), selects the best model on the validation split, and verifies that the VAE
*preserves the rain-rate distribution and power spectrum* (a blurry VAE would
defeat the whole point). See docs/VAE-Explanation.md for the full walkthrough.

    conda activate nowcast
    pip install torch --index-url https://download.pytorch.org/whl/cu124   # once, for the L4
    python train_vae.py --epochs 40                 # full cache
    python train_vae.py --limit 4000 --epochs 5     # quick smoke test

Outputs (to ~/dissertation_outputs/vae/):
    vae_best.pt        checkpoint (weights + config + normalisation + latent scale)
    recon_epochXX.png  reconstructions during training
    vae_verify.png     histogram + PSD: observations vs VAE reconstruction
    train_log.json     per-epoch losses (reproducibility)
"""
import os, glob, json, getpass, argparse, random, contextlib, time
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
OUT        = os.path.expanduser("~/dissertation_outputs/vae")
DBR_THRESH = 0.1            # mm/h below this is "dry"
DBR_ZERO   = -15.0          # dBR value assigned to dry pixels (~0.03 mm/h)


# ----------------------------------------------------------------------------
# Data: each item is one 256x256 dBR field (the y or A of a crop), normalised.
# ----------------------------------------------------------------------------
def to_dbr(R):
    """mm/h -> dBR (log). Dry / non-finite pixels -> DBR_ZERO."""
    R = np.asarray(R, dtype="float32")
    out = np.full_like(R, DBR_ZERO)
    wet = np.isfinite(R) & (R >= DBR_THRESH)
    out[wet] = 10.0 * np.log10(R[wet])
    return out


def from_dbr(dbr):
    """dBR -> mm/h (inverse), with the dry floor mapped back to 0."""
    R = np.power(10.0, dbr / 10.0)
    R[dbr <= DBR_ZERO + 1e-3] = 0.0
    return R


class RadarFields(Dataset):
    """(file, field) pairs -> normalised dBR tensor (1,H,W) + valid mask (1,H,W)."""
    def __init__(self, files, mean, std, fields=("y_mmh", "A_mmh")):
        self.items = [(f, k) for f in files for k in fields]
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, key = self.items[i]
        R = np.load(f, allow_pickle=True)[key].astype("float32")
        if R.ndim == 3:                     # safety (x_mmh is 4xHxW)
            R = R[-1]
        valid = np.isfinite(R).astype("float32")
        x = (to_dbr(R) - self.mean) / self.std
        return torch.from_numpy(x)[None], torch.from_numpy(valid)[None]


def compute_norm(files, fields=("y_mmh", "A_mmh"), n=400):
    """dBR mean/std over a random sample (for zero-mean/unit-var normalisation)."""
    rng = random.Random(0)
    vals = []
    for f in rng.sample(files, min(n, len(files))):
        z = np.load(f, allow_pickle=True)
        for k in fields:
            vals.append(to_dbr(z[k].astype("float32")).ravel())
    v = np.concatenate(vals)
    return float(v.mean()), float(v.std())


# ----------------------------------------------------------------------------
# Model: a small convolutional VAE (encoder -> latent distribution -> decoder).
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


class VAE(nn.Module):
    """256x256x1  <->  (zc)x64x64 latent  (x4 spatial compression)."""
    def __init__(self, w=64, zc=4):
        super().__init__()
        self.zc = zc
        self.enc = nn.Sequential(
            nn.Conv2d(1, w, 3, padding=1), ResBlock(w),
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), ResBlock(2 * w),   # 256->128
            nn.Conv2d(2 * w, 2 * w, 3, stride=2, padding=1), ResBlock(2 * w),  # 128->64
            nn.GroupNorm(8, 2 * w), nn.SiLU(),
            nn.Conv2d(2 * w, 2 * zc, 1),                                    # -> mean,logvar
        )
        self.dec = nn.Sequential(
            nn.Conv2d(zc, 2 * w, 3, padding=1), ResBlock(2 * w),
            nn.ConvTranspose2d(2 * w, 2 * w, 4, stride=2, padding=1), ResBlock(2 * w),  # 64->128
            nn.ConvTranspose2d(2 * w, w, 4, stride=2, padding=1), ResBlock(w),          # 128->256
            nn.GroupNorm(8, w), nn.SiLU(),
            nn.Conv2d(w, 1, 3, padding=1),
        )

    def encode(self, x):
        h = self.enc(x)
        return h[:, :self.zc], h[:, self.zc:]          # mean, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)   # reparameterisation
        return self.dec(z), mu, logvar


def recon_loss(recon, x, mask):
    """Masked L1 (mean absolute error) over valid pixels only."""
    return ((recon - x).abs() * mask).sum() / mask.sum().clamp(min=1.0)


def kl_loss(mu, logvar):
    """KL( N(mu,sigma) || N(0,1) ) — keeps the latent space well-behaved."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()


# ----------------------------------------------------------------------------
# Verification: does the VAE preserve the rain distribution + power spectrum?
# ----------------------------------------------------------------------------
def radial_psd(field):
    f = np.nan_to_num(field).astype("float64")
    f = f - f.mean()
    P = np.fft.fftshift(np.abs(np.fft.fft2(f)) ** 2)
    H, W = f.shape
    Y, X = np.ogrid[:H, :W]
    r = np.hypot(Y - H // 2, X - W // 2).astype(int)
    return np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)


@torch.no_grad()
def verify(model, files, mean, std, device, out_png, n=300):
    rng = random.Random(1)
    sample = rng.sample(files, min(n, len(files)))
    rbins = np.logspace(np.log10(DBR_THRESH), np.log10(128), 41)
    h_obs, h_rec = np.zeros(40), np.zeros(40)
    psd_obs = psd_rec = None
    model.eval()
    for f in sample:
        R = np.load(f, allow_pickle=True)["y_mmh"].astype("float32")
        x = torch.from_numpy(((to_dbr(R) - mean) / std))[None, None].to(device)
        rec = model(x)[0][0, 0].float().cpu().numpy()
        R_rec = from_dbr(rec * std + mean)
        R_obs = np.nan_to_num(R)
        h_obs += np.histogram(R_obs[R_obs >= DBR_THRESH], bins=rbins)[0]
        h_rec += np.histogram(R_rec[R_rec >= DBR_THRESH], bins=rbins)[0]
        po, pr = radial_psd(R_obs), radial_psd(R_rec)
        psd_obs = po if psd_obs is None else psd_obs + po
        psd_rec = pr if psd_rec is None else psd_rec + pr
    rc = np.sqrt(rbins[:-1] * rbins[1:])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].loglog(rc, h_obs / h_obs.sum(), "k", lw=2, label="observed")
    ax[0].loglog(rc, h_rec / h_rec.sum(), "C0", lw=1.5, label="VAE reconstruction")
    ax[0].set(title="Rain-rate distribution", xlabel="mm/h", ylabel="frequency")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    k = np.arange(1, len(psd_obs)); wl = 256.0 / k
    ax[1].loglog(wl, psd_obs[1:] / len(sample), "k", lw=2, label="observed")
    ax[1].loglog(wl, psd_rec[1:] / len(sample), "C0", lw=1.5, label="VAE reconstruction")
    ax[1].set(title="Power spectrum (PSD)", xlabel="wavelength (km)", ylabel="power")
    ax[1].invert_xaxis(); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print("saved verification ->", out_png)


@torch.no_grad()
def save_recons(model, batch, mean, std, device, png):
    x = batch[0][:4].to(device)
    rec = model(x)[0]
    fig, ax = plt.subplots(2, 4, figsize=(14, 7))
    for j in range(min(4, x.shape[0])):
        for row, img in ((0, x), (1, rec)):
            a = img[j, 0].float().cpu().numpy() * std + mean
            ax[row, j].imshow(from_dbr(a), vmin=0, vmax=8, cmap="viridis")
            ax[row, j].axis("off")
    ax[0, 0].set_ylabel("input"); ax[1, 0].set_ylabel("recon")
    plt.tight_layout(); plt.savefig(png, dpi=100, bbox_inches="tight"); plt.close()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=1e-6, help="KL weight")
    ap.add_argument("--width", type=int, default=64, help="base channels (÷8)")
    ap.add_argument("--zc", type=int, default=4, help="latent channels")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cap #crops (smoke test)")
    ap.add_argument("--root", default=PRIOR)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0); random.seed(0); np.random.seed(0)

    tr = sorted(glob.glob(os.path.join(args.root, "train", "*", "*.npz")))
    va = sorted(glob.glob(os.path.join(args.root, "val", "*", "*.npz")))
    if args.limit:
        tr, va = tr[:args.limit], va[:max(200, args.limit // 10)]
    if not tr:
        raise SystemExit(f"no train crops under {args.root}/train")
    print(f"train crops {len(tr)} | val crops {len(va)}")

    mean, std = compute_norm(tr)
    print(f"dBR normalisation: mean={mean:.3f} std={std:.3f}")

    dl_tr = DataLoader(RadarFields(tr, mean, std), batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(RadarFields(va, mean, std), batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAE(w=args.width, zc=args.zc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"device={device} | VAE params={nparams/1e6:.2f}M")

    use_amp = device == "cuda"
    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    if device == "cuda":
        torch.backends.cudnn.benchmark = True                 # fixed input size -> faster
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} | {p.total_memory/1e9:.0f} GB total", flush=True)
    else:
        print("WARNING: no GPU found — running on CPU (very slow)", flush=True)

    nsteps = len(dl_tr)
    print(f"{nsteps} steps/epoch (batch {args.batch})", flush=True)
    log, best, t_all = [], float("inf"), time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tl = tk = 0.0; seen = 0; t0 = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for i, (x, m) in enumerate(dl_tr, 1):
            x, m = x.to(device, non_blocking=True), m.to(device, non_blocking=True)
            with amp():
                rec, mu, lv = model(x)
                lr_ = recon_loss(rec, x, m); lkl = kl_loss(mu, lv)
                loss = lr_ + args.beta * lkl
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tl += lr_.item(); tk += lkl.item(); seen += x.size(0)
            if i % 200 == 0:                                    # live progress every 200 steps
                ips = seen / (time.time() - t0)
                print(f"  ep{ep} step {i}/{nsteps} | recon {tl/i:.4f} | {ips:.0f} img/s",
                      flush=True)
        dt = time.time() - t0
        model.eval(); vl = 0.0
        with torch.no_grad():
            for x, m in dl_va:
                x, m = x.to(device), m.to(device)
                with amp():
                    rec, mu, lv = model(x)
                vl += recon_loss(rec, x, m).item()
        vl /= max(len(dl_va), 1)
        rec_tr, kl_tr = tl / nsteps, tk / nsteps
        gpu = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        print(f"epoch {ep:3d}/{args.epochs} | train recon {rec_tr:.4f} KL {kl_tr:.2f} "
              f"| val recon {vl:.4f} | {dt:.0f}s/epoch | peak GPU {gpu:.1f} GB", flush=True)
        log.append({"epoch": ep, "train_recon": rec_tr, "kl": kl_tr, "val_recon": vl,
                    "epoch_sec": round(dt, 1), "gpu_gb": round(gpu, 2)})
        if vl < best:
            best = vl
            # latent scale: std of latents, so the diffusion sees ~unit-variance latents
            with torch.no_grad():
                xb = next(iter(dl_va))[0].to(device)
                lat_std = float(model.encode(xb)[0].std().item())
            torch.save({"model": model.state_dict(),
                        "config": {"width": args.width, "zc": args.zc},
                        "norm": {"mean": mean, "std": std,
                                 "dbr_thresh": DBR_THRESH, "dbr_zero": DBR_ZERO},
                        "latent_scale": 1.0 / max(lat_std, 1e-6)},
                       os.path.join(OUT, "vae_best.pt"))
        if ep % 5 == 0 or ep == args.epochs:
            save_recons(model, next(iter(dl_va)), mean, std, device,
                        os.path.join(OUT, f"recon_ep{ep:02d}.png"))
        json.dump(log, open(os.path.join(OUT, "train_log.json"), "w"), indent=2)

    # reload best and verify it preserves the distribution + spectrum
    ck = torch.load(os.path.join(OUT, "vae_best.pt"), map_location=device)
    model.load_state_dict(ck["model"])
    print(f"\ntotal train time {(time.time()-t_all)/60:.1f} min | "
          f"best val recon {best:.4f} | latent_scale {ck['latent_scale']:.3f}", flush=True)
    verify(model, va, mean, std, device, os.path.join(OUT, "vae_verify.png"))
    print(f"checkpoint -> {os.path.join(OUT, 'vae_best.pt')}")


if __name__ == "__main__":
    main()
