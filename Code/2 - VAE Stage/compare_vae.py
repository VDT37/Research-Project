#!/usr/bin/env python3
"""
compare_vae.py - apples-to-apples domain metrics for VAE v1 vs VAE v2.

v1 never logged reconstruction CSI / tail / PSD during training (its log holds
only recon, kl, val_recon). This script loads both checkpoints and runs the
IDENTICAL measurement code (train_vae_v2.domain_metrics) on the SAME validation
crops, each codec using its own stored normalisation. The v1 column then becomes
measured, not estimated, so the v1-vs-v2 comparison in the dissertation rests on
real numbers.

    conda activate nowcast
    python compare_vae.py
        --v1 ~/dissertation_outputs/vae_v1/vae_best.pt
        --v2 ~/dissertation_outputs/vae_v2/vae_best.pt

Requires train_vae.py (v1 VAE class) and train_vae_v2.py (v2 VAE + metrics) in
the same directory.
"""
import os, glob, json, getpass, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_vae import VAE as VAE1                     # v1 architecture
from train_vae_v2 import (VAE as VAE2, NpzFields, domain_metrics, masked_l1,
                          CSI_T, TAIL_T)

USER  = getpass.getuser()
PRIOR = f"/scratch/{USER}/dissertation/prior"
OUT   = os.path.expanduser("~/dissertation_outputs")


def eval_ckpt(ckpt_path, VAECls, val_files, device, n_domain=512, n_l1=1024, workers=8):
    ck = torch.load(ckpt_path, map_location=device)
    model = VAECls(w=ck["config"]["width"], zc=ck["config"]["zc"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    mean, std = ck["norm"]["mean"], ck["norm"]["std"]

    ds = NpzFields(val_files, mean, std)              # rows: y,A,y,A,... per file
    obs = np.arange(0, len(ds), 2)                    # even rows = y (observations)
    idx = obs[np.linspace(0, len(obs) - 1, min(n_domain, len(obs))).astype(int)]
    dm = domain_metrics(model, ds, idx, mean, std, device)

    # plain masked L1 over the SAME kind of y-subset, both codecs, same data
    l1_idx = obs[np.linspace(0, len(obs) - 1, min(n_l1, len(obs))).astype(int)]
    sub = torch.utils.data.Subset(ds, l1_idx.tolist())
    dl = DataLoader(sub, batch_size=32, num_workers=workers)
    tot, nb = 0.0, 0
    with torch.no_grad():
        for x, m in dl:
            x, m = x.to(device), m.to(device)
            rec = model(x)[0]
            tot += masked_l1(rec, x, m).item(); nb += 1
    dm["val_l1_plain"] = tot / max(nb, 1)
    dm["n_params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
    dm["norm"] = (round(mean, 3), round(std, 3))
    return dm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default=os.path.join(OUT, "vae_v1", "vae_best.pt"))
    ap.add_argument("--v2", default=os.path.join(OUT, "vae_v2", "vae_best.pt"))
    ap.add_argument("--root", default=PRIOR)
    ap.add_argument("--n", type=int, default=512, help="crops for CSI/PSD")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_files = sorted(glob.glob(os.path.join(args.root, "val", "*", "*.npz")))
    if not val_files:
        raise SystemExit(f"no val npz under {args.root}/val")
    for p in (args.v1, args.v2):
        if not os.path.exists(p):
            raise SystemExit(f"checkpoint not found: {p}")
    print(f"comparing on {len(val_files)} val files, {args.n} crops for CSI/PSD, "
          f"device={device}\n", flush=True)

    r1 = eval_ckpt(args.v1, VAE1, val_files, device, args.n)
    print("v1 done", flush=True)
    r2 = eval_ckpt(args.v2, VAE2, val_files, device, args.n)
    print("v2 done\n", flush=True)

    rows = [("params (M)", "n_params_M", "{:.2f}"),
            ("val L1 (plain)", "val_l1_plain", "{:.4f}"),
            ("mu_std (latent)", "mu_std", "{:.2f}")]
    rows += [(f"CSI @ {t:g}", f"csi_{t:g}", "{:.3f}") for t in CSI_T]
    rows += [(f"tail ratio {t:g}", f"tail_ratio_{t:g}", "{:.3f}") for t in TAIL_T]
    rows += [("PSD ratio 2-8km", "psd_ratio_2_8km", "{:.3f}")]

    print(f"{'metric':>18} | {'v1':>9} | {'v2':>9} | {'winner':>7}")
    print("-" * 54)
    for label, key, fmt in rows:
        a, b = r1[key], r2[key]
        # winner: for L1 lower is better; for CSI higher is better;
        # for tail/PSD closer to 1 is better
        if key == "val_l1_plain":
            win = "v1" if a < b else "v2"
        elif key.startswith("csi_"):
            win = "v1" if a > b else "v2"
        elif key.startswith("tail_") or key.startswith("psd_"):
            win = "v1" if abs(a - 1) < abs(b - 1) else "v2"
        else:
            win = ""
        print(f"{label:>18} | {fmt.format(a):>9} | {fmt.format(b):>9} | {win:>7}")

    json.dump({"v1": r1, "v2": r2}, open(os.path.join(OUT, "compare_vae.json"), "w"),
              indent=2, default=float)
    print(f"\nsaved -> {os.path.join(OUT, 'compare_vae.json')}")


if __name__ == "__main__":
    main()
