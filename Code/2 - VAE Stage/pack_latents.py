#!/usr/bin/env python3
"""
pack_latents.py - encode the prior npz cache into latent memmaps for the
diffusion stage (LDM.md, Stage 0).

For every crop the frozen VAE v2 encoder maps six 256x256 dBR fields
(x1..x4, A, y) to scaled 4x64x64 latents z = latent_scale * mu(E(field)),
stored as one float16 row of 24 channels per crop:

    channels  0:4   z_x1  (t0-45 min)
              4:8   z_x2  (t0-30 min)
              8:12  z_x3  (t0-15 min)
             12:16  z_x4  (t0)
             16:20  z_A   (advection prior)
             20:24  z_y   (ground truth)

The diffusion target delta = z_y - z_A is derived on the fly by the trainer;
storing z_y (not delta) keeps the predict-z_y fallback open (LDM.md section 4).

While packing, sigma_data = std(z_y - z_A) is accumulated exactly (running
sum / sum-of-squares over every latent cell). It is the EDM noise constant
the trainer needs, and is written into the meta json.

Safety pattern mirrors pack_vae_data.py: pack into a .part file, spot-check
rows against a fresh re-encode, then atomically rename into place; the meta
json is only written after verification, and a pack is invalidated (repacked)
if the VAE checkpoint hash changes. Re-running skips complete packs.

    conda activate nowcast
    python pack_latents.py --vae ~/dissertation_outputs/vae_v2/vae_best.pt
    python pack_latents.py --check-only            # re-verify an existing pack

Paths honour the DISS_SCRATCH environment variable and default to
/scratch/<user>/dissertation (the Exeter convention). On JASMIN:
    export DISS_SCRATCH=/work/scratch-pw4/$USER/dissertation
(see README_jasmin.md, Stage C).

Outputs (in --out, default $DISS_SCRATCH/latents/):
    {split}_latents.npy         float16 memmap, (n_crops, 24, 64, 64)
    {split}_latents_meta.json   layout, latent_scale, sigma_data, VAE sha
    {split}_latents_index.json  the exact npz file list (row provenance)
"""
import os, glob, json, time, getpass, hashlib, argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

# Reuse the exact codec class and dBR transform so pack and training can never
# drift apart (train_vae_v2.py sits in the same directory on the server).
from train_vae_v2 import VAE, to_dbr

USER    = getpass.getuser()
SCRATCH = os.environ.get("DISS_SCRATCH", f"/scratch/{USER}/dissertation")
PRIOR   = os.path.join(SCRATCH, "prior")
OUT     = os.path.join(SCRATCH, "latents")
VAE_CKPT = os.path.expanduser("~/dissertation_outputs/vae_v2/vae_best.pt")

DBR_THRESH = 0.1
DBR_ZERO   = -15.0
FIELDS     = ("x1", "x2", "x3", "x4", "A", "y")     # channel-block order
H = W = 256
HL = WL = 64


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_json(obj, path, **kw):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, **kw)
    os.replace(tmp, path)


def load_crop_dbr(path):
    """One npz -> (6, 256, 256) float32 dBR, invalid pixels filled with the dry
    floor (to_dbr maps non-finite to DBR_ZERO, matching what the codec saw in
    training via PackedFields' fill)."""
    z = np.load(path, allow_pickle=True)
    out = np.empty((len(FIELDS), H, W), dtype="float32")
    x = z["x_mmh"].astype("float32")
    for j in range(4):
        out[j] = to_dbr(x[j])
    out[4] = to_dbr(z["A_mmh"].astype("float32"))
    out[5] = to_dbr(z["y_mmh"].astype("float32"))
    return out


def load_vae(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    for k in ("model", "config", "norm", "latent_scale"):
        if k not in ck:
            raise SystemExit(f"ERROR: {ckpt_path} has no '{k}' key; expected a "
                             "vae_best.pt / vae_last.pt checkpoint from train_vae_v2.py")
    nrm = ck["norm"]
    if abs(nrm["dbr_thresh"] - DBR_THRESH) > 1e-9 or abs(nrm["dbr_zero"] - DBR_ZERO) > 1e-9:
        raise SystemExit("ERROR: dBR constants in the VAE checkpoint do not match "
                         "this script; the data contract has diverged.")
    model = VAE(w=ck["config"]["width"], zc=ck["config"]["zc"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, float(ck["latent_scale"]), float(nrm["mean"]), float(nrm["std"]), ck


@torch.no_grad()
def encode_batch(vae, batch_np, mean, std, latent_scale, device, zc):
    """(B, 6, 256, 256) float32 dBR -> (B, 6*zc, 64, 64) float32 scaled latents."""
    B = batch_np.shape[0]
    x = torch.from_numpy(batch_np).to(device)
    x = (x - mean) / std
    mu, _ = vae.encode(x.reshape(B * len(FIELDS), 1, H, W))
    z = (latent_scale * mu).reshape(B, len(FIELDS) * zc, HL, WL)
    return z.float().cpu().numpy()


def spot_check(npy_path, files, vae, mean, std, latent_scale, device, zc, n=8, seed=0, atol=2e-2):
    """Re-encode a few random crops from the source npz and compare to the pack.
    atol covers float16 storage rounding plus tiny GPU non-determinism; latents
    are O(1) so 2e-2 is far below signal."""
    mm = np.load(npy_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), size=min(n, len(files)), replace=False)
    bad = 0
    for i in idx:
        fresh = encode_batch(vae, load_crop_dbr(files[i])[None], mean, std,
                             latent_scale, device, zc)[0]
        got = np.asarray(mm[i], dtype="float32")
        if np.isnan(got).any() or np.max(np.abs(fresh - got)) > atol:
            bad += 1
            print(f"  MISMATCH row {i} ({files[i]}): "
                  f"max|diff| {np.max(np.abs(fresh - got)):.4f}  nan {np.isnan(got).any()}")
    print(f"  spot check: {len(idx) - bad}/{len(idx)} rows verified against source npz")
    return bad == 0


def pack_split(split, args, vae, latent_scale, mean, std, zc, vae_sha, vae_ck, device):
    files = sorted(glob.glob(os.path.join(args.root, split, "*", "*.npz")))
    if not files:
        print(f"[{split}] no crops found under {os.path.join(args.root, split)}, skipping")
        return
    nch = len(FIELDS) * zc
    npy    = os.path.join(args.out, f"{split}_latents.npy")
    meta_p = os.path.join(args.out, f"{split}_latents_meta.json")

    if os.path.exists(npy) and os.path.exists(meta_p) and not args.force:
        try:
            meta = json.load(open(meta_p))
        except (json.JSONDecodeError, ValueError):
            print(f"[{split}] {meta_p} is corrupt (interrupted pack?): repacking")
            meta = {}
        n_disk = np.load(npy, mmap_mode="r").shape[0]
        if meta.get("n_files") == len(files) and n_disk == len(files) \
                and meta.get("vae_sha256") == vae_sha:
            print(f"[{split}] already packed ({len(files)} rows, same VAE), "
                  "skipping (use --force to redo)")
            return
        print(f"[{split}] existing pack is stale (meta n {meta.get('n_files')}, "
              f"disk {n_disk}, expected {len(files)}, "
              f"vae match {meta.get('vae_sha256') == vae_sha}): repacking")

    part = npy + ".part"
    gb = len(files) * nch * HL * WL * 2 / 1e9
    print(f"[{split}] encoding {len(files)} crops -> ({len(files)}, {nch}, {HL}, {WL}) "
          f"float16 ({gb:.1f} GB) on {device} ...", flush=True)
    mm = np.lib.format.open_memmap(part, mode="w+", dtype="float16",
                                   shape=(len(files), nch, HL, WL))
    # exact accumulators for sigma_data = std(z_y - z_A) and diagnostics
    d_sum = d_sumsq = 0.0
    d_cnt = 0
    zy_sum = zy_sumsq = 0.0

    ctx = mp.get_context("fork")
    t0, done, buf, i0 = time.time(), 0, [], 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        def flush():
            nonlocal buf, i0, d_sum, d_sumsq, d_cnt, zy_sum, zy_sumsq
            if not buf:
                return
            z = encode_batch(vae, np.stack(buf), mean, std, latent_scale, device, zc)
            d = z[:, 5 * zc:6 * zc] - z[:, 4 * zc:5 * zc]        # z_y - z_A
            d_sum   += float(d.sum());  d_sumsq += float((d * d).sum())
            d_cnt   += d.size
            zy = z[:, 5 * zc:6 * zc]
            zy_sum  += float(zy.sum()); zy_sumsq += float((zy * zy).sum())
            mm[i0:i0 + len(buf)] = z.astype("float16")
            i0 += len(buf)
            buf = []

        for arr in ex.map(load_crop_dbr, files, chunksize=8):
            buf.append(arr)
            if len(buf) == args.batch:
                flush()
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(files)} crops | {done / (time.time() - t0):.0f} crops/s",
                      flush=True)
        flush()
    mm.flush()
    del mm

    d_mean = d_sum / d_cnt
    sigma_data = float(np.sqrt(max(d_sumsq / d_cnt - d_mean ** 2, 0.0)))
    zy_mean = zy_sum / d_cnt
    zy_std = float(np.sqrt(max(zy_sumsq / d_cnt - zy_mean ** 2, 0.0)))
    print(f"[{split}] latent stats: sigma_data (std of z_y - z_A) = {sigma_data:.4f} "
          f"| delta mean {d_mean:+.4f} | z_y std {zy_std:.3f}", flush=True)

    if not spot_check(part, files, vae, mean, std, latent_scale, device, zc, n=args.spot):
        print(f"[{split}] VERIFICATION FAILED: leaving {part} for inspection, "
              "pack NOT installed", flush=True)
        return
    # Install order matters: the npy is renamed into place FIRST, and the meta
    # json (the commit record the skip-check trusts) is written LAST. A SIGKILL
    # anywhere in between leaves new-npy + stale-meta, which the staleness check
    # correctly repacks. The reverse order would leave new-meta + old-npy, which
    # would wrongly pass the skip-check after a VAE change (silent corruption).
    os.replace(part, npy)
    atomic_json(files, os.path.join(args.out, f"{split}_latents_index.json"))
    atomic_json({"n_files": len(files), "n_channels": nch, "h": HL, "w": WL,
                 "dtype": "float16",
                 "channel_layout": "0:4 z_x1, 4:8 z_x2, 8:12 z_x3, 12:16 z_x4, "
                                   "16:20 z_A, 20:24 z_y (zc=4 blocks, files sorted)",
                 "fields": list(FIELDS), "zc": zc,
                 "latent_scale": latent_scale,
                 "norm": {"mean": mean, "std": std,
                          "dbr_thresh": DBR_THRESH, "dbr_zero": DBR_ZERO},
                 "sigma_data": sigma_data,
                 "delta_mean": d_mean, "zy_std": zy_std,
                 "vae_ckpt": os.path.abspath(args.vae),
                 "vae_sha256": vae_sha,
                 "vae_epoch": vae_ck.get("epoch"),
                 "vae_config": vae_ck.get("config")},
                meta_p, indent=2)
    print(f"[{split}] done -> {npy} (verified)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default=VAE_CKPT, help="frozen codec checkpoint")
    ap.add_argument("--root", default=PRIOR, help="prior npz cache root")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--batch", type=int, default=16, help="crops per encoder pass (x6 fields)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    ap.add_argument("--spot", type=int, default=8, help="spot-check sample size")
    ap.add_argument("--force", action="store_true", help="repack even if present")
    ap.add_argument("--check-only", action="store_true", help="only run the spot checks")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found; encoding on CPU will be very slow.", flush=True)
    torch.backends.cudnn.benchmark = False       # keep encoding reproducible

    if not os.path.exists(args.vae):
        raise SystemExit(f"ERROR: VAE checkpoint not found: {args.vae}")
    vae, latent_scale, mean, std, vae_ck = load_vae(args.vae, device)
    zc = vae_ck["config"]["zc"]
    vae_sha = sha256_file(args.vae)
    print(f"VAE: {args.vae} (epoch {vae_ck.get('epoch')}, sha {vae_sha[:12]}) | "
          f"latent_scale {latent_scale:.3f} | norm mean {mean:.3f} std {std:.3f}", flush=True)

    for split in args.splits.split(","):
        split = split.strip()
        if args.check_only:
            idx_p = os.path.join(args.out, f"{split}_latents_index.json")
            npy = os.path.join(args.out, f"{split}_latents.npy")
            if not (os.path.exists(idx_p) and os.path.exists(npy)):
                print(f"[{split}] no pack to check")
                continue
            files = json.load(open(idx_p))
            spot_check(npy, files, vae, mean, std, latent_scale, device, zc, n=10)
        else:
            pack_split(split, args, vae, latent_scale, mean, std, zc, vae_sha, vae_ck, device)


if __name__ == "__main__":
    main()
