#!/usr/bin/env python3
"""
pack_vae_data.py - pack the VAE training fields into float16 memmaps.

Why: VAE v1 trained at ~52 img/s because every sample required opening and
zlib-decompressing an .npz file. This script does that decompression exactly
once, writing the y_mmh and A_mmh fields of every prior crop into one
contiguous .npy memmap per split (dBR domain, NaN where the radar is invalid).
Training then reads 131 KB slices, and after the first epoch the OS page cache
holds the whole array in RAM (the packed train split is ~40 GB, the server has
384 GB), so epochs 2+ stream from memory.

Row layout: row 2*i   = y_mmh of the i-th file (sorted order),
            row 2*i+1 = A_mmh of the i-th file.
Even rows are therefore ground-truth observations; the training code relies on
this convention for its observation-only diagnostics.

    conda activate nowcast
    python pack_vae_data.py                  # packs train and val (~10-25 min)
    python pack_vae_data.py --check-only     # re-verify an existing pack

Outputs (in --out, default /scratch/<user>/dissertation/packed/):
    {split}_fields.npy    float16 memmap, shape (2*n_files, 256, 256), dBR
    {split}_meta.json     counts, shape, field order, dtype
    {split}_index.json    the exact file list (row provenance)
"""
import os, glob, json, getpass, argparse, random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

USER   = getpass.getuser()
PRIOR  = f"/scratch/{USER}/dissertation/prior"
PACKED = f"/scratch/{USER}/dissertation/packed"
DBR_THRESH = 0.1
DBR_ZERO   = -15.0
FIELDS = ("y_mmh", "A_mmh")
H = W = 256


def to_dbr_nan(R):
    """mm/h -> dBR; wet pixels get 10*log10(R), dry get DBR_ZERO, invalid get NaN."""
    R = np.asarray(R, dtype="float32")
    out = np.full_like(R, DBR_ZERO)
    finite = np.isfinite(R)
    wet = finite & (R >= DBR_THRESH)
    out[wet] = 10.0 * np.log10(R[wet])
    out[~finite] = np.nan
    return out


def atomic_json(obj, path, **kw):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, **kw)
    os.replace(tmp, path)


def load_fields(path):
    """One npz -> (2, 256, 256) float16 dBR array (y then A)."""
    z = np.load(path, allow_pickle=True)
    out = np.empty((len(FIELDS), H, W), dtype="float16")
    for j, k in enumerate(FIELDS):
        out[j] = to_dbr_nan(z[k].astype("float32")).astype("float16")
    return out


def spot_check(npy_path, files, n=5, seed=0):
    """Recompute a few random rows from the source npz and compare."""
    mm = np.load(npy_path, mmap_mode="r")
    rng = random.Random(seed)
    bad = 0
    for _ in range(n):
        i = rng.randrange(len(files))
        want = load_fields(files[i])
        got = np.asarray(mm[2 * i:2 * i + 2])
        same_nan = np.array_equal(np.isnan(want), np.isnan(got))
        close = np.allclose(np.nan_to_num(want), np.nan_to_num(got), atol=1e-3)
        if not (same_nan and close):
            bad += 1
            print(f"  MISMATCH at file {files[i]}")
    print(f"  spot check: {n - bad}/{n} rows verified against source npz")
    return bad == 0


def pack_split(split, root, out_dir, workers, force):
    files = sorted(glob.glob(os.path.join(root, split, "*", "*.npz")))
    if not files:
        print(f"[{split}] no crops found, skipping")
        return
    n_rows = len(files) * len(FIELDS)
    npy = os.path.join(out_dir, f"{split}_fields.npy")
    meta_p = os.path.join(out_dir, f"{split}_meta.json")

    if os.path.exists(npy) and os.path.exists(meta_p) and not force:
        try:
            meta = json.load(open(meta_p))
        except (json.JSONDecodeError, ValueError):
            print(f"[{split}] {meta_p} is corrupt (interrupted pack?): repacking")
            meta = {}
        n_disk = np.load(npy, mmap_mode="r").shape[0]
        if meta.get("n_rows") == n_rows and n_disk == n_rows:
            print(f"[{split}] already packed ({n_rows} rows), skipping (use --force to redo)")
            return
        print(f"[{split}] existing pack is stale (meta {meta.get('n_rows')}, on disk "
              f"{n_disk}, expected {n_rows}): repacking")

    # Pack into a .part file and only rename it into place after the spot
    # check passes, so a killed or in-progress pack can never be mistaken for
    # a complete one (the trainer additionally requires the meta json, which
    # is only written here, after verification).
    part = npy + ".part"
    print(f"[{split}] packing {len(files)} files -> {n_rows} rows "
          f"({n_rows * H * W * 2 / 1e9:.1f} GB) with {workers} workers ...", flush=True)
    mm = np.lib.format.open_memmap(part, mode="w+", dtype="float16", shape=(n_rows, H, W))
    ctx = mp.get_context("fork")
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for i, arr in enumerate(ex.map(load_fields, files, chunksize=32)):
            mm[2 * i:2 * i + 2] = arr
            done += 1
            if done % 5000 == 0:
                print(f"  {done}/{len(files)} files", flush=True)
    mm.flush()
    del mm

    if not spot_check(part, files):
        print(f"[{split}] VERIFICATION FAILED: leaving {part} for inspection, "
              "pack NOT installed", flush=True)
        return
    atomic_json(files, os.path.join(out_dir, f"{split}_index.json"))
    atomic_json({"n_files": len(files), "n_rows": n_rows, "h": H, "w": W,
                 "dtype": "float16", "fields": list(FIELDS),
                 "row_layout": "row 2i = y_mmh, row 2i+1 = A_mmh, files sorted",
                 "dbr_thresh": DBR_THRESH, "dbr_zero": DBR_ZERO,
                 "invalid_encoding": "NaN"},
                meta_p, indent=2)
    os.replace(part, npy)

    a = np.load(npy, mmap_mode="r")
    sample = np.asarray(a[np.linspace(0, n_rows - 1, 200).astype(int)], dtype="float32")
    print(f"  stats over 200 sampled rows: NaN frac {np.isnan(sample).mean():.3f} | "
          f"min {np.nanmin(sample):.1f} | max {np.nanmax(sample):.1f} dBR")
    print(f"[{split}] done -> {npy} (verified)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=PRIOR)
    ap.add_argument("--out", default=PACKED)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    ap.add_argument("--force", action="store_true", help="repack even if present")
    ap.add_argument("--check-only", action="store_true", help="only run the spot checks")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for split in args.splits.split(","):
        split = split.strip()
        if args.check_only:
            files = json.load(open(os.path.join(args.out, f"{split}_index.json")))
            spot_check(os.path.join(args.out, f"{split}_fields.npy"), files, n=10)
        else:
            pack_split(split, args.root, args.out, args.workers, args.force)


if __name__ == "__main__":
    main()
