#!/usr/bin/env python3
"""
build_advection_prior.py
========================
Server-side pipeline for the MSc dissertation "Nowcasting using physics-informed
diffusion models". It builds the prescribed semi-Lagrangian **advection prior** A
and the **residual target** r = y - A for UK Met Office radar, ready for the
latent-diffusion stage.

Designed for the University of Exeter undergraduate compute servers
(mcrugcomp01/02/03 - 64-core EPYC, 384 GB RAM, 13 TB /scratch):
  * streams ODIM HDF5 radar from the public AWS S3 bucket (boto3, anonymous),
  * caches everything on fast local /scratch (NOT the slow 30 GB home dir),
  * parallelises across all CPU cores,
  * is idempotent / resumable (safe to re-run after an SSH disconnect).

Two phases:
  1. DOWNLOAD  (threaded)  : pull every needed .h5 once -> /scratch/<user>/.../frames
  2. PROCESS   (processes) : crop -> Lucas-Kanade flow -> semi-Lagrangian advect
                             -> residual -> /scratch/<user>/.../prior/<split>/*.npz

Typical headless run (survives disconnect):
    tmux new -s prior
    conda activate nowcast
    python build_advection_prior.py --start 2024-11-21 --end 2025-04-30
    #   detach: Ctrl-b then d      reattach: tmux attach -t prior

Connectivity self-test only (no data pulled):
    python build_advection_prior.py --check

See README_server.md for the full setup. The notebook documents every design
choice; this script is the same logic wrapped for headless, parallel runs.
"""
import argparse
import datetime as dt
import getpass
import io
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
import h5py
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError

# ----------------------------------------------------------------------------
# Fixed methodology constants (documented in Dissertation.ipynb, Part 2)
# ----------------------------------------------------------------------------
BUCKET   = "met-office-radar-obs-data"
REGION   = "eu-west-2"
FSUFFIX  = "_ODIM_ng_radar_rainrate_composite_1km_UK.h5"

CADENCE_MIN    = 15
N_INPUT        = 4
LEAD_MIN       = 60
CROP           = 256
MARGIN         = 64
CONTEXT        = CROP + 2 * MARGIN          # 384
STRIDE         = 256
N_LEADTIMES    = LEAD_MIN // CADENCE_MIN     # 4 steps to +60 min
CAP_MMH        = 128.0
RAIN_THRESH    = 0.1
MIN_VALID_FRAC = 0.90
MIN_RAIN_FRAC  = 0.05
DBR_THRESH     = 0.1
DBR_ZERO       = -15.0
DBR_INV_THR    = -10.0
TEST_YEAR      = 2026
VAL_DAYS       = 2                            # first 2 days of each month -> val

# module globals set in main() and inherited by forked workers
FRAMES_DIR = None
PRIOR_DIR  = None

# ----------------------------------------------------------------------------
# S3 + ODIM reading
# ----------------------------------------------------------------------------
def make_s3():
    return boto3.client("s3", region_name=REGION,
                        config=Config(signature_version=UNSIGNED,
                                      retries={"max_attempts": 5, "mode": "standard"}))


def radar_key(t):
    return f"radar/{t:%Y/%m/%d}/{t:%Y%m%d%H%M}{FSUFFIX}"


def read_odim_rainrate(source):
    """Decode ODIM HDF5 (path or bytes) -> 2D mm/h array, masked.
    nodata -> NaN (unknown), undetect -> 0 (dry), value -> value*gain+offset."""
    fh = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    with h5py.File(fh, "r") as f:
        node = f["dataset1"]["data1"]
        raw  = node["data"][...].astype("float32")
        a    = node["what"].attrs
        gain     = float(a.get("gain", 1.0))
        offset   = float(a.get("offset", 0.0))
        nodata   = float(a.get("nodata", -1.0))
        undetect = float(a.get("undetect", 0.0))
    R = raw * gain + offset
    R[raw == nodata]   = np.nan
    R[raw == undetect] = 0.0
    return np.clip(R, 0.0, CAP_MMH)           # np.clip leaves NaN as NaN


def frame_path(t):
    return os.path.join(FRAMES_DIR, f"{t:%Y/%m/%d}", f"{t:%Y%m%d%H%M}.h5")


def download_frame(t, s3):
    """Download one frame to the local frame cache if missing. Returns
    'ok' | 'skip' | 'missing' (not on S3) | 'error:<msg>'."""
    dst = frame_path(t)
    if os.path.exists(dst):
        return "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=radar_key(t))
        data = obj["Body"].read()
    except ClientError:
        return "missing"
    except Exception as e:                     # transient network etc.
        return f"error:{e}"
    tmp = dst + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, dst)                        # atomic
    return "ok"


def read_frame_local(t):
    p = frame_path(t)
    if not os.path.exists(p):
        return None
    try:
        return read_odim_rainrate(p)
    except Exception:
        return None


def list_day_keys(date, s3):
    prefix = f"radar/{date:%Y/%m/%d}/"
    present = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            stamp = o["Key"].rsplit("/", 1)[-1][:12]
            try:
                present.add(dt.datetime.strptime(stamp, "%Y%m%d%H%M"))
            except ValueError:
                pass
    return present


# ----------------------------------------------------------------------------
# Sampling + split + cropping
# ----------------------------------------------------------------------------
def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def build_availability(start, end, workers):
    s3 = make_s3()                              # boto3 clients are thread-safe -> share one
    days = list(daterange(start, end))
    avail = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(list_day_keys, d, s3): d for d in days}
        for f in as_completed(futs):
            avail |= f.result()
    return avail


def make_temporal_samples(avail):
    step = dt.timedelta(minutes=CADENCE_MIN)
    lead = dt.timedelta(minutes=LEAD_MIN)
    out = []
    for t0 in sorted(avail):
        inputs = [t0 - k * step for k in range(N_INPUT - 1, -1, -1)]
        target = t0 + lead
        if all(ti in avail for ti in inputs) and target in avail:
            out.append({"input_times": inputs, "target_time": target})
    return out


def assign_split(date, first_days):
    """test = test year; val = first VAL_DAYS available days of the month."""
    if date.year == TEST_YEAR:
        return "test"
    if date.day <= VAL_DAYS:
        return "val"
    if first_days and date in first_days:      # months whose 1st/2nd are absent
        return "val"
    return "train"


def split_samples(temporal, avail):
    month_days = {}
    for t in avail:
        month_days.setdefault((t.year, t.month), set()).add(t.date())
    first_by_month = {k: set(sorted(v)[:VAL_DAYS]) for k, v in month_days.items()}
    for s in temporal:
        d = s["target_time"].date()
        s["split"] = assign_split(d, first_by_month.get((d.year, d.month)))
    return temporal


def candidate_windows(shape):
    H, W = shape
    return [(r, c)
            for r in range(0, H - CONTEXT + 1, STRIDE)
            for c in range(0, W - CONTEXT + 1, STRIDE)]


def center_crop(a, size=CROP):
    H, W = a.shape[-2:]
    r = (H - size) // 2
    c = (W - size) // 2
    return a[..., r:r + size, c:c + size]


def crop_passes(target_ctx):
    center = center_crop(target_ctx, CROP)
    valid  = np.isfinite(center)
    if valid.mean() < MIN_VALID_FRAC:
        return False
    wet = np.nan_to_num(center) >= RAIN_THRESH
    return (wet[valid].mean() if valid.any() else 0.0) >= MIN_RAIN_FRAC


# ----------------------------------------------------------------------------
# Advection prior (pysteps) - imported lazily so --check needs no pysteps
# ----------------------------------------------------------------------------
_OFLOW = _EXTRAP = _TF = None

def _init_pysteps():
    global _OFLOW, _EXTRAP, _TF
    if _OFLOW is None:
        from pysteps import motion, nowcasts
        from pysteps.utils import transformation
        _OFLOW  = motion.get_method("LK")
        _EXTRAP = nowcasts.get_method("extrapolation")
        _TF     = transformation


def to_dbr(R):
    _init_pysteps()
    return _TF.dB_transform(R, threshold=DBR_THRESH, zerovalue=DBR_ZERO)[0]


def from_dbr(R):
    _init_pysteps()
    return _TF.dB_transform(R, threshold=DBR_INV_THR, inverse=True)[0]


def advection_prior(x_ctx_mmh):
    """Semi-Lagrangian advection of the input stack to +LEAD_MIN.
    Returns A_dbr (256), A_mmh (256), V (2,256,256)."""
    _init_pysteps()
    Rd = np.stack([to_dbr(f) for f in x_ctx_mmh], axis=0)
    Rd[~np.isfinite(Rd)] = DBR_ZERO
    try:
        V = _OFLOW(Rd)
        if not np.isfinite(V).all():
            V = np.nan_to_num(V)
    except Exception:
        V = np.zeros((2,) + Rd.shape[-2:], dtype="float32")
    fc = _EXTRAP(Rd[-1], V, N_LEADTIMES)
    A_dbr_ctx = np.where(np.isfinite(fc[-1]), fc[-1], DBR_ZERO)
    A_dbr = center_crop(A_dbr_ctx)
    return A_dbr, from_dbr(A_dbr), center_crop(V)


def prior_path(target_time, window, split):
    sub = os.path.join(PRIOR_DIR, split, f"{target_time:%Y%m%d}")
    return os.path.join(sub, f"{target_time:%Y%m%d%H%M}_r{window[0]:04d}_c{window[1]:04d}.npz")


def process_sample(sample):
    """Worker: read local frames -> crop -> advect -> save (A, r). Returns count."""
    inputs, target, split = sample["input_times"], sample["target_time"], sample["split"]
    frames = [read_frame_local(t) for t in inputs] + [read_frame_local(target)]
    if any(f is None for f in frames):
        return 0
    x_full = np.stack(frames[:-1], axis=0)
    y_full = frames[-1]
    n = 0
    for (r, c) in candidate_windows(x_full.shape[-2:]):
        out = prior_path(target, (r, c), split)
        if os.path.exists(out):
            n += 1
            continue
        sl = (slice(r, r + CONTEXT), slice(c, c + CONTEXT))
        y_ctx = y_full[sl]
        if not crop_passes(y_ctx):
            continue
        x_ctx = x_full[:, sl[0], sl[1]]
        A_dbr, A_mmh, V = advection_prior(x_ctx)
        y_mmh = center_crop(y_ctx)
        valid = np.isfinite(y_mmh)
        y_dbr = np.where(valid, to_dbr(y_mmh), DBR_ZERO)
        r_dbr = y_dbr - A_dbr
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".part"                      # write then atomically rename:
        with open(tmp, "wb") as fh:              # a half-written crop is never
            np.savez_compressed(                 # seen as a finished .npz, so
                fh,                              # stop/resume is always clean
                x_mmh=center_crop(x_ctx).astype("float16"),
                A_dbr=A_dbr.astype("float16"), A_mmh=A_mmh.astype("float16"),
                y_mmh=y_mmh.astype("float16"), r_dbr=r_dbr.astype("float16"),
                valid=valid, split=split, target=target.isoformat(),
            )
        os.replace(tmp, out)
        n += 1
    return n


# ----------------------------------------------------------------------------
# Advection-only baseline (CSI / MAE)
# ----------------------------------------------------------------------------
THRESHOLDS = [0.5, 1.0, 2.0, 4.0, 8.0]

def _score_file(path):
    try:
        z = np.load(path, allow_pickle=True)
        A = z["A_mmh"].astype("float32"); y = z["y_mmh"].astype("float32")
        valid = z["valid"] & np.isfinite(A)
        cont = {}
        for t in THRESHOLDS:
            o, p = (y >= t), (A >= t)
            cont[t] = (int(np.sum(valid & o & p)),
                       int(np.sum(valid & o & ~p)),
                       int(np.sum(valid & ~o & p)))
        mae = float(np.sum(np.abs(np.nan_to_num(A) - np.nan_to_num(y))[valid]))
        return cont, mae, int(valid.sum())
    except Exception:
        return None        # corrupt/unreadable crop -> skip, never crash the baseline


def evaluate(files, workers):
    tot = {t: np.zeros(3) for t in THRESHOLDS}
    mae_sum, npix, skipped = 0.0, 0, 0
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=mp.get_context("fork")) as ex:
        for res in ex.map(_score_file, files, chunksize=64):
            if res is None:
                skipped += 1
                continue
            cont, mae, n = res
            for t in THRESHOLDS:
                tot[t] += cont[t]
            mae_sum += mae; npix += n
    out = {"n_crops": len(files), "n_skipped": skipped,
           "MAE_mmh": mae_sum / max(npix, 1), "CSI": {}}
    for t in THRESHOLDS:
        H, M, F = tot[t]
        out["CSI"][t] = (H / (H + M + F)) if (H + M + F) > 0 else float("nan")
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def connectivity_check():
    print("S3 connectivity self-test ...", flush=True)
    try:
        s3 = make_s3()
        keys = list_day_keys(dt.date(2024, 11, 21), s3)
        print(f"  OK - reached s3://{BUCKET}, found {len(keys)} frames on 2024-11-21.")
        return True
    except (EndpointConnectionError, Exception) as e:
        print(f"  FAILED to reach AWS S3: {e}")
        print("  -> The server likely has no outbound internet. Fallback:")
        print("     download frames on an internet-connected machine and rsync")
        print("     them into the --frames-dir, then re-run with --skip-download.")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    user = getpass.getuser()
    default_root = f"/scratch/{user}/dissertation"
    ap.add_argument("--start", default="2024-11-21", help="subset start (YYYY-MM-DD)")
    ap.add_argument("--end",   default="2025-04-30", help="subset end (YYYY-MM-DD)")
    ap.add_argument("--root",  default=default_root, help="scratch root for caches")
    ap.add_argument("--frames-dir", default=None, help="override frame cache dir")
    ap.add_argument("--prior-dir",  default=None, help="override prior cache dir")
    ap.add_argument("--out", default=os.path.expanduser("~/dissertation_outputs"),
                    help="persistent (home) dir for the manifest + baseline JSON")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    ap.add_argument("--io-workers", type=int, default=32, help="threads for download")
    ap.add_argument("--check", action="store_true", help="connectivity test then exit")
    ap.add_argument("--skip-download", action="store_true",
                    help="assume frames already in --frames-dir (offline servers)")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--baseline-only", action="store_true",
                    help="skip download+process; just (re)score the existing cache "
                         "and (re)write the manifest. No internet needed.")
    args = ap.parse_args()

    global FRAMES_DIR, PRIOR_DIR
    FRAMES_DIR = args.frames_dir or os.path.join(args.root, "frames")
    PRIOR_DIR  = args.prior_dir  or os.path.join(args.root, "prior")
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(PRIOR_DIR, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    if args.check:
        sys.exit(0 if connectivity_check() else 1)

    if args.baseline_only:
        from collections import Counter
        import glob as _glob
        files = sorted(_glob.glob(os.path.join(PRIOR_DIR, "*", "*", "*.npz")))
        if not files:
            print("no crops in", PRIOR_DIR); sys.exit(1)
        by_split = dict(Counter(f.split(os.sep)[-3] for f in files))
        print(f"[baseline-only] scoring {len(files)} crops  by split: {by_split}")
        base = evaluate(files, args.workers)
        manifest = {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "start": args.start, "end": args.end,
            "n_crops": len(files), "by_split": by_split,
            "config": {"crop": CROP, "margin": MARGIN, "n_input": N_INPUT,
                       "lead_min": LEAD_MIN, "val_days": VAL_DAYS},
            "prior_dir": PRIOR_DIR, "baseline_advection_only": base,
        }
        print(f"  scored {len(files) - base['n_skipped']} crops "
              f"({base['n_skipped']} skipped as unreadable)")
        print(f"  MAE = {base['MAE_mmh']:.4f} mm/h")
        for t in THRESHOLDS:
            print(f"  CSI @ {t:>4} mm/h = {base['CSI'][t]:.3f}")
        for d in (args.out, PRIOR_DIR):
            with open(os.path.join(d, "manifest.json"), "w") as fh:
                json.dump(manifest, fh, indent=2)
        print(f"manifest -> {os.path.join(args.out, 'manifest.json')}")
        return

    if not args.skip_download and not connectivity_check():
        sys.exit(1)

    start = dt.date.fromisoformat(args.start)
    end   = dt.date.fromisoformat(args.end)
    t_run = time.time()

    # --- sampling ---------------------------------------------------------
    print(f"\n[1/4] Listing S3 availability {start}..{end} ...", flush=True)
    avail = build_availability(start, end, args.io_workers)
    temporal = split_samples(make_temporal_samples(avail), avail)
    from collections import Counter
    by_split = Counter(s["split"] for s in temporal)
    print(f"      {len(avail)} frames -> {len(temporal)} samples  by split: {dict(by_split)}")

    # --- download frames once --------------------------------------------
    if not args.skip_download:
        need = set()
        for s in temporal:
            need.update(s["input_times"]); need.add(s["target_time"])
        print(f"\n[2/4] Downloading {len(need)} unique frames "
              f"({args.io_workers} threads) ...", flush=True)
        stats = Counter()
        s3dl = make_s3()                        # one shared thread-safe client
        with ThreadPoolExecutor(max_workers=args.io_workers) as ex:
            futs = [ex.submit(download_frame, t, s3dl) for t in sorted(need)]
            for i, f in enumerate(as_completed(futs), 1):
                stats[f.result().split(":")[0]] += 1
                if i % 1000 == 0:
                    print(f"      {i}/{len(need)}  {dict(stats)}", flush=True)
        print(f"      done: {dict(stats)}")
    else:
        print("\n[2/4] --skip-download set: using existing frame cache.")

    # --- process (parallel advection) ------------------------------------
    print(f"\n[3/4] Building advection prior + residual "
          f"({args.workers} processes) ...", flush=True)
    total = 0
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        for i, n in enumerate(ex.map(process_sample, temporal, chunksize=8), 1):
            total += n
            if i % 500 == 0:
                print(f"      {i}/{len(temporal)} samples, {total} crops cached", flush=True)
    print(f"      cached {total} crops to {PRIOR_DIR}")

    # --- baseline + manifest ---------------------------------------------
    import glob
    files = sorted(glob.glob(os.path.join(PRIOR_DIR, "*", "*", "*.npz")))
    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "start": args.start, "end": args.end,
        "n_crops": len(files), "by_split": dict(by_split),
        "config": {"crop": CROP, "margin": MARGIN, "n_input": N_INPUT,
                   "lead_min": LEAD_MIN, "val_days": VAL_DAYS},
        "prior_dir": PRIOR_DIR,
    }
    if not args.no_baseline and files:
        print(f"\n[4/4] Advection-only baseline over {len(files)} crops ...", flush=True)
        base = evaluate(files, args.workers)
        manifest["baseline_advection_only"] = base
        print(f"      MAE = {base['MAE_mmh']:.4f} mm/h")
        for t in THRESHOLDS:
            print(f"      CSI @ {t:>4} mm/h = {base['CSI'][t]:.3f}")

    for d in (args.out, PRIOR_DIR):
        with open(os.path.join(d, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
    print(f"\nDone in {(time.time()-t_run)/60:.1f} min. "
          f"Manifest -> {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
