#!/usr/bin/env python3
"""
clean_priors.py — find and remove corrupt advection-prior crops.

Unlike a quick index check, this *decompresses every member of every .npz*, so it
catches files whose compressed data is damaged (the `zlib.error: invalid code
lengths set` kind left by an interrupted run). Parallel across cores.

    conda activate nowcast
    python clean_priors.py            # report + remove
    python clean_priors.py --dry-run  # report only, delete nothing
"""
import os, glob, getpass, zipfile, argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

USER  = getpass.getuser()
PRIOR = f"/scratch/{USER}/dissertation/prior"


def is_bad(f):
    """Return f if the file is unreadable/corrupt, else None."""
    try:
        with zipfile.ZipFile(f) as z:
            if z.testzip() is not None:     # CRC-checks every member (forces decompress)
                return f
    except Exception:
        return f
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=min(64, (os.cpu_count() or 8)))
    args = ap.parse_args()

    files = glob.glob(os.path.join(PRIOR, "*", "*", "*.npz"))
    print(f"checking {len(files)} crops with {args.workers} workers ...")
    bad = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=mp.get_context("fork")) as ex:
        for r in ex.map(is_bad, files, chunksize=128):
            if r:
                bad.append(r)

    parts = glob.glob(os.path.join(PRIOR, "*", "*", "*.part"))
    print(f"\ncorrupt .npz : {len(bad)}")
    print(f"leftover .part: {len(parts)}")
    for f in bad[:20]:
        print("   bad:", f)
    if args.dry_run:
        print("\n--dry-run: nothing deleted.")
        return
    for f in bad + parts:
        try:
            os.remove(f)
        except OSError:
            pass
    print(f"\nremoved {len(bad)} corrupt + {len(parts)} .part file(s).")
    print(f"remaining crops: {len(files) - len(bad)}")


if __name__ == "__main__":
    main()
