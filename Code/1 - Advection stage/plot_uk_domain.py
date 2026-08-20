#!/usr/bin/env python3
"""
plot_uk_domain.py - project the ODIM radar composite and the crop tiling onto a
map of the UK.

WHY THIS IS RECOVERABLE. The npz crop cache stores no geolocation: build_advection_prior.py
writes x_mmh, y_mmh, A_mmh, A_dbr, r_dbr, valid, split, target and lead_min, and
nothing about where on the grid the crop came from. But the filename does carry
it. Every crop is named

    {YYYYMMDDHHMM}_r{ROW:04d}_c{COL:04d}_L{LEAD:02d}.npz

where ROW and COL are the top-left corner of the 384x384 CONTEXT window in the
full composite, tiled by candidate_windows() at STRIDE = 256. The scored 256x256
crop sits inside that context with a 64-pixel margin, so it occupies

    rows  ROW+64 : ROW+320
    cols  COL+64 : COL+320

of the composite array. The composite itself is a projected grid whose geometry
lives in the ODIM file's root /where attributes (projdef, xsize, ysize, xscale,
yscale and the four corner latitudes and longitudes), so one cached frame plus
the crop filenames is enough to place every crop on a map exactly.

Run this on the server, where the frame cache and h5py already are. It needs
numpy, h5py and matplotlib. Coastlines are drawn only if cartopy and pyproj are
importable; without them the figure is still correct, just in projection
coordinates with the corner latitudes and longitudes annotated. Do not install
cartopy in a hurry to get this figure: it adds nothing the argument depends on.

    conda activate nowcast
    export DISS_SCRATCH=/work/scratch-nopw2/$USER/dissertation

    # pick any cached frame; --frame auto takes the first one it finds
    python plot_uk_domain.py --frames-dir $DISS_SCRATCH/frames \
        --prior-dir $DISS_SCRATCH/prior_ml --out ~/dissertation_outputs/figures

    # a specific timestamp, for a case-study panel
    python plot_uk_domain.py --frames-dir $DISS_SCRATCH/frames \
        --prior-dir $DISS_SCRATCH/prior_ml --frame 202506151200 \
        --out ~/dissertation_outputs/figures

Outputs into --out:
    uk_domain.png        the composite for one time, the full candidate tiling,
                         and the tiles that actually passed the quality filter
    uk_crop_density.png  how many accepted crops each tile contributed, which is
                         where the training data actually comes from
    uk_domain.json       the recovered grid geometry and per-tile counts
"""

import argparse
import collections
import glob
import json
import os
import re
import sys

import numpy as np
import h5py

CROP = 256
MARGIN = 64
CONTEXT = CROP + 2 * MARGIN          # 384, must match build_advection_prior.py
STRIDE = 256
CAP_MMH = 200.0
NAME_RE = re.compile(r"^(\d{12})_r(\d{4})_c(\d{4})(?:_L(\d{2}))?\.npz$")


def read_odim(path):
    """Rain rate in mm/h plus the grid geometry from the root /where group.
    The decode matches build_advection_prior.read_odim_rainrate exactly:
    nodata becomes NaN (out of radar range), undetect becomes 0 (dry)."""
    with h5py.File(path, "r") as f:
        node = f["dataset1"]["data1"]
        raw = node["data"][...].astype("float32")
        a = node["what"].attrs
        gain = float(a.get("gain", 1.0))
        offset = float(a.get("offset", 0.0))
        nodata = float(a.get("nodata", -1.0))
        undetect = float(a.get("undetect", 0.0))
        where = dict(f["where"].attrs) if "where" in f else {}
    R = raw * gain + offset
    R[raw == nodata] = np.nan
    R[raw == undetect] = 0.0
    R = np.clip(R, 0.0, CAP_MMH)

    def get(k, default=None):
        v = where.get(k, default)
        if isinstance(v, bytes):
            return v.decode()
        return v

    geo = {"projdef": get("projdef"),
           "xsize": int(get("xsize", R.shape[1])),
           "ysize": int(get("ysize", R.shape[0])),
           "xscale": float(get("xscale", 1000.0)),
           "yscale": float(get("yscale", 1000.0))}
    for c in ("LL", "LR", "UL", "UR"):
        for ax in ("lat", "lon"):
            v = get(f"{c}_{ax}")
            if v is not None:
                geo[f"{c}_{ax}"] = float(v)
    return R, geo


def find_frame(frames_dir, stamp=None):
    if stamp:
        hits = glob.glob(os.path.join(frames_dir, "*", "*", "*", f"{stamp}.h5"))
        if not hits:
            raise SystemExit(f"ERROR: no cached frame {stamp}.h5 under {frames_dir}")
        return hits[0]
    for root, _dirs, files in os.walk(frames_dir):
        for fn in sorted(files):
            if fn.endswith(".h5"):
                return os.path.join(root, fn)
    raise SystemExit(f"ERROR: no .h5 frames found under {frames_dir}")


def scan_crops(prior_dir, max_files=None):
    """Count accepted crops per (row, col) tile and per split. Only filenames
    are read, never the arrays, so this is fast even on 667k crops."""
    per_tile = collections.Counter()        # unique (time, tile), lead-deduplicated
    per_split = collections.Counter()
    tile_split = collections.defaultdict(collections.Counter)
    # A crop is only cached if it is >=90% in radar range AND >=5% wet, so the
    # number of accepted crops at a timestamp is a free proxy for how widespread
    # the rain was. Ranking timestamps by it finds candidate case studies without
    # opening a single HDF5 file.
    per_time = collections.Counter()
    time_split = {}
    seen = set()                             # (stamp, r, c) already counted
    n, n_files, bad = 0, 0, 0
    for split in sorted(os.listdir(prior_dir)):
        sp = os.path.join(prior_dir, split)
        if not os.path.isdir(sp):
            continue
        for day in sorted(os.listdir(sp)):
            dp = os.path.join(sp, day)
            if not os.path.isdir(dp):
                continue
            with os.scandir(dp) as it:
                for e in it:
                    m = NAME_RE.match(e.name)
                    if not m:
                        bad += 1
                        continue
                    r, c = int(m.group(2)), int(m.group(3))
                    n_files += 1
                    # prior_ml stores one npz per (crop, lead), so the same
                    # geographic crop appears four times. Counting files would
                    # report four times the real number of crops per tile, which
                    # would be wrong on a figure captioned "crops per tile".
                    key = (m.group(1), r, c)
                    if key in seen:
                        continue
                    seen.add(key)
                    per_time[m.group(1)] += 1
                    time_split[m.group(1)] = split
                    per_tile[(r, c)] += 1
                    per_split[split] += 1
                    tile_split[(r, c)][split] += 1
                    n += 1
                    if max_files and n_files >= max_files:
                        return (per_tile, per_split, tile_split, per_time, time_split,
                                n, n_files, bad)
    return (per_tile, per_split, tile_split, per_time, time_split,
            n, n_files, bad)


def try_cartopy(geo):
    """Return (crs, feature_module) if a projection can be built, else (None, None)."""
    projdef = geo.get("projdef")
    if not projdef:
        return None, None
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import pyproj                                  # noqa: F401
        crs = ccrs.Projection(projdef)
        return crs, cfeature
    except Exception as e:
        print(f"note: coastlines unavailable ({type(e).__name__}); plotting in "
              "projection coordinates instead. This is cosmetic.")
        return None, None


def main():
    ap = argparse.ArgumentParser(
        description="Project the ODIM composite and the crop tiling onto a UK map.")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--prior-dir", required=True,
                    help="prior_ml (multi-lead) or prior (the +60 cache)")
    ap.add_argument("--frame", default=None, help="YYYYMMDDHHMM; default: any")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--max-files", type=int, default=None,
                    help="cap the filename scan; the cap is reported, never silent")
    ap.add_argument("--list-events", type=int, default=0, metavar="N",
                    help="rank timestamps by how many crops they contributed, "
                         "read the frames of the top N, and print their rain "
                         "statistics so a case study can be chosen. Prints and "
                         "exits; draws nothing.")
    ap.add_argument("--events-split", default=None,
                    help="restrict --list-events to one split, e.g. test")
    ap.add_argument("--vmax", type=float, default=8.0, help="colour cap, mm/h")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    frame = find_frame(args.frames_dir, args.frame)
    R, geo = read_odim(frame)
    H, W = R.shape
    print(f"frame: {frame}")
    print(f"  grid {W} x {H} at {geo['xscale']:.0f} x {geo['yscale']:.0f} m")
    print(f"  projdef: {geo.get('projdef')}")
    corners = {k: v for k, v in geo.items() if k[:2] in ("LL", "LR", "UL", "UR")}
    if corners:
        print(f"  corners: {corners}")
    print(f"  in-range pixels: {100 * np.isfinite(R).mean():.1f}% | "
          f"wet (>=0.1 mm/h): {100 * np.nanmean(R >= 0.1):.1f}%")

    tiles_all = [(r, c)
                 for r in range(0, H - CONTEXT + 1, STRIDE)
                 for c in range(0, W - CONTEXT + 1, STRIDE)]
    print(f"  candidate tiling: {len(tiles_all)} windows of {CONTEXT}px "
          f"at stride {STRIDE}")

    (per_tile, per_split, tile_split, per_time, time_split,
     n, n_files, bad) = scan_crops(args.prior_dir, args.max_files)
    if args.max_files and n_files >= args.max_files:
        print(f"  NOTE: filename scan capped at {args.max_files} files; counts "
              "below are a partial sample, not the whole cache.")
    print(f"  files scanned: {n_files} -> {n} unique crops (lead-deduplicated) "
          f"across {len(per_tile)} tiles ({bad} unparsable names)")
    print(f"  by split: {dict(per_split)}")

    if args.list_events:
        cands = [(t, c) for t, c in per_time.items()
                 if args.events_split is None or time_split.get(t) == args.events_split]
        cands.sort(key=lambda tc: -tc[1])
        cands = cands[:args.list_events]
        print(f"\nTop {len(cands)} timestamps by accepted crops"
              + (f" in split '{args.events_split}'" if args.events_split else "")
              + ". Higher crop counts mean more widespread rain; the rain\n"
              "columns separate widespread drizzle from intense convection.\n")
        print(f"  {'timestamp':<14}{'split':>7}{'crops':>7}{'wet%':>8}"
              f"{'mean':>8}{'p99.9':>9}{'max':>8}  frame")
        rows = []
        for t, cnt in cands:
            try:
                fp = find_frame(args.frames_dir, t)
                Rt, _g = read_odim(fp)
            except SystemExit:
                print(f"  {t:<14}{time_split.get(t,''):>7}{cnt:>7}"
                      f"{'':>8}{'':>8}{'':>9}{'':>8}  (frame not cached)")
                continue
            fin = np.isfinite(Rt)
            wet = float(np.mean(Rt[fin] >= 0.1)) if fin.any() else float("nan")
            mean = float(np.mean(Rt[fin])) if fin.any() else float("nan")
            p999 = float(np.percentile(Rt[fin], 99.9)) if fin.any() else float("nan")
            mx = float(np.nanmax(Rt)) if fin.any() else float("nan")
            rows.append((t, cnt, wet, mean, p999, mx))
            print(f"  {t:<14}{time_split.get(t,''):>7}{cnt:>7}{100*wet:>7.1f}%"
                  f"{mean:>8.3f}{p999:>9.2f}{mx:>8.1f}  {os.path.basename(fp)}")
        os.makedirs(args.out, exist_ok=True)
        csvp = os.path.join(args.out, "uk_events.csv")
        with open(csvp, "w") as fh:
            fh.write("timestamp,split,n_crops,wet_fraction,mean_mmh,"
                     "p99_9_mmh,max_mmh\n")
            for t, cnt, wet, mean, p999, mx in rows:
                fh.write(f"{t},{time_split.get(t, '')},{cnt},{wet:.6f},"
                         f"{mean:.4f},{p999:.3f},{mx:.2f}\n")
        print(f"\nwrote {csvp} ({len(rows)} rows)")

        print("\nPick one and pass it as --frame. For a talk, prefer a high p99.9 "
              "(intense, convective) over a merely high wet% (widespread drizzle),")
        print("and prefer split 'test' so the case study sits on held-out 2026 data.")
        return

    # Extent in projection metres, with the array's origin at the top-left.
    ex = W * geo["xscale"]
    ey = H * geo["yscale"]
    extent = (0.0, ex, 0.0, ey)

    def rect_xy(r, c, size):
        """Array (row, col) to plot (x, y) of the lower-left corner. Row 0 is
        the northern edge in ODIM, so y is measured from the bottom."""
        x = c * geo["xscale"]
        y = (H - r - size) * geo["yscale"]
        return x, y, size * geo["xscale"], size * geo["yscale"]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.colors import LogNorm
    except Exception as e:
        print(f"ERROR: matplotlib unavailable ({e}); geometry printed above, "
              "no figures written.")
        return

    crs, cfeature = try_cartopy(geo)

    # ---- Figure 1: the domain, the tiling, and which tiles were used --------
    subplot_kw = {"projection": crs} if crs is not None else {}
    fig, ax = plt.subplots(figsize=(9, 10), subplot_kw=subplot_kw)
    shown = np.where(np.isfinite(R), R, np.nan)
    im = ax.imshow(shown, origin="upper", extent=extent, cmap="viridis",
                   vmin=0.0, vmax=args.vmax, interpolation="nearest",
                   **({"transform": crs} if crs is not None else {}))
    # Out-of-range radar as a light grey wash, so coverage is visible.
    ax.imshow(np.where(np.isfinite(R), np.nan, 1.0), origin="upper", extent=extent,
              cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest",
              **({"transform": crs} if crs is not None else {}))
    if crs is not None and cfeature is not None:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="white")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="white")

    used = set(per_tile)
    for (r, c) in tiles_all:
        x, y, w, h = rect_xy(r + MARGIN, c + MARGIN, CROP)
        on = (r, c) in used
        ax.add_patch(Rectangle((x, y), w, h, fill=False,
                               edgecolor=("red" if on else "0.6"),
                               lw=(1.4 if on else 0.5),
                               ls=("-" if on else ":"),
                               **({"transform": crs} if crs is not None else {})))
    ax.set_title(f"UK radar composite and the {CROP} km crop tiling\n"
                 f"{os.path.basename(frame)[:12]}  |  red = tiles that passed the "
                 f"quality filter ({len(used)} of {len(tiles_all)})", fontsize=11)
    if crs is None:
        ax.set_xlabel("projection easting (m)")
        ax.set_ylabel("projection northing (m)")
        if corners:
            ax.text(0.01, 0.01,
                    "corners: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(corners.items())),
                    transform=ax.transAxes, fontsize=6, color="0.3", va="bottom")
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("rain rate (mm/h)")
    fig.tight_layout()
    p1 = os.path.join(args.out, "uk_domain.png")
    fig.savefig(p1, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p1}")

    # ---- Figure 2: where the training data actually comes from -------------
    fig, ax = plt.subplots(figsize=(9, 10), subplot_kw=subplot_kw)
    ax.imshow(np.where(np.isfinite(R), 0.0, np.nan), origin="upper", extent=extent,
              cmap="Greys", vmin=0, vmax=1, interpolation="nearest",
              **({"transform": crs} if crs is not None else {}))
    if crs is not None and cfeature is not None:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="0.3")
    vmax = max(per_tile.values()) if per_tile else 1
    cmap = plt.get_cmap("magma")
    for (r, c), cnt in sorted(per_tile.items()):
        x, y, w, h = rect_xy(r + MARGIN, c + MARGIN, CROP)
        ax.add_patch(Rectangle((x, y), w, h, facecolor=cmap(cnt / vmax),
                               edgecolor="white", lw=0.6, alpha=0.85,
                               **({"transform": crs} if crs is not None else {})))
        ax.text(x + w / 2, y + h / 2, f"{cnt:,}", ha="center", va="center",
                fontsize=7, color="white",
                **({"transform": crs} if crs is not None else {}))
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=vmax))
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("accepted crops per tile")
    ax.set_title(f"Where the training data comes from\n{n:,} accepted crops "
                 f"across {len(per_tile)} tiles", fontsize=11)
    if crs is None:
        ax.set_xlabel("projection easting (m)")
        ax.set_ylabel("projection northing (m)")
    fig.tight_layout()
    p2 = os.path.join(args.out, "uk_crop_density.png")
    fig.savefig(p2, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p2}")

    meta = {"frame": frame, "grid": {"H": H, "W": W}, "geo": geo,
            "crop": CROP, "margin": MARGIN, "context": CONTEXT, "stride": STRIDE,
            "n_candidate_tiles": len(tiles_all), "n_used_tiles": len(used),
            "n_crops_unique": n, "n_files_scanned": n_files, "scan_capped": bool(args.max_files and n_files >= args.max_files),
            "by_split": dict(per_split),
            "per_tile": {f"r{r:04d}_c{c:04d}": cnt for (r, c), cnt in sorted(per_tile.items())},
            "per_tile_by_split": {f"r{r:04d}_c{c:04d}": dict(v)
                                  for (r, c), v in sorted(tile_split.items())}}
    p3 = os.path.join(args.out, "uk_domain.json")
    tmp = p3 + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    os.replace(tmp, p3)
    print(f"wrote {p3}")


if __name__ == "__main__":
    main()
