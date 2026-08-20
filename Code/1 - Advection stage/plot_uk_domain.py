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


def true_origin(geo):
    """Projected coordinates of the grid's lower-left corner.

    The ODIM /where group gives corner LATITUDES and LONGITUDES, not projected
    coordinates, so an extent of (0, W*xscale, 0, H*yscale) is offsets from the
    array corner and NOT British National Grid eastings. Labelling it as easting
    would be a false claim on a figure. With pyproj the true origin is one
    forward transform; without it the axes are labelled as distances instead."""
    if "LL_lat" not in geo or "LL_lon" not in geo or not geo.get("projdef"):
        return None
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", geo["projdef"], always_xy=True)
        x0, y0 = tr.transform(geo["LL_lon"], geo["LL_lat"])
        return float(x0), float(y0)
    except Exception as e:
        print(f"note: pyproj unavailable ({type(e).__name__}); axes will show "
              "distance across the composite rather than projected coordinates.")
        return None


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
                    help="open N frames, evenly spaced through the record, score "
                         "their rain statistics and print convective and frontal "
                         "shortlists. Use 0 for every timestamp in the split. "
                         "Prints and exits; draws nothing.")
    ap.add_argument("--events-split", default=None,
                    help="restrict --list-events to one split, e.g. test")
    ap.add_argument("--style", default="pysteps", choices=["pysteps", "plain"],
                    help="pysteps draws coastlines, borders and the standard "
                         "discrete log intensity scale; plain is a raw array "
                         "plot and always works")
    ap.add_argument("--vmax", type=float, default=8.0,
                    help="colour cap for --style plain, mm/h (pysteps uses its "
                         "own log scale)")
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
        # Ranking by accepted-crop count does not work on this domain: only 12
        # of the 42 candidate tiles ever pass the filter, so the count saturates
        # at 12 for any timestamp with widespread rain and the ordering becomes
        # arbitrary. Worse, convective cells are LOCALISED, so a crop count
        # would rank them below frontal rain, which is the opposite of what a
        # case study wants. So the frames are actually opened and ranked on
        # their own rain statistics.
        stamps = sorted(t for t in per_time
                        if args.events_split is None
                        or time_split.get(t) == args.events_split)
        if not stamps:
            raise SystemExit(f"ERROR: no timestamps in split "
                             f"{args.events_split!r}.")
        want = args.list_events if args.list_events > 0 else len(stamps)
        if want >= len(stamps):
            pick = stamps
        else:
            # Evenly spaced through the record, so the sample spans the whole
            # period rather than clustering in whichever month sorts first.
            step = len(stamps) / float(want)
            pick = [stamps[min(len(stamps) - 1, int(i * step))] for i in range(want)]
            pick = sorted(set(pick))
        print(f"\n{len(stamps)} timestamps available"
              + (f" in split '{args.events_split}'" if args.events_split else "")
              + f"; opening {len(pick)} of them (evenly spaced).")
        print("Reading frames. Each is a full composite, so this is I/O bound; "
              "expect roughly 0.1 to 0.3 s each.\n")

        rows, misses = [], 0
        for i, t in enumerate(pick, 1):
            if i % 100 == 0 or i == len(pick):
                print(f"  {i}/{len(pick)} frames read ({misses} not cached)",
                      flush=True)
            try:
                fp = find_frame(args.frames_dir, t)
                Rt, _g = read_odim(fp)
            except SystemExit:
                misses += 1
                continue
            except Exception:
                misses += 1
                continue
            fin = np.isfinite(Rt)
            if not fin.any():
                continue
            v = Rt[fin]
            wet = float(np.mean(v >= 0.1))
            heavy = float(np.mean(v >= 8.0))
            mean = float(np.mean(v))
            p99 = float(np.percentile(v, 99.0))
            p999 = float(np.percentile(v, 99.9))
            mx = float(np.max(v))
            # Concentration of intensity: high when a lot of rain falls in a
            # small fraction of the domain (convection), low when moderate rain
            # is spread widely (frontal). Guarded against a zero wet fraction.
            conc = p999 / max(wet * 100.0, 1e-6)
            rows.append(dict(timestamp=t, split=time_split.get(t, ""),
                             n_crops=per_time[t], wet=wet, heavy=heavy,
                             mean=mean, p99=p99, p999=p999, mx=mx, conc=conc))

        if not rows:
            raise SystemExit("ERROR: no frames could be read. Check --frames-dir "
                             "and that the cache still holds these timestamps.")

        os.makedirs(args.out, exist_ok=True)
        csvp = os.path.join(args.out, "uk_events.csv")
        with open(csvp, "w") as fh:
            fh.write("timestamp,split,n_crops,wet_fraction,heavy_fraction,"
                     "mean_mmh,p99_mmh,p99_9_mmh,max_mmh,concentration\n")
            for r in sorted(rows, key=lambda r: r["timestamp"]):
                fh.write(f"{r['timestamp']},{r['split']},{r['n_crops']},"
                         f"{r['wet']:.6f},{r['heavy']:.8f},{r['mean']:.4f},"
                         f"{r['p99']:.3f},{r['p999']:.3f},{r['mx']:.2f},"
                         f"{r['conc']:.4f}\n")
        print(f"\nwrote {csvp} ({len(rows)} frames, {misses} not cached)")

        def table(title, sel, note):
            print(f"\n{title}")
            print(f"  {note}")
            print(f"  {'timestamp':<14}{'split':>6}{'wet%':>7}{'heavy%':>8}"
                  f"{'mean':>7}{'p99':>7}{'p99.9':>8}{'max':>7}{'conc':>7}")
            for r in sel:
                print(f"  {r['timestamp']:<14}{r['split']:>6}{100*r['wet']:>6.1f}%"
                      f"{100*r['heavy']:>7.3f}%{r['mean']:>7.3f}{r['p99']:>7.2f}"
                      f"{r['p999']:>8.2f}{r['mx']:>7.1f}{r['conc']:>7.2f}")

        n_show = max(5, min(15, len(rows) // 4))
        conv = sorted(rows, key=lambda r: -r["conc"])[:n_show]
        table("CONVECTIVE candidates (intensity concentrated in a small area)",
              "ranked by concentration = p99.9 / wet%. These are where advection "
              "fails and the model should show its value.", conv)
        # Frontal: widespread, and deliberately excluding anything that also has
        # a convective signature, so the two shortlists are genuinely different.
        conv_cut = sorted((r["conc"] for r in rows), reverse=True)
        conv_cut = conv_cut[max(0, len(conv_cut) // 2)] if conv_cut else 0.0
        front = sorted((r for r in rows if r["conc"] <= conv_cut),
                       key=lambda r: -r["wet"])[:n_show]
        table("FRONTAL / STRATIFORM candidates (widespread, low peak intensity)",
              "widest rain among the less concentrated half. Advection is already "
              "good here, which is the honest contrast case.", front)

        print("\nHow to choose. For the convective panel take a high p99.9 with a "
              "modest wet%, and check max is not a single bright pixel by "
              "comparing p99.9 against max.")
        print("For the frontal panel take a high wet% with a low p99.9. Prefer "
              "split 'test' for both, so the case study sits on held-out 2026 data.")
        print("Every scanned frame is in the CSV, so a different rule can be "
              "applied to it without re-reading anything.")
        return

    # Extent in projection metres, with the array's origin at the top-left.
    ex = W * geo["xscale"]
    ey = H * geo["yscale"]
    origin = true_origin(geo)
    if origin is not None:
        x0, y0 = origin
        extent = (x0, x0 + ex, y0, y0 + ey)
        axis_x, axis_y = "easting (m, British National Grid)", "northing (m, BNG)"
        print(f"  projected origin (lower-left): {x0:.0f}, {y0:.0f} m")
    else:
        x0 = y0 = 0.0
        extent = (0.0, ex, 0.0, ey)
        axis_x = "distance east across the composite (m, not BNG easting)"
        axis_y = "distance north across the composite (m, not BNG northing)"

    def rect_xy(r, c, size):
        """Array (row, col) to plot (x, y) of the lower-left corner. Row 0 is
        the northern edge in ODIM, so y is measured from the bottom."""
        x = x0 + c * geo["xscale"]
        y = y0 + (H - r - size) * geo["yscale"]
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

    # ---- basemap helper ----------------------------------------------------
    # pysteps already knows how to draw a UK radar composite properly: coastlines,
    # borders, land and ocean shading, and the standard discrete log-spaced
    # intensity colour scale that every nowcasting paper uses. It is a hard
    # dependency of this project, so there is no reason to hand-roll a raw
    # imshow. It needs the grid corners in PROJECTION coordinates, which is why
    # true_origin() exists; without pyproj those corners are unknown and the
    # plain fallback is used instead.
    def basemap(title, field=None, vmax=None):
        """Return (fig, ax, in_projection_coords). Draws the composite if a
        field is given, otherwise an empty basemap to place patches on."""
        if args.style != "plain" and origin is not None:
            try:
                from pysteps.visualization import plot_precip_field
                geodata = {"projection": geo["projdef"],
                           "x1": extent[0], "x2": extent[1],
                           "y1": extent[2], "y2": extent[3],
                           "yorigin": "upper"}
                # Do NOT pre-create a figure. plot_precip_field makes its own
                # GeoAxes, and a figure created first leaves a stray normalised
                # 0-1 axes frame showing through behind the map. Let pysteps
                # build it, then take the figure it actually drew on.
                base = field if field is not None else np.full_like(R, np.nan)
                ax = plot_precip_field(base, geodata=geodata, units="mm/h",
                                       title=None, colorbar=field is not None,
                                       map_kwargs={"drawlonlatlines": False,
                                                   "scale": "50m"})
                fig = ax.figure
                fig.set_size_inches(9, 10)
                # pysteps places its own title tight against the axes, which
                # clips a two-line one. Set it afterwards with padding instead.
                ax.set_title(title, fontsize=11, pad=14)
                return fig, ax, True
            except Exception as e:
                print(f"note: pysteps map plotting unavailable "
                      f"({type(e).__name__}: {e}); falling back to a plain "
                      "array plot. Pass --style plain to silence this.")
        fig, ax = plt.subplots(figsize=(9, 10))
        if field is not None:
            im = ax.imshow(field, origin="upper", extent=extent, cmap="viridis",
                           vmin=0.0, vmax=vmax or args.vmax,
                           interpolation="nearest")
            ax.imshow(np.where(np.isfinite(field), np.nan, 1.0), origin="upper",
                      extent=extent, cmap="Greys", vmin=0, vmax=1.6,
                      interpolation="nearest")
            cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
            cb.set_label("rain rate (mm/h)")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(axis_x)
        ax.set_ylabel(axis_y)
        return fig, ax, False

    # ---- Figure 1: the domain, the tiling, and which tiles were used --------
    used = set(per_tile)
    fig, ax, mapped = basemap(
        f"UK radar composite and the {CROP} km crop tiling\n"
        f"{os.path.basename(frame)[:12]}  |  red = tiles that passed the quality "
        f"filter ({len(used)} of {len(tiles_all)})", field=R)
    for (r, c) in tiles_all:
        x, y, w, h = rect_xy(r + MARGIN, c + MARGIN, CROP)
        on = (r, c) in used
        ax.add_patch(Rectangle((x, y), w, h, fill=False,
                               edgecolor=("red" if on else "0.55"),
                               lw=(1.6 if on else 0.5),
                               ls=("-" if on else ":"), zorder=5))
    if not mapped and corners:
        ax.text(0.01, 0.01,
                "corners: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(corners.items())),
                transform=ax.transAxes, fontsize=6, color="0.3", va="bottom")
    p1 = os.path.join(args.out, "uk_domain.png")
    fig.savefig(p1, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p1}")

    # ---- Figure 2: where the training data actually comes from -------------
    fig, ax, mapped = basemap(
        f"Where the training data comes from\n{n:,} accepted crops "
        f"across {len(per_tile)} tiles", field=None)
    vmax_t = max(per_tile.values()) if per_tile else 1
    cmap = plt.get_cmap("magma")
    for (r, c), cnt in sorted(per_tile.items()):
        x, y, w, h = rect_xy(r + MARGIN, c + MARGIN, CROP)
        ax.add_patch(Rectangle((x, y), w, h, facecolor=cmap(cnt / vmax_t),
                               edgecolor="white", lw=0.6, alpha=0.85, zorder=5))
        ax.text(x + w / 2, y + h / 2, f"{cnt:,}", ha="center", va="center",
                fontsize=7, color="white", zorder=6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax_t))
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("accepted crops per tile")
    p2 = os.path.join(args.out, "uk_crop_density.png")
    fig.savefig(p2, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p2}")

    meta = {"frame": frame, "grid": {"H": H, "W": W}, "geo": geo,
            "crop": CROP, "margin": MARGIN, "context": CONTEXT, "stride": STRIDE,
            "n_candidate_tiles": len(tiles_all), "n_used_tiles": len(used),
            "projected_origin": origin,
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
