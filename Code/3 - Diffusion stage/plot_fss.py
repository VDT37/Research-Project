#!/usr/bin/env python3
"""
plot_fss.py - Fractions Skill Score figures from evaluation scorecards.

Every scorecard written by evaluate_diffusion.py already contains the full FSS
grid: four methods (advection, persistence, model_mean, model_m0 which is a
single ensemble member), five rain-rate thresholds (0.5, 1, 2, 4, 8 mm/h) and
six neighbourhood scales (1, 5, 11, 21, 51, 101 km). Nothing is recomputed here
and no GPU is involved; this reads the JSON and draws it.

WHY FSS BELONGS BESIDE THE SPECTRUM. docs/designs/Metrics_Catalogue.md records
that on this data FSS and the power spectrum point in opposite directions: a
single member can beat advection at every neighbourhood scale at 8 mm/h while
carrying less small-scale variance than the observation. Quoting only one of them
is selective. FSS also answers the question a spectrum cannot, namely at what
spatial scale the forecast is actually skilful, which is what a nowcasting
audience wants to know.

POOLING. FSS is a ratio of sums and must never be averaged across leads or
crops. Each scorecard stores the numerator and denominator alongside the value
(`method|threshold|scale|num` and `|den`), so pooling is
FSS = 1 - sum(num) / sum(den) over whatever set is being combined. Averaging the
per-lead FSS values instead would silently weight rare-event leads wrongly.

THE SKILFUL-SCALE REFERENCE. The conventional target line is
FSS_useful = 0.5 + f0/2, where f0 is the observed base rate at that threshold: a
forecast is said to be skilful at scales where FSS exceeds it. f0 is recovered
here from the contingency counts, since (hits + misses) / n_pixels is the
observed exceedance frequency and every method in a scorecard shares one
observation. The smallest scale at which each method crosses that line is the
"skilful scale", and it is tabulated as well as drawn.

    # one arm, all leads pooled, plus the per-lead breakdown
    python plot_fss.py --arm "CorrDiff ep25=<eval dir>/diffusion_eval_cd_ep025_L*.json" \
        --out Results/figures

    # two arms overlaid, which is the figure the ablation needs
    python plot_fss.py \
        --arm "CorrDiff=<dir>/diffusion_eval_cd_ep025_L*.json" \
        --arm "ml_v2=<dir>/diffusion_eval_mlv2_ep25_L*.json" \
        --out Results/figures

    # numbers only, no matplotlib needed, for checking
    python plot_fss.py --arm "CorrDiff=..." --csv-only

Outputs into --out:
    fss_vs_scale.png        FSS against neighbourhood scale, one panel per
                            threshold, with the skilful-scale reference line
    fss_vs_lead.png         FSS against lead time at a chosen scale
    fss_skilful_scale.png   the smallest skilful scale, per threshold and lead
    fss_table.csv           every pooled value, for the appendix
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

# Method display order and styling. model_m0 is a single member, which is the
# physically realistic field, and it is the one that matters for the sharpness
# argument; model_mean is the smooth field that wins pointwise scores.
METHOD_ORDER = ["persistence", "advection", "model_mean", "model_m0"]
METHOD_LABEL = {"persistence": "persistence", "advection": "advection (pysteps)",
                "model_mean": "ensemble mean", "model_m0": "single member"}
# Short forms for the fixed-width console table, where the long labels overflow.
METHOD_SHORT = {"persistence": "persist", "advection": "advect",
                "model_mean": "ens mean", "model_m0": "member"}
METHOD_STYLE = {"persistence": ("0.55", ":"), "advection": ("C3", "--"),
                "model_mean": ("C0", "-"), "model_m0": ("C2", "-")}


def parse_fss(d):
    """Pull the FSS block out of one scorecard into
    {(method, threshold, scale): (num, den)}. Scorecards that predate the
    numerator/denominator fields are rejected rather than silently averaged."""
    out, missing = {}, 0
    for key, val in (d.get("fss") or {}).items():
        parts = key.split("|")
        if len(parts) != 3:
            continue                       # this is a |num or |den entry
        method, thr, scale = parts[0], float(parts[1]), int(parts[2])
        num = d["fss"].get(f"{key}|num")
        den = d["fss"].get(f"{key}|den")
        if num is None or den is None:
            missing += 1
            continue
        out[(method, thr, scale)] = (float(num), float(den))
    if missing:
        raise SystemExit(
            f"ERROR: {missing} FSS entries carry no numerator/denominator. This "
            "scorecard predates the pooling fix and cannot be combined with "
            "others; re-run the evaluation at the current commit.")
    return out


def base_rate(d):
    """Observed exceedance frequency per threshold, from the contingency counts.
    Every method shares one observation, so hits + misses is the observed count
    however the forecast behaved."""
    det = d.get("deterministic", {})
    for method in ("advection", "model_mean", "persistence"):
        bt = det.get(method, {}).get("by_threshold")
        npix = det.get(method, {}).get("n_pixels")
        if not bt or not npix:
            continue
        out = {}
        for thr, v in bt.items():
            counts = v.get("counts")
            if not counts:
                continue
            hits, misses, _false = counts
            out[float(thr)] = (hits + misses) / float(npix)
        if out:
            return out
    return {}


def load_arm(spec):
    """spec is 'Label=glob'. Returns (label, {lead: parsed}, {lead: base_rate})."""
    if "=" not in spec:
        raise SystemExit(f"ERROR: --arm must be 'Label=glob', got {spec!r}")
    label, pattern = spec.split("=", 1)
    files = sorted(glob.glob(os.path.expanduser(pattern)))
    if not files:
        raise SystemExit(f"ERROR: --arm {label!r} matched no files: {pattern}")
    per_lead, rates, meta = {}, {}, []
    for f in files:
        d = json.load(open(f))
        lead = d.get("lead")
        if lead in per_lead:
            raise SystemExit(
                f"ERROR: {label} has two scorecards for lead {lead}. Narrow the "
                "glob; pooling the same lead twice would double-count it.")
        per_lead[lead] = parse_fss(d)
        rates[lead] = base_rate(d)
        meta.append((lead, d.get("n_fss_crops"), d.get("ckpt_epoch"), d.get("split")))
    return label, per_lead, rates, meta


def pool(per_lead, leads=None):
    """FSS = 1 - sum(num)/sum(den) over the chosen leads. Never an average."""
    acc = defaultdict(lambda: [0.0, 0.0])
    for lead, table in per_lead.items():
        if leads is not None and lead not in leads:
            continue
        for k, (num, den) in table.items():
            acc[k][0] += num
            acc[k][1] += den
    return {k: (1.0 - n / dd if dd > 0 else float("nan")) for k, (n, dd) in acc.items()}


def skilful_scale(fss_at, scales, target):
    """Smallest neighbourhood at which FSS exceeds 0.5 + f0/2. None if never."""
    for s in scales:
        v = fss_at.get(s)
        if v is not None and v == v and v >= target:
            return s
    return None


def write_csv(path, arms, scales, thresholds):
    rows = ["arm,lead,method,threshold_mmh,scale_km,fss"]
    for label, per_lead, _rates, _meta in arms:
        for lead in sorted(per_lead, key=lambda x: (x is None, x)):
            table = pool(per_lead, leads={lead})
            for (method, thr, sc), v in sorted(table.items()):
                rows.append(f"{label},{lead},{method},{thr:g},{sc},{v:.6f}")
        table = pool(per_lead)
        for (method, thr, sc), v in sorted(table.items()):
            rows.append(f"{label},pooled,{method},{thr:g},{sc},{v:.6f}")
    with open(path, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    return len(rows) - 1


def main():
    ap = argparse.ArgumentParser(description="FSS figures from evaluation scorecards.")
    ap.add_argument("--arm", action="append", required=True,
                    help="'Label=glob' pointing at per-lead scorecards. Repeat to "
                         "overlay arms.")
    ap.add_argument("--out", default="Results/figures")
    ap.add_argument("--lead-for-scale-plot", type=int, default=None,
                    help="lead for fss_vs_scale.png (default: all leads pooled)")
    ap.add_argument("--scale-for-lead-plot", type=int, default=21,
                    help="neighbourhood in km for fss_vs_lead.png")
    ap.add_argument("--csv-only", action="store_true",
                    help="write the CSV and the skilful-scale table, no figures. "
                         "Needs only the standard library.")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    arms = [load_arm(s) for s in args.arm]
    os.makedirs(args.out, exist_ok=True)

    keys = set()
    for _l, per_lead, _r, _m in arms:
        for t in per_lead.values():
            keys |= set(t)
    thresholds = sorted({k[1] for k in keys})
    scales = sorted({k[2] for k in keys})
    methods = [m for m in METHOD_ORDER if m in {k[0] for k in keys}]

    print(f"arms: {[a[0] for a in arms]}")
    print(f"thresholds (mm/h): {thresholds}")
    print(f"scales (km): {scales}")
    print(f"methods: {methods}")
    for label, per_lead, _r, meta in arms:
        for lead, ncrops, ep, split in sorted(meta, key=lambda m: (m[0] is None, m[0])):
            print(f"  {label:<16} lead {lead} | {ncrops} FSS crops | ckpt ep{ep} | {split}")

    csv_path = os.path.join(args.out, "fss_table.csv")
    n = write_csv(csv_path, arms, scales, thresholds)
    print(f"\nwrote {csv_path} ({n} rows)")

    # ---- skilful-scale table, printed and returned regardless of matplotlib ---
    print("\nSmallest neighbourhood (km) at which FSS >= 0.5 + f0/2")
    print("a dash means the method never reaches the target at any scale tested")
    hdr = f"  {'arm':<16}{'lead':>6}{'thr':>6}" + "".join(f"{METHOD_SHORT[m]:>10}" for m in methods)
    print(hdr)
    skil = {}
    for label, per_lead, rates, _m in arms:
        for lead in sorted(per_lead, key=lambda x: (x is None, x)):
            table = pool(per_lead, leads={lead})
            f0 = rates.get(lead, {})
            for thr in thresholds:
                target = 0.5 + f0.get(thr, 0.0) / 2.0
                cells = []
                for m in methods:
                    at = {s: table.get((m, thr, s)) for s in scales}
                    s = skilful_scale(at, scales, target)
                    skil[(label, lead, thr, m)] = s
                    cells.append(f"{s if s is not None else '-':>10}")
                print(f"  {label:<16}{lead:>6}{thr:>6g}" + "".join(cells))

    if args.csv_only:
        print("\n--csv-only: figures skipped.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\nmatplotlib unavailable ({e}); the CSV and tables above are still "
              "written. Re-run without --csv-only where matplotlib is installed.")
        return

    # ---- Figure 1: FSS against neighbourhood scale, one panel per threshold ---
    leads_sel = None if args.lead_for_scale_plot is None else {args.lead_for_scale_plot}
    ncol = min(3, len(thresholds))
    nrow = (len(thresholds) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow),
                             squeeze=False, sharex=True)
    for i, thr in enumerate(thresholds):
        ax = axes[i // ncol][i % ncol]
        target = None
        for ai, (label, per_lead, rates, _m) in enumerate(arms):
            table = pool(per_lead, leads=leads_sel)
            rel = [rates[l] for l in (leads_sel or per_lead) if l in rates]
            if rel:
                f0 = sum(r.get(thr, 0.0) for r in rel) / len(rel)
                target = 0.5 + f0 / 2.0
            for m in methods:
                # Baselines are identical across arms, so draw them once.
                if m in ("advection", "persistence") and ai > 0:
                    continue
                y = [table.get((m, thr, s)) for s in scales]
                colour, ls = METHOD_STYLE[m]
                name = METHOD_LABEL[m]
                if m.startswith("model"):
                    name = f"{label} {name}"
                    ls = "-" if ai == 0 else "-."
                ax.plot(scales, y, ls, color=colour, marker="o", ms=3.5,
                        lw=1.6, label=name, alpha=0.9 if ai == 0 else 0.75)
        if target is not None:
            ax.axhline(target, color="0.3", lw=1.0, ls=(0, (4, 3)))
            ax.text(scales[-1], target, f" target {target:.2f}", va="bottom",
                    ha="right", fontsize=7, color="0.3")
        ax.set_title(f"{thr:g} mm/h", fontsize=10)
        ax.set_xscale("log")
        ax.set_xticks(scales)
        ax.set_xticklabels([str(s) for s in scales], fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        if i % ncol == 0:
            ax.set_ylabel("FSS")
        if i // ncol == nrow - 1:
            ax.set_xlabel("neighbourhood width (km)")
    for j in range(len(thresholds), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8,
               bbox_to_anchor=(0.98, 0.04))
    lead_txt = "all leads pooled" if leads_sel is None else f"+{args.lead_for_scale_plot} min"
    fig.suptitle(f"Fractions Skill Score against neighbourhood scale ({lead_txt})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p1 = os.path.join(args.out, "fss_vs_scale.png")
    fig.savefig(p1, dpi=args.dpi)
    plt.close(fig)
    print(f"wrote {p1}")

    # ---- Figure 2: FSS against lead time at one neighbourhood ---------------
    sc = args.scale_for_lead_plot
    if sc not in scales:
        sc = min(scales, key=lambda s: abs(s - args.scale_for_lead_plot))
        print(f"note: scale {args.scale_for_lead_plot} km not available, using {sc} km")
    leads_all = sorted({l for _lb, pl, _r, _m in arms for l in pl if l is not None})
    if leads_all:
        fig, axes = plt.subplots(1, len(thresholds),
                                 figsize=(3.2 * len(thresholds), 3.4),
                                 squeeze=False, sharey=True)
        for i, thr in enumerate(thresholds):
            ax = axes[0][i]
            for ai, (label, per_lead, _r, _m) in enumerate(arms):
                for m in methods:
                    if m in ("advection", "persistence") and ai > 0:
                        continue
                    y = []
                    for lead in leads_all:
                        t = pool(per_lead, leads={lead})
                        y.append(t.get((m, thr, sc)))
                    colour, ls = METHOD_STYLE[m]
                    name = METHOD_LABEL[m]
                    if m.startswith("model"):
                        name = f"{label} {name}"
                        ls = "-" if ai == 0 else "-."
                    ax.plot(leads_all, y, ls, color=colour, marker="o", ms=3.5,
                            lw=1.6, label=name)
            ax.set_title(f"{thr:g} mm/h", fontsize=10)
            ax.set_xticks(leads_all)
            ax.set_xlabel("lead (min)")
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1)
            if i == 0:
                ax.set_ylabel(f"FSS at {sc} km")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
                   fontsize=8, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"Fractions Skill Score against lead time, {sc} km neighbourhood",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0.06, 1, 0.94))
        p2 = os.path.join(args.out, "fss_vs_lead.png")
        fig.savefig(p2, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {p2}")

    # ---- Figure 3: skilful scale ------------------------------------------
    if leads_all:
        fig, axes = plt.subplots(1, len(arms), figsize=(4.4 * len(arms), 3.6),
                                 squeeze=False, sharey=True)
        cap = max(scales) * 1.6
        for ai, (label, per_lead, _r, _m) in enumerate(arms):
            ax = axes[0][ai]
            for m in methods:
                y = []
                for lead in leads_all:
                    vals = [skil.get((label, lead, thr, m)) for thr in thresholds]
                    vals = [v for v in vals if v is not None]
                    y.append(max(vals) if vals else cap)
                colour, ls = METHOD_STYLE[m]
                ax.plot(leads_all, y, ls, color=colour, marker="s", ms=4, lw=1.6,
                        label=METHOD_LABEL[m])
            ax.set_yscale("log")
            ax.set_yticks(scales)
            ax.set_yticklabels([str(s) for s in scales])
            ax.set_xticks(leads_all)
            ax.set_xlabel("lead (min)")
            ax.set_title(label, fontsize=10)
            ax.grid(alpha=0.3)
            if ai == 0:
                ax.set_ylabel("coarsest skilful scale needed (km)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
                   bbox_to_anchor=(0.5, -0.03))
        fig.suptitle("Neighbourhood needed for skill, across all thresholds "
                     "(lower is better)", fontsize=11)
        fig.tight_layout(rect=(0, 0.08, 1, 0.93))
        p3 = os.path.join(args.out, "fss_skilful_scale.png")
        fig.savefig(p3, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {p3}")

    print("\nReminder for the write-up: FSS rests on a different, smaller sample "
          "than the deterministic scores (n_fss_crops above, against 13,281 for "
          "MAE and CSI). State which sample each number uses.")


if __name__ == "__main__":
    main()
