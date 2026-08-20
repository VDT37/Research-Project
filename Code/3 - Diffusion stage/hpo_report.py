#!/usr/bin/env python3
"""
hpo_report.py - turn an hpo_search.py study directory into the tables and
figures the results chapter needs.

Reads only what the study already wrote (study.json, ranking.json, trials.jsonl,
baselines.json) and recomputes nothing, so the report can be regenerated from a
frozen study long after the GPUs are gone.

It produces four things, in order of how much they matter to the write-up:

1. hpo_leaderboard.md
   Every trial, ranked, with its objective, its admissibility gate, and its
   scores beside BOTH baselines the brief requires: persistence and the
   pysteps advection prior, on the identical crops. Trials that failed their
   gate stay in the table with the reason. Nothing is silently dropped.

2. The one-factor-at-a-time effects table.
   A coarse screen built as OFAT around an incumbent is read as marginals, not
   as a winner. This table gives, per parameter, the change in objective caused
   by moving that one parameter away from the incumbent while everything else
   is held. That is the form the discussion section actually needs: "learning
   rate mattered, weight decay did not" is a sentence you can defend, whereas
   "trial 17 won" is not.

3. The rung-transfer diagnostic.
   Multi-fidelity search is only sound if the cheap rung ranks configurations
   the way the expensive rung would. This computes the Spearman rank
   correlation between consecutive rungs on the trials that appear in both,
   which is the honest measure of whether the screening worked. A low value is
   itself a reportable result: it says the screen did not transfer and the
   winner needs confirming at full fidelity before anything is claimed. It is
   printed with its sample size, because a rank correlation on four promoted
   trials is a number, not evidence.

4. hpo_curves.png / hpo_marginals.png / hpo_transfer.png
   Learning curves coloured by outcome, per-parameter marginals, and the
   fidelity-transfer scatter.

    python hpo_report.py --study $HPO/ldm_coarse
    python hpo_report.py --study $HPO/ldm_coarse --top 10 --no-figures
"""

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hpo_spaces as S                                            # noqa: E402


# ----------------------------------------------------------------------------
# Statistics, kept local so the report has no scipy dependency
# ----------------------------------------------------------------------------
def rankdata(xs):
    """Average ranks, ties shared, matching scipy.stats.rankdata('average')."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((x - mb) ** 2 for x in b))
    if va == 0 or vb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


def spearman(a, b):
    return pearson(rankdata(a), rankdata(b))


def fnum(v, nd=4, na="-"):
    if v is None:
        return na
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return na
    return f"{f:.{nd}f}"


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_study(study_dir):
    def maybe(name):
        p = os.path.join(study_dir, name)
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except Exception:
                return None
        return None

    spec = maybe("study.json")
    if spec is None:
        raise SystemExit(f"ERROR: {study_dir} has no study.json. Point --study at "
                         "an hpo_search.py output directory.")
    ranking = (maybe("ranking.json") or {}).get("trials", [])
    plan = maybe("plan.json") or {}
    # hpo_baselines.py writes one file per ARM, and writes it beside the study
    # directories rather than inside any one of them, because several studies
    # share an arm's baselines. Look in the study dir first, then beside it, and
    # try the arm-suffixed name at both, so a report never silently drops its
    # baseline table just because of where the file landed.
    def maybe_up(name):
        p = os.path.join(os.path.dirname(os.path.abspath(study_dir)), name)
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except Exception:
                return None
        return None

    _arm = (spec or {}).get("arm")
    _cands = ([("baselines_%s.json" % _arm)] if _arm else []) + ["baselines.json"]
    baselines = {}
    for _c in _cands:
        baselines = maybe(_c) or maybe_up(_c) or {}
        if baselines:
            break
    events = []
    ep = os.path.join(study_dir, "trials.jsonl")
    if os.path.exists(ep):
        for line in open(ep):
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    return spec, ranking, plan, baselines, events


def changed(params, incumbent=None):
    inc = incumbent or S.INCUMBENT
    d = {k: v for k, v in (params or {}).items() if v != inc.get(k)}
    return ", ".join(f"{k}={v}" for k, v in sorted(d.items())) or "(incumbent)"


# Validation-log keys, in preference order, with the axis label each implies.
# train_ldm.py emits loss_w; train_regression.py emits mse. A study is scanned
# for whichever its trainer actually wrote.
VAL_KEYS = (("loss_w", "validation loss (weighted, EMA)"),
            ("mse", "validation MSE (latent space)"))


def learning_curves(study_dir):
    """Per-trial validation curves, read from each trainer's own train_log.json.

    The trainers do NOT share a validation-log schema: train_ldm.py writes
    `val.loss_w` (weighted EMA loss) and train_regression.py writes `val.mse`
    (latent-space MSE). Assuming one of them crashes the report for the other
    arm, so the key is discovered per study rather than hardcoded.

    Returns (curves, ylabel). ylabel is None when no trial carried a usable key,
    in which case the caller should skip the curves figure rather than draw an
    unlabelled one.
    """
    out, ylabel = {}, None
    for name in sorted(os.listdir(study_dir)):
        d = os.path.join(study_dir, name)
        lp = os.path.join(d, "train_log.json")
        if not (name.startswith("trial_") and os.path.exists(lp)):
            continue
        try:
            log = json.load(open(lp))
        except Exception:
            continue
        if not isinstance(log, list):
            continue
        for key, lab in VAL_KEYS:
            pts = [(r["epoch"], r["val"][key]) for r in log
                   if isinstance(r, dict) and "epoch" in r
                   and isinstance(r.get("val"), dict) and key in r["val"]]
            if pts:
                out[name] = pts
                ylabel = ylabel or lab
                break
    return out, ylabel


# ----------------------------------------------------------------------------
# Report sections
# ----------------------------------------------------------------------------
def leaderboard_rows(ranking, baselines, arm):
    rows = []
    for r in ranking:
        m = r.get("metrics") or {}
        b = baselines.get(r.get("rung"), {}) if isinstance(baselines, dict) else {}
        mae, crps = m.get("mae"), m.get("crps")
        mae_adv, crps_adv = m.get("mae_adv"), m.get("crps_adv")
        mae_pers = m.get("mae_pers", b.get("mae_pers"))
        crps_pers = m.get("crps_pers", b.get("crps_pers"))
        rows.append({
            "trial": r.get("trial"), "rung": r.get("rung"),
            "origin": r.get("origin"), "changed": changed(r.get("params")),
            "objective": r.get("objective"), "gate": r.get("gate_pass"),
            "mae": mae, "mae_over_adv": (mae / mae_adv) if (mae and mae_adv) else None,
            "mae_over_pers": (mae / mae_pers) if (mae and mae_pers) else None,
            "crps": crps,
            "crps_ss_adv": (1 - crps / crps_adv) if (crps and crps_adv) else None,
            "crps_ss_pers": (1 - crps / crps_pers) if (crps and crps_pers) else None,
            "csi_1": m.get("csi_1"), "csi_8": m.get("csi_8"),
            "psd": m.get("psd_ratio_2_8km"),
            "val_loss": m.get("val_loss_w"),
            "gpu_h": (r.get("wall_min") / 60.0) if r.get("wall_min") else None,
            "params": r.get("params") or {},
        })
    return rows


def ofat_effects(rows, objective_name):
    """Marginal effect of each parameter, measured against the incumbent cell in
    the same rung. Only meaningful for a grid study built as one factor at a
    time; returns an empty list otherwise."""
    by_rung = {}
    for r in rows:
        by_rung.setdefault(r["rung"], []).append(r)
    out = []
    for rung, rs in by_rung.items():
        base = next((x for x in rs if x["origin"] == "incumbent"), None)
        if base is None or base["objective"] is None:
            continue
        for r in rs:
            if not str(r["origin"]).startswith("ofat:"):
                continue
            name = r["origin"].split(":", 1)[1]
            if r["objective"] is None:
                continue
            out.append({"rung": rung, "param": name,
                        "value": r["params"].get(name),
                        "incumbent": S.INCUMBENT.get(name),
                        "objective": r["objective"],
                        "delta": r["objective"] - base["objective"],
                        "trial": r["trial"]})
    out.sort(key=lambda d: (-abs(d["delta"]), d["param"]))
    return out


def transfer(rows, rungs):
    """Spearman rank correlation between consecutive rungs on the trials that
    appear in both. This is the number that justifies (or refutes) the whole
    multi-fidelity design."""
    by_rung = {}
    for r in rows:
        if r["objective"] is None:
            continue
        idx = r["trial"].split("_")[1] if "_" in r["trial"] else r["trial"]
        by_rung.setdefault(r["rung"], {})[idx] = r["objective"]
    names = [r["name"] for r in rungs if r["name"] in by_rung]
    out = []
    for a, b in zip(names, names[1:]):
        shared = sorted(set(by_rung[a]) & set(by_rung[b]))
        if len(shared) < 3:
            out.append({"from": a, "to": b, "n": len(shared),
                        "spearman": None,
                        "note": "too few trials promoted to estimate a rank correlation"})
            continue
        rho = spearman([by_rung[a][k] for k in shared],
                       [by_rung[b][k] for k in shared])
        out.append({"from": a, "to": b, "n": len(shared), "spearman": rho,
                    "note": ("the cheap rung ranked these the same way the "
                             "expensive one did" if (rho or 0) > 0.6 else
                             "weak transfer: treat the screen's ordering as "
                             "provisional and confirm more candidates")})
    return out


# ----------------------------------------------------------------------------
# Markdown
# ----------------------------------------------------------------------------
def write_markdown(path, spec, rows, effects, trans, plan, baselines, top):
    L = []
    a = L.append
    arm = spec.get("arm")
    a(f"# HPO study: {spec.get('space')} ({arm} arm)\n")
    a(f"Generated from `{os.path.dirname(path)}`. Search code at git "
      f"`{spec.get('git')}`, study created {spec.get('created')}.\n")
    a(f"{spec.get('space_note', '')}\n")

    a("## What was searched\n")
    a(f"- sampler: `{spec.get('sampler')}`")
    a(f"- objective (maximised): `{spec.get('objective')}` = `{spec.get('objective_expr')}`")
    a(f"- admissibility gate: `{spec.get('gate')}` = `{spec.get('gate_expr')}`")
    a(f"- fidelity ladder: " + ", ".join(
        f"`{r['name']}` ({r['rows']} rows x {r['epochs']} epochs)"
        for r in spec.get("rungs", [])))
    a(f"- successive halving with eta = {spec.get('eta')}, seed pinned at "
      f"{spec.get('seed')} on every trial so comparisons are paired")
    if plan.get("gpu_h_planned") is not None:
        a(f"- planned cost: {plan['gpu_h_planned']} A100-hours")
    if plan.get("gpu_h_if_full_fidelity"):
        f = plan["gpu_h_if_full_fidelity"]
        p = plan.get("gpu_h_planned") or 0
        a(f"- the same configurations trained to completion at full fidelity would "
          f"be {f:.0f} A100-hours, so the multi-fidelity search costs "
          f"{100.0 * p / max(f, 1e-9):.1f} percent of the naive grid")
    a("")

    a("## Baselines\n")
    a("Every trial below is scored against both baselines the brief requires, on "
      "the identical validation crops. The advection column is the pysteps "
      "baseline: dense Lucas-Kanade optical flow plus Germann-Zawadzki "
      "extrapolation, computed once in `build_advection_prior.py` and never "
      "learned.\n")
    if baselines:
        a("| rung | crops | persistence MAE | advection MAE | climatology MAE | "
          "persistence CSI@1 | advection CSI@1 |")
        a("|---|---|---|---|---|---|---|")
        for k, b in sorted(baselines.items()):
            a(f"| {k} | {b.get('n_crops')} | {fnum(b.get('mae_pers'))} | "
              f"{fnum(b.get('mae_advcheck'))} | {fnum(b.get('mae_clim'))} | "
              f"{fnum(b.get('csi_1_pers'))} | {fnum(b.get('csi_1_advcheck'))} |")
        a("")
        a("The advection row is an integrity check, not a result: the trainer "
          "computes its own advection control on these same crops and the two "
          "must agree to float noise. Disagreement means the crop sets diverged "
          "and no paired comparison in this study is valid.\n")
    else:
        a("No baselines file was found in this study directory or beside it, "
          "so the per-rung baseline table is omitted. The persistence and "
          "advection COLUMNS in the leaderboard below are unaffected: those "
          "come from each trial's own record, not from this file. Run "
          "`hpo_baselines.py` and re-run the report to restore the table.\n")

    a("## Leaderboard\n")
    a("Ordered by fidelity rung first (deepest rung, and therefore the most "
      "evidence, at the top), then by the objective, with trials that failed the "
      "admissibility gate sorted last within their rung but kept visible. The "
      "ordering is deliberately not a single global sort on the objective: a "
      "configuration that scored well once on the cheapest rung has far less "
      "behind it than one that survived to the expensive rung, and letting the "
      "two compete directly is the same error as quoting a 16-crop diagnostic as "
      "a result.\n")
    a("| # | trial | rung | changed from incumbent | objective | gate | MAE | "
      "MAE/adv | MAE/pers | CRPSS vs adv | CRPSS vs pers | CSI@1 | CSI@8 | "
      "PSD 2-8km | val loss | GPU-h |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows[:top], 1):
        a(f"| {i} | {r['trial']} | {r['rung']} | {r['changed']} | "
          f"{fnum(r['objective'], 5)} | {'pass' if r['gate'] else 'FAIL'} | "
          f"{fnum(r['mae'])} | {fnum(r['mae_over_adv'], 3)} | "
          f"{fnum(r['mae_over_pers'], 3)} | {fnum(r['crps_ss_adv'], 4)} | "
          f"{fnum(r['crps_ss_pers'], 4)} | {fnum(r['csi_1'])} | {fnum(r['csi_8'])} | "
          f"{fnum(r['psd'], 3)} | {fnum(r['val_loss'], 5)} | {fnum(r['gpu_h'], 2)} |")
    if len(rows) > top:
        a(f"\n({len(rows) - top} further rows in `ranking.json`; nothing was "
          f"dropped, the table is truncated for reading only.)")
    a("")

    if effects:
        a("## One-factor-at-a-time effects\n")
        a("Change in the objective from moving one parameter away from the "
          "incumbent, everything else held. Sorted by absolute effect, which is "
          "the order in which these parameters deserve GPU hours.\n")
        a("| parameter | brief's name | value | incumbent | objective | delta | rung |")
        a("|---|---|---|---|---|---|---|")
        for e in effects:
            brief = S.PARAMS.get(e["param"], {}).get("brief", "")
            a(f"| `{e['param']}` | {brief} | {e['value']} | {e['incumbent']} | "
              f"{fnum(e['objective'], 5)} | {fnum(e['delta'], 5)} | {e['rung']} |")
        a("")
        near_zero = [e for e in effects if abs(e["delta"]) < 1e-3]
        if near_zero:
            a(f"{len(near_zero)} of {len(effects)} cells moved the objective by "
              f"less than 0.001. Report those as measured null results rather "
              f"than as tuning wins; a flat axis is information about the model, "
              f"and this project has already been bitten once by reporting an "
              f"optimum from a curve that was flat to the eighth decimal (the "
              f"ridge-gate alpha scan).\n")

    a("## Fidelity transfer\n")
    a("Multi-fidelity search assumes the cheap rung orders configurations the way "
      "the expensive rung would. That assumption is measured here rather than "
      "asserted.\n")
    if trans:
        a("| from | to | trials in both | Spearman rho | reading |")
        a("|---|---|---|---|---|")
        for t in trans:
            a(f"| {t['from']} | {t['to']} | {t['n']} | "
              f"{fnum(t['spearman'], 3)} | {t['note']} |")
    else:
        a("Only one rung has scored trials, so there is nothing to correlate yet.")
    a("")

    a("## How to read this\n")
    a("- A rung result is a ranking statistic, not a result. Nothing in this "
      "table may be quoted as a model score. The winner is confirmed by a "
      "full-fidelity run and a full-split evaluation through "
      "`evaluate_diffusion.py`, and only those numbers go in the results "
      "chapter.")
    a("- The objective is deliberately not the validation EDM loss. Across three "
      "runs in this project, validation loss and small-scale power move in "
      "opposite directions, so ranking on loss selects the smoothest model the "
      "search produced. CRPS is used instead because it is strictly proper: it "
      "cannot be gamed by smoothing, which flatters MAE through the "
      "double-penalty effect, nor by sharpening, which flatters PSD.")
    a("- The gate encodes `LDM.md` section 6 criterion (a): matching the "
      "advection prior on deterministic accuracy is a precondition for a "
      "configuration to be considered at all, not something to be traded away "
      "for a better CRPS.")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def figures(study_dir, spec, rows, effects, trans):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"figures skipped ({e})")
        return []
    made = []

    # -- learning curves, coloured by whether the trial survived its rung ------
    curves, curve_label = learning_curves(study_dir)
    if curves and curve_label:
        best = {r["trial"]: r["objective"] for r in rows if r["objective"] is not None}
        fig, ax = plt.subplots(figsize=(8, 5))
        vals = [v for v in best.values()]
        lo, hi = (min(vals), max(vals)) if vals else (0, 1)
        for name, c in curves.items():
            if not c:
                continue
            o = best.get(name)
            frac = 0.5 if o is None else (o - lo) / max(hi - lo, 1e-12)
            ax.plot([e for e, _ in c], [v for _, v in c],
                    color=plt.cm.viridis(frac), alpha=0.85, lw=1.4)
        ax.set_xlabel("epoch")
        ax.set_ylabel(curve_label)
        ax.set_title(f"{spec.get('space')}: per-trial validation curves\n"
                     "(colour = objective, dark low to bright high)")
        ax.grid(alpha=0.3)
        p = os.path.join(study_dir, "hpo_curves.png")
        fig.tight_layout()
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append(p)

    # -- per-parameter marginals ----------------------------------------------
    names = sorted({k for r in rows for k in r["params"]})
    names = [n for n in names if len({str(r["params"].get(n)) for r in rows}) > 1]
    if names and any(r["objective"] is not None for r in rows):
        n = len(names)
        cols = min(4, n)
        nrow = int(math.ceil(n / cols))
        fig, axes = plt.subplots(nrow, cols, figsize=(3.6 * cols, 3.0 * nrow),
                                 squeeze=False)
        for i, name in enumerate(names):
            ax = axes[i // cols][i % cols]
            xs, ys, cs = [], [], []
            for r in rows:
                if r["objective"] is None or name not in r["params"]:
                    continue
                v = r["params"][name]
                xs.append(v if isinstance(v, (int, float)) else str(v))
                ys.append(r["objective"])
                cs.append("tab:green" if r["gate"] else "tab:red")
            if not xs:
                ax.axis("off")
                continue
            if all(isinstance(x, (int, float)) for x in xs):
                ax.scatter(xs, ys, c=cs, s=26)
                if name in ("lr", "weight_decay") and all(x > 0 for x in xs):
                    ax.set_xscale("log")
            else:
                cats = sorted(set(xs))
                ax.scatter([cats.index(x) for x in xs], ys, c=cs, s=26)
                ax.set_xticks(range(len(cats)))
                ax.set_xticklabels(cats, rotation=30, fontsize=7)
            inc = S.INCUMBENT.get(name)
            if isinstance(inc, (int, float)) and all(isinstance(x, (int, float))
                                                     for x in xs):
                ax.axvline(inc, color="0.4", ls="--", lw=1)
            ax.set_title(f"{name}\n{S.PARAMS.get(name, {}).get('brief', '')}",
                         fontsize=8)
            ax.grid(alpha=0.25)
        for j in range(n, nrow * cols):
            axes[j // cols][j % cols].axis("off")
        fig.suptitle(f"{spec.get('space')}: objective by parameter "
                     "(green passes the gate, red fails; dashed line = incumbent)",
                     fontsize=10)
        p = os.path.join(study_dir, "hpo_marginals.png")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append(p)

    # -- fidelity transfer ----------------------------------------------------
    by_rung = {}
    for r in rows:
        if r["objective"] is None:
            continue
        idx = r["trial"].split("_")[1] if "_" in r["trial"] else r["trial"]
        by_rung.setdefault(r["rung"], {})[idx] = r["objective"]
    pairs = [(t["from"], t["to"], t) for t in trans if t.get("spearman") is not None]
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 4.0),
                                 squeeze=False)
        for i, (a, b, t) in enumerate(pairs):
            ax = axes[0][i]
            shared = sorted(set(by_rung[a]) & set(by_rung[b]))
            ax.scatter([by_rung[a][k] for k in shared],
                       [by_rung[b][k] for k in shared], s=34, color="tab:blue")
            for k in shared:
                ax.annotate(k, (by_rung[a][k], by_rung[b][k]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel(f"objective at {a}")
            ax.set_ylabel(f"objective at {b}")
            ax.set_title(f"Spearman rho = {t['spearman']:.3f} (n = {t['n']})",
                         fontsize=9)
            ax.grid(alpha=0.3)
        fig.suptitle("does the cheap rung rank configurations like the expensive one?",
                     fontsize=10)
        p = os.path.join(study_dir, "hpo_transfer.png")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append(p)
    return made


def main():
    ap = argparse.ArgumentParser(description="Report on an hpo_search.py study.")
    ap.add_argument("--study", required=True)
    ap.add_argument("--top", type=int, default=30, help="rows in the markdown table")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", default=None, help="markdown path (default: in --study)")
    args = ap.parse_args()

    spec, ranking, plan, baselines, events = load_study(args.study)
    if not ranking:
        raise SystemExit(f"ERROR: {args.study} has no ranking.json yet. The study "
                         "has not completed a rung; nothing to report.")
    rows = leaderboard_rows(ranking, baselines, spec.get("arm"))
    # Rank by rung depth FIRST, then by objective. A trial that survived to the
    # expensive rung has more evidence behind it than one that scored well once
    # on the cheapest, and mixing the two into a single ordering would let a
    # lucky low-fidelity result outrank a confirmed one. That is exactly the
    # error this project has already made twice with 16-crop diagnostics.
    depth = {r["name"]: i for i, r in enumerate(spec.get("rungs", []))}
    rows.sort(key=lambda r: (-depth.get(r["rung"], -1),
                             0 if r["gate"] else 1,
                             -(r["objective"] if r["objective"] is not None else -1e300)))
    effects = ofat_effects(rows, spec.get("objective"))
    trans = transfer(rows, spec.get("rungs", []))

    md = args.out or os.path.join(args.study, "hpo_leaderboard.md")
    write_markdown(md, spec, rows, effects, trans, plan, baselines, args.top)

    summary = {"study": os.path.abspath(args.study), "spec": spec,
               "n_trials": len(rows),
               "n_gate_pass": sum(1 for r in rows if r["gate"]),
               "best": rows[0] if rows else None,
               "ofat_effects": effects, "transfer": trans,
               "gpu_h_measured": round(sum((r["gpu_h"] or 0) for r in rows), 2),
               "gpu_h_planned": plan.get("gpu_h_planned"),
               "gpu_h_if_full_fidelity": plan.get("gpu_h_if_full_fidelity")}
    tmp = os.path.join(args.study, "hpo_summary.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    os.replace(tmp, os.path.join(args.study, "hpo_summary.json"))

    print(f"wrote {md}")
    print(f"wrote {os.path.join(args.study, 'hpo_summary.json')}")
    if not args.no_figures:
        for p in figures(args.study, spec, rows, effects, trans):
            print(f"wrote {p}")

    best = rows[0] if rows else None
    if best:
        print(f"\nbest: {best['trial']}  objective {fnum(best['objective'], 5)}  "
              f"({best['changed']})")
    for t in trans:
        if t.get("spearman") is not None:
            print(f"transfer {t['from']} -> {t['to']}: rho {t['spearman']:.3f} "
                  f"on {t['n']} trials")
    print("\nA rung result is a ranking statistic. Confirm the winner at full "
          "fidelity and score it through evaluate_diffusion.py on the full split "
          "before quoting any number from it.")


if __name__ == "__main__":
    main()
