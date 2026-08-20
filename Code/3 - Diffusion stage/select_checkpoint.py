#!/usr/bin/env python3
"""
select_checkpoint.py - re-run the precommitted checkpoint-selection rule and
emit an auditable record of how it decided.

WHY THIS EXISTS. The rule that picked the reported checkpoint for each diffusion
arm was applied by hand during the run, and its outcome (epoch 25 for both arms)
was recorded in STATE.md rather than in any artefact an examiner could re-run.
That is a reproducibility gap, not a correctness one: every input the rule
consumes is already in each run's `train_log.json`. This script reads those logs,
applies the rule mechanically, prints the full decision table and writes it to
disk, so the selection can be audited without trusting a prose summary.

THE RULE, unchanged from the one precommitted before either arm was scored:

  1. HARD GATE. Keep only checkpoints whose sampled model-mean MAE is at or
     below the advection control measured on the same crops. A checkpoint that
     cannot beat pysteps on its own diagnostic is not a candidate.
  2. MAXIMISE CRPS SKILL SCORE, CRPSS = 1 - crps / crps_adv, against the same
     advection control.
  3. TIEBREAK. Among checkpoints within --tolerance CRPSS of the best, take the
     one whose member PSD ratio is closest to the codec ceiling. Power above the
     ceiling is synthesised rather than recovered, power below it is
     over-smoothing, so "closest" is the right target in both directions.

WHAT THE INPUTS ARE, AND THE CAVEAT THAT MUST TRAVEL WITH THEM. Every field the
rule reads comes from the `sampled` block of `train_log.json`, which is the
in-training diagnostic: 16 crops, 8 members, drawn during training. It is NOT the
full-split scorecard. docs/Diffusion_MultiLead_Results.md section 12 measures that
this diagnostic understates member PSD by 15 to 32 percent, and that the bias
grows as the true value falls. So the tiebreak ran on a statistic now known to be
biased. That is a real limitation of the selection procedure and this script does
not paper over it: it reproduces what the rule actually did, not what it would
have done with better inputs. Re-running the rule against full-split PSD would be
a different (and retrospective) procedure, which is why it is not offered here.

Only the archived checkpoints are candidates. Both arms archived every fifth
epoch, so the default ladder is 5, 10, ... 50; pass --epochs to override.

No GPU, no torch, no network. Reads JSON and prints a table.

USAGE

    python select_checkpoint.py \
        --arm "ml_v2=analysis_bundle/dissertation_outputs/diffusion_ml_v2/train_log.json" \
        --arm "CorrDiff=analysis_bundle/dissertation_outputs/diffusion_corrdiff_v1/train_log.json" \
        --out analysis_bundle/dissertation_outputs/checkpoint_selection

Writes `<out>.json` and `<out>.md`. Exit status is 1 if any arm has no
gate-passing checkpoint, so this can be used as a check in a pipeline.
"""
import argparse
import json
import os
import sys

# The measured pooled codec ceiling on the mean-of-ratios PSD axis, from
# docs/designs/Metrics_Catalogue.md Part 4. The in-training diagnostic emits
# mean-of-ratios only, so this is the axis the rule ran on.
DEFAULT_CEILING = 0.903
DEFAULT_TOLERANCE = 0.005
DEFAULT_EPOCHS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

REQUIRED = ("mae", "mae_adv", "crps", "crps_adv", "psd_ratio_2_8km")


def load_rows(path, epochs):
    """Pull the sampled diagnostic for each archived epoch out of a train log."""
    log = json.load(open(path, encoding="utf-8"))
    if not isinstance(log, list):
        raise SystemExit(f"ERROR: {path} is not a list of epoch records.")
    by_epoch = {r.get("epoch"): r for r in log if isinstance(r, dict)}
    rows, missing = [], []
    for e in epochs:
        rec = by_epoch.get(e)
        s = (rec or {}).get("sampled") or {}
        if not all(k in s for k in REQUIRED):
            missing.append(e)
            continue
        rows.append({
            "epoch": e,
            "mae": s["mae"],
            "mae_adv": s["mae_adv"],
            "crps": s["crps"],
            "crps_adv": s["crps_adv"],
            "psd": s["psd_ratio_2_8km"],
            "n_crops": s.get("n_crops"),
            "n_members": s.get("n_members"),
            "gate": s["mae"] <= s["mae_adv"],
            "crpss": 1.0 - s["crps"] / s["crps_adv"],
        })
    return rows, missing


def select(rows, ceiling, tolerance):
    """Apply gate, then CRPSS, then the PSD tiebreak. Returns the decision."""
    passed = [r for r in rows if r["gate"]]
    if not passed:
        return None
    best = max(r["crpss"] for r in passed)
    tied = [r for r in passed if best - r["crpss"] <= tolerance]
    for r in rows:
        r["tied"] = any(t["epoch"] == r["epoch"] for t in tied)
        r["psd_distance"] = abs(r["psd"] - ceiling)
    winner = min(tied, key=lambda r: abs(r["psd"] - ceiling))
    # How close the tie tolerance came to admitting the next checkpoint. A rule
    # whose outcome hinges on the fourth decimal of a tolerance is worth saying
    # so about, which is why this is reported rather than left implicit.
    outside = [r for r in passed if not r["tied"]]
    nearest_miss = None
    if outside:
        nm = max(outside, key=lambda r: r["crpss"])
        nearest_miss = {"epoch": nm["epoch"], "crpss": nm["crpss"],
                        "short_by": (best - nm["crpss"]) - tolerance,
                        "psd_distance": abs(nm["psd"] - ceiling)}
    return {"winner": winner["epoch"], "best_crpss": best,
            "tied_epochs": [r["epoch"] for r in tied],
            "n_gate_pass": len(passed), "n_candidates": len(rows),
            "nearest_miss": nearest_miss}


def render(arms, ceiling, tolerance):
    out = []
    a = out.append
    a("# Checkpoint selection, reproduced from the training logs\n")
    a("Generated by `select_checkpoint.py`. Every figure below is read from the "
      "`sampled` block of each arm's `train_log.json`; nothing is recomputed and "
      "no GPU is involved.\n")
    a("Rule, precommitted before either arm was scored: hard gate on sampled "
      "model-mean MAE at or below the advection control on the same crops, then "
      f"maximise CRPSS = 1 - crps/crps_adv, then break ties within {tolerance} "
      f"CRPSS on member PSD closest to the codec ceiling of {ceiling}.\n")
    a("CAVEAT. These are 16-crop in-training diagnostics, not full-split "
      "scorecards. `docs/Diffusion_MultiLead_Results.md` section 12 measures that "
      "this statistic understates member PSD by 15 to 32 percent, so the tiebreak "
      "ran on a biased input. This table reproduces what the rule did, not what a "
      "better-informed rule would have done.\n")
    for arm in arms:
        d = arm["decision"]
        a(f"## {arm['label']}\n")
        if d is None:
            a("No checkpoint passed the MAE gate. No selection possible.\n")
            continue
        a(f"Selected **epoch {d['winner']}**. {d['n_gate_pass']} of "
          f"{d['n_candidates']} archived checkpoints passed the gate; best CRPSS "
          f"{d['best_crpss']:.5f}; tied set "
          f"{', '.join('ep%d' % e for e in d['tied_epochs'])}.\n")
        a("| epoch | gate | sampled MAE | advection MAE | CRPSS | member PSD | "
          "distance from ceiling | in tied set | selected |")
        a("|---|---|---|---|---|---|---|---|---|")
        for r in arm["rows"]:
            a(f"| {r['epoch']} | {'pass' if r['gate'] else 'FAIL'} | "
              f"{r['mae']:.4f} | {r['mae_adv']:.4f} | {r['crpss']:.5f} | "
              f"{r['psd']:.4f} | {r['psd_distance']:.4f} | "
              f"{'yes' if r['tied'] else 'no'} | "
              f"{'**yes**' if r['epoch'] == d['winner'] else ''} |")
        a("")
        nm = d.get("nearest_miss")
        if nm is not None:
            win_dist = next(r["psd_distance"] for r in arm["rows"]
                            if r["epoch"] == d["winner"])
            a(f"Nearest checkpoint outside the tied set: ep{nm['epoch']}, CRPSS "
              f"{nm['crpss']:.5f}, short of the tolerance by {nm['short_by']:.5f}. "
              f"Its PSD distance is {nm['psd_distance']:.4f} against the winner's "
              f"{win_dist:.4f}, ")
            if nm["psd_distance"] < win_dist:
                a(f"so a tolerance loose enough to admit it would have selected "
                  f"ep{nm['epoch']} INSTEAD, on a PSD distance {win_dist / nm['psd_distance']:.1f} "
                  f"times smaller. THE SELECTION IS SENSITIVE TO THE TIE TOLERANCE "
                  f"for this arm, and the reported result depends on that tolerance "
                  f"having been fixed at {tolerance} before the arm was scored "
                  f"rather than chosen afterwards. State this wherever the "
                  f"selection is defended.\n")
            else:
                a("so admitting it would not change the outcome. The selection is "
                  "insensitive to the tie tolerance in this direction.\n")
    labels = [a_["label"] for a_ in arms if a_["decision"]]
    winners = {a_["label"]: a_["decision"]["winner"] for a_ in arms if a_["decision"]}
    if len(set(winners.values())) == 1 and len(winners) > 1:
        a(f"## Convergence\n")
        a(f"All {len(labels)} arms independently select epoch "
          f"{next(iter(winners.values()))}. The arms differ in conditioning, loss "
          "landscape and training history, so independent convergence on the same "
          "epoch is evidence the rule is selecting on its criterion rather than on "
          "noise, and evidence against cherry-picking.\n")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=LOG",
                    help='an arm to select for, e.g. --arm "ml_v2=.../train_log.json". '
                         "Repeat for each arm.")
    ap.add_argument("--epochs", default=None,
                    help="comma-separated archived epochs (default 5,10,...,50)")
    ap.add_argument("--ceiling", type=float, default=DEFAULT_CEILING,
                    help="codec PSD ceiling the tiebreak targets")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help="CRPSS window that counts as a tie")
    ap.add_argument("--out", default=None,
                    help="output basename; .json and .md are appended")
    args = ap.parse_args()

    epochs = ([int(x) for x in args.epochs.split(",")] if args.epochs
              else list(DEFAULT_EPOCHS))

    arms, failed = [], False
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit(f"ERROR: --arm must be LABEL=LOG, got {spec!r}")
        label, path = spec.split("=", 1)
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise SystemExit(f"ERROR: --arm {label!r} log not found: {path}")
        rows, missing = load_rows(path, epochs)
        if missing:
            print(f"note: {label} has no complete sampled diagnostic at epoch(s) "
                  f"{', '.join(str(m) for m in missing)}; they are excluded.")
        if not rows:
            raise SystemExit(f"ERROR: {label} has no usable epochs in {path}.")
        decision = select(rows, args.ceiling, args.tolerance)
        if decision is None:
            print(f"WARNING: {label} has no gate-passing checkpoint.")
            failed = True
        arms.append({"label": label, "log": path, "rows": rows,
                     "decision": decision})

    for arm in arms:
        d = arm["decision"]
        print(f"\n== {arm['label']}")
        if d is None:
            print("   no checkpoint passed the MAE gate")
            continue
        print(f"   {d['n_gate_pass']}/{d['n_candidates']} pass gate | best CRPSS "
              f"{d['best_crpss']:.5f} | tied {d['tied_epochs']}")
        for r in arm["rows"]:
            mark = "  <-- SELECTED" if r["epoch"] == d["winner"] else ""
            print(f"   ep{r['epoch']:03d} gate={'pass' if r['gate'] else 'FAIL'} "
                  f"CRPSS {r['crpss']:.5f} PSD {r['psd']:.4f} "
                  f"dist {r['psd_distance']:.4f}"
                  f"{' (tied)' if r['tied'] else ''}{mark}")

    if args.out:
        payload = {"rule": {"gate": "sampled mae <= mae_adv",
                            "objective": "maximise 1 - crps/crps_adv",
                            "tiebreak": "member PSD closest to ceiling",
                            "ceiling": args.ceiling,
                            "tolerance": args.tolerance,
                            "epochs": epochs},
                   "source": "train_log.json `sampled` block, in-training "
                             "diagnostic (16 crops), NOT the full-split scorecard",
                   "arms": arms}
        tmp = args.out + ".json.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        os.replace(tmp, args.out + ".json")
        tmp = args.out + ".md.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(render(arms, args.ceiling, args.tolerance))
        os.replace(tmp, args.out + ".md")
        print(f"\nwrote {args.out}.json and {args.out}.md")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
