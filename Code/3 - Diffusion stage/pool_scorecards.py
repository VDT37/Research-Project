#!/usr/bin/env python3
"""
pool_scorecards.py - combine the per-lead scorecards of one model into a single
multi-lead scorecard (CorrDiff_Design.md section 3.9). CPU only, no GPU, seconds.

open_split() takes one shard, so a lead-conditioned model is always evaluated as
four independent jobs and always produces four JSONs. The multi-lead numbers the
dissertation reports have to come from those four, and they have to be pooled
EXACTLY. Averaging four ratios is not pooling and is not done anywhere here:

    MAE, RMSE, bias        recombine through n_pixels
    POD/FAR/CSI/freq_bias  recombine from the summed [H, M, F] contingency counts
    FSS                    recombines as 1 - sum(num)/sum(den)
    CRPS, spread/RMSE      recombine through CRPS_n_pixels and spread_n_pixels
    PSD arrays             recombine through n_psd, then band ratios are recomputed
    rank histogram         recombines through rank_n; reliability counts add
    rain-rate histogram    raw counts, adds directly

Pooling is refused, loudly, if the inputs disagree on git, vae_sha256, batch,
seed, members or steps: two scorecards produced by different code, a different
codec or a different noise seed are not the same experiment. --batch matters
because evaluate_diffusion seeds per crop-batch as seed + 7919*s0.

Works on both scorecard families: evaluate_diffusion.py's
diffusion_eval*_L*.json (which carry a probabilistic block) and
evaluate_deterministic.py's det_eval*_L*.json (which do not). The method list is
read from the files rather than assumed.

    python "Code/3 - Diffusion stage/pool_scorecards.py" \
        --eval-dir ~/dissertation_outputs/diffusion_corrdiff_v1/eval \
        --pattern "diffusion_eval*_L*.json"
    python "Code/3 - Diffusion stage/pool_scorecards.py" \
        --eval-dir ~/dissertation_outputs/regression_delta_ep17/eval \
        --pattern "det_eval*_L*.json"

Outputs: <out>.json and <out>.md (default <eval-dir>/pooled_scorecard).
"""
import os, sys, json, glob, time, socket, argparse, subprocess

import numpy as np

# Numpy is this script's only hard dependency, deliberately: it is meant to run on
# a CPU box or a JASMIN sci node with no GPU and possibly no torch. The one shared
# function it wants (psd_band_metrics) lives in evaluate_diffusion, which imports
# torch and scipy, so it is imported LAZILY and the band table degrades to "not
# recomputed" rather than taking the whole script down. Thresholds and scales are
# read out of the input JSONs rather than imported, which is more robust anyway.
try:
    from train_ldm import git_hash
except ImportError:
    def git_hash():
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None


def _band_metrics(model, obs):
    try:
        from evaluate_diffusion import psd_band_metrics
    except ImportError:
        return None
    return psd_band_metrics(model, obs)


MUST_MATCH = ("git", "vae_sha256", "batch", "seed", "members", "steps",
              "guidance", "churn", "split", "ckpt", "field", "anchor")


def _t(d, t):
    """by_threshold keys survive a JSON round trip as strings."""
    return d[str(t)] if str(t) in d else d[t]


def thresholds_of(card, method):
    """Read the thresholds from the scorecard rather than importing them, so a
    change to THRESHOLDS upstream cannot make an old scorecard unpoolable."""
    return sorted(float(k) for k in card["deterministic"][method]["by_threshold"])


def scales_of(card):
    out = set()
    for k in card.get("fss", {}):
        parts = k.split("|")
        if len(parts) == 3:
            try:
                out.add(int(parts[2]))
            except ValueError:
                pass
    return sorted(out)


def check_consistency(cards, strict=True):
    bad = []
    for k in MUST_MATCH:
        vals = {json.dumps(c.get(k), sort_keys=True) for c in cards if k in c}
        if len(vals) > 1:
            bad.append(f"{k}: {sorted(vals)}")
    leads = [c.get("lead") for c in cards]
    if len(set(leads)) != len(leads):
        bad.append(f"duplicate leads: {leads}")
    if bad:
        msg = ("ERROR: these scorecards are not the same experiment and must not "
               "be pooled:\n  " + "\n  ".join(bad))
        if strict:
            raise SystemExit(msg + "\nPass --force to pool anyway (and say so in "
                                   "the write-up).")
        print(msg + "\n  --force given: pooling anyway.", flush=True)
    return bad


def pool_deterministic(cards, methods, thresholds):
    out = {}
    for m in methods:
        n = sum(c["deterministic"][m]["n_pixels"] for c in cards)
        sa = sum(c["deterministic"][m]["MAE_mmh"] * c["deterministic"][m]["n_pixels"]
                 for c in cards)
        ss = sum(c["deterministic"][m]["RMSE_mmh"] ** 2 *
                 c["deterministic"][m]["n_pixels"] for c in cards)
        se = sum(c["deterministic"][m]["bias_mmh"] * c["deterministic"][m]["n_pixels"]
                 for c in cards)
        e = {"MAE_mmh": sa / max(n, 1), "RMSE_mmh": (ss / max(n, 1)) ** 0.5,
             "bias_mmh": se / max(n, 1), "n_pixels": n, "by_threshold": {}}
        for t in thresholds:
            H = M = F = 0
            for c in cards:
                cnt = _t(c["deterministic"][m]["by_threshold"], t).get("counts")
                if cnt is None:
                    raise SystemExit(
                        "ERROR: this scorecard has no raw contingency counts, so "
                        "CSI cannot be pooled (only averaged, which is wrong). It "
                        "was produced before the design 3.6 edit landed. Re-score "
                        "with the current evaluate_*.py.")
                H += cnt[0]; M += cnt[1]; F += cnt[2]
            e["by_threshold"][str(t)] = {
                "POD": H / (H + M) if (H + M) else float("nan"),
                "FAR": F / (H + F) if (H + F) else float("nan"),
                "CSI": H / (H + M + F) if (H + M + F) else float("nan"),
                "freq_bias": (H + F) / (H + M) if (H + M) else float("nan"),
                "counts": [H, M, F]}
        out[m] = e
    return out


def pool_fss(cards):
    out, keys = {}, set()
    for c in cards:
        keys |= {k for k in c["fss"] if k.endswith("|num")}
    for k in sorted(keys):
        base = k[:-4]
        num = sum(c["fss"].get(base + "|num", 0.0) for c in cards)
        den = sum(c["fss"].get(base + "|den", 0.0) for c in cards)
        out[base] = (1 - num / den) if den > 0 else float("nan")
        out[base + "|num"], out[base + "|den"] = num, den
    if not out:
        print("WARNING: no FSS numerator/denominator keys found; these scorecards "
              "predate the design 3.6 edit, so FSS is omitted from the pool rather "
              "than averaged.", flush=True)
    return out


def pool_probabilistic(cards, M, thresholds):
    if not all("probabilistic" in c for c in cards):
        return None
    P = [c["probabilistic"] for c in cards]
    if any(p.get("CRPS_n_pixels") is None or p.get("spread_n_pixels") is None
           or p.get("rank_n") is None for p in P):
        raise SystemExit(
            "ERROR: a scorecard is missing CRPS_n_pixels / spread_n_pixels / "
            "rank_n, so the probabilistic block cannot be pooled exactly. Re-score "
            "with the current evaluate_diffusion.py (design 3.6).")
    cn = sum(p["CRPS_n_pixels"] for p in P)
    sn = sum(p["spread_n_pixels"] for p in P)
    rn = sum(p["rank_n"] for p in P)
    crps = sum(p["CRPS_fair_mmh"] * p["CRPS_n_pixels"] for p in P) / max(cn, 1)
    spread_sq = sum(p["spread_mmh"] ** 2 * p["spread_n_pixels"] for p in P) / max(sn, 1)
    err_sq = sum(p["rmse_ens_mean_mmh"] ** 2 * p["spread_n_pixels"]
                 for p in P) / max(sn, 1)
    rh = np.zeros(M + 1)
    for p in P:
        rh += np.array(p["rank_histogram"]) * p["rank_n"]
    rh = rh / max(rh.sum(), 1)
    rel = {}
    for t in thresholds:
        n = np.zeros(M + 1)
        o = np.zeros(M + 1)
        for p in P:
            r = _t(p["reliability"], t)
            n += np.array(r["n"])
            o += np.array(r["obs_freq"]) * np.array(r["n"])
        rel[str(t)] = {"prob": (np.arange(M + 1) / M).tolist(), "n": n.tolist(),
                       "obs_freq": (o / np.maximum(n, 1)).tolist()}
    return {"CRPS_fair_mmh": crps, "CRPS_n_pixels": cn, "spread_n_pixels": sn,
            "rank_n": rn,
            "CRPS_estimator": P[0].get("CRPS_estimator"),
            "spread_mmh": spread_sq ** 0.5,
            "rmse_ens_mean_mmh": err_sq ** 0.5,
            "spread_rmse_ratio": (spread_sq ** 0.5) / max(err_sq ** 0.5, 1e-9),
            "outlier_count": sum(p.get("outlier_count", 0) for p in P),
            "outlier_rate": sum(p.get("outlier_count", 0) for p in P) / max(rn, 1),
            "outlier_rate_ideal": 2.0 / (M + 1),
            "rank_histogram": rh.tolist(),
            "rank_flatness_rmse": float(np.sqrt(np.mean((rh - 1.0 / (M + 1)) ** 2))),
            "reliability": rel}


def pool_distribution(cards):
    D = [c["distribution"] for c in cards]
    keys = sorted({k for d in D for k in d["hist"]})
    hist = {k: np.sum([np.array(d["hist"][k]) for d in D if k in d["hist"]], axis=0)
            for k in keys}
    npsd = sum(d["n_psd"] for d in D)
    psd = {}
    for k in keys:
        parts = [(np.array(d["psd"][k]) * d["n_psd"]) for d in D
                 if d["psd"].get(k) is not None and d["n_psd"]]
        if not parts:
            psd[k] = None
            continue
        L = min(len(p) for p in parts)
        psd[k] = np.sum([p[:L] for p in parts], axis=0) / max(npsd, 1)
    wn = [d.get("wet_area_n") for d in D]
    if any(w is None for w in wn):
        wn = [c["n_crops"] for c in cards]     # pre-3.6 fallback, near-exact
    wet = {k: float(np.sum([d["wet_area"][k] * w for d, w in zip(D, wn)
                            if k in d["wet_area"]]) / max(sum(wn), 1))
           for k in keys}
    return {"rbins": D[0]["rbins"],
            "hist": {k: v.tolist() for k, v in hist.items()},
            "psd": {k: (None if v is None else v.tolist()) for k, v in psd.items()},
            "n_psd": npsd,
            "psd_bands": {k: _band_metrics(psd[k], psd.get("obs"))
                          for k in psd if k != "obs"},
            "wet_area": wet, "wet_area_n": sum(wn)}


def write_markdown(r, md):
    d = r["deterministic"]
    names = r["methods"]
    thresholds = sorted(float(k) for k in d[names[0]]["by_threshold"])
    scales = r.get("scales") or []
    hdr = "| metric | " + " | ".join(names) + " |"
    sep = "|---" * (len(names) + 1) + "|"
    L = [f"# Pooled multi-lead scorecard ({r['n_files']} leads: "
         f"{', '.join(str(x) for x in r['leads'])})\n",
         f"_{r['n_crops']} crops pooled from {r['n_files']} per-lead scorecards, "
         f"split `{r['split']}`, {r.get('members', 'n/a')} members, "
         f"{r.get('steps', 'n/a')} steps, batch {r.get('batch')}, seed "
         f"{r.get('seed')}. Source git `{r.get('src_git')}`, codec "
         f"`{str(r.get('vae_sha256'))[:12]}`._\n",
         "Pooling is exact throughout: pixel errors recombine through `n_pixels`, "
         "CSI and its relatives from summed contingency counts, FSS as "
         "`1 - sum(num)/sum(den)`, CRPS and spread through their pixel counts, PSD "
         "through `n_psd`. No ratio was averaged.\n",
         "## Pixel error\n", hdr, sep,
         "| MAE (mm/h) | " + " | ".join(f"{d[m]['MAE_mmh']:.4f}" for m in names) + " |",
         "| RMSE (mm/h) | " + " | ".join(f"{d[m]['RMSE_mmh']:.4f}" for m in names) + " |",
         "| bias (mm/h) | " + " | ".join(f"{d[m]['bias_mmh']:+.4f}" for m in names) + " |",
         "\n## CSI by threshold\n", "| mm/h | " + " | ".join(names) + " |", sep]
    for t in thresholds:
        L.append(f"| {t:g} | " + " | ".join(
            f"{_t(d[m]['by_threshold'], t)['CSI']:.4f}" for m in names) + " |")
    if r.get("probabilistic"):
        p = r["probabilistic"]
        L += ["\n## Probabilistic (pooled)\n",
              f"- Fair CRPS: **{p['CRPS_fair_mmh']:.4f}** mm/h over "
              f"{p['CRPS_n_pixels']:,} pixels. A deterministic forecast's CRPS is "
              f"its MAE, so compare against advection "
              f"{d['advection']['MAE_mmh']:.4f}.",
              f"- Spread / RMSE: {p['spread_rmse_ratio']:.4f} (spread "
              f"{p['spread_mmh']:.4f}, ens-mean RMSE "
              f"{p['rmse_ens_mean_mmh']:.4f}).",
              f"- Outlier rate: {p['outlier_rate']:.4f} vs ideal "
              f"{p['outlier_rate_ideal']:.4f}.",
              f"- Rank-histogram flatness (RMSE from flat): "
              f"{p['rank_flatness_rmse']:.5f}."]
    if r.get("fss") and scales:
        L += ["\n## FSS (pooled as 1 - sum(num)/sum(den))\n",
              "| field, threshold | " + " | ".join(f"{s} km" for s in scales) + " |",
              "|---" * (len(scales) + 1) + "|"]
        for key in sorted({k.split("|")[0] for k in r["fss"] if "|" in k}):
            for t in (1.0, 8.0):
                cells = [r["fss"].get(f"{key}|{t}|{s}") for s in scales]
                if any(c is None for c in cells):
                    continue
                L.append(f"| {key}, {t:g} mm/h | " +
                         " | ".join(f"{c:.3f}" for c in cells) + " |")
    bands = r["distribution"].get("psd_bands") or {}
    have = [m for m in names if bands.get(m)]
    if have:
        ref = bands[have[0]]["bands"]
        L += [f"\n## Power spectrum by band ({r['distribution']['n_psd']} crops)\n",
              "| field | " + " | ".join(b for b in ref) +
              " | 2-8 km band power | 2-8 km mean-of-ratios |",
              "|---" * (len(ref) + 3) + "|"]
        for m in have:
            b = bands[m]
            L.append(f"| {m} | " +
                     " | ".join(f"{b['bands'][k]['ratio']:.3f}" for k in ref) +
                     f" | {b['psd_band_power']:.3f} | {b['psd_mean_ratio']:.3f} |")
        L.append("| _obs share of variance_ | " +
                 " | ".join(f"{ref[k]['obs_share']*100:.1f}%" for k in ref) +
                 " | | |")
    L += ["\n## Per-lead inputs\n",
          "| lead | crops | " + " | ".join(f"MAE {m}" for m in names) + " |",
          "|---" * (len(names) + 2) + "|"]
    for c in r["per_lead"]:
        L.append(f"| {c['lead']} | {c['n_crops']} | " +
                 " | ".join(f"{c['MAE'][m]:.4f}" for m in names) + " |")
    open(md, "w").write("\n".join(L) + "\n")
    print("tables ->", md, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True,
                    help="directory holding the per-lead scorecard JSONs")
    ap.add_argument("--pattern", default="diffusion_eval*_L*.json")
    ap.add_argument("--out", default=None,
                    help="output basename; .json and .md are appended "
                         "(default <eval-dir>/pooled_scorecard)")
    ap.add_argument("--force", action="store_true",
                    help="pool even if the inputs disagree on git / codec / batch / "
                         "seed / members / steps. Say so in the write-up if used.")
    args = ap.parse_args()
    out = args.out or os.path.join(args.eval_dir, "pooled_scorecard")

    paths = sorted(glob.glob(os.path.join(args.eval_dir, args.pattern)))
    paths = [p for p in paths if not os.path.basename(p).startswith("pooled")]
    if len(paths) < 2:
        raise SystemExit(f"ERROR: found {len(paths)} scorecards matching "
                         f"{args.pattern!r} in {args.eval_dir}; nothing to pool.")
    cards = [json.load(open(p)) for p in paths]
    print(f"pooling {len(cards)} scorecards:")
    for p, c in zip(paths, cards):
        print(f"  {os.path.basename(p)}  lead {c.get('lead')}  "
              f"{c.get('n_crops')} crops  git {c.get('git')}")
    disagreements = check_consistency(cards, strict=not args.force)

    methods = [m for m in cards[0]["deterministic"]
               if all(m in c["deterministic"] for c in cards)]
    dropped = [m for m in cards[0]["deterministic"] if m not in methods]
    if dropped:
        print(f"WARNING: dropping {dropped}: not scored in every input.", flush=True)
    if not methods:
        raise SystemExit("ERROR: these scorecards share no scored method; they are "
                         "not the same kind of run and cannot be pooled.")
    M = cards[0].get("members")
    thresholds = thresholds_of(cards[0], methods[0])
    scales = scales_of(cards[0])

    r = {"pooled_from": [os.path.abspath(p) for p in paths],
         "thresholds": thresholds, "scales": scales,
         "n_files": len(cards),
         "leads": [c.get("lead") for c in cards],
         "n_crops": sum(c["n_crops"] for c in cards),
         "n_crops_available": sum(c.get("n_crops_available", c["n_crops"])
                                  for c in cards),
         "n_fss_crops": sum(c.get("n_fss_crops", 0) for c in cards),
         "methods": methods, "split": cards[0].get("split"),
         "members": M, "steps": cards[0].get("steps"),
         "guidance": cards[0].get("guidance"), "churn": cards[0].get("churn"),
         "batch": cards[0].get("batch"), "seed": cards[0].get("seed"),
         "ckpt": cards[0].get("ckpt"), "field": cards[0].get("field"),
         "vae_sha256": cards[0].get("vae_sha256"),
         "mu_dir": cards[0].get("mu_dir"),
         "reg_sha256": cards[0].get("reg_sha256"),
         "src_git": cards[0].get("git"), "pool_git": git_hash(),
         "host": socket.gethostname(), "argv": sys.argv,
         "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "forced": bool(args.force and disagreements),
         "disagreements": disagreements,
         "deterministic": pool_deterministic(cards, methods, thresholds),
         "fss": pool_fss(cards),
         "distribution": pool_distribution(cards),
         "per_lead": [{"lead": c.get("lead"), "n_crops": c["n_crops"],
                       "MAE": {m: c["deterministic"][m]["MAE_mmh"] for m in methods},
                       "CSI8": {m: _t(c["deterministic"][m]["by_threshold"],
                                      8.0)["CSI"] for m in methods}}
                      for c in cards]}
    prob = pool_probabilistic(cards, M, thresholds) if M else None
    if prob:
        r["probabilistic"] = prob

    tmp = out + ".json.tmp"
    with open(tmp, "w") as fh:
        json.dump(r, fh, indent=2)
    os.replace(tmp, out + ".json")
    write_markdown(r, out + ".md")

    d = r["deterministic"]
    print(f"\n=== pooled over {r['n_files']} leads, {r['n_crops']} crops ===")
    for m in methods:
        print(f"  {m:16s} MAE {d[m]['MAE_mmh']:.4f} | RMSE {d[m]['RMSE_mmh']:.4f} | "
              f"CSI@1 {_t(d[m]['by_threshold'], 1.0)['CSI']:.4f} | "
              f"CSI@8 {_t(d[m]['by_threshold'], 8.0)['CSI']:.4f}")
    if prob:
        print(f"  CRPS (fair)      {prob['CRPS_fair_mmh']:.4f} | spread/RMSE "
              f"{prob['spread_rmse_ratio']:.4f}")
    print(f"\npooled -> {out}.json", flush=True)


if __name__ == "__main__":
    main()
