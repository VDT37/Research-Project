#!/usr/bin/env python3
"""
check_vae_multilead.py - decide, with evidence, whether the EXISTING VAE v2 codec
can be reused for the multi-lead stage, or whether it must be retrained on
prior_ml/.

The question. The codec was trained only on the +60 min cache: on y_mmh (real
radar observations) and A_mmh (advection extrapolated 4 steps, so noticeably
blurred). The multi-lead cache adds +15/+30/+45. What actually changes?

  y_mmh   a real radar observation at t0+L. Identically distributed at every
          lead, and already the codec's core training distribution.
  A_mmh   advection extrapolated L/15 steps. At +15 it is ONE step from the last
          real frame, so it is much sharper than the +60 fields the codec saw.

So the only genuine distribution shift is A becoming sharper at short leads, and
"sharper, like real radar" is exactly what the codec learned from y. The prior
expectation is therefore that reuse is fine, but that is an argument, not a
measurement. This script measures it.

Method: encode and decode a sample of crops with the frozen codec, per lead and
per field, and report codec fidelity (reconstruction error, small-scale power,
heavy-rain tail, threshold agreement). The +60 row is the reference, because that
is what the codec was trained on. If +15/+30/+45 match the +60 row, one codec
serves every lead and no retrain is needed.

Note this measures CODEC FIDELITY (reconstruction vs its own input), not forecast
skill. A is reconstructed against A, y against y. Advection being a poor forecast
of y is irrelevant here and is measured elsewhere.

    conda activate nowcast
    python "Code/2 - VAE Stage/check_vae_multilead.py" \
        --vae ~/dissertation_outputs/vae_v2/vae_best.pt \
        --root /scratch/$USER/dissertation/prior_ml --n 200

Outputs (to --out, default ~/dissertation_outputs/vae_v2/):
    multilead_check.json    the numbers
    multilead_check.png     histogram + PSD per lead, for the A field
    and a verdict table on stdout
"""
import os, glob, json, getpass, argparse, random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_vae_v2 import VAE, to_dbr, from_dbr, radial_psd, DBR_THRESH

USER  = getpass.getuser()
PRIOR_ML = f"/scratch/{USER}/dissertation/prior_ml"
OUT   = os.path.expanduser("~/dissertation_outputs/vae_v2")
CSI_T  = [1.0, 8.0]
TAIL_T = [16.0, 32.0]

# Acceptance bands, from docs/Analysis with VAE v1.md section 6 (the same gates
# used to accept the +60 codec).
BANDS = {"rec_l1": (0.0, 0.022), "psd_ratio_2_8km": (0.8, 1.2),
         "tail_ratio_16": (0.8, 1.2), "tail_ratio_32": (0.8, 1.2),
         "csi_8": (0.95, 1.01)}


def lead_of(path):
    b = os.path.basename(path)
    if "_L" in b:
        try:
            return int(b.rsplit("_L", 1)[-1].split(".")[0])
        except ValueError:
            return None
    return None


@torch.no_grad()
def codec_metrics(vae, files, field, mean, std, device, batch=16, max_psd=64):
    """Encode/decode `field` for these crops; compare reconstruction to input."""
    cont = {t: np.zeros(3) for t in CSI_T}
    tail = {t: np.zeros(2) for t in TAIL_T}
    l1_sum, l1_n = 0.0, 0
    psd_in = psd_rec = None
    n_psd = 0
    rbins = np.logspace(np.log10(DBR_THRESH), np.log10(128), 41)
    h_in, h_rec = np.zeros(40), np.zeros(40)

    for s in range(0, len(files), batch):
        chunk = files[s:s + batch]
        X, V = [], []
        for f in chunk:
            z = np.load(f, allow_pickle=True)
            R = z[field].astype("float32")
            X.append(to_dbr(R))
            V.append(z["valid"] & np.isfinite(R))
        Xn = (np.stack(X) - mean) / std
        t = torch.from_numpy(Xn)[:, None].to(device)
        mu, _ = vae.encode(t)
        rec = vae.decode(mu)[:, 0].float().cpu().numpy()
        Vs = np.stack(V)

        # masked L1 in NORMALISED dBR units: directly comparable to the trainer's
        # "val rec (plain)" number and its <= 0.022 acceptance gate
        d = np.abs(rec - Xn)
        l1_sum += float(d[Vs].sum()); l1_n += int(Vs.sum())

        R_in  = from_dbr(np.stack(X))
        R_rec = from_dbr(rec * std + mean)
        for i in range(len(chunk)):
            v = Vs[i]
            for t_ in CSI_T:
                o, p = (R_in[i] >= t_), (R_rec[i] >= t_)
                cont[t_] += [np.sum(v & o & p), np.sum(v & o & ~p), np.sum(v & ~o & p)]
            for t_ in TAIL_T:
                tail[t_] += [np.sum(v & (R_rec[i] >= t_)), np.sum(v & (R_in[i] >= t_))]
            a, b = R_in[i][v], R_rec[i][v]
            h_in  += np.histogram(a[a >= DBR_THRESH], bins=rbins)[0]
            h_rec += np.histogram(b[b >= DBR_THRESH], bins=rbins)[0]
            if v.mean() > 0.99 and n_psd < max_psd:
                pi, pr = radial_psd(R_in[i]), radial_psd(R_rec[i])
                psd_in  = pi if psd_in  is None else psd_in + pi
                psd_rec = pr if psd_rec is None else psd_rec + pr
                n_psd += 1

    out = {"rec_l1": l1_sum / max(l1_n, 1), "n_crops": len(files), "n_psd": n_psd}
    for t_ in CSI_T:
        H, M, F = cont[t_]
        out[f"csi_{t_:g}"] = float(H / (H + M + F)) if (H + M + F) > 0 else float("nan")
    for t_ in TAIL_T:
        r, o = tail[t_]
        out[f"tail_ratio_{t_:g}"] = float(r / o) if o > 0 else float("nan")
    if n_psd:
        k = np.arange(1, len(psd_in)); wl = 256.0 / k
        band = (wl >= 2.0) & (wl <= 8.0)
        out["psd_ratio_2_8km"] = float(np.mean(psd_rec[1:][band] /
                                               np.maximum(psd_in[1:][band], 1e-12)))
        out["_psd_in"], out["_psd_rec"] = psd_in / n_psd, psd_rec / n_psd
    else:
        out["psd_ratio_2_8km"] = float("nan")
    out["_hist_in"], out["_hist_rec"], out["_rbins"] = h_in, h_rec, rbins
    return out


def verdict(v, key):
    lo, hi = BANDS[key]
    if not np.isfinite(v):
        return "n/a"
    return "ok" if lo <= v <= hi else "OUT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default=os.path.join(OUT, "vae_best.pt"))
    ap.add_argument("--root", default=PRIOR_ML, help="multi-lead prior cache")
    ap.add_argument("--split", default="val", choices=["val", "train", "test"])
    ap.add_argument("--n", type=int, default=200, help="crops sampled per lead")
    ap.add_argument("--leads", default="15,30,45,60")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.vae):
        raise SystemExit(f"ERROR: VAE checkpoint not found: {args.vae}")
    ck = torch.load(args.vae, map_location=device)
    vae = VAE(w=ck["config"]["width"], zc=ck["config"]["zc"]).to(device)
    vae.load_state_dict(ck["model"]); vae.eval()
    mean, std = ck["norm"]["mean"], ck["norm"]["std"]
    print(f"codec: {args.vae} (epoch {ck.get('epoch')}) | norm mean {mean:.3f} "
          f"std {std:.3f} | device {device}", flush=True)

    files = sorted(glob.glob(os.path.join(args.root, args.split, "*", "*.npz")))
    if not files:
        raise SystemExit(f"ERROR: no crops under {os.path.join(args.root, args.split)}")
    by_lead = {}
    for f in files:
        by_lead.setdefault(lead_of(f), []).append(f)
    if set(by_lead) == {None}:
        raise SystemExit("ERROR: no lead-tagged crops found. Point --root at the "
                         "multi-lead cache (prior_ml), not the +60 prior.")
    leads = [int(x) for x in args.leads.split(",") if int(x) in by_lead]
    rng = random.Random(args.seed)
    print(f"{len(files)} {args.split} crops; leads found "
          f"{sorted(k for k in by_lead if k is not None)}\n", flush=True)

    res = {}
    for L in leads:
        sample = rng.sample(by_lead[L], min(args.n, len(by_lead[L])))
        for field in ("y_mmh", "A_mmh"):
            res[(L, field)] = codec_metrics(vae, sample, field, mean, std, device)
            print(f"  done lead +{L} {field}", flush=True)

    # ---- report ----
    hdr = (f"\n{'lead':>5} {'field':>6} | {'rec_l1':>8} {'psd2-8':>7} "
           f"{'tail16':>7} {'tail32':>7} {'csi1':>6} {'csi8':>6}")
    print(hdr); print("-" * len(hdr))
    for L in leads:
        for field in ("y_mmh", "A_mmh"):
            m = res[(L, field)]
            print(f"{L:>5} {field:>6} | {m['rec_l1']:>8.4f} "
                  f"{m['psd_ratio_2_8km']:>7.3f} {m['tail_ratio_16']:>7.3f} "
                  f"{m['tail_ratio_32']:>7.3f} {m['csi_1']:>6.3f} {m['csi_8']:>6.3f}")

    print("\nagainst the +60 acceptance bands (the codec's own training distribution):")
    worst = []
    for L in leads:
        for field in ("y_mmh", "A_mmh"):
            m = res[(L, field)]
            flags = {k: verdict(m.get(k, float("nan")), k) for k in BANDS}
            bad = [k for k, v in flags.items() if v == "OUT"]
            if bad:
                worst.append((L, field, bad))
            print(f"  +{L:>2} {field:>6}: " +
                  "  ".join(f"{k}={flags[k]}" for k in BANDS))

    ref = {f: res[(60, f)] for f in ("y_mmh", "A_mmh")} if 60 in leads else None
    print("\ndrift vs the +60 reference (what the codec was trained on):")
    if ref:
        for L in leads:
            for field in ("y_mmh", "A_mmh"):
                m, r = res[(L, field)], ref[field]
                print(f"  +{L:>2} {field:>6}: rec_l1 {m['rec_l1'] - r['rec_l1']:+.4f}  "
                      f"psd {m['psd_ratio_2_8km'] - r['psd_ratio_2_8km']:+.3f}  "
                      f"csi8 {m['csi_8'] - r['csi_8']:+.3f}")
    print("\nVERDICT: " + (
        "reuse the existing codec for the multi-lead stage; no retrain needed."
        if not worst else
        "at least one lead/field is outside the acceptance band -> retrain on prior_ml:\n  "
        + "\n  ".join(f"+{L} {f}: {', '.join(b)}" for L, f, b in worst)), flush=True)

    # ---- figure: the A field is the one that shifts, so plot it per lead ----
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    rb = res[(leads[0], "A_mmh")]["_rbins"]
    rc = np.sqrt(rb[:-1] * rb[1:])
    for i, L in enumerate(leads):
        m = res[(L, "A_mmh")]
        c = f"C{i}"
        hi = m["_hist_in"] / max(m["_hist_in"].sum(), 1)
        hr = m["_hist_rec"] / max(m["_hist_rec"].sum(), 1)
        ax[0].loglog(rc, hi, c, lw=2, label=f"+{L} input")
        ax[0].loglog(rc, hr, c, lw=1.2, ls="--", label=f"+{L} recon")
        if "_psd_in" in m:
            k = np.arange(1, len(m["_psd_in"])); wl = 256.0 / k
            ax[1].loglog(wl, m["_psd_in"][1:], c, lw=2, label=f"+{L} input")
            ax[1].loglog(wl, m["_psd_rec"][1:], c, lw=1.2, ls="--", label=f"+{L} recon")
    ax[0].set(title="Advection field A: rain-rate distribution",
              xlabel="mm/h", ylabel="frequency")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    ax[1].set(title="Advection field A: power spectrum (solid input, dashed recon)",
              xlabel="wavelength (km)", ylabel="power")
    ax[1].invert_xaxis(); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    png = os.path.join(args.out, "multilead_check.png")
    plt.tight_layout(); plt.savefig(png, dpi=120, bbox_inches="tight"); plt.close()
    print(f"\nfigure -> {png}")

    clean = {f"{L}|{f}": {k: v for k, v in res[(L, f)].items() if not k.startswith("_")}
             for L in leads for f in ("y_mmh", "A_mmh")}
    jp = os.path.join(args.out, "multilead_check.json")
    json.dump({"vae": os.path.abspath(args.vae), "vae_epoch": ck.get("epoch"),
               "split": args.split, "n_per_lead": args.n, "bands": BANDS,
               "metrics": clean}, open(jp, "w"), indent=2, default=float)
    print(f"json   -> {jp}")


if __name__ == "__main__":
    main()
