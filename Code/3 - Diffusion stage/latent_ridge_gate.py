#!/usr/bin/env python3
"""
latent_ridge_gate.py - the zero-GPU lower bound on the predictable fraction of
the latent advection residual (CorrDiff_Design.md sections 3.5 and 7.1).

This is the cheapest informative experiment in the CorrDiff arm and it runs
before a single GPU is requested. The packed latents already contain z_x1..z_x4,
z_A and z_y for every crop, so a closed-form linear regression of
delta = z_y - z_A on the conditioning latents puts a floor under EV in minutes,
on CPU, with no Slurm allocation.

What is regressed. Per lead, every latent cell is a sample. The target is the
4-vector delta[:, i, j]. The features are the C conditioning channels over a KxK
neighbourhood of cell (i, j), flattened, plus a bias: for the full 20-channel
stack at K=3 that is 20*9 + 1 = 181 features. Weights are shared across cells,
which makes the estimator EXACTLY a single KxK convolution with 181*4 = 724
parameters, solved by ridge rather than by gradient descent. Sharing is the right
constraint: the field is approximately translation-invariant, and it is what
makes the problem small enough to solve exactly.

Three defensible uses, and one indefensible one.
  - Plumbing check. If EV is exactly 0, negative or NaN, the slicing or the
    memmap alignment is wrong and no GPU should be spent until it is fixed. This
    alone justifies the stage.
  - A floor for ABORT-B. After one epoch the 65.7M-parameter UNet must beat a
    724-parameter closed-form linear filter, or it has a training problem rather
    than a data problem.
  - A reportable result: "a closed-form linear predictor recovers EV_ridge of the
    latent advection residual, rising from X at +15 to Y at +60" is a genuine
    measurement of how much structure the pysteps prior leaves behind, at zero
    GPU cost.
  - NOT an estimate of what the UNet will reach. EV_ridge is a LOWER bound: a
    linear KxK filter has no nonlinearity and a 4K-km receptive field, where the
    UNet is nonlinear with global attention at resolution 16. A small EV_ridge
    does not predict failure.

Two EV definitions are emitted, because the two consumers normalise differently:
  ev_eval      1 - SSE / sum((delta - mean(delta))^2)   variance-normalised, the
               classical R^2, which is how design 7.1 writes it
  ev_eval_sm   1 - SSE / sum(delta^2)                   second-moment normalised,
               which is what train_regression.py reports, so the ABORT-B
               comparison is like for like
They differ only by the target's mean, which is small here, but stating both
removes the ambiguity rather than leaving it to be discovered later.

The a-only and x-only feature sets come almost free: they are submatrices of the
same accumulation, so one pass over the data answers all three.

    conda activate nowcast
    export DISS_SCRATCH=/work/scratch-pw4/$USER/dissertation
    python "Code/3 - Diffusion stage/latent_ridge_gate.py" \
        --latents-dir $DISS_SCRATCH/latents_ml_ep17 \
        --leads 15,30,45,60 --kernel 3 --fit-rows 20000 --eval-rows 5000 \
        --features full --alphas 1e-4,1e-3,1e-2,1e-1,1,10 \
        --out ~/dissertation_outputs/regression/ridge_gate.json

--kernel is one integer per invocation. Run it at 1, 3 and 5 into three --out
files: if EV barely moves from 1 to 5 cells the predictable part is local and a
deep UNet will not add much, and if it climbs steeply it will.

Cost: 20,000 train rows plus 5,000 val rows is 3.93 + 0.98 GB of sequential
memmap IO per lead, and 4096 * 181^2 * 2 = 268 MFLOP per row, so about 5.4 TFLOP
per lead at K=3. Realistically 5 to 15 minutes per lead including IO, under an
hour for four leads on a 64-core box. No GPU, no Slurm job.
"""
import os, sys, json, time, socket, getpass, argparse

import numpy as np

from train_ldm import ZC, shard_suffix, load_pack_meta, git_hash

USER    = getpass.getuser()
SCRATCH = os.environ.get("DISS_SCRATCH", f"/work/scratch-nopw2/{USER}/dissertation")
LATENTS = os.path.join(SCRATCH, "latents")
OUT     = os.path.expanduser("~/dissertation_outputs/regression/ridge_gate.json")

# Conditioning channel ranges in the 24-channel pack row, so a feature subset is
# a submatrix selection rather than a second pass over 5 GB of latents.
FEATURE_SETS = {"full":   (0, 5 * ZC),      # z_x1..z_x4, z_A
                "x-only": (0, 4 * ZC),      # z_x1..z_x4
                "a-only": (4 * ZC, 5 * ZC)}  # z_A


def strided_rows(n, k, seed=0):
    """k row indices spread over the WHOLE shard. Pack row order is chronological
    (sorted(glob(prior_ml/{split}/YYYYMMDD/*.npz))), so taking the first k rows
    would sample one season. A deterministic stride needs no seed and mirrors what
    every other script in this project now does."""
    if k >= n:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, k).round().astype(np.int64))


def windows(cond, K):
    """(C, 64, 64) -> (4096, C*K*K) neighbourhood features, zero-padded.

    Zero padding, not edge padding, so the estimator is exactly the linear part
    of an nn.Conv2d(padding=K//2) and the comparison with the UNet is honest."""
    C, H, W = cond.shape
    p = K // 2
    if p:
        cond = np.pad(cond, ((0, 0), (p, p), (p, p)))
    win = np.lib.stride_tricks.sliding_window_view(cond, (K, K), axis=(1, 2))
    # (C, H, W, K, K) -> (H, W, C, K, K) -> (H*W, C*K*K)
    return np.ascontiguousarray(win.transpose(1, 2, 0, 3, 4)).reshape(H * W, C * K * K)


def accumulate(path, rows, K, c_lo, c_hi, block=16):
    """Stream rows and accumulate XtX, XtY and the target moments.

    Blocked so BLAS sees one (block*4096, F) gemm instead of 4096-row ones, and
    accumulated in float64 because 625 blocks of float32 gemm would drift."""
    mm = np.load(path, mmap_mode="r")
    C = c_hi - c_lo
    F = C * K * K + 1
    XtX = np.zeros((F, F))
    XtY = np.zeros((F, ZC))
    y_sum = np.zeros(ZC)
    y_sq = np.zeros(ZC)
    n_cells = 0
    for s in range(0, len(rows), block):
        sel = rows[s:s + block]
        raw = np.asarray(mm[sel], dtype="float32")
        cond = raw[:, c_lo:c_hi]
        tgt = raw[:, 5 * ZC:6 * ZC] - raw[:, 4 * ZC:5 * ZC]     # delta
        Xs = np.empty((len(sel) * 4096, F))
        Ys = np.empty((len(sel) * 4096, ZC))
        for b in range(len(sel)):
            Xs[b * 4096:(b + 1) * 4096, :F - 1] = windows(cond[b], K)
            Ys[b * 4096:(b + 1) * 4096] = tgt[b].reshape(ZC, -1).T
        Xs[:, F - 1] = 1.0                                      # bias column
        XtX += Xs.T @ Xs
        XtY += Xs.T @ Ys
        y_sum += Ys.sum(axis=0)
        y_sq += (Ys * Ys).sum(axis=0)
        n_cells += Ys.shape[0]
    return {"XtX": XtX, "XtY": XtY, "y_sum": y_sum, "y_sq": y_sq, "n": n_cells}


def subset_index(K, c_lo, c_hi, sub_lo, sub_hi):
    """Feature indices of a channel sub-range within the accumulated matrices,
    plus the bias. Channel c contributes offsets c*K*K .. (c+1)*K*K."""
    F = (c_hi - c_lo) * K * K + 1
    idx = []
    for c in range(sub_lo - c_lo, sub_hi - c_lo):
        idx.extend(range(c * K * K, (c + 1) * K * K))
    idx.append(F - 1)
    return np.array(idx, dtype=np.int64)


def solve_ridge(XtX, XtY, alpha):
    """(XtX + alpha I) W = XtY with the BIAS column left unpenalised: penalising
    the intercept would shrink the target's mean toward zero and show up as a
    spurious loss of explained variance."""
    A = XtX.copy()
    F = A.shape[0]
    A[np.arange(F - 1), np.arange(F - 1)] += alpha
    return np.linalg.solve(A, XtY)


def score(acc, W, idx=None):
    """SSE and the two EV normalisers, all from the accumulated moments: no
    second pass over the data is needed because
        SSE = sum(y^2) - 2 tr(W^T XtY) + tr(W^T XtX W)."""
    XtX = acc["XtX"] if idx is None else acc["XtX"][np.ix_(idx, idx)]
    XtY = acc["XtY"] if idx is None else acc["XtY"][idx]
    sse = float(acc["y_sq"].sum() - 2 * np.sum(W * XtY) + np.sum(W * (XtX @ W)))
    n = acc["n"]
    sq = float(acc["y_sq"].sum())                           # sum(y^2)
    var = float(sq - (acc["y_sum"] ** 2).sum() / n)         # sum((y - mean)^2)
    return {"sse": sse, "ev_var": 1 - sse / max(var, 1e-12),
            "ev_sm": 1 - sse / max(sq, 1e-12)}


def per_channel_ev(acc, W, idx=None):
    """EV broken down by latent channel, which is where a dead channel shows up."""
    XtX = acc["XtX"] if idx is None else acc["XtX"][np.ix_(idx, idx)]
    XtY = acc["XtY"] if idx is None else acc["XtY"][idx]
    out = []
    for c in range(ZC):
        w = W[:, c]
        sse = float(acc["y_sq"][c] - 2 * w @ XtY[:, c] + w @ (XtX @ w))
        sq = float(acc["y_sq"][c])
        out.append(1 - sse / max(sq, 1e-12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", default=LATENTS)
    ap.add_argument("--leads", default="15,30,45,60")
    ap.add_argument("--fit-rows", type=int, default=20000,
                    help="train rows sampled by stride")
    ap.add_argument("--eval-rows", type=int, default=5000,
                    help="held-out val rows sampled by stride")
    ap.add_argument("--kernel", type=int, default=3,
                    help="receptive field in latent cells: 1, 3 or 5")
    ap.add_argument("--features", choices=["full", "a-only", "x-only"],
                    default="full",
                    help="full also reports the x-only and a-only submatrices, "
                         "which cost nothing extra")
    ap.add_argument("--alphas", default="1e-4,1e-3,1e-2,1e-1,1,10")
    ap.add_argument("--block", type=int, default=16,
                    help="rows per BLAS block (memory ~ block * 4096 * F * 8 B)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    K = args.kernel
    if K % 2 == 0 or K < 1:
        raise SystemExit(f"ERROR: --kernel must be odd and >= 1, got {K}.")
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    leads = [int(x) for x in args.leads.split(",") if x.strip()] or [None]
    c_lo, c_hi = FEATURE_SETS[args.features]
    subsets = (["full", "x-only", "a-only"] if args.features == "full"
               else [args.features])

    result = {"kernel": K, "features": args.features, "alphas": alphas,
              "fit_rows": args.fit_rows, "eval_rows": args.eval_rows,
              "latents_dir": os.path.abspath(args.latents_dir),
              "git": git_hash(), "host": socket.gethostname(),
              "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv,
              "by_lead": {}}

    for L in leads:
        suf = shard_suffix(L)
        tr_p = os.path.join(args.latents_dir, f"train_latents{suf}.npy")
        va_p = os.path.join(args.latents_dir, f"val_latents{suf}.npy")
        if not (os.path.exists(tr_p) and os.path.exists(va_p)):
            print(f"[+{L}min] shard missing ({tr_p}); skipping", flush=True)
            continue
        tr_meta = load_pack_meta(args.latents_dir, "train", L)
        n_tr = np.load(tr_p, mmap_mode="r").shape[0]
        n_va = np.load(va_p, mmap_mode="r").shape[0]
        rows_tr = strided_rows(n_tr, args.fit_rows)
        rows_va = strided_rows(n_va, args.eval_rows)
        t0 = time.time()
        print(f"[+{L}min] accumulating {len(rows_tr)} train rows "
              f"({(c_hi - c_lo)}ch x {K}x{K} = {(c_hi-c_lo)*K*K + 1} features) ...",
              flush=True)
        A_tr = accumulate(tr_p, rows_tr, K, c_lo, c_hi, args.block)
        A_va = accumulate(va_p, rows_va, K, c_lo, c_hi, args.block)
        print(f"[+{L}min] accumulated in {time.time() - t0:.0f}s "
              f"({A_tr['n']:,} train cells, {A_va['n']:,} eval cells)", flush=True)

        entry = {"n_train_rows": int(len(rows_tr)), "n_eval_rows": int(len(rows_va)),
                 "n_train_cells": A_tr["n"], "n_eval_cells": A_va["n"],
                 "sigma_data_pack": tr_meta.get("sigma_data")}
        for sub in subsets:
            s_lo, s_hi = FEATURE_SETS[sub]
            idx = (None if (s_lo, s_hi) == (c_lo, c_hi)
                   else subset_index(K, c_lo, c_hi, s_lo, s_hi))
            XtX = A_tr["XtX"] if idx is None else A_tr["XtX"][np.ix_(idx, idx)]
            XtY = A_tr["XtY"] if idx is None else A_tr["XtY"][idx]
            best = None
            scan = []
            for a in alphas:
                try:
                    W = solve_ridge(XtX, XtY, a)
                except np.linalg.LinAlgError:
                    print(f"  alpha {a:g}: singular system, skipped", flush=True)
                    continue
                fit = score(A_tr, W, idx)
                ev = score(A_va, W, idx)
                scan.append({"alpha": a, "ev_fit": fit["ev_var"],
                             "ev_eval": ev["ev_var"], "ev_eval_sm": ev["ev_sm"]})
                # Selected on the HELD-OUT rows, never on the fit rows: a ridge
                # scan chosen on its own training data always picks alpha -> 0.
                if best is None or ev["ev_var"] > best["ev_eval"]:
                    best = {"alpha": a, "ev_fit": fit["ev_var"],
                            "ev_fit_sm": fit["ev_sm"], "ev_eval": ev["ev_var"],
                            "ev_eval_sm": ev["ev_sm"],
                            "ev_eval_per_channel": per_channel_ev(A_va, W, idx),
                            "n_features": int(XtX.shape[0])}
            if best is None:
                continue
            best["scan"] = scan
            entry[sub] = best
            print(f"  [+{L}min] {sub:7s} K={K} best alpha {best['alpha']:g} | "
                  f"EV fit {best['ev_fit']:.4f} | EV held-out {best['ev_eval']:.4f} "
                  f"(second-moment {best['ev_eval_sm']:.4f}) | per channel "
                  + ", ".join(f"{v:.3f}" for v in best["ev_eval_per_channel"]),
                  flush=True)
        result["by_lead"][str(L)] = entry

    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=2)
    os.replace(tmp, args.out)
    print(f"\nridge gate -> {args.out}", flush=True)

    print("\nEV_ridge is a LOWER bound on what R_phi can reach, not an estimate of "
          "it. Use it three ways: as a plumbing check (exactly 0, negative or NaN "
          "means the slicing is wrong), as the ABORT-B floor after epoch 1, and as "
          "a result in its own right. Compare it against train_regression.py's val "
          "EV using ev_eval_sm, which is the same normaliser.", flush=True)
    for L, e in result["by_lead"].items():
        row = e.get("full") or e.get(args.features)
        if row:
            print(f"  +{L}min  EV_ridge(held out) = {row['ev_eval']:.4f} "
                  f"[sm {row['ev_eval_sm']:.4f}]", flush=True)


if __name__ == "__main__":
    main()
