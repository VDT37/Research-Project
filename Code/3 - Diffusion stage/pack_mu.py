#!/usr/bin/env python3
"""
pack_mu.py - freeze the CorrDiff conditional mean into float16 memmaps that sit
alongside the latent packs (CorrDiff_Design.md section 3.2).

CorrDiff re-runs its regression network inside the loss on every training
iteration (physicsnemo legacy_losses.ResidualLoss.__call__ calls the net and then
forms y = y - self.y_mean, caching only across the patch-accumulation branches of
one batch). Here that would be pure waste: the regression inputs are a fixed
packed memmap that never changes and the network is frozen, so mu_r is a
deterministic function of a fixed table and is computed exactly once. That
removes roughly a third of the per-step cost of the 26-hour diffusion retrain
and, more importantly, removes any chance of the two stages drifting apart.

The frozen mean becomes a first-class artefact: hashed, spot-checked, and
carrying the residual statistics the diffusion trainer needs.

    {split}_mu{_LNN}.npy        float16 (n_rows, 4, 64, 64), row-aligned with
                                {split}_latents{_LNN}.npy by construction
    {split}_mu{_LNN}_meta.json  reg_sha256, latents_meta_sha, sigma_data_resid,
                                resid_moments (raw, so a lead-conditioned model
                                can pool ONE sigma_data exactly), ev

Chain of custody: pack_mu records reg_sha256; train_corrdiff records it
into its config and can assert it with --reg-sha; evaluate_diffusion --mu-dir
reads the same pack. Every mismatch is fatal, never a warning, because a CorrDiff
checkpoint evaluated against the wrong mu pack produces silently wrong fields.

    conda activate nowcast
    export DISS_SCRATCH=/work/scratch-pw4/$USER/dissertation
    # validate on the small split first: ~1 minute, catches every plumbing error
    python "Code/3 - Diffusion stage/pack_mu.py" \
        --reg ~/dissertation_outputs/regression_delta_ep17/reg_best.pt \
        --latents-dir $DISS_SCRATCH/latents_ml_ep17 \
        --out $DISS_SCRATCH/latents_ml_ep17_mu_delta \
        --splits val --leads 15,30,45,60 --batch 256
    # then the train split
    python "Code/3 - Diffusion stage/pack_mu.py" --reg ... --splits train,val ...

Cross-check afterwards: the printed per-shard `ev` must agree to three decimal
places with the val EV in the regression's train_log.json (different code, same
rows), and `evaluate_deterministic.py --field mu-pack` must reproduce
`--field regression` on the same crops.

Disk: 667,016 rows x 4 x 64 x 64 x 2 bytes = 21.86 GB (train 20.11, val 1.74).
Check the quota first: lfs quota -u $USER /work/scratch-pw4
"""
import os, sys, json, time, math, socket, getpass, argparse, contextlib

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from train_ldm import ZC, shard_suffix, load_pack_meta, git_hash, sha256_file
from train_corrdiff import mu_shard_names          # the on-disk format contract
from train_regression import load_regression, slice_cond, slice_target
from train_vae_v2 import atomic_json

USER    = getpass.getuser()
SCRATCH = os.environ.get("DISS_SCRATCH", f"/work/scratch-nopw2/{USER}/dissertation")
LATENTS = os.path.join(SCRATCH, "latents")


class PackRows(Dataset):
    """One latent shard, in order, as (conditioning, target) float32 tensors.

    Only the two slices the regression needs are materialised, so the DataLoader
    workers move 20+4 of the 24 channels rather than the whole row twice."""

    def __init__(self, path, cond_mode, target):
        self.path, self.cond_mode, self.target = path, cond_mode, target
        self.n = np.load(path, mmap_mode="r").shape[0]
        self._mm = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self._mm is None:
            self._mm = np.load(self.path, mmap_mode="r")
        row = torch.from_numpy(np.asarray(self._mm[i], dtype="float32"))[None]
        return slice_cond(row, self.cond_mode)[0], slice_target(row, self.target)[0]


@torch.no_grad()
def pack_shard(split, lead, args, net, ck, cond_mode, target, device):
    suf = shard_suffix(lead)
    lat_npy = os.path.join(args.latents_dir, f"{split}_latents{suf}.npy")
    lat_meta_p = os.path.join(args.latents_dir, f"{split}_latents{suf}_meta.json")
    if not os.path.exists(lat_npy):
        print(f"[{split}{suf}] no latent shard at {lat_npy}, skipping")
        return None
    lat_meta = load_pack_meta(args.latents_dir, split, lead)
    n_rows = np.load(lat_npy, mmap_mode="r").shape[0]
    npy, meta_p = mu_shard_names(args.out, split, lead)
    tag = f"{split}{suf}"

    if os.path.exists(npy) and os.path.exists(meta_p) and not args.force:
        try:
            meta = json.load(open(meta_p))
        except (json.JSONDecodeError, ValueError):
            print(f"[{tag}] {meta_p} is corrupt (interrupted pack?): repacking")
            meta = {}
        n_disk = np.load(npy, mmap_mode="r").shape[0]
        if (meta.get("n_files") == n_rows and n_disk == n_rows
                and meta.get("reg_sha256") == args._reg_sha):
            print(f"[{tag}] already packed ({n_rows} rows, same regression), "
                  "skipping (use --force to redo)")
            return meta
        print(f"[{tag}] existing pack is stale (meta n {meta.get('n_files')}, "
              f"disk {n_disk}, expected {n_rows}, reg match "
              f"{meta.get('reg_sha256') == args._reg_sha}): repacking")

    part = npy + ".part"
    gb = n_rows * ZC * 64 * 64 * 2 / 1e9
    print(f"[{tag}] {n_rows} rows -> ({n_rows}, {ZC}, 64, 64) float16 "
          f"({gb:.2f} GB) on {device} ...", flush=True)
    mm = np.lib.format.open_memmap(part, mode="w+", dtype="float16",
                                   shape=(n_rows, ZC, 64, 64))

    ds = PackRows(lat_npy, cond_mode, target)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                    persistent_workers=False,
                    prefetch_factor=4 if args.workers > 0 else None)
    lead_idx = None
    ck_leads = ck.get("leads") or (ck.get("config") or {}).get("leads")
    if ck_leads:
        if lead is None or lead not in ck_leads:
            raise SystemExit(f"ERROR: [{tag}] the regression is lead-conditioned on "
                             f"{ck_leads} but this shard is lead {lead}; the lead "
                             "embedding row would be wrong.")
        lead_idx = ck_leads.index(lead)

    r_sum = r_sumsq = d_sum = d_sumsq = 0.0
    count = 0
    i0, since_sync, t0 = 0, 0, time.time()
    use_amp = device == "cuda"
    for cond, tgt in dl:
        cond = cond.to(device, non_blocking=True)
        b = cond.shape[0]
        li = (torch.full((b,), lead_idx, dtype=torch.long, device=device)
              if lead_idx is not None else None)
        with (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp
              else contextlib.nullcontext()):
            mu = net(cond, li)
        mu16 = mu.float().cpu().numpy().astype("float16")
        mm[i0:i0 + b] = mu16
        # Accumulate against the float16 that was actually STORED, not against the
        # float32 forward: the trainer reads the rounded values, so sigma_data_resid
        # has to describe the target the trainer will really see.
        r = tgt.numpy().astype("float32") - mu16.astype("float32")
        d = tgt.numpy().astype("float32")
        r_sum += float(r.sum()); r_sumsq += float((r * r).sum())
        d_sum += float(d.sum()); d_sumsq += float((d * d).sum())
        count += int(r.size)
        i0 += b
        since_sync += b
        if args.flush_rows and since_sync >= args.flush_rows:
            mm.flush()
            since_sync = 0
        if i0 % (100 * args.batch) < args.batch:
            el = time.time() - t0
            print(f"  {i0}/{n_rows} rows | {i0/max(el,1e-9):.0f} rows/s | "
                  f"eta {(n_rows-i0)/max(i0/max(el,1e-9),1e-9)/60:.1f} min", flush=True)
    mm.flush()
    del mm
    if i0 != n_rows:
        raise SystemExit(f"ERROR: [{tag}] wrote {i0} rows, expected {n_rows}.")

    r_mean = r_sum / count
    d_mean = d_sum / count
    var_r = max(r_sumsq / count - r_mean ** 2, 0.0)
    var_d = max(d_sumsq / count - d_mean ** 2, 0.0)
    sq_r, sq_d = r_sumsq / count, d_sumsq / count
    sd_resid = float(math.sqrt(var_r))
    # EV against the target's SECOND MOMENT, which is the single definition the
    # whole arm uses (design 3.1) and what train_regression.py prints, so the two
    # are directly comparable. The variance-normalised form is kept alongside
    # because it is what an R^2 conventionally means, and the two differ by
    # mean(delta)^2, which is 0.0038 here and shifts EV by about 0.003.
    ev = float(1.0 - sq_r / sq_d) if sq_d > 0 else float("nan")
    ev_var = float(1.0 - var_r / var_d) if var_d > 0 else float("nan")
    print(f"[{tag}] sigma_data_resid {sd_resid:.4f} (target sd "
          f"{math.sqrt(var_d):.4f}) | EV {ev:.4f} (variance-normalised "
          f"{ev_var:.4f}) | resid mean {r_mean:+.5f} "
          f"| {time.time() - t0:.0f}s", flush=True)

    if not spot_check(part, lat_npy, net, cond_mode, lead_idx, device,
                      n=args.spot, seed=args.seed, atol=args.spot_atol):
        print(f"[{tag}] VERIFICATION FAILED: leaving {part} for inspection, "
              "pack NOT installed", flush=True)
        return None

    meta = {"n_files": n_rows, "n_channels": ZC, "h": 64, "w": 64,
            "dtype": "float16", "split": split, "lead_min": lead,
            "target": target, "cond_mode": cond_mode,
            "reg_ckpt": os.path.abspath(args.reg), "reg_sha256": args._reg_sha,
            "reg_epoch": ck.get("epoch"), "reg_val_mse": ck.get("val_mse"),
            "reg_config": ck.get("config"),
            "latents_dir": os.path.abspath(args.latents_dir),
            "latents_meta_sha": sha256_file(lat_meta_p),
            "latent_scale": lat_meta.get("latent_scale"),
            "vae_sha256": lat_meta.get("vae_sha256"),
            "sigma_data_resid": sd_resid,
            "resid_moments": {"sum": r_sum, "sumsq": r_sumsq, "count": count},
            "resid_mean": r_mean,
            "ev": ev, "ev_var": ev_var,
            "sigma_data_delta": lat_meta.get("sigma_data"),
            "sigma_data_target_measured": float(math.sqrt(var_d)),
            "git": git_hash(), "host": socket.gethostname(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv}
    # Install order copies pack_latents.py exactly: the npy is renamed into place
    # FIRST and the meta json (the record the skip-check trusts) LAST. A SIGKILL in
    # between leaves new-npy + stale-meta, which the staleness check repacks. The
    # reverse would leave new-meta + old-npy, which would silently pass.
    os.replace(part, npy)
    atomic_json(meta, meta_p)
    print(f"[{tag}] done -> {npy} (verified)", flush=True)
    return meta


@torch.no_grad()
def spot_check(mu_path, lat_path, net, cond_mode, lead_idx, device, n=8, seed=0,
               atol=6e-2):
    """Re-run the frozen regression on a random sample of rows and compare to the
    stored float16.

    The tolerance is NOT pack_latents.py's 2e-2. That figure covers an fp32
    encoder forward plus float16 storage rounding, which is tiny. This forward
    runs under bf16 autocast (an 8-bit mantissa, roughly 3.9e-3 relative) and the
    check re-runs single rows at batch 1 against values produced at batch 256, so
    it compares two equally valid bf16 evaluations that select different kernels
    and reduction orders. Observed spread on real data is 0.02 to 0.035 on
    latents of order 1. Real corruption (wrong rows, wrong network, wrong slice)
    produces O(1) differences, so 6e-2 still catches everything that matters
    while not failing on arithmetic noise. Float16 STORAGE contributes about
    1.7e-4 here and is not the issue.

    A useful tell if this ever fires for real: rounding noise makes the SAME rows
    fail across independent shards, because it scales with latent magnitude.
    Corruption hits different rows in each shard."""
    mu_mm = np.load(mu_path, mmap_mode="r")
    lat_mm = np.load(lat_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(mu_mm.shape[0], size=min(n, mu_mm.shape[0]),
                             replace=False))
    bad, worst = 0, 0.0
    for i in idx:
        row = torch.from_numpy(np.asarray(lat_mm[i:i + 1], dtype="float32"))
        cond = slice_cond(row, cond_mode).to(device)
        li = (torch.full((1,), lead_idx, dtype=torch.long, device=device)
              if lead_idx is not None else None)
        fresh = net(cond, li).float().cpu().numpy()[0]
        got = np.asarray(mu_mm[i], dtype="float32")
        d = float(np.max(np.abs(fresh - got)))
        worst = max(worst, d)
        if np.isnan(got).any() or d > atol:
            bad += 1
            print(f"  MISMATCH row {i}: max|diff| {d:.4f} "
                  f"nan {bool(np.isnan(got).any())}")
    print(f"  spot check: {len(idx) - bad}/{len(idx)} rows verified "
          f"(worst |diff| {worst:.5f}, tol {atol})")
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", required=True,
                    help="frozen regression checkpoint (reg_best.pt or ckpt_epNNN.pt)")
    ap.add_argument("--latents-dir", default=LATENTS)
    ap.add_argument("--out", default=None, help="default <latents-dir>_mu")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--leads", default=None,
                    help="e.g. 15,30,45,60; omit to pack every shard found")
    ap.add_argument("--batch", type=int, default=256,
                    help="rows per forward pass (forward-only, bf16)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--flush-rows", type=int, default=8192,
                    help="msync the output mapping every N rows, so dirty pages "
                         "stay bounded on Lustre (0 = only at the end)")
    ap.add_argument("--spot", type=int, default=8,
                    help="rows re-run through the net and compared")
    ap.add_argument("--spot-atol", type=float, default=6e-2,
                    help="spot-check tolerance. Sized for a bf16 forward compared "
                         "across batch sizes, not for fp32; see spot_check()")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="repack even if present")
    ap.add_argument("--check-only", action="store_true",
                    help="only run the spot checks against existing packs")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.abspath(args.latents_dir.rstrip("/\\")) + "_mu"
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found; the regression forward will be slow.", flush=True)
    torch.backends.cudnn.benchmark = False       # keep the forward reproducible

    net, ck, cond_mode, target, ck_leads = load_regression(args.reg, device)
    loss_kind = (ck.get("config") or {}).get("loss", "mse")
    if loss_kind != "mse":
        # L1 yields the conditional MEDIAN, and the entire two-stage argument
        # assumes the MEAN: var(delta - mu_r) <= var(delta) is the law of total
        # variance applied to a conditional mean, and EV, sigma_data_resid and
        # GATE-C's arithmetic all follow from it. Refuse rather than print a
        # number that silently means something else (design 4.2).
        raise SystemExit(f"ERROR: {args.reg} records loss={loss_kind!r}. Only a "
                         "plain-MSE regression is a conditional mean, and every "
                         "statistic this pack carries assumes one.")
    args._reg_sha = sha256_file(args.reg)
    print(f"frozen regression sha {args._reg_sha[:12]} | target={target} "
          f"cond={cond_mode} | out -> {args.out}", flush=True)

    leads = ([int(x) for x in args.leads.split(",") if x.strip()]
             if args.leads else None)
    metas = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        shard_leads = leads if leads else discover_leads(args.latents_dir, split)
        for L in shard_leads:
            if args.check_only:
                npy, _ = mu_shard_names(args.out, split, L)
                lat = os.path.join(args.latents_dir,
                                   f"{split}_latents{shard_suffix(L)}.npy")
                if not (os.path.exists(npy) and os.path.exists(lat)):
                    print(f"[{split}{shard_suffix(L)}] no pack to check")
                    continue
                li = ck_leads.index(L) if ck_leads and L in ck_leads else None
                spot_check(npy, lat, net, cond_mode, li, device, n=max(args.spot, 10),
                           seed=args.seed)
                continue
            m = pack_shard(split, L, args, net, ck, cond_mode, target, device)
            if m:
                metas[f"{split}{shard_suffix(L)}"] = m

    if metas:
        S = sum(m["resid_moments"]["sum"] for k, m in metas.items()
                if k.startswith("train"))
        SS = sum(m["resid_moments"]["sumsq"] for k, m in metas.items()
                 if k.startswith("train"))
        N = sum(m["resid_moments"]["count"] for k, m in metas.items()
                if k.startswith("train"))
        if N:
            sd = math.sqrt(max(SS / N - (S / N) ** 2, 0.0))
            print(f"\npooled over the packed TRAIN shards: sigma_data_resid "
                  f"{sd:.4f}. This is the number train_corrdiff.py will "
                  f"read; the EDM weighted loss should start near 1.0 with it.",
                  flush=True)
        print("per shard: " + ", ".join(
            f"{k} EV {m['ev']:.4f} sd' {m['sigma_data_resid']:.4f}"
            for k, m in sorted(metas.items())), flush=True)


def discover_leads(latents_dir, split):
    """Lead shards actually present for this split. [None] for a legacy
    single-lead pack, so the +60-only layout still works."""
    import glob as _glob
    found = []
    for p in sorted(_glob.glob(os.path.join(latents_dir, f"{split}_latents_L*.npy"))):
        base = os.path.basename(p)
        try:
            found.append(int(base.rsplit("_L", 1)[-1].split(".")[0]))
        except ValueError:
            continue
    if found:
        return sorted(found)
    if os.path.exists(os.path.join(latents_dir, f"{split}_latents.npy")):
        return [None]
    return []


if __name__ == "__main__":
    main()
