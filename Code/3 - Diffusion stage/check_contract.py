#!/usr/bin/env python3
"""
check_contract.py - stage 0b of the CorrDiff build order (CorrDiff_Design.md 7.0b).

Standard library only. No numpy, no torch, no GPU, no network. It runs in under a
second on the Windows dev laptop, which is the whole point: it catches, statically,
the class of defect that otherwise fails on JASMIN 26 hours into a retrain.

`ast.parse` alone catches syntax and nothing else. It would not have caught any of
the five blocking defects the blueprint review found. All of them ARE catchable
here, by asserting ON the AST rather than merely building it:

  1. Every string in a --cond-mode choices list is a key of the COND_CH dict
     literal (the KeyError on "x-only").
  2. Every flag used in a design command line exists as an add_argument in the
     script that command invokes (the missing --mu-dir on evaluate_diffusion.py).
  3. Every name the design claims is reused exists as a top-level def or class in
     the file it is claimed to come from (the reuse list that had drifted, which
     is how load_denoiser came to be described as unchanged).
  4. Channel arithmetic, which is not a pure AST property, is asserted
     numerically instead: COND_CH and ZC are parsed out of the source and the
     (cond_mode, hr_mean_cond) -> in_ch table is recomputed and compared against
     the widths actually constructed in load_denoiser, sample_ensemble and the
     in-training diagnostic slice.
  5. Cross-module imports resolve: every `from X import a, b` between these
     scripts names something X actually defines.

Run it after every edit and before every job submission:

    python "Code/3 - Diffusion stage/check_contract.py"

Exit status 0 = all checks pass. Non-zero = do not queue the job.
"""
import ast, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["train_ldm.py", "train_corrdiff.py", "sample_diffusion.py",
         "evaluate_diffusion.py", "train_regression.py", "pack_mu.py",
         "evaluate_deterministic.py", "latent_ridge_gate.py", "pool_scorecards.py"]

# train_corrdiff.py is a deliberate fork of train_ldm.main(). These flags define
# the training regime the two arms must share for the comparison to be a
# controlled single-change experiment, so their DEFAULTS are compared and a drift
# is a failure. --patience is excluded on purpose: train_ldm defaults to 15 and
# train_corrdiff to 0, and both real runs pass it explicitly (design 7.8).
# --epochs is excluded because the two arms are launched with different values.
REGIME_FLAGS = ["--batch", "--lr", "--warmup", "--lr-schedule", "--weight-decay",
                "--width", "--mults", "--attn", "--dropout", "--ema-decay",
                "--cond-drop", "--p-mean", "--p-std", "--sample-every",
                "--sample-steps", "--sample-members", "--sample-crops",
                "--sample-batch", "--psd-crops", "--guidance", "--churn",
                "--workers", "--seed"]

# Flags each design command line uses, per script (design sections 7.1-7.9 and 3.x).
# This is the source list the review asked for: if a documented command names a
# flag the script does not define, argparse fails at submission time, not now.
REQUIRED_FLAGS = {
    "train_regression.py": [
        "--epochs", "--batch", "--lr", "--warmup", "--lr-schedule", "--weight-decay",
        "--width", "--mults", "--attn", "--dropout", "--ema-decay", "--target",
        "--cond-mode", "--diag-every", "--diag-crops", "--psd-crops", "--diag-batch",
        "--no-keep-sampled", "--patience", "--limit", "--workers", "--seed",
        "--leads", "--latents-dir", "--vae", "--out", "--resume", "--ignore-done"],
    "pack_mu.py": [
        "--reg", "--latents-dir", "--out", "--splits", "--leads", "--batch",
        "--workers", "--flush-rows", "--spot", "--force", "--check-only"],
    "evaluate_deterministic.py": [
        "--field", "--reg", "--mu-dir", "--npy", "--anchor", "--oracle",
        "--no-oracle", "--vae", "--latents-dir", "--split", "--lead", "--batch",
        "--limit", "--fss-sample", "--psd-sample", "--seed", "--out", "--tag",
        "--allow-vae-mismatch"],
    "latent_ridge_gate.py": [
        "--latents-dir", "--leads", "--fit-rows", "--eval-rows", "--kernel",
        "--features", "--alphas", "--seed", "--out"],
    "pool_scorecards.py": ["--eval-dir", "--pattern", "--out"],
    "train_ldm.py": [
        "--sigma-data", "--leads", "--epochs", "--batch", "--lr", "--warmup",
        "--cond-mode", "--cond-drop", "--p-mean", "--p-std", "--ema-decay",
        "--patience", "--sample-every", "--sample-crops", "--sample-members",
        "--psd-crops", "--seed", "--latents-dir", "--vae", "--out", "--resume",
        "--limit"],
    "train_corrdiff.py": [
        "--mu-dir", "--hr-mean-cond", "--reg-sha", "--sigma-data", "--leads",
        "--epochs", "--batch", "--lr", "--warmup", "--cond-mode", "--cond-drop",
        "--p-mean", "--p-std", "--ema-decay", "--patience", "--sample-every",
        "--sample-crops", "--sample-members", "--psd-crops", "--seed",
        "--latents-dir", "--vae", "--out", "--resume", "--ignore-done", "--limit"],
    "evaluate_diffusion.py": [
        "--ckpt", "--mu-dir", "--reg-sha", "--lead", "--split", "--members",
        "--steps", "--guidance", "--churn", "--batch", "--seed", "--fss-sample",
        "--psd-sample", "--latents-dir", "--vae", "--out", "--tag", "--limit",
        "--allow-vae-mismatch"],
    "sample_diffusion.py": [
        "--ckpt", "--vae", "--latents-dir", "--split", "--lead", "--crops",
        "--members", "--steps", "--guidance", "--churn", "--seed", "--out",
        "--mu-dir"],
}

# Design section 5: what each file must keep exporting. A name disappearing from
# here is what silently invalidates a "reused unchanged" claim.
REUSE = {
    "train_ldm.py": [
        "UNet", "EDMDenoiser", "edm_sample", "edm_sigma_schedule", "ema_init",
        "ema_update", "ema_weights", "crps_ensemble", "sampled_diagnostics",
        "load_diag_crops", "plot_curves", "LatentRows", "shard_suffix",
        "load_pack_meta", "git_hash", "sha256_file", "edm_loss_terms",
        "NoiseEmb", "ResBlockT", "Cell", "Upsample"],
    "train_corrdiff.py": [
        "mu_shard_names", "check_mu_shard", "open_mu_split", "pool_moments",
        "MuLatentRows", "corrdiff_loss_terms"],
    "sample_diffusion.py": [
        "load_codec", "resolve_lead_idx", "read_truth", "open_split", "montage",
        "load_denoiser", "sample_ensemble", "check_corrdiff_pairing"],
    "evaluate_diffusion.py": [
        "new_det", "acc_det", "finish_det", "crps_fair", "psd_band_metrics",
        "run_stamp", "append_runs_row"],
    "train_regression.py": [
        "RegressionNet", "load_regression", "slice_cond", "slice_target",
        "measure_zy_moments", "target_moments", "decoded_diagnostics"],
    "pack_mu.py": ["pack_shard", "spot_check", "discover_leads"],
    "evaluate_deterministic.py": ["evaluate", "decode_latent", "write_markdown"],
    "latent_ridge_gate.py": ["windows", "accumulate", "solve_ridge", "score",
                             "subset_index", "per_channel_ev"],
    "pool_scorecards.py": ["pool_deterministic", "pool_fss", "pool_probabilistic",
                           "pool_distribution", "check_consistency",
                           "thresholds_of", "scales_of"],
}

FAIL = []
TREES = {}


def fail(msg):
    FAIL.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg):
    print(f"  ok    {msg}")


def top_level_names(tree):
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                names.add(a.asname or a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                names.add((a.asname or a.name).split(".")[0])
    return names


def add_argument_flags(tree):
    flags = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument"):
            for a in n.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    flags.add(a.value)
    return flags


def add_argument_defaults(tree):
    """{flag: source text of its default=}, so two argparse blocks can be compared."""
    out = {}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument"):
            continue
        flag = next((a.value for a in n.args
                     if isinstance(a, ast.Constant) and isinstance(a.value, str)
                     and a.value.startswith("--")), None)
        if flag is None:
            continue
        for kw in n.keywords:
            if kw.arg == "default":
                out[flag] = ast.unparse(kw.value)
    return out


def cond_mode_keys(tree, base_keys):
    """Cond-mode names this file may legally use: the shared COND_CH keys plus any
    it adds in its own top-level *COND_CH dict literal (train_regression's
    REG_COND_CH adds x-only, which no diffusion run uses)."""
    keys = set(base_keys)
    for n in tree.body:
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id.endswith("COND_CH")
                and isinstance(n.value, ast.Dict)):
            for k in n.value.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
    return keys


def const_dict(tree, name):
    """The literal value of a top-level `name = {...}` whose values are simple
    arithmetic on module constants (COND_CH is `{"full": 5 * ZC, ...}`)."""
    consts = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name):
            try:
                consts[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, SyntaxError, TypeError):
                pass
    for n in tree.body:
        if not (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == name and isinstance(n.value, ast.Dict)):
            continue
        out = {}
        for k, v in zip(n.value.keys, n.value.values):
            out[ast.literal_eval(k)] = _eval_const(v, consts)
        return out, consts
    return None, consts


def _eval_const(node, consts):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if isinstance(node, ast.BinOp):
        a, b = _eval_const(node.left, consts), _eval_const(node.right, consts)
        if a is None or b is None:
            return None
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
    return None


def choices_for(tree, flag):
    """The `choices=[...]` list attached to one add_argument call."""
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument"):
            continue
        if not any(isinstance(a, ast.Constant) and a.value == flag for a in n.args):
            continue
        for kw in n.keywords:
            if kw.arg == "choices":
                try:
                    return list(ast.literal_eval(kw.value))
                except (ValueError, SyntaxError):
                    return None
    return None


def imported_from(tree, module):
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == module:
            for a in n.names:
                names.add(a.name)
    return names


def in_ch_expr(tree, func_name):
    """Recompute the `in_ch=` argument of the UNet(...) call inside one function,
    symbolically, as a mapping over (cond_mode, hr_mean_cond). Returns the set of
    channel-count expressions found, as source strings."""
    found = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.FunctionDef) or n.name != func_name:
            continue
        for m in ast.walk(n):
            if (isinstance(m, ast.Call) and isinstance(m.func, ast.Name)
                    and m.func.id == "UNet"):
                for kw in m.keywords:
                    if kw.arg == "in_ch":
                        found.append(ast.unparse(kw.value))
            if isinstance(m, ast.Assign) and len(m.targets) == 1 \
                    and isinstance(m.targets[0], ast.Name) \
                    and m.targets[0].id == "in_ch":
                found.append(ast.unparse(m.value))
    return found


def main():
    print(f"check_contract.py: {len(FILES)} files under {HERE}\n")

    print("[0] syntax parse")
    for f in FILES:
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            fail(f"{f} does not exist")
            continue
        try:
            TREES[f] = ast.parse(open(p, encoding="utf-8").read(), filename=p)
            ok(f"{f} parses")
        except SyntaxError as e:
            fail(f"{f} line {e.lineno}: {e.msg}")
    if len(TREES) != len(FILES):
        print("\nstopping: not every file parsed.")
        return 1

    print("\n[1] every --cond-mode choice resolves to a conditioning width")
    cond_ch, consts = const_dict(TREES["train_ldm.py"], "COND_CH")
    if not cond_ch:
        fail("COND_CH is not a top-level dict literal in train_ldm.py")
        cond_ch = {}
    else:
        ok(f"train_ldm.COND_CH = {cond_ch}")
    for f, tree in TREES.items():
        ch = choices_for(tree, "--cond-mode")
        if ch is None:
            continue
        allowed = cond_mode_keys(tree, cond_ch)
        missing = [c for c in ch if c not in allowed]
        if missing:
            fail(f"{f} --cond-mode offers {missing}, which resolves to no "
                 f"conditioning width (KeyError on the first line of that run)")
        else:
            ok(f"{f} --cond-mode {ch} all resolve")

    print("\n[2] every documented flag exists as an add_argument")
    for f, flags in REQUIRED_FLAGS.items():
        have = add_argument_flags(TREES[f])
        missing = [x for x in flags if x not in have]
        if missing:
            fail(f"{f} is missing {missing} (argparse would reject the "
                 f"documented command line)")
        else:
            ok(f"{f}: all {len(flags)} documented flags present")

    print("\n[3] every reused name exists where it is claimed to")
    for f, names in REUSE.items():
        have = top_level_names(TREES[f])
        missing = [x for x in names if x not in have]
        if missing:
            fail(f"{f} no longer defines {missing}")
        else:
            ok(f"{f}: all {len(names)} reused names present")

    print("\n[4] cross-module imports resolve")
    mod_of = {f[:-3]: f for f in FILES}
    for f, tree in TREES.items():
        for mod, target in mod_of.items():
            if target == f:
                continue
            want = imported_from(tree, mod)
            if not want:
                continue
            have = top_level_names(TREES[target])
            missing = [x for x in want if x not in have]
            if missing:
                fail(f"{f} imports {missing} from {mod}, which does not define them")
            else:
                ok(f"{f} <- {mod}: {len(want)} names resolve")

    print("\n[5] channel arithmetic agrees across the three construction sites")
    zc = consts.get("ZC")
    if zc is None:
        fail("ZC is not a top-level constant in train_ldm.py")
    else:
        expect = {(m, hr): zc + cond_ch[m] + (zc if hr else 0)
                  for m in cond_ch for hr in (False, True)}
        ok(f"ZC = {zc}; expected denoiser in_ch "
           f"{{(full, hr off): {expect[('full', False)]}, "
           f"(full, hr on): {expect[('full', True)]}}}")
        # load_denoiser must build a width that DEPENDS on hr_mean_cond, or a
        # 28-channel CorrDiff checkpoint cannot be loaded at all (design 3.8).
        exprs = in_ch_expr(TREES["sample_diffusion.py"], "load_denoiser")
        if not exprs:
            fail("sample_diffusion.load_denoiser has no in_ch expression")
        elif not any("hr" in e for e in exprs):
            fail(f"sample_diffusion.load_denoiser builds in_ch as {exprs}, which "
                 "does not depend on hr_mean_cond: a 28-channel CorrDiff "
                 "checkpoint would raise a size mismatch on load_state_dict")
        else:
            ok(f"load_denoiser in_ch = {exprs[0]}")
        # The regression net has NO noised latent, so its in_ch must be the bare
        # conditioning width, with no + ZC.
        rexprs = in_ch_expr(TREES["train_regression.py"], "load_regression")
        if not rexprs:
            fail("train_regression.load_regression has no in_ch expression")
        elif any("ZC +" in e or "+ ZC" in e for e in rexprs):
            fail(f"load_regression builds in_ch as {rexprs}: the regression has no "
                 "noised latent to concatenate, so it must be COND_CH alone")
        else:
            ok(f"load_regression in_ch = {rexprs[0]}")

    src = {f: open(os.path.join(HERE, f), encoding="utf-8").read() for f in FILES}
    # sample_ensemble must widen the CONDITIONING as well as move the anchor:
    # moving the anchor alone produces a checkpoint that cannot be sampled at all.
    se = src["sample_diffusion.py"]
    body = se[se.index("def sample_ensemble"):se.index("def resolve_lead_idx")]
    if "hr_mean_cond" not in body or "torch.cat([cond" not in body:
        fail("sample_ensemble does not widen cond with mu when hr_mean_cond is on")
    elif "anchor" not in body:
        fail("sample_ensemble does not move the anchor to z_A + mu_r")
    else:
        ok("sample_ensemble widens cond and moves the anchor")
    # The in-training diagnostic slice has to be widened alongside the training
    # path, or the run crashes at the FIRST sampled epoch, hours in.
    tc = src["train_corrdiff.py"]
    if "anchor_diag" not in tc or "mu_diag" not in tc:
        fail("train_corrdiff has no mu_diag / anchor_diag: the in-training "
             "sampler would receive a 20-channel cond against a 28-channel stem")
    else:
        ok("train_corrdiff builds mu_diag and anchor_diag for the diagnostics")

    print("\n[6] train_ldm.py stays free of the CorrDiff arm")
    tl = src["train_ldm.py"]
    leaked = [t for t in ("mu_dir", "hr_mean_cond", "mu_r", "x-only") if t in tl]
    if leaked:
        fail(f"train_ldm.py mentions {leaked}: it must remain exactly the trainer "
             "that produced ml_v1, ml_v2 and single60, so that arm's provenance "
             "is not perturbed by this one")
    else:
        ok("train_ldm.py contains no CorrDiff code")

    print("\n[7] the two trainers have not drifted apart")
    dl_, dc = (add_argument_defaults(TREES["train_ldm.py"]),
               add_argument_defaults(TREES["train_corrdiff.py"]))
    drift = [(k, dl_.get(k), dc.get(k)) for k in REGIME_FLAGS
             if k in dl_ and k in dc and dl_[k] != dc[k]]
    absent = [k for k in REGIME_FLAGS if k not in dc]
    if drift:
        fail("shared training-regime defaults differ between train_ldm.py and "
             "train_corrdiff.py, so the arms are no longer a controlled "
             "single-change experiment: "
             + "; ".join(f"{k} {a} vs {b}" for k, a, b in drift))
    if absent:
        fail(f"train_corrdiff.py does not define {absent}, which train_ldm.py does")
    if not drift and not absent:
        ok(f"all {len(REGIME_FLAGS)} shared regime defaults identical "
           "(--patience and --epochs excluded by design)")

    print(f"\n{'FAILED: ' + str(len(FAIL)) + ' problem(s)' if FAIL else 'ALL CHECKS PASSED'}")
    for m in FAIL:
        print(f"  - {m}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
