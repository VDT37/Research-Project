#!/usr/bin/env python3
"""
hpo_spaces.py - declarative search spaces, fidelity rungs and the cost model for
the hyperparameter optimisation of both diffusion arms.

This file holds no logic that runs anything. It is the single place where "what
is being searched, over what range, at what fidelity, for what estimated cost"
is written down, so that the study specification can be committed to git before
any GPU hour is spent and quoted verbatim in the methods chapter. hpo_search.py
imports it, freezes the chosen space into the study directory, and executes.

Pure standard library plus nothing else, so it imports on a login node, on the
Windows dev machine, and inside a Slurm job identically.


WHY THIS EXISTS, AND WHY THE NAIVE GRID IS NOT RUN
--------------------------------------------------
The supervisor's protocol is a coarse grid over the main parameters first, then
a more efficient search (random, Bayesian, Optuna/Hyperband), with every run
compared against persistence and pysteps/advection on the same validation set.

Taken literally at full training fidelity that protocol is unaffordable here.
One full multi-lead training cell is 50 epochs over 613,892 rows at a measured
326 img/s, which is 26.2 A100-hours. The eight-cell lr x batch x conditioning
grid originally written into docs/LDM.md section 7 is therefore 209 A100-hours,
2.6 times the project's entire committed GPU budget of 80.6 hours.

The resolution is not to skip the search, and not to shrink it to a token two
cells. It is to run the search at REDUCED FIDELITY and confirm only the winner
at full fidelity. That is exactly what Hyperband and successive halving do, and
it is the reason the supervisor named them. The arithmetic:

    full-fidelity cell   613,892 rows x 50 epochs = 30.7 M row-epochs = 26.2 h
    rung 0 screening      60,000 rows x  3 epochs =  0.18 M           =  0.15 h
    rung 1 refinement    150,000 rows x  4 epochs =  0.60 M           =  0.51 h
    rung 2 confirmation  613,892 rows x  6 epochs =  3.68 M           =  3.14 h

so a 24-cell screen costs 3.7 A100-hours at rung 0 rather than 629, and a full
three-rung successive-halving pass over the same 24 cells costs about 17. The
same scientific question is answered for roughly 8 percent of the naive cost.

The assumption this buys its saving with is that the RANK ORDER of configurations
at low fidelity predicts the rank order at high fidelity. That assumption is not
asserted here, it is measured: hpo_report.py computes the Spearman rank
correlation between consecutive rungs on the trials that were promoted, and that
number is the honest defence of the method in the write-up. If it comes out low,
the correct conclusion is reported (the screen did not transfer), which is still
a result.


THE SUPERVISOR'S PARAMETER LIST, MAPPED
---------------------------------------
Every parameter named in the brief, and where it lives:

  learning rate          --lr                        searched, stage 1 and 2
  batch size             --batch                     searched, stage 1
  latent dimension       FIXED at 4 x 64 x 64        see note below
  number of diffusion    --steps  (evaluate_*.py)    searched, stage 4, inference
    steps                                            only, costs no training
  noise schedule         --p-mean / --p-std          searched, stage 1 and 2
                         (EDM training noise dist.)
                         --churn  (sampler)          searched, stage 4
  UNet depth             --mults                     searched, stage 1
  UNet width             --width                     searched, stage 1 and 2
  dropout                --dropout                   searched, stage 1 and 2
  weight decay           --weight-decay              searched, stage 1 and 2
  conditioning method    --cond-mode / --cond-drop   searched, stage 1
                         --hr-mean-cond (CorrDiff)   searched, CorrDiff arm

Latent dimension is the one entry that cannot be searched inside this harness
and the write-up must say so plainly. The latent geometry is set by the frozen
VAE codec: pack_latents.py encodes a 256x256 km crop to 4 x 64 x 64, so changing
it means retraining the codec (a separate multi-hour stage) and repacking 131 GB
of latents per candidate. The project does however already own the relevant
ablation: the epoch-9 and epoch-17 codecs are two different latent
representations of the same data, they were compared head to head on identical
crops (ml_v1 against ml_v2), and epoch 17 won on every measured metric. That is
a latent-capacity comparison in everything but name, and it is what should be
cited in place of a latent-dimension sweep.


BASELINES
---------
The brief requires every run to be compared against persistence and
pysteps/advection on the same validation set. Two of those three come for free
and one had to be added:

  advection   the trainer's own sampled diagnostics already score the advection
              prior on the identical crops (mae_adv, crps_adv, csi_1_adv,
              csi_8_adv in train_log.json). This IS the pysteps baseline: A is
              a pysteps dense Lucas-Kanade flow plus Germann-Zawadzki
              extrapolation, computed in build_advection_prior.py.
  persistence hpo_baselines.py reproduces the trial's exact diagnostic crop
              indices and scores the last input frame held still on them, so the
              comparison is paired rather than approximately matched.
  climatology also emitted by hpo_baselines.py as a floor, not required by the
              brief but free once the crops are open.
"""

import math

# ----------------------------------------------------------------------------
# Arms
# ----------------------------------------------------------------------------
# Every arm names the script it drives, the flag that carries the output
# directory, and the flags a trial is not allowed to search because they define
# the experiment rather than the model. `required` flags must be supplied on the
# hpo_search.py command line via --pass-through; the harness refuses to plan a
# study without them rather than launching jobs that die on argparse.

ARMS = {
    "ldm": {
        "script": "train_ldm.py",
        "kind": "train",
        "out_flag": "--out",
        "required": [],
        "n_train_rows": 613892,      # multi-lead, 4 leads x ~153k
        "note": "residual LDM: learns delta = z_y - z_A. The ml_v2 arm.",
    },
    "corrdiff": {
        "script": "train_corrdiff.py",
        "kind": "train",
        "out_flag": "--out",
        "required": ["--mu-dir"],
        "n_train_rows": 613892,
        "note": "CorrDiff stage two: learns r' = delta - mu_r on the frozen mu pack.",
    },
    "regression": {
        "script": "train_regression.py",
        "kind": "train",
        "out_flag": "--out",
        "required": [],
        "n_train_rows": 613892,
        "note": "CorrDiff stage one: deterministic conditional mean of the residual.",
    },
    "inference": {
        "script": "evaluate_diffusion.py",
        "kind": "infer",
        "out_flag": "--out",
        "required": ["--ckpt"],
        "n_train_rows": 0,
        "note": "sampler-only sweep on a frozen checkpoint; no training cost.",
    },
}

# ----------------------------------------------------------------------------
# Parameter registry
# ----------------------------------------------------------------------------
# Each entry: the CLI flag, the type used for casting and for the TPE sampler,
# the incumbent value (ml_v2's config, the point every one-factor-at-a-time
# screen is measured against), and the supervisor's own name for it so the
# leaderboard and the methods chapter can be written straight off this table.

PARAMS = {
    "lr":            {"flag": "--lr",            "type": "logfloat", "incumbent": 1e-4,
                      "brief": "learning rate"},
    "batch":         {"flag": "--batch",         "type": "int",      "incumbent": 64,
                      "brief": "batch size"},
    "width":         {"flag": "--width",         "type": "int",      "incumbent": 128,
                      "brief": "UNet width"},
    "mults":         {"flag": "--mults",         "type": "cat",      "incumbent": "1,2,4",
                      "brief": "UNet depth"},
    "attn":          {"flag": "--attn",          "type": "cat",      "incumbent": "16",
                      "brief": "UNet attention resolutions"},
    "dropout":       {"flag": "--dropout",       "type": "float",    "incumbent": 0.0,
                      "brief": "dropout"},
    "weight_decay":  {"flag": "--weight-decay",  "type": "logfloat", "incumbent": 0.0,
                      "brief": "weight decay"},
    "p_mean":        {"flag": "--p-mean",        "type": "float",    "incumbent": -1.2,
                      "brief": "noise schedule (EDM log-sigma mean)"},
    "p_std":         {"flag": "--p-std",         "type": "float",    "incumbent": 1.2,
                      "brief": "noise schedule (EDM log-sigma std)"},
    "cond_mode":     {"flag": "--cond-mode",     "type": "cat",      "incumbent": "full",
                      "brief": "conditioning method"},
    "cond_drop":     {"flag": "--cond-drop",     "type": "float",    "incumbent": 0.1,
                      "brief": "conditioning method (CFG dropout rate)"},
    "ema_decay":     {"flag": "--ema-decay",     "type": "cat",      "incumbent": 0.999,
                      "brief": "EMA horizon"},
    "lr_schedule":   {"flag": "--lr-schedule",   "type": "cat",      "incumbent": "cosine",
                      "brief": "learning rate schedule"},
    "warmup":        {"flag": "--warmup",        "type": "int",      "incumbent": 1000,
                      "brief": "learning rate warmup"},
    # CorrDiff arm only
    "hr_mean_cond":  {"flag": "--hr-mean-cond",  "type": "cat",      "incumbent": "on",
                      "brief": "conditioning method (mu_r in the conditioning stack)"},
    # regression arm only
    "target":        {"flag": "--target",        "type": "cat",      "incumbent": "delta",
                      "brief": "regression target"},
    # inference arm only (no training cost)
    "steps":         {"flag": "--steps",         "type": "int",      "incumbent": 25,
                      "brief": "number of diffusion steps"},
    "members":       {"flag": "--members",       "type": "int",      "incumbent": 8,
                      "brief": "ensemble size"},
    "guidance":      {"flag": "--guidance",      "type": "float",    "incumbent": 1.0,
                      "brief": "classifier-free guidance weight"},
    "churn":         {"flag": "--churn",         "type": "float",    "incumbent": 0.0,
                      "brief": "noise schedule (sampler stochasticity S_churn)"},
}

INCUMBENT = {k: v["incumbent"] for k, v in PARAMS.items()}


# ----------------------------------------------------------------------------
# Constraints
# ----------------------------------------------------------------------------
def n_levels(mults):
    return len(str(mults).split(","))


def valid_attn(attn, mults):
    """Attention resolutions are feature-map sizes. The UNet starts at 64 and
    halves once per level, so `mults` of length L reaches 64 / 2**(L-1). An
    attention resolution below that is never constructed and would silently be
    a no-op, producing a duplicate of the no-attention cell under a different
    label. Reject it at planning time instead."""
    if attn in ("", None):
        return True
    lo = 64 // (2 ** (n_levels(mults) - 1))
    return all(int(r) >= lo and int(r) <= 64 and (64 % int(r) == 0)
               for r in str(attn).split(","))


def check_constraints(params):
    """Return a list of human-readable reasons this configuration is invalid.
    An empty list means it is fine. Invalid cells are logged and skipped, never
    silently dropped."""
    bad = []
    w = params.get("width", INCUMBENT["width"])
    if int(w) % 8 != 0:
        bad.append(f"width {w} is not divisible by 8 (GroupNorm assertion)")
    if not valid_attn(params.get("attn", INCUMBENT["attn"]),
                      params.get("mults", INCUMBENT["mults"])):
        bad.append(f"attn {params.get('attn')} unreachable with mults "
                   f"{params.get('mults')} (deepest resolution is "
                   f"{64 // 2 ** (n_levels(params.get('mults', INCUMBENT['mults'])) - 1)})")
    if float(params.get("cond_drop", 0.1)) > 0 and \
            params.get("cond_mode", "full") == "a-only" and \
            float(params.get("cond_drop", 0.1)) >= 0.5:
        bad.append("cond_drop >= 0.5 with cond-mode a-only drops almost all signal")
    return bad


# ----------------------------------------------------------------------------
# Fidelity rungs
# ----------------------------------------------------------------------------
# A rung is (rows, epochs). `--limit` on the trainers strides across the WHOLE
# concatenated index (train_ldm.py LatentRows.__init__), so a reduced-fidelity
# rung sees every lead and the whole date range rather than a chronological
# prefix. That property is what makes low-fidelity screening scientifically
# admissible here, and it is the same stride fix that had to be applied to
# evaluate_diffusion.py --limit for the same reason.
#
# warmup and sample_every are derived per rung rather than searched: a fixed
# --warmup 1000 would span the entire first epoch of a 937-step rung and turn
# the learning-rate comparison into a warmup comparison.

RUNGS = {
    "ldm": [
        {"name": "r0", "rows": 60000,  "epochs": 3, "sample_crops": 96, "psd_crops": 96},
        {"name": "r1", "rows": 150000, "epochs": 4, "sample_crops": 96, "psd_crops": 96},
        {"name": "r2", "rows": 613892, "epochs": 6, "sample_crops": 128, "psd_crops": 128},
    ],
    "corrdiff": [
        {"name": "r0", "rows": 60000,  "epochs": 3, "sample_crops": 96, "psd_crops": 96},
        {"name": "r1", "rows": 150000, "epochs": 4, "sample_crops": 96, "psd_crops": 96},
        {"name": "r2", "rows": 613892, "epochs": 6, "sample_crops": 128, "psd_crops": 128},
    ],
    "regression": [
        {"name": "r0", "rows": 60000,  "epochs": 2, "sample_crops": 96, "psd_crops": 96},
        {"name": "r1", "rows": 200000, "epochs": 3, "sample_crops": 96, "psd_crops": 96},
    ],
    # The inference arm has no fidelity ladder in the training sense. Its single
    # rung is a crop count on a frozen checkpoint.
    "inference": [
        {"name": "r0", "rows": 3000, "epochs": 0, "sample_crops": 0, "psd_crops": 0},
    ],
}


def rung_flags(arm, rung, params):
    """Derived per-rung flags that must move with the fidelity, not be searched."""
    rows = rung["rows"]
    if ARMS[arm]["kind"] == "infer":
        return {"--limit": rows}
    batch = int(params.get("batch", INCUMBENT["batch"]))
    steps_per_epoch = max(1, rows // batch)
    total_steps = steps_per_epoch * rung["epochs"]
    # 10 percent of the run in warmup, capped at the production value of 1000 and
    # floored at 50 so the very first steps are still tamed. The Phase-0 overfit
    # diverged at a constant 3e-4 and passed under cosine annealing, so the
    # schedule stays cosine unless a trial explicitly searches it.
    warmup = int(min(1000, max(50, round(0.10 * total_steps))))
    out = {
        "--limit": rows,
        "--epochs": rung["epochs"],
        "--warmup": warmup,
        "--psd-crops": rung["psd_crops"],
        # Patience is meaningless on a 3-epoch rung and would fire on noise.
        "--patience": 0,
        # Per-epoch checkpoint archiving is 263 MB a time and a screening rung
        # never needs the intermediate weights.
        "--no-keep-sampled": True,
    }
    # The two diffusion trainers and the regression trainer name their decoded
    # diagnostic flags differently (--sample-every/--sample-crops against
    # --diag-every/--diag-crops). Getting this wrong is an argparse death at
    # minute zero of a queued job, so it is resolved here rather than assumed.
    # Diagnose exactly once, at the final epoch: the trainers always run the
    # diagnostic at ep == args.epochs regardless of the interval, so this is the
    # cheapest setting that still produces the decoded scores the objective needs.
    if arm == "regression":
        out["--diag-every"] = rung["epochs"]
        out["--diag-crops"] = rung["sample_crops"]
    else:
        out["--sample-every"] = rung["epochs"]
        out["--sample-crops"] = rung["sample_crops"]
    return out


# ----------------------------------------------------------------------------
# Cost model
# ----------------------------------------------------------------------------
# Measured anchors, all on the JASMIN Orchid A100-SXM4-40GB:
#   1884.5 s/epoch over 613,892 rows at batch 64, width 128, mults 1,2,4,
#   attn 16  =>  326 img/s. Confirmed independently by the regression net's own
#   smoke test at 324 img/s.
#   195.8 min for a full-split evaluation of 13,281 crops at 8 members and
#   25 Heun steps (Results/Diffusion/eval/diffusion_eval.json).
#
# The architecture factors below are analytic first approximations. They are
# deliberately not trusted: hpo_search.py records the measured imgs_per_s of
# every completed trial keyed by its architecture signature and prefers the
# measurement over the estimate for every later trial in the same study. Any
# cost printed before the first trial completes is labelled "estimated".

ANCHOR_IMGS_PER_S = 326.0
ANCHOR_WIDTH = 128
ANCHOR_MULTS = "1,2,4"
ANCHOR_ATTN = "16"
ANCHOR_BATCH = 64
ANCHOR_EVAL_MIN = 195.8          # minutes
ANCHOR_EVAL_CROPS = 13281
ANCHOR_EVAL_MEMBERS = 8
ANCHOR_EVAL_STEPS = 25
L4_SLOWDOWN = 3.5                # planning rule from docs/PLAN_to_Aug28.md


def _depth_factor(mults):
    """Relative convolutional cost of a channel-multiplier ladder.

    A UNet level at resolution 64/2**l with channels width*m_l costs roughly
    (width*m_l)**2 * (64/2**l)**2, so the ladder's relative cost is
    sum_l m_l**2 / 4**l, normalised by the anchor ladder 1,2,4 (which sums to
    3.0). Note this correctly makes 1,2,2,4 CHEAPER than 1,2,4: the extra level
    sits at a quarter of the spatial resolution."""
    ms = [float(m) for m in str(mults).split(",")]
    s = sum(m * m / (4.0 ** l) for l, m in enumerate(ms))
    anchor = sum(m * m / (4.0 ** l)
                 for l, m in enumerate(float(x) for x in ANCHOR_MULTS.split(",")))
    return s / anchor


def _attn_factor(attn):
    """Self-attention is quadratic in the number of tokens, so it is negligible
    at resolution 16 (256 tokens) and dominant at 64 (4096 tokens). Rough
    multipliers relative to the anchor's single attention block at 16."""
    if attn in ("", None):
        return 0.96
    f = 1.0
    for r in str(attn).split(","):
        r = int(r)
        if r >= 64:
            f += 4.0
        elif r >= 32:
            f += 0.45
        elif r >= 16:
            f += 0.0          # the anchor already contains one of these
        else:
            f += 0.02
    return f


def _batch_factor(batch):
    """Throughput is roughly flat in images per second above batch 32 and falls
    off below it as kernel launches stop being amortised."""
    b = float(batch)
    if b >= ANCHOR_BATCH:
        return 0.97 if b > ANCHOR_BATCH else 1.0
    return 1.0 + 0.25 * max(0.0, math.log2(ANCHOR_BATCH / b))


def arch_signature(params):
    """The key under which measured throughput is cached. Only the flags that
    change FLOPs per image belong in it."""
    return "|".join(str(params.get(k, INCUMBENT[k]))
                    for k in ("width", "mults", "attn", "batch"))


def imgs_per_s(params, measured=None):
    """Estimated (or, if available, measured) training throughput."""
    if measured:
        hit = measured.get(arch_signature(params))
        if hit:
            return float(hit)
    w = float(params.get("width", INCUMBENT["width"]))
    factor = ((w / ANCHOR_WIDTH) ** 2
              * _depth_factor(params.get("mults", INCUMBENT["mults"]))
              * _attn_factor(params.get("attn", INCUMBENT["attn"]))
              * _batch_factor(params.get("batch", INCUMBENT["batch"])))
    return ANCHOR_IMGS_PER_S / max(factor, 1e-6)


def train_cost_h(arm, rung, params, measured=None):
    """Estimated A100-hours for one trial at one rung.

    Three terms: the training pass, the decoded diagnostic (K crops x M members
    through 2*steps - 1 denoiser forwards, which the regression arm does not
    pay because it has no sampler), and the per-epoch validation pass. The
    diagnostic and validation passes are charged at training throughput, which
    over-counts because neither runs a backward pass; over-counting is the
    correct direction for a budget guard."""
    if ARMS[arm]["kind"] == "infer":
        return infer_cost_h(params, n_crops=rung["rows"])
    rows, epochs = rung["rows"], rung["epochs"]
    ips = imgs_per_s(params, measured)
    train_s = rows * epochs / max(ips, 1.0)
    K = rung.get("sample_crops", 0)
    if arm == "regression":
        samp_s = (K / max(ips, 1.0)) if K else 0.0     # one forward, no sampler
    else:
        samp_s = (K * 8 * (2 * 25 - 1) / max(ips, 1.0)) if K else 0.0
    # One validation pass per epoch over max(400, rows//10) rows, forward only.
    val_s = epochs * max(400, rows // 10) / max(ips * 3.0, 1.0)
    return (train_s + samp_s + val_s) / 3600.0


def infer_cost_h(params, n_crops):
    """Estimated A100-hours for one evaluate_diffusion.py cell.

    Scaling validated against the measured 195.8 min at 13,281 crops, 8 members
    and 25 steps: cost is linear in crops and members, linear in the number of
    denoiser forwards (2*steps - 1 for Heun), and exactly doubles above
    guidance 1.0 because classifier-free guidance runs a second, unconditional
    denoiser pass at every step."""
    M = float(params.get("members", INCUMBENT["members"]))
    S = float(params.get("steps", INCUMBENT["steps"]))
    g = float(params.get("guidance", INCUMBENT["guidance"]))
    minutes = (ANCHOR_EVAL_MIN
               * (float(n_crops) / ANCHOR_EVAL_CROPS)
               * (M / ANCHOR_EVAL_MEMBERS)
               * ((2 * S - 1) / (2 * ANCHOR_EVAL_STEPS - 1))
               * (2.0 if g > 1.0 else 1.0))
    return minutes / 60.0


def full_fidelity_cost_h(arm, params, epochs=50, measured=None):
    """What one cell of this configuration would cost trained to completion.
    Printed beside every screening cost so the saving is explicit."""
    rung = {"rows": ARMS[arm]["n_train_rows"], "epochs": epochs,
            "sample_crops": 16, "psd_crops": 64}
    return train_cost_h(arm, rung, params, measured)


# ----------------------------------------------------------------------------
# Search spaces
# ----------------------------------------------------------------------------
# A space is a dict with:
#   stage    "grid" | "random" | "tpe"   the intended search strategy
#   arm      which ARM it targets
#   design   "ofat" | "cartesian"        how a grid space is expanded
#   axes     ordered dict of name -> list of values (grid) or range spec (tpe)
#   note     one line for the methods chapter
#
# design "ofat": one-factor-at-a-time around INCUMBENT. Each axis contributes
# len(values) - 1 new cells (the incumbent value is the shared centre cell), so
# an 8-axis screen is additive rather than multiplicative. This is the reading
# of "coarse grid search" that respects the brief's own warning not to tune too
# many parameters at once, and it is what makes 24 cells rather than 5,832.
#
# design "cartesian": the full product. Used only for the small blocks where an
# interaction is expected on physical grounds (learning rate with batch size,
# because the effective step size is lr per sample; conditioning mode with
# conditioning dropout, because CFG dropout is meaningless without conditioning).

SPACES = {

    # ------------------------------------------------------------------ stage 1
    "ldm_coarse": {
        "arm": "ldm",
        "stage": "grid",
        "design": "ofat+blocks",
        "note": ("Stage 1 coarse screen, one factor at a time around the ml_v2 "
                 "incumbent, plus two small cartesian blocks where an interaction "
                 "is expected on physical grounds."),
        "blocks": [
            # lr and batch interact: the per-sample step size is lr/batch, so the
            # two cannot be screened independently without confounding them.
            {"name": "lr_x_batch", "design": "cartesian",
             "axes": {"lr": [5e-5, 1e-4, 2e-4], "batch": [32, 64, 128]}},
            # CFG dropout only means anything when there is conditioning to drop.
            {"name": "conditioning", "design": "cartesian",
             "axes": {"cond_mode": ["full", "a-only"], "cond_drop": [0.0, 0.1, 0.2]}},
        ],
        "axes": {
            "width":        [96, 128, 192],
            "mults":        ["1,2,4", "1,2,2,4"],
            "attn":         ["16", "16,32"],
            "dropout":      [0.0, 0.05, 0.10],
            "weight_decay": [0.0, 1e-4, 1e-2],
            # EDM's P_mean = -1.2 was calibrated at sigma_data = 0.5. This arm's
            # measured sigma_data is 0.7407, and preserving EDM's sigma/sigma_data
            # ratio implies P_mean = ln(0.6 * 0.7407) = -0.81, which puts MORE
            # weight at large sigma. The over-smoothing symptom argues the other
            # way, for -1.6. Both directions are in the screen precisely because
            # the principled value and the symptom-driven value disagree.
            "p_mean":       [-1.6, -1.2, -0.81],
            "p_std":        [1.2, 1.6],
        },
    },

    # ------------------------------------------------------------------ stage 2
    "ldm_refine": {
        "arm": "ldm",
        "stage": "tpe",
        "design": "sampled",
        "note": ("Stage 2 Bayesian refinement. Tree-structured Parzen estimator "
                 "over the continuous axes only, seeded with every stage 1 trial "
                 "so the surrogate starts from real observations rather than cold."),
        "axes": {
            "lr":           {"type": "logfloat", "low": 3e-5, "high": 4e-4},
            "width":        {"type": "int",      "low": 96,  "high": 192, "step": 32},
            "dropout":      {"type": "float",    "low": 0.0, "high": 0.15},
            "weight_decay": {"type": "logfloat", "low": 1e-6, "high": 3e-2,
                             "allow_zero": True},
            "p_mean":       {"type": "float",    "low": -2.0, "high": -0.6},
            "p_std":        {"type": "float",    "low": 1.0,  "high": 1.8},
            "cond_drop":    {"type": "float",    "low": 0.0,  "high": 0.25},
        },
    },

    # ------------------------------------------------------- CorrDiff, stage 1
    "corrdiff_coarse": {
        "arm": "corrdiff",
        "stage": "grid",
        "design": "ofat+blocks",
        "note": ("Stage 1 coarse screen for the CorrDiff arm. Narrower than the "
                 "LDM screen by design: the two trainers share their training "
                 "regime by construction (check_contract.py fails if the argparse "
                 "defaults drift), so the LDM screen's verdict on lr, batch and "
                 "architecture transfers, and this screen spends its cells on what "
                 "is genuinely different, namely the second-residual target's own "
                 "noise schedule and the mu_r conditioning."),
        "blocks": [
            {"name": "mu_conditioning", "design": "cartesian",
             "axes": {"hr_mean_cond": ["on", "off"], "cond_drop": [0.0, 0.1, 0.2]}},
        ],
        "axes": {
            "lr":      [5e-5, 1e-4, 2e-4],
            # The second residual r' is by construction lower-variance than
            # delta, so its sigma_data is smaller and the EDM noise distribution
            # that was right for delta need not be right here. This is the axis
            # most likely to matter in this arm.
            "p_mean":  [-1.6, -1.2, -0.81],
            "p_std":   [1.2, 1.6],
            "width":   [128, 192],
        },
    },

    "corrdiff_refine": {
        "arm": "corrdiff",
        "stage": "tpe",
        "design": "sampled",
        "note": "Stage 2 Bayesian refinement for the CorrDiff arm.",
        "axes": {
            "lr":           {"type": "logfloat", "low": 3e-5, "high": 4e-4},
            "p_mean":       {"type": "float",    "low": -2.0, "high": -0.6},
            "p_std":        {"type": "float",    "low": 1.0,  "high": 1.8},
            "dropout":      {"type": "float",    "low": 0.0,  "high": 0.15},
            "weight_decay": {"type": "logfloat", "low": 1e-6, "high": 3e-2,
                             "allow_zero": True},
            "cond_drop":    {"type": "float",    "low": 0.0,  "high": 0.25},
        },
    },

    # --------------------------------------------------- CorrDiff stage one net
    "regression_coarse": {
        "arm": "regression",
        "stage": "grid",
        "design": "ofat+blocks",
        "note": ("Screen for the deterministic conditional-mean network that "
                 "feeds the CorrDiff arm. No noise schedule here: this net is "
                 "trained under a plain regression loss, so the searchable set is "
                 "optimisation and capacity only."),
        "blocks": [],
        "axes": {
            "lr":           [5e-5, 1e-4, 2e-4, 4e-4],
            "width":        [96, 128, 192],
            "weight_decay": [0.0, 1e-4, 1e-2],
            "dropout":      [0.0, 0.05],
        },
    },

    # ------------------------------------------------------------------ stage 4
    # Inference-only. Costs zero training hours and is the highest value per hour
    # in the whole programme, which is why the supervisor put it in the protocol.
    "inference_grid": {
        "arm": "inference",
        "stage": "grid",
        "design": "ofat+blocks",
        "note": ("Stage 4 sampler sweep on a frozen checkpoint. One factor at a "
                 "time around the production sampler setting."),
        "blocks": [],
        "axes": {
            "steps":    [18, 25, 50],
            "members":  [4, 8, 16],
            "guidance": [1.0, 1.25, 1.5],
            # Reported with a standing caveat: this implementation is not Karras
            # Algorithm 2 (gamma is applied at every sigma including the last,
            # there is no S_tmin/S_tmax gate, S_noise is 1.0 not 1.007), and at
            # 25 steps gamma saturates at 0.41421 for any S_churn >= 10.36, so
            # {10, 40} tests 0.400 against 0.414 and is near-degenerate. It is in
            # the grid because the brief asks for the noise schedule to be swept,
            # and a measured near-null result is a reportable one.
            "churn":    [0.0, 10.0, 40.0],
        },
    },

    "inference_grid_full": {
        "arm": "inference",
        "stage": "grid",
        "design": "cartesian",
        "note": ("Full cartesian sampler grid, 81 cells. Only affordable at a "
                 "small --limit; print the plan before running it."),
        "blocks": [],
        "axes": {
            "steps":    [18, 25, 50],
            "members":  [4, 8, 16],
            "guidance": [1.0, 1.25, 1.5],
            "churn":    [0.0, 10.0, 40.0],
        },
    },
}


# ----------------------------------------------------------------------------
# The codec PSD ceiling
# ----------------------------------------------------------------------------
# Measured 10 Aug 2026 by stage 5 (evaluate_deterministic.py --oracle) on the
# full validation split, 13,281 crops per lead, epoch-17 codec. The oracle is y
# encoded and decoded, so it is what NO latent method in this arm can beat.
#
# WHICH ESTIMATOR. The project reports two 2-8 km PSD numbers and they disagree
# materially, so the one used here has to be named. Everything called
# `psd_ratio_2_8km` in this codebase is the MEAN-OF-RATIOS estimator,
# mean over wavenumbers of (summed model power / summed observed power):
# train_ldm.sampled_diagnostics computes it inline, train_regression.py assigns
# it from `psd_mean_ratio`, and evaluate_diffusion.py defines `psd_mean_ratio`
# as np.mean(m[band] / o[band]). The composite objective below reads that name,
# so the ceiling it is compared against must be the mean-of-ratios oracle, 0.903
# pooled, NOT the band-power oracle of 0.970. An earlier draft of this file used
# 0.980, which was the band-power figure from a 512-crop L4 run, and mixing the
# two would have put the target 8 percent too high.

PSD_CEILING_MEAN_RATIO = {15: 0.904, 30: 0.895, 45: 0.908, 60: 0.904}
PSD_CEILING_BAND_POWER = {15: 0.971, 30: 0.968, 45: 0.969, 60: 0.970}
PSD_CEILING_POOLED = 0.903               # mean-of-ratios, the default target

# For context, and it changes how the composite objective should be read:
# ml_v2's member field on the same full split reaches mean-of-ratios 0.895 at
# +15, 0.882 at +30, 0.918 at +45 and 0.946 at +60. At the two long leads it is
# ABOVE the oracle ceiling. That is not extra skill, it is synthesised
# high-frequency power, which is exactly the failure the two-sided penalty in the
# composite objective exists to catch. A search that rewarded raw PSD would have
# selected for it.


# ----------------------------------------------------------------------------
# Objectives
# ----------------------------------------------------------------------------
# Every objective is expressed as a python expression over the metrics namespace
# that hpo_search.py assembles from train_log.json (or the evaluation JSON for
# the inference arm) plus the paired baselines from hpo_baselines.py. All are
# written so that LARGER IS BETTER; hpo_search maximises unconditionally.
#
# The default is deliberately not the validation EDM loss. This project has
# established across three runs that val loss and small-scale power move in
# opposite directions, so ranking on val loss systematically selects the
# smoothest model the search produced. CRPS is used instead because it is a
# strictly proper score: it cannot be gamed by smoothing (which flatters MAE
# through the double-penalty effect) and cannot be gamed by sharpening (which
# flatters PSD).

OBJECTIVES = {
    "crps_ss":   {"expr": "1.0 - crps / crps_adv",
                  "desc": "CRPS skill score against the advection prior on identical crops"},
    "crps_ss_pers": {"expr": "1.0 - crps / crps_pers",
                     "desc": "CRPS skill score against persistence"},
    "mae_ss":    {"expr": "1.0 - mae / mae_adv",
                  "desc": "MAE skill score against advection (ensemble mean)"},
    "csi8_ss":   {"expr": "csi_8 - csi_8_adv",
                  "desc": "heavy-rain CSI advantage over advection (noisy; not a primary)"},
    "val_loss":  {"expr": "-val_loss_w",
                  "desc": "negated weighted EDM validation loss (smoothness-biased; diagnostic only)"},
    # The regression arm has no sampler and no CRPS, so it is ranked on the
    # metric the project already gates it with: held-out explained variance of
    # the residual, second-moment normalised. This is GATE-C's own quantity, so
    # the search and the gate are measured in the same units.
    "val_ev":    {"expr": "val_ev",
                  "desc": "held-out explained variance of the regression target (GATE-C's metric)"},
    "psd_gap":   {"expr": "-abs(psd_ratio_2_8km - psd_ceiling)",
                  "desc": "closeness of member small-scale power to the measured codec ceiling"},
    # The recommended default. CRPS skill score, penalised for departing from the
    # codec's own physical PSD ceiling in EITHER direction. Exceeding the ceiling
    # is not a win: it is evidence of synthesised high-frequency noise, which is
    # exactly what ml_v2 epoch 5 produced at PSD 1.28 and CSI@8 0.079.
    "composite": {"expr": "(1.0 - crps / crps_adv) - 0.25 * abs(psd_ratio_2_8km - psd_ceiling)",
                  "desc": ("CRPS skill score against advection, penalised for "
                           "departing from the codec PSD ceiling in either direction")},
    # Inference arm: the evaluation JSON carries fair CRPS and a spread/RMSE ratio,
    # so calibration can enter the objective directly.
    "infer_composite": {
        "expr": ("(1.0 - crps_fair / mae_adv) "
                 "- 0.50 * abs(spread_rmse_ratio - spread_rmse_ideal)"),
        "desc": ("fair-CRPS skill score against advection, penalised for "
                 "dispersion error against the M-dependent ideal sqrt((M+1)/M)")},
}

# A trial that fails its gate is ranked last but never deleted: a configuration
# that is fast and inaccurate must stay visible in the leaderboard, otherwise the
# search looks better than it was.
GATES = {
    "none":        {"expr": "True", "desc": "no admissibility gate"},
    "mae_vs_adv":  {"expr": "mae <= 1.01 * mae_adv",
                    "desc": ("ensemble-mean MAE must be within 1 percent of the "
                             "advection prior on identical crops. This is LDM.md "
                             "section 6 criterion (a) written as a hard gate.")},
    "beats_persistence": {"expr": "mae <= mae_pers",
                          "desc": "must at least beat persistence on MAE"},
    "both":        {"expr": "(mae <= 1.01 * mae_adv) and (mae <= mae_pers)",
                    "desc": "within 1 percent of advection AND better than persistence"},
    # GATE-C from docs/designs/CorrDiff_Design.md, stated as an expression so the
    # regression search and the arm's own authorisation gate use one definition.
    "gate_c":      {"expr": "val_ev >= 0.10",
                    "desc": ("CorrDiff GATE-C: pooled held-out explained variance "
                             "of at least 0.10 for the learned mean to be worth "
                             "the 26.2-hour diffusion retrain")},
}

DEFAULT_OBJECTIVE = {"ldm": "composite", "corrdiff": "composite",
                     "regression": "val_ev", "inference": "infer_composite"}
DEFAULT_GATE = {"ldm": "mae_vs_adv", "corrdiff": "mae_vs_adv",
                "regression": "gate_c", "inference": "none"}


# ----------------------------------------------------------------------------
# Successive halving
# ----------------------------------------------------------------------------
def sha_schedule(n_trials, rungs, eta=3):
    """Synchronous successive halving: how many trials survive to each rung.

    Synchronous rather than asynchronous (ASHA) on purpose. The asynchronous
    variant is faster on a busy cluster but its promotion decisions depend on
    job completion order, so a rerun does not reproduce the same set of
    promotions. A dissertation needs the search to be describable and repeatable
    in one sentence, and 24 trials over 3 rungs is small enough that the
    synchronisation barrier costs little.

    Returns a list of (rung, n_kept)."""
    out, n = [], int(n_trials)
    for r in rungs:
        out.append((r, max(1, n)))
        n = max(1, int(math.floor(n / float(eta))))
    return out


def hyperband_brackets(rungs, eta=3, max_trials=None):
    """Hyperband's outer loop: several successive-halving brackets that trade
    the number of configurations against the fidelity each one starts at.
    Bracket 0 starts everything at the cheapest rung (maximum exploration),
    the last bracket starts a few configurations straight at the top rung
    (maximum exploitation, and the safety net against a screen that does not
    transfer). Returns a list of dicts."""
    s_max = len(rungs) - 1
    brackets = []
    for s in range(s_max, -1, -1):
        n = int(math.ceil((s_max + 1) / (s + 1.0) * eta ** s))
        if max_trials:
            n = min(n, max_trials)
        brackets.append({"bracket": s_max - s, "n_configs": n,
                         "start_rung": s_max - s, "rungs": rungs[s_max - s:]})
    return brackets
