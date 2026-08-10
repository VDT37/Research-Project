#!/usr/bin/env python3
"""
hpo_search.py - hyperparameter optimisation harness for both diffusion arms.

Implements the supervisor's protocol end to end:

  stage 1  coarse grid over the main parameters (one factor at a time around the
           incumbent, plus small cartesian blocks where an interaction is
           expected), at reduced training fidelity
  stage 2  a more efficient search over the same space: random, or Bayesian
           optimisation by tree-structured Parzen estimator, seeded with every
           stage 1 observation
  stage 3  Hyperband-style early stopping: synchronous successive halving over a
           fidelity ladder, so a bad configuration is killed after 0.15
           A100-hours rather than 26.2, with optional median pruning inside a rung
  stage 4  the inference-only sampler sweep (steps, members, guidance, churn) on
           a frozen checkpoint, which costs no training hours at all

and it enforces the two rules the brief attaches to all of them: every run is
scored against persistence and the pysteps advection prior on the same validation
crops, and only a small number of parameters move at a time.

The search space, the fidelity ladder, the cost model and the objective all live
in hpo_spaces.py, which is imported and then FROZEN into the study directory, so
what was searched is recoverable from disk even if the space is later edited.

Everything is stdlib only. It runs on a login node, on the Windows dev machine
(for --exec dry), and inside a Slurm job identically. It never imports torch,
never touches the GPU, and never opens a latent pack: it launches the existing
stage scripts unmodified and reads the JSON they already write.


USAGE
-----
Plan first, always. --exec dry is the default and writes nothing but a plan.

    # what would a full stage 1 screen cost?
    python hpo_search.py --space ldm_coarse --out $HPO/ldm_coarse \
        -- --latents-dir $DISS_SCRATCH/latents_ml_ep17 --vae $VAE17 --leads 15,30,45,60

    # run it on JASMIN, eight jobs at a time (the orchid QoS MaxJobsPU)
    python hpo_search.py --space ldm_coarse --out $HPO/ldm_coarse \
        --exec slurm --parallel 8 --budget-gpu-h 20 \
        -- --latents-dir $DISS_SCRATCH/latents_ml_ep17 --vae $VAE17 --leads 15,30,45,60

    # stage 2, Bayesian, seeded from the stage 1 study
    python hpo_search.py --space ldm_refine --out $HPO/ldm_refine \
        --sampler tpe --n-trials 24 --seed-from $HPO/ldm_coarse \
        --exec slurm --parallel 8 --budget-gpu-h 12 \
        -- --latents-dir $DISS_SCRATCH/latents_ml_ep17 --vae $VAE17 --leads 15,30,45,60

    # stage 4, inference only, on the frozen ml_v2 checkpoint
    python hpo_search.py --space inference_grid --out $HPO/sampler \
        --exec slurm --parallel 4 --budget-gpu-h 9 \
        -- --ckpt $ML/ckpt_ep050.pt --vae $VAE17 \
           --latents-dir $DISS_SCRATCH/latents_ml_ep17 --split val --lead 60 --batch 16

    # the CorrDiff arm
    python hpo_search.py --space corrdiff_coarse --out $HPO/corrdiff_coarse \
        --exec slurm --parallel 8 --budget-gpu-h 12 \
        -- --latents-dir $DISS_SCRATCH/latents_ml_ep17 --vae $VAE17 --leads 15,30,45,60 \
           --mu-dir $DISS_SCRATCH/latents_ml_ep17_mu_delta --hr-mean-cond on

Everything after the bare `--` is appended verbatim to every trial command. That
is where the paths, the lead set, and any arm-specific required flag go.

Resumability: a trial is complete if and only if its output directory holds the
DONE marker the trainers already write (or, for the inference arm, its evaluation
JSON). Re-running the same command picks up exactly where it stopped, which is
what makes this safe under a Slurm requeue. The study also refuses to reuse a
trial directory whose recorded parameters do not match the ones it is about to
run, so editing hpo_spaces.py between runs cannot silently corrupt a study.

Outputs -> <out>/
    study.json            frozen specification: space, rungs, objective, gate, git
    plan.json             the planned trials and their estimated cost
    trials.jsonl          append-only event log, one row per state change
    throughput.json       measured img/s per architecture, refines the cost model
    ranking.json          final ranking, written after every rung
    trial_NNN_rM/         one directory per trial per rung, the trainer's
                          own --out, holding its submit.sh and slurm-*.out
    winner.json           the winner and its full-fidelity confirm command
    hpo.log               harness log

Then: python hpo_report.py --study <out>
"""

import argparse
import glob
import hashlib
import itertools
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hpo_spaces as S                                            # noqa: E402


# ----------------------------------------------------------------------------
# Small utilities, deliberately duplicated rather than imported from
# train_vae_v2 so this file has no numpy/torch dependency at all.
# ----------------------------------------------------------------------------
def atomic_json(obj, path):
    """Write JSON through a temp file plus os.replace, so a SIGKILL mid-write
    can never leave a truncated study file behind."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def param_hash(params):
    blob = json.dumps({k: params[k] for k in sorted(params)}, sort_keys=True,
                      default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


class Tee:
    """Harness log: everything printed also lands in <out>/hpo.log, because a
    Slurm stdout file is not where anyone will look six weeks later."""

    def __init__(self, path):
        self.fh = open(path, "a", buffering=1)

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.fh.write(msg + "\n")


# ----------------------------------------------------------------------------
# Search-space expansion
# ----------------------------------------------------------------------------
def expand_grid(space):
    """Turn a grid space into an ordered list of parameter dicts.

    design "ofat+blocks":
        the incumbent configuration is cell 0. Each axis then contributes one
        cell per value that is NOT the incumbent's, holding every other axis at
        the incumbent. Each block additionally contributes its full cartesian
        product. This is additive in the number of axes rather than
        multiplicative, which is the whole reason a nine-parameter screen is 24
        cells instead of several thousand, and it is what the brief means by
        "avoid tuning too many parameters at once".

    design "cartesian":
        the full product of all axes. Used for the sampler sweep, where cells
        are cheap.
    """
    axes = space.get("axes", {})
    blocks = space.get("blocks", [])
    design = space.get("design", "ofat+blocks")
    seen, out = set(), []

    def add(params, origin):
        base = dict(S.INCUMBENT)
        base.update(params)
        # Keep only the keys this space actually talks about plus anything the
        # cell explicitly set, so a trial command does not restate every default.
        keys = set(axes) | set(params)
        for b in blocks:
            keys |= set(b["axes"])
        cell = {k: base[k] for k in sorted(keys)}
        h = param_hash(cell)
        if h in seen:
            return
        bad = S.check_constraints(cell)
        if bad:
            out.append({"params": cell, "origin": origin, "invalid": bad})
            seen.add(h)
            return
        seen.add(h)
        out.append({"params": cell, "origin": origin, "invalid": []})

    if design == "cartesian":
        names = list(axes)
        for combo in itertools.product(*(axes[n] for n in names)):
            add(dict(zip(names, combo)), "cartesian")
        return out

    add({}, "incumbent")
    for name, values in axes.items():
        for v in values:
            if v == S.INCUMBENT.get(name):
                continue
            add({name: v}, f"ofat:{name}")
    for b in blocks:
        names = list(b["axes"])
        for combo in itertools.product(*(b["axes"][n] for n in names)):
            add(dict(zip(names, combo)), f"block:{b['name']}")
    return out


# ----------------------------------------------------------------------------
# Samplers for the non-grid stages
# ----------------------------------------------------------------------------
def _sample_axis(spec, rng):
    t = spec["type"]
    if t == "cat":
        return rng.choice(spec["choices"])
    if t == "int":
        step = spec.get("step", 1)
        n = int((spec["high"] - spec["low"]) // step)
        return int(spec["low"] + step * rng.randint(0, n))
    if t == "logfloat":
        if spec.get("allow_zero") and rng.random() < 0.15:
            return 0.0
        lo, hi = math.log(spec["low"]), math.log(spec["high"])
        return math.exp(rng.uniform(lo, hi))
    return rng.uniform(spec["low"], spec["high"])


class RandomSampler:
    """Uniform random search. Included because it is the honest control for any
    Bayesian claim: if TPE does not beat random on the same budget, the
    write-up should say so rather than assert that TPE helped."""

    name = "random"

    def __init__(self, space, seed=0):
        self.space, self.rng = space, random.Random(seed)

    def ask(self, n, observed):
        out = []
        for _ in range(n):
            p = {k: _sample_axis(v, self.rng) for k, v in self.space["axes"].items()}
            out.append(p)
        return out


class TPESampler:
    """Tree-structured Parzen estimator, the Bayesian optimiser the brief names.

    The idea in one paragraph, because the methods chapter needs it. Standard
    Bayesian optimisation models p(objective | params) with a Gaussian process.
    TPE inverts that: it splits the observations at a quantile into a "good" set
    and a "bad" set, fits a density to the parameters of each (l(x) for good,
    g(x) for bad), and proposes the candidate maximising l(x)/g(x). That ratio
    is monotone in expected improvement, so it selects points that look like the
    good observations and unlike the bad ones. It handles categorical and
    log-scaled parameters natively, it does not invert an n-by-n covariance
    matrix, and it treats each parameter's marginal independently, which is
    exactly the regime here: few observations, mixed types, and a search that
    must be explainable.

    Densities are Parzen mixtures: one Gaussian per observation with an adaptive
    bandwidth (the larger gap to its sorted neighbours, floored so a cluster of
    near-identical observations cannot collapse to a spike), plus one broad prior
    component covering the whole range so the sampler can never be trapped by its
    first few observations. Categoricals use Laplace-smoothed counts.

    This is deliberately a self-contained implementation rather than a hard
    dependency on Optuna, because the conda environment on the compute servers
    does not carry Optuna and a study must never fail to start over a missing
    package. --sampler optuna-tpe uses the real Optuna TPESampler for the ask
    step instead, if it is installed; the harness still owns execution, pruning
    and logging in both cases.
    """

    name = "tpe"

    def __init__(self, space, seed=0, gamma=0.20, n_candidates=32, n_startup=8):
        self.space, self.rng = space, random.Random(seed)
        self.gamma, self.n_candidates, self.n_startup = gamma, n_candidates, n_startup

    # -- density helpers ----------------------------------------------------
    @staticmethod
    def _to_internal(spec, v):
        return math.log(max(v, spec["low"] * 1e-3)) if spec["type"] == "logfloat" else float(v)

    @staticmethod
    def _bounds(spec):
        if spec["type"] == "logfloat":
            return math.log(spec["low"]), math.log(spec["high"])
        return float(spec["low"]), float(spec["high"])

    def _parzen(self, spec, values):
        """Return (mus, sigmas, weights) of the mixture, including the prior."""
        lo, hi = self._bounds(spec)
        rng_width = hi - lo
        xs = sorted(self._to_internal(spec, v) for v in values)
        mus = list(xs) + [(lo + hi) / 2.0]
        sig_floor = rng_width / 20.0
        sigmas = []
        for i, x in enumerate(xs):
            left = x - xs[i - 1] if i > 0 else rng_width
            right = xs[i + 1] - x if i + 1 < len(xs) else rng_width
            sigmas.append(min(max(max(left, right), sig_floor), rng_width))
        sigmas.append(rng_width)                      # broad prior component
        w = [1.0] * len(xs) + [1.0]
        tot = sum(w)
        return mus, sigmas, [x / tot for x in w], (lo, hi)

    def _logpdf(self, mix, x):
        mus, sigmas, ws, _ = mix
        acc = 0.0
        for m, s, w in zip(mus, sigmas, ws):
            acc += w * math.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))
        return math.log(max(acc, 1e-300))

    def _draw(self, mix, rng):
        mus, sigmas, ws, (lo, hi) = mix
        r, acc = rng.random(), 0.0
        for m, s, w in zip(mus, sigmas, ws):
            acc += w
            if r <= acc:
                return min(max(rng.gauss(m, s), lo), hi)
        return rng.uniform(lo, hi)

    @staticmethod
    def _from_internal(spec, x):
        if spec["type"] == "logfloat":
            return math.exp(x)
        if spec["type"] == "int":
            step = spec.get("step", 1)
            return int(round((x - spec["low"]) / step) * step + spec["low"])
        return float(x)

    # -- ask ----------------------------------------------------------------
    def ask(self, n, observed):
        """observed: list of (params, objective) with LARGER objective better."""
        usable = [(p, o) for p, o in observed if o is not None and math.isfinite(o)]
        out = []
        for _ in range(n):
            if len(usable) < self.n_startup:
                out.append({k: _sample_axis(v, self.rng)
                            for k, v in self.space["axes"].items()})
                continue
            ranked = sorted(usable, key=lambda t: -t[1])
            n_below = max(1, min(int(math.ceil(self.gamma * len(ranked))), 25))
            below = [p for p, _ in ranked[:n_below]]
            above = [p for p, _ in ranked[n_below:]] or below
            cand_best, score_best = None, -1e300
            for _c in range(self.n_candidates):
                cand, score = {}, 0.0
                for name, spec in self.space["axes"].items():
                    if spec["type"] == "cat":
                        ch = spec["choices"]
                        cb = {c: 1.0 for c in ch}
                        ca = {c: 1.0 for c in ch}
                        for p in below:
                            cb[p[name]] = cb.get(p[name], 1.0) + 1.0
                        for p in above:
                            ca[p[name]] = ca.get(p[name], 1.0) + 1.0
                        tb, ta = sum(cb.values()), sum(ca.values())
                        pick = self.rng.choices(ch, weights=[cb[c] for c in ch])[0]
                        cand[name] = pick
                        score += math.log(cb[pick] / tb) - math.log(ca[pick] / ta)
                        continue
                    mb = self._parzen(spec, [p[name] for p in below])
                    ma = self._parzen(spec, [p[name] for p in above])
                    x = self._draw(mb, self.rng)
                    score += self._logpdf(mb, x) - self._logpdf(ma, x)
                    cand[name] = self._from_internal(spec, x)
                if score > score_best:
                    cand_best, score_best = cand, score
            out.append(cand_best)
            # Optimistic in-batch update: assume the proposal lands at the median
            # so a batch of --parallel proposals does not collapse onto one point.
            if len(out) < n:
                med = sorted(o for _, o in usable)[len(usable) // 2]
                usable = usable + [(cand_best, med)]
        return out


class OptunaTPESampler:
    """Optuna's own TPESampler used purely as an ask interface. Execution,
    pruning, logging and resumability stay in this harness, so the two sampler
    paths are interchangeable and produce identical study artefacts."""

    name = "optuna-tpe"

    def __init__(self, space, seed=0):
        import optuna                                             # noqa: F401
        from optuna.samplers import TPESampler as _T
        self.optuna = optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.space = space
        self.study = optuna.create_study(direction="maximize",
                                         sampler=_T(seed=seed, multivariate=True))
        self._n_told = 0

    def _suggest(self, trial):
        p = {}
        for name, spec in self.space["axes"].items():
            if spec["type"] == "cat":
                p[name] = trial.suggest_categorical(name, spec["choices"])
            elif spec["type"] == "int":
                p[name] = trial.suggest_int(name, spec["low"], spec["high"],
                                            step=spec.get("step", 1))
            elif spec["type"] == "logfloat":
                p[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
            else:
                p[name] = trial.suggest_float(name, spec["low"], spec["high"])
        return p

    def ask(self, n, observed):
        for params, obj in observed[self._n_told:]:
            if obj is not None and math.isfinite(obj):
                self.study.add_trial(self.optuna.trial.create_trial(
                    params=params,
                    distributions=self._distributions(),
                    value=float(obj)))
            self._n_told += 1
        out = []
        for _ in range(n):
            t = self.study.ask()
            out.append(self._suggest(t))
        return out

    def _distributions(self):
        from optuna import distributions as D
        d = {}
        for name, spec in self.space["axes"].items():
            if spec["type"] == "cat":
                d[name] = D.CategoricalDistribution(spec["choices"])
            elif spec["type"] == "int":
                d[name] = D.IntDistribution(spec["low"], spec["high"],
                                            step=spec.get("step", 1))
            elif spec["type"] == "logfloat":
                d[name] = D.FloatDistribution(spec["low"], spec["high"], log=True)
            else:
                d[name] = D.FloatDistribution(spec["low"], spec["high"])
        return d


# ----------------------------------------------------------------------------
# Command construction
# ----------------------------------------------------------------------------
def flagify(name, value):
    spec = S.PARAMS.get(name)
    flag = spec["flag"] if spec else "--" + name.replace("_", "-")
    return flag, value


def default_python(executor):
    """Which interpreter a trial command should name.

    On Slurm the answer is the BARE WORD "python", never sys.executable. Trials
    run through gpu.sbatch, which does `conda activate nowcast` and then executes
    its arguments, so a bare `python` resolves inside the activated environment
    while an absolute path bypasses it entirely. Naming sys.executable here means
    the login node's /usr/bin/python is baked into every job and every trial dies
    on `import torch` in about two seconds. That is exactly what happened on the
    first real run of this harness, so the default is now chosen per executor
    rather than assumed.

    Locally there is no wrapper and no activation step, so the interpreter
    running this file is the right one."""
    return "python" if executor == "slurm" else sys.executable


def build_cmd(arm, script_dir, params, rung, trial_dir, passthrough, seed,
              tag=None, python=None):
    """Assemble one trial's argv. Order is: interpreter, script, searched
    parameters, rung-derived fidelity flags, output, seed, then the caller's
    pass-through (which therefore always wins on a collision, deliberately, so a
    user can pin anything the harness got wrong without editing this file)."""
    meta = S.ARMS[arm]
    cmd = [python or sys.executable, os.path.join(script_dir, meta["script"])]
    for name in sorted(params):
        flag, value = flagify(name, params[name])
        cmd += [flag, str(value)]
    for flag, value in S.rung_flags(arm, rung, params).items():
        if value is True:
            cmd.append(flag)
        elif value is False or value is None:
            continue
        else:
            cmd += [flag, str(value)]
    cmd += ["--seed", str(seed)]
    if meta["kind"] == "infer":
        cmd += [meta["out_flag"], os.path.dirname(trial_dir.rstrip("/")) + "/eval"]
        cmd += ["--tag", tag or ("_" + os.path.basename(trial_dir))]
    else:
        cmd += [meta["out_flag"], trial_dir]
    cmd += list(passthrough)
    return cmd


# ----------------------------------------------------------------------------
# Metrics extraction
# ----------------------------------------------------------------------------
def _last_block(log, *names):
    """The most recent non-empty diagnostic block. The two diffusion trainers
    call theirs "sampled" and the regression trainer calls its "decoded"; both
    carry mae, mae_adv and psd_ratio_2_8km on the same crops, so one reader
    serves all three arms."""
    for rec in reversed(log):
        for n in names:
            if rec.get(n):
                return rec[n]
    return None


def metrics_from_training(trial_dir):
    """Read a completed trainer's own artefacts. Nothing is recomputed here: the
    numbers are exactly the ones the trainer wrote, including its advection
    controls on the identical diagnostic crops."""
    lp = os.path.join(trial_dir, "train_log.json")
    if not os.path.exists(lp):
        return None
    try:
        log = json.load(open(lp))
    except Exception:
        return None
    if not log:
        return None
    m = {}
    vs = [r["val"] for r in log if r.get("val")]
    # The diffusion trainers minimise a weighted EDM loss (val.loss_w); the
    # regression trainer minimises an MSE and reports explained variance
    # (val.mse, val.ev). Expose whichever exist, under both a schema-specific
    # and a schema-neutral name, so one objective expression can serve either.
    if vs and "loss_w" in vs[0]:
        m["val_loss_w"] = min(v["loss_w"] for v in vs)
        m["val_loss_w_last"] = vs[-1]["loss_w"]
        m["val_metric"] = m["val_loss_w"]
    if vs and "mse" in vs[0]:
        m["val_mse"] = min(v["mse"] for v in vs)
        m["val_mse_last"] = vs[-1]["mse"]
        m.setdefault("val_metric", m["val_mse"])
    if vs and "ev" in vs[0]:
        m["val_ev"] = max(v["ev"] for v in vs)
        m["val_ev_last"] = vs[-1]["ev"]
    tr = log[-1].get("train", {})
    for k in ("loss_w", "mse", "ev"):
        if k in tr:
            m["train_" + k + "_last"] = tr[k]
    m["epochs_run"] = log[-1]["epoch"]
    sys_ = log[-1].get("sys", {})
    m["imgs_per_s"] = sys_.get("imgs_per_s", float("nan"))
    m["epoch_sec"] = sys_.get("epoch_sec", float("nan"))
    m["gpu_gb"] = sys_.get("gpu_gb", float("nan"))
    samp = _last_block(log, "sampled", "decoded")
    if samp:
        for k, v in samp.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                m[k] = v
        # Ratios, so a gate expression reads the way it is written in the plan.
        for a, b in (("mae", "mae_adv"), ("crps", "crps_adv")):
            if samp.get(b):
                m[a + "_ratio"] = samp[a] / samp[b]
    dp = os.path.join(trial_dir, "DONE")
    if os.path.exists(dp):
        try:
            d = json.load(open(dp))
            m["wall_min"] = d.get("wall_min")
            m["finish_reason"] = d.get("reason")
        except Exception:
            pass
    return m


def metrics_from_eval(json_path):
    """Read one evaluate_diffusion.py scorecard into the same flat namespace, so
    the objective expressions do not have to care which arm produced them."""
    if not os.path.exists(json_path):
        return None
    try:
        d = json.load(open(json_path))
    except Exception:
        return None
    det, prob = d.get("deterministic", {}), d.get("probabilistic", {})
    dist = d.get("distribution", {})
    m = {"n_crops": d.get("n_crops"), "wall_min": d.get("wall_min"),
         "members": d.get("members"), "steps": d.get("steps"),
         "guidance": d.get("guidance"), "churn": d.get("churn"),
         "ckpt_epoch": d.get("ckpt_epoch"), "git": d.get("git")}

    def g(method, key, default=float("nan")):
        return det.get(method, {}).get(key, default)

    m["mae"] = g("model_mean", "MAE_mmh")
    m["mae_member"] = g("model_member", "MAE_mmh")
    m["mae_adv"] = g("advection", "MAE_mmh")
    m["mae_pers"] = g("persistence", "MAE_mmh")
    m["rmse"] = g("model_mean", "RMSE_mmh")
    m["rmse_adv"] = g("advection", "RMSE_mmh")
    for meth, pre in (("model_mean", ""), ("model_member", "member_"),
                      ("advection", "adv_"), ("persistence", "pers_")):
        bt = det.get(meth, {}).get("by_threshold", {})
        for t in ("1.0", "8.0"):
            if t in bt:
                m[f"{pre}csi_{float(t):g}"] = bt[t].get("CSI", float("nan"))
    # Alias the baseline CSI columns onto the names the training-side
    # objective expressions already use, so one expression serves both arms
    # (the model_mean loop above already wrote csi_1 and csi_8).
    m["csi_1_adv"], m["csi_8_adv"] = m.get("adv_csi_1"), m.get("adv_csi_8")
    m["csi_1_pers"], m["csi_8_pers"] = m.get("pers_csi_1"), m.get("pers_csi_8")
    m["crps_fair"] = prob.get("CRPS_fair_mmh", float("nan"))
    m["crps"] = m["crps_fair"]
    m["crps_adv"] = m["mae_adv"]          # CRPS of a point forecast is its MAE
    m["crps_pers"] = m["mae_pers"]
    m["spread_rmse_ratio"] = prob.get("spread_rmse_ratio", float("nan"))
    M = float(d.get("members") or 8)
    m["spread_rmse_ideal"] = math.sqrt(M / (M + 1.0))
    m["outlier_rate"] = prob.get("outlier_rate", float("nan"))
    m["outlier_rate_ideal"] = prob.get("outlier_rate_ideal", 2.0 / (M + 1.0))
    m["rank_flatness_rmse"] = prob.get("rank_flatness_rmse", float("nan"))
    bands = dist.get("psd_bands", {})
    for meth, pre in (("model_member", ""), ("model_mean", "mean_"),
                      ("advection", "adv_"), ("persistence", "pers_")):
        b = bands.get(meth, {})
        if b:
            m[pre + "psd_ratio_2_8km"] = b.get("psd_mean_ratio", float("nan"))
            m[pre + "psd_band_power"] = b.get("psd_band_power", float("nan"))
    for a, b in (("mae", "mae_adv"), ("crps", "crps_adv")):
        if m.get(b):
            m[a + "_ratio"] = m[a] / m[b]
    return m


SAFE_FUNCS = {"abs": abs, "min": min, "max": max, "sqrt": math.sqrt,
              "log": math.log, "exp": math.exp, "isfinite": math.isfinite}


def evaluate_expr(expr, metrics):
    """Evaluate an objective or gate expression in the metrics namespace.

    The expression comes from this file's own OBJECTIVES/GATES tables or from the
    user's own command line, never from anything read off disk or off the
    network, so a restricted eval is the right tool: it keeps the objective
    readable in the methods chapter instead of hiding it inside a lookup table."""
    env = dict(SAFE_FUNCS)
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            env[k] = v
    try:
        return eval(expr, {"__builtins__": {}}, env)              # noqa: S307
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Executors
# ----------------------------------------------------------------------------
class Executor:
    def submit(self, trial):
        raise NotImplementedError

    def poll(self, trial):
        """Return True when the trial is no longer running."""
        raise NotImplementedError

    def kill(self, trial):
        pass


class LocalExecutor:
    """Run trainers as child processes on this machine. Used on the Exeter L4
    and inside a single interactive Slurm allocation."""

    name = "local"

    def __init__(self, log):
        self.log = log

    def submit(self, trial):
        os.makedirs(trial["dir"], exist_ok=True)
        out = open(os.path.join(trial["dir"], "trial.out"), "a", buffering=1)
        out.write(f"\n==== {now()} :: {' '.join(shlex.quote(c) for c in trial['cmd'])}\n")
        trial["_proc"] = subprocess.Popen(trial["cmd"], stdout=out,
                                          stderr=subprocess.STDOUT, cwd=HERE)
        trial["_out"] = out
        trial["pid"] = trial["_proc"].pid
        return trial

    def poll(self, trial):
        p = trial.get("_proc")
        if p is None:
            return True
        rc = p.poll()
        if rc is None:
            return False
        trial["returncode"] = rc
        if trial.get("_out"):
            trial["_out"].close()
        return True

    def kill(self, trial):
        p = trial.get("_proc")
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=30)
            except Exception:
                p.kill()


class SlurmExecutor:
    """Submit one job per trial through the project's gpu.sbatch wrapper.

    The wrapper is used rather than a bespoke sbatch header on purpose: it
    already carries --exclude=gpuhost007 (whose CUDA driver fails to initialise
    while NVML still reports the GPU present, which once cost about ten hours of
    silent CPU training) and a CUDA assert that fails a job in one second rather
    than degrading to CPU. Bypassing it reintroduces both failure modes."""

    name = "slurm"

    def __init__(self, log, wrapper, extra_sbatch=(), safety=2.0,
                 min_time="00:20:00", max_time="23:30:00", grace=90.0):
        self.log, self.wrapper = log, os.path.expanduser(wrapper)
        self.extra, self.safety = list(extra_sbatch), safety
        self.min_time, self.max_time = min_time, max_time
        self.grace = grace
        if not os.path.exists(self.wrapper):
            raise SystemExit(
                f"ERROR: gpu.sbatch wrapper not found at {self.wrapper}. Create it "
                "(README_jasmin.md section 8) or pass --gpu-sbatch. Never submit a "
                "GPU job to Orchid without the gpuhost007 exclusion.")

    def _walltime(self, hours):
        secs = int(max(self._to_s(self.min_time),
                       min(self._to_s(self.max_time), hours * 3600 * self.safety)))
        return f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"

    @staticmethod
    def _to_s(hhmmss):
        h, m, s = (int(x) for x in hhmmss.split(":"))
        return h * 3600 + m * 60 + s

    def submit(self, trial):
        os.makedirs(trial["dir"], exist_ok=True)
        sb = ["sbatch", "--parsable",
              f"--job-name={trial['job_name']}",
              f"--time={self._walltime(trial['cost_h_est'])}",
              f"--output={os.path.join(trial['dir'], 'slurm-%j.out')}"]
        sb += self.extra + [self.wrapper] + trial["cmd"]
        trial["sbatch"] = sb
        with open(os.path.join(trial["dir"], "submit.sh"), "w") as fh:
            fh.write("#!/bin/sh\n# emitted by hpo_search.py " + now() + "\n")
            fh.write(" ".join(shlex.quote(c) for c in sb) + "\n")
        res = subprocess.run(sb, capture_output=True, text=True, cwd=HERE)
        if res.returncode != 0:
            self.log(f"  SUBMIT FAILED {trial['id']}: {res.stderr.strip()[:300]}")
            trial["error"] = res.stderr.strip()[:500]
            trial["jobid"] = None
            return trial
        trial["jobid"] = res.stdout.strip().split(";")[0]
        trial["_submitted_at"] = time.time()
        self.log(f"  submitted {trial['id']} as job {trial['jobid']}")
        return trial

    def poll(self, trial):
        jid = trial.get("jobid")
        if not jid:
            return True
        # A freshly submitted job can be absent from squeue for a few seconds
        # while the controller registers it. Without this grace period an empty
        # squeue immediately after submission reads as "finished", the trial is
        # scored with no metrics, and successive halving promotes on noise.
        if time.time() - trial.get("_submitted_at", 0) < self.grace:
            return False
        res = subprocess.run(["squeue", "-h", "-j", str(jid), "-o", "%T"],
                             capture_output=True, text=True)
        if res.returncode != 0 and "Invalid job id" not in (res.stderr or ""):
            # squeue itself failed (controller busy, transient network). Treat
            # the job as still running rather than silently declaring it done.
            self.log(f"  squeue failed for {trial['id']}: "
                     f"{(res.stderr or '').strip()[:160]}; treating as running")
            return False
        state = res.stdout.strip()
        if not state:
            return True                    # gone from the queue: finished or failed
        trial["slurm_state"] = state.splitlines()[0]
        return False

    def kill(self, trial):
        if trial.get("jobid"):
            subprocess.run(["scancel", str(trial["jobid"])], capture_output=True)


# ----------------------------------------------------------------------------
# The study
# ----------------------------------------------------------------------------
class Study:
    def __init__(self, out, spec, log):
        self.out, self.spec, self.log = out, spec, log
        os.makedirs(out, exist_ok=True)
        self.trials_path = os.path.join(out, "trials.jsonl")
        self.tp_path = os.path.join(out, "throughput.json")
        self.throughput = {}
        if os.path.exists(self.tp_path):
            try:
                self.throughput = json.load(open(self.tp_path))
            except Exception:
                self.throughput = {}
        sp = os.path.join(out, "study.json")
        if os.path.exists(sp):
            old = json.load(open(sp))
            drift = [k for k in ("space", "arm", "objective", "gate", "seed")
                     if old.get(k) != spec.get(k)]
            if drift:
                raise SystemExit(
                    f"ERROR: {sp} already describes a different study (differs on "
                    f"{', '.join(drift)}). Point --out at a new directory rather "
                    "than mixing two studies in one trials table.")
            spec["created"] = old.get("created", spec["created"])
            spec["reruns"] = old.get("reruns", 0) + 1
        atomic_json(spec, sp)

    def event(self, **row):
        row = {"ts": now(), **row}
        with open(self.trials_path, "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def note_throughput(self, params, metrics):
        ips = metrics.get("imgs_per_s") if metrics else None
        if ips and math.isfinite(ips) and ips > 0:
            self.throughput[S.arch_signature(params)] = round(float(ips), 1)
            atomic_json(self.throughput, self.tp_path)

    def spent_gpu_h(self):
        """GPU hours actually consumed, for the budget guard.

        The trainer's own wall_min (from its DONE marker) is preferred over the
        harness's wall clock, and this matters on Slurm: the harness clock starts
        at submission, so queue wait would otherwise be charged as GPU time and
        the budget guard would stop a study that had barely used the allocation."""
        tot = 0.0
        if not os.path.exists(self.trials_path):
            return 0.0
        for line in open(self.trials_path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("event") != "finish":
                continue
            wm = (r.get("metrics") or {}).get("wall_min")
            if wm is None:
                wm = r.get("wall_min")
            if wm:
                tot += float(wm) / 60.0
        return tot


def failure_tail(trial, n=12):
    """The last few meaningful lines a failed trial printed, so the cause is
    visible in the harness log instead of only in a Slurm file nobody opens."""
    cands = sorted(glob.glob(os.path.join(trial["dir"], "slurm-*.out")),
                   key=os.path.getmtime, reverse=True)
    path = cands[0] if cands else os.path.join(trial["dir"], "trial.out")
    if not os.path.exists(path):
        return None
    try:
        with open(path, errors="replace") as fh:
            lines = [ln.rstrip() for ln in fh.read().splitlines() if ln.strip()]
    except Exception:
        return None
    return lines[-n:] if lines else None


def trial_done(arm, trial):
    if S.ARMS[arm]["kind"] == "infer":
        return os.path.exists(trial["eval_json"])
    return os.path.exists(os.path.join(trial["dir"], "DONE"))


def collect(arm, trial, baselines, psd_ceiling, objective_expr, gate_expr):
    m = (metrics_from_eval(trial["eval_json"]) if S.ARMS[arm]["kind"] == "infer"
         else metrics_from_training(trial["dir"]))
    if m is None:
        return None, None, None
    b = (baselines or {}).get(trial["rung"]["name"], {})
    for k, v in b.items():
        m.setdefault(k, v)
    m.setdefault("psd_ceiling", psd_ceiling)
    obj = evaluate_expr(objective_expr, m)
    gate = evaluate_expr(gate_expr, m)
    return m, (float(obj) if isinstance(obj, (int, float)) else None), bool(gate)


# ----------------------------------------------------------------------------
# Median pruning
# ----------------------------------------------------------------------------
def median_prune_check(running, rung, log):
    """Optuna's MedianPruner, applied across the trials currently in flight.

    A trial is killed if, at an epoch that at least three other trials in this
    rung have already passed, its validation loss is worse than the median of
    theirs. Successive halving between rungs is the primary early-stopping
    mechanism here and this is the secondary one inside a rung; on a 3-epoch
    screening rung it rarely fires, which is why it is off by default."""
    curves = {}
    for t in running:
        lp = os.path.join(t["dir"], "train_log.json")
        if not os.path.exists(lp):
            continue
        try:
            rec = json.load(open(lp))
        except Exception:
            continue
        curves[t["id"]] = {r["epoch"]: r["val"]["loss_w"] for r in rec if r.get("val")}
    killed = []
    for t in running:
        c = curves.get(t["id"], {})
        if not c:
            continue
        ep = max(c)
        peers = [v[ep] for k, v in curves.items() if k != t["id"] and ep in v]
        if len(peers) < 3:
            continue
        med = sorted(peers)[len(peers) // 2]
        if c[ep] > med:
            log(f"  prune {t['id']}: val {c[ep]:.4f} > median {med:.4f} at epoch {ep}")
            killed.append(t)
    return killed


# ----------------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------------
def plan_table(arm, cells, schedule, throughput, log):
    """Print, and return, the estimated cost of the whole study, alongside what
    the same search would have cost at full training fidelity. That comparison
    is the justification for the method and belongs in the write-up."""
    rows, total = [], 0.0
    for rung, n_keep in schedule:
        per = [S.train_cost_h(arm, rung, c["params"], throughput)
               for c in cells[:n_keep]]
        sub = sum(per)
        total += sub
        rows.append({"rung": rung["name"], "rows": rung["rows"],
                     "epochs": rung["epochs"], "n_trials": len(per),
                     "gpu_h": round(sub, 2),
                     "gpu_h_per_trial": round(sub / max(len(per), 1), 3)})
    full = (sum(S.full_fidelity_cost_h(arm, c["params"], measured=throughput)
                for c in cells) if S.ARMS[arm]["kind"] == "train" else 0.0)
    log("")
    log(f"  {'rung':<6} {'rows':>8} {'epochs':>7} {'trials':>7} {'GPU-h/trial':>12} {'GPU-h':>8}")
    for r in rows:
        log(f"  {r['rung']:<6} {r['rows']:>8} {r['epochs']:>7} {r['n_trials']:>7} "
            f"{r['gpu_h_per_trial']:>12.3f} {r['gpu_h']:>8.2f}")
    log(f"  {'TOTAL':<6} {'':>8} {'':>7} {'':>7} {'':>12} {total:>8.2f}")
    if full:
        log(f"\n  the same {len(cells)} configurations trained to 50 epochs at full "
            f"fidelity: {full:.0f} A100-hours")
        log(f"  multi-fidelity screening cost: {total:.1f} A100-hours "
            f"({100.0 * total / max(full, 1e-9):.1f} percent of that)")
    log("")
    return rows, total, full


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Hyperparameter optimisation for the LDM and CorrDiff arms.",
        epilog="Everything after a bare -- is appended verbatim to every trial command.")
    ap.add_argument("--space", required=True, choices=sorted(S.SPACES),
                    help="search space from hpo_spaces.SPACES")
    ap.add_argument("--out", required=True, help="study directory")
    ap.add_argument("--exec", dest="executor", default="dry",
                    choices=["dry", "local", "slurm"],
                    help="dry (default) plans and costs the study without running it")
    ap.add_argument("--sampler", default=None,
                    choices=["grid", "random", "tpe", "optuna-tpe"],
                    help="default: the space's own declared stage")
    ap.add_argument("--n-trials", type=int, default=24,
                    help="number of configurations for random/tpe (grid uses its own size)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="trials in flight at once. On Slurm this is the job "
                         "concurrency (the orchid QoS allows 8). With --exec "
                         "local every trial shares ONE GPU, so anything above 1 "
                         "will contend for memory and probably OOM.")
    ap.add_argument("--eta", type=int, default=3,
                    help="successive-halving reduction factor")
    ap.add_argument("--rungs", default=None,
                    help="comma-separated rung names to use, e.g. r0,r1")
    ap.add_argument("--hyperband", action="store_true",
                    help="run Hyperband's outer loop of brackets instead of a "
                         "single successive-halving pass")
    ap.add_argument("--objective", default=None,
                    help="name from hpo_spaces.OBJECTIVES, or a raw expression "
                         "(larger is better)")
    ap.add_argument("--gate", default=None,
                    help="name from hpo_spaces.GATES, or a raw boolean expression")
    ap.add_argument("--psd-ceiling", type=float, default=S.PSD_CEILING_POOLED,
                    help="codec oracle 2-8 km PSD ratio, MEAN-OF-RATIOS estimator, "
                         "which is the one every psd_ratio_2_8km in this codebase "
                         "uses. Default 0.903, measured by stage 5 on the full "
                         "validation split (13,281 crops per lead, epoch-17 codec). "
                         "Do not pass the band-power oracle of 0.970 here: the two "
                         "estimators disagree by 8 percent and the objective reads "
                         "mean-of-ratios.")
    ap.add_argument("--baselines", default=None,
                    help="baselines.json from hpo_baselines.py (paired persistence)")
    ap.add_argument("--seed", type=int, default=0,
                    help="pinned on every trial so comparisons stay paired")
    ap.add_argument("--sampler-seed", type=int, default=0)
    ap.add_argument("--seed-from", default=None,
                    help="another study directory whose finished trials seed the "
                         "surrogate for a tpe run")
    ap.add_argument("--seed-from-rung", default=None,
                    help="which rung of --seed-from to take observations from "
                         "(default: that study's cheapest rung, the only one "
                         "where every configuration was evaluated)")
    ap.add_argument("--budget-gpu-h", type=float, default=None,
                    help="refuse to launch a study whose plan exceeds this")
    ap.add_argument("--max-trials", type=int, default=None,
                    help="truncate the candidate list; the drop is logged, never silent")
    ap.add_argument("--prune", default="none", choices=["none", "median"])
    ap.add_argument("--poll", type=int, default=60, help="seconds between polls")
    ap.add_argument("--gpu-sbatch", default="~/dissertation/gpu.sbatch")
    ap.add_argument("--sbatch-extra", default="",
                    help="extra sbatch flags, e.g. '--qos=orchid --account=orchid'")
    ap.add_argument("--python", default=None,
                    help="interpreter for trial commands. Default: the bare word "
                         "'python' under --exec slurm, so gpu.sbatch's activated "
                         "conda environment resolves it, and sys.executable "
                         "otherwise. Only override if you know why.")
    ap.add_argument("--script-dir", default=HERE,
                    help="where the stage scripts live (default: next to this "
                         "file; override when the scripts are deployed flat "
                         "somewhere else, or to point at a stub for testing)")
    ap.add_argument("--force", action="store_true",
                    help="launch even if the plan exceeds --budget-gpu-h")
    ap.add_argument("passthrough", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    passthrough = [a for a in args.passthrough if a != "--"]
    if args.python is None:
        args.python = default_python(args.executor)
    space = S.SPACES[args.space]
    arm = space["arm"]
    meta = S.ARMS[arm]
    os.makedirs(args.out, exist_ok=True)
    log = Tee(os.path.join(args.out, "hpo.log"))

    missing = [f for f in meta["required"] if f not in passthrough]
    if missing and args.executor != "dry":
        raise SystemExit(f"ERROR: the {arm} arm requires {', '.join(missing)}; pass "
                         f"them after a bare -- on the command line.")

    obj_name = args.objective or S.DEFAULT_OBJECTIVE[arm]
    gate_name = args.gate or S.DEFAULT_GATE[arm]
    obj_expr = S.OBJECTIVES.get(obj_name, {}).get("expr", obj_name)
    gate_expr = S.GATES.get(gate_name, {}).get("expr", gate_name)

    rungs = S.RUNGS[arm]
    if args.rungs:
        want = [r.strip() for r in args.rungs.split(",")]
        rungs = [r for r in rungs if r["name"] in want]
        if not rungs:
            raise SystemExit(f"ERROR: --rungs {args.rungs} matched no rung of arm {arm}.")

    sampler_kind = args.sampler or space["stage"]
    baselines = json.load(open(args.baselines)) if args.baselines else None
    if baselines is None and arm != "inference":
        log("WARNING: no --baselines given, so persistence columns will be absent "
            "and any objective or gate that mentions mae_pers/crps_pers will be "
            "undefined. Run hpo_baselines.py first; the brief asks for every run "
            "to be compared against persistence.")

    spec = {"space": args.space, "arm": arm, "sampler": sampler_kind,
            "objective": obj_name, "objective_expr": obj_expr,
            "gate": gate_name, "gate_expr": gate_expr,
            "rungs": rungs, "eta": args.eta, "seed": args.seed,
            "sampler_seed": args.sampler_seed, "n_trials": args.n_trials,
            "psd_ceiling": args.psd_ceiling, "passthrough": passthrough,
            "executor": args.executor, "parallel": args.parallel,
            "prune": args.prune, "git": git_hash(), "created": now(),
            "space_note": space.get("note"), "reruns": 0,
            "host": os.uname().nodename if hasattr(os, "uname") else "windows"}
    study = Study(args.out, spec, log)

    log(f"\n=== study {args.space} :: arm {arm} :: sampler {sampler_kind} "
        f":: git {spec['git']} ===")
    log(f"objective (maximised): {obj_name} = {obj_expr}")
    log(f"        admissibility: {gate_name} = {gate_expr}")
    log(f"           rung ladder: " +
        ", ".join(f"{r['name']}({r['rows']}x{r['epochs']})" for r in rungs))

    # ---- candidates --------------------------------------------------------
    if sampler_kind == "grid":
        cells = expand_grid(space)
        bad = [c for c in cells if c["invalid"]]
        for c in bad:
            log(f"  SKIP invalid cell {c['origin']}: {'; '.join(c['invalid'])}")
        cells = [c for c in cells if not c["invalid"]]
        sampler = None
    else:
        sampler = {"random": RandomSampler,
                   "tpe": TPESampler,
                   "optuna-tpe": OptunaTPESampler}[sampler_kind](
            space, seed=args.sampler_seed)
        cells = None                       # generated adaptively below

    seeded = []
    if args.seed_from:
        rp = os.path.join(args.seed_from, "ranking.json")
        if os.path.exists(rp):
            prev = json.load(open(rp))
            # Seed from ONE rung, not from the whole ranking. A ranking table
            # holds the same configuration at several fidelities, scored on
            # different crop sets, and feeding a surrogate two contradictory
            # objectives for one point is worse than feeding it none. The
            # cheapest rung is the default because it is the only one where every
            # configuration was evaluated, so the observation set is complete and
            # internally consistent.
            want = args.seed_from_rung or (prev.get("spec", {}).get("rungs")
                                           or [{}])[0].get("name")
            keys = set(space.get("axes", {}))
            skipped = 0
            for r in prev.get("trials", []):
                if r.get("objective") is None:
                    continue
                if want and r.get("rung") != want:
                    skipped += 1
                    continue
                p = {k: v for k, v in (r.get("params") or {}).items() if k in keys}
                if len(p) == len(keys):
                    seeded.append((p, r["objective"]))
            log(f"seeded the surrogate with {len(seeded)} observations from "
                f"{args.seed_from} at rung '{want}' ({skipped} rows at other rungs "
                "ignored so the surrogate sees one fidelity only)")
            if not seeded:
                log("  WARNING: nothing usable was seeded. Check that the earlier "
                    "study's parameters cover every axis of this space.")
        else:
            log(f"WARNING: --seed-from {args.seed_from} has no ranking.json; "
                "starting cold.")

    if cells is None:
        proposals = sampler.ask(args.n_trials, seeded)
        cells = []
        for p in proposals:
            full = dict(S.INCUMBENT)
            full.update(p)
            keys = sorted(set(space["axes"]))
            cell = {k: full[k] for k in keys}
            bad = S.check_constraints(cell)
            cells.append({"params": cell, "origin": sampler.name, "invalid": bad})
        for c in [c for c in cells if c["invalid"]]:
            log(f"  SKIP invalid proposal: {'; '.join(c['invalid'])}")
        cells = [c for c in cells if not c["invalid"]]

    if args.max_trials and len(cells) > args.max_trials:
        log(f"  NOTE: candidate list truncated from {len(cells)} to {args.max_trials} "
            f"by --max-trials. Dropped cells: "
            f"{', '.join(c['origin'] for c in cells[args.max_trials:])}")
        cells = cells[:args.max_trials]

    log(f"\n{len(cells)} candidate configurations")
    for i, c in enumerate(cells):
        diff = {k: v for k, v in c["params"].items() if v != S.INCUMBENT.get(k)}
        log(f"  {i:>3} {c['origin']:<22} " +
            (", ".join(f"{k}={v}" for k, v in sorted(diff.items())) or "(incumbent)"))

    schedule = S.sha_schedule(len(cells), rungs, eta=args.eta)
    rows, total, full = plan_table(arm, cells, schedule, study.throughput, log)

    # Show one fully assembled command before anything is spent. Every flag a
    # trial will actually carry is visible here, which is the cheapest possible
    # place to catch a wrong pack directory or a missing arm flag.
    example = build_cmd(arm, args.script_dir, cells[0]["params"], rungs[0],
                        os.path.join(args.out, "trial_000_" + rungs[0]["name"]),
                        passthrough, args.seed, tag="_trial_000_" + rungs[0]["name"],
                        python=args.python)
    log("example trial command (candidate 0 at rung " + rungs[0]["name"] + "):")
    log("  " + " ".join(shlex.quote(c) for c in example))
    log("")

    atomic_json({"spec": spec, "candidates": cells, "schedule": rows,
                 "gpu_h_planned": round(total, 2),
                 "gpu_h_if_full_fidelity": round(full, 1),
                 "example_cmd": example},
                os.path.join(args.out, "plan.json"))

    if args.budget_gpu_h and total > args.budget_gpu_h and not args.force:
        raise SystemExit(
            f"\nERROR: planned {total:.1f} A100-hours exceeds --budget-gpu-h "
            f"{args.budget_gpu_h}. Reduce --n-trials or --rungs, drop a rung, or "
            "pass --force if the budget figure is the thing that is wrong.")

    if args.executor == "dry":
        log(f"\ndry run: plan written to {os.path.join(args.out, 'plan.json')}. "
            "Re-run with --exec slurm (JASMIN) or --exec local (L4) to execute.")
        return

    # ---- execution ---------------------------------------------------------
    if args.executor == "slurm":
        ex = SlurmExecutor(log, args.gpu_sbatch,
                           extra_sbatch=shlex.split(args.sbatch_extra))
    else:
        ex = LocalExecutor(log)

    survivors = list(range(len(cells)))
    ranking = []
    n_finished = n_ok = 0          # study-wide, for the fail-fast guard
    for rung, n_keep in schedule:
        survivors = survivors[:n_keep]
        log(f"\n---- rung {rung['name']} ({rung['rows']} rows x {rung['epochs']} "
            f"epochs) :: {len(survivors)} trials ----")
        trials = []
        for idx in survivors:
            params = cells[idx]["params"]
            tid = f"trial_{idx:03d}_{rung['name']}"
            tdir = os.path.join(args.out, tid)
            t = {"id": tid, "index": idx, "rung": rung, "params": params,
                 "dir": tdir, "origin": cells[idx]["origin"],
                 "param_hash": param_hash(params),
                 "job_name": f"hpo-{args.space}-{idx:03d}{rung['name']}",
                 "eval_json": os.path.join(args.out, "eval",
                                           f"diffusion_eval_{tid}.json"),
                 "cost_h_est": S.train_cost_h(arm, rung, params, study.throughput)}
            t["cmd"] = build_cmd(arm, args.script_dir, params, rung, tdir, passthrough,
                                 args.seed, tag=f"_{tid}", python=args.python)
            # A trial directory belongs to exactly one parameter set. If the
            # space was edited between runs, refuse rather than resume onto a
            # mismatched checkpoint.
            cp = os.path.join(tdir, "hpo_trial.json")
            if os.path.exists(cp):
                old = json.load(open(cp))
                if old.get("param_hash") != t["param_hash"]:
                    raise SystemExit(
                        f"ERROR: {tdir} was created for a different configuration "
                        f"({old.get('param_hash')} != {t['param_hash']}). The search "
                        "space changed since this study started. Use a new --out.")
            os.makedirs(tdir, exist_ok=True)
            atomic_json({k: t[k] for k in
                         ("id", "index", "params", "param_hash", "origin",
                          "cmd", "cost_h_est")} | {"rung": rung["name"]}, cp)
            trials.append(t)

        pending, resumed = [], []
        for t in trials:
            (resumed if trial_done(arm, t) else pending).append(t)
        for t in resumed:
            log(f"  resume: {t['id']} already has its completion marker, skipping")
            study.event(event="skip", trial=t["id"], rung=rung["name"],
                        params=t["params"])

        inflight, queue = [], list(pending)
        while queue or inflight:
            while queue and len(inflight) < max(1, args.parallel):
                t = queue.pop(0)
                spent = study.spent_gpu_h()
                if args.budget_gpu_h and spent > args.budget_gpu_h and not args.force:
                    log(f"  BUDGET STOP: {spent:.1f} A100-hours already spent, "
                        f"limit {args.budget_gpu_h}. {len(queue) + 1} trials not "
                        "launched; re-run with a higher --budget-gpu-h to continue.")
                    queue = []
                    break
                t["t0"] = time.time()
                study.event(event="launch", trial=t["id"], rung=rung["name"],
                            params=t["params"], cost_h_est=round(t["cost_h_est"], 3),
                            cmd=" ".join(shlex.quote(c) for c in t["cmd"]))
                ex.submit(t)
                inflight.append(t)
            if not inflight:
                break
            time.sleep(args.poll)
            if args.prune == "median" and len(inflight) > 3:
                for t in median_prune_check(inflight, rung, log):
                    ex.kill(t)
                    t["pruned"] = True
                    study.event(event="prune", trial=t["id"], rung=rung["name"])
            still = []
            for t in inflight:
                if ex.poll(t) or t.get("pruned"):
                    m, obj, gate = collect(arm, t, baselines, args.psd_ceiling,
                                           obj_expr, gate_expr)
                    t["metrics"], t["objective"], t["gate_pass"] = m, obj, gate
                    t["wall_min"] = round((time.time() - t["t0"]) / 60.0, 1)
                    study.note_throughput(t["params"], m)
                    ok = trial_done(arm, t)
                    study.event(event="finish", trial=t["id"], rung=rung["name"],
                                params=t["params"], objective=obj, gate_pass=gate,
                                complete=ok, pruned=bool(t.get("pruned")),
                                wall_min=t["wall_min"], jobid=t.get("jobid"),
                                returncode=t.get("returncode"), metrics=m)
                    tag = ("PRUNED" if t.get("pruned") else
                           ("ok" if ok else "INCOMPLETE"))
                    log(f"  {t['id']:<24} {tag:<10} objective "
                        f"{('%.5f' % obj) if obj is not None else '   n/a':>10} "
                        f"gate {'pass' if gate else 'FAIL':<4} "
                        f"{t['wall_min']:.1f} min")
                    # An INCOMPLETE trial means the command itself failed, and the
                    # reason is sitting in the job's stdout. Surface it here rather
                    # than making the reader go and find it: the first real run of
                    # this harness lost nine trials to a wrong interpreter that one
                    # line of traceback would have named immediately.
                    if not ok and not t.get("pruned"):
                        for line in (failure_tail(t) or ["(no output captured)"]):
                            log(f"      | {line}")
                    n_finished += 1
                    n_ok += 1 if ok else 0
                else:
                    still.append(t)
            inflight = still
            # Fail fast on a systematically broken command. If the first two
            # trials of a study both produced nothing, every later trial will fail
            # the same way, and continuing just spends the queue on it.
            if n_finished >= 2 and n_ok == 0:
                for u in inflight:
                    ex.kill(u)
                raise SystemExit(
                    f"\nERROR: the first {n_finished} trials produced no metrics, so the "
                    "trial command is broken rather than the configurations being bad. "
                    "The lines above are that command's own output. Common causes: the "
                    "interpreter cannot import torch (check --python; under --exec slurm "
                    "it must be the bare word 'python' so gpu.sbatch's conda activation "
                    "resolves it), a --latents-dir or --mu-dir that does not exist on the "
                    "compute node, or a missing gpu.sbatch wrapper. Nothing is lost: fix "
                    "the cause and re-run the identical command, and completed trials are "
                    "skipped.")

        scored = []
        for t in trials:
            if "objective" not in t:
                m, obj, gate = collect(arm, t, baselines, args.psd_ceiling,
                                       obj_expr, gate_expr)
                t["metrics"], t["objective"], t["gate_pass"] = m, obj, gate
            scored.append(t)
        # Ranking: admissible trials first, then by objective. A trial that
        # failed its gate or failed to produce metrics stays in the table with
        # its reason, it is never dropped.
        scored.sort(key=lambda t: (0 if t.get("gate_pass") else 1,
                                   -(t["objective"] if t["objective"] is not None
                                     else -1e300)))
        survivors = [t["index"] for t in scored]
        ranking = [{"rung": rung["name"], "trial": t["id"], "index": t["index"],
                    "origin": t["origin"], "params": t["params"],
                    "objective": t["objective"], "gate_pass": t.get("gate_pass"),
                    "wall_min": t.get("wall_min"), "metrics": t.get("metrics")}
                   for t in scored] + ranking
        atomic_json({"spec": spec, "trials": ranking},
                    os.path.join(args.out, "ranking.json"))
        log(f"  rung {rung['name']} ranking: " +
            ", ".join(f"{t['id'].split('_')[1]}"
                      f"({'%.4f' % t['objective'] if t['objective'] is not None else 'na'})"
                      for t in scored[:6]))

    best = next((r for r in ranking if r["gate_pass"]), ranking[0] if ranking else None)
    if best:
        log(f"\nBEST: {best['trial']}  objective {best['objective']}")
        diff = {k: v for k, v in best["params"].items() if v != S.INCUMBENT.get(k)}
        log("  differs from the incumbent on: " +
            (", ".join(f"{k}={v}" for k, v in sorted(diff.items())) or "nothing"))
        if meta["kind"] == "train":
            full_h = S.full_fidelity_cost_h(arm, best["params"],
                                            measured=study.throughput)
            full_rung = {"name": "full", "rows": meta["n_train_rows"], "epochs": 50,
                         "sample_crops": 16, "psd_crops": 64}
            conf = build_cmd(arm, args.script_dir, best["params"], full_rung,
                             os.path.join(os.path.dirname(args.out.rstrip("/\\")),
                                          f"{args.space}_winner"),
                             passthrough, args.seed, python=args.python)
            # The screening rung suppressed checkpoint archiving and set patience
            # to zero; a confirmation run wants both back. --limit is dropped
            # rather than pinned at the row count this harness believes in: if
            # the pack has even one row more than n_train_rows, a pinned --limit
            # silently subsamples the confirmation run.
            conf = [c for c in conf if c != "--no-keep-sampled"]
            if "--limit" in conf:
                i = conf.index("--limit")
                del conf[i:i + 2]
            for a, b in (("--patience", "15"), ("--sample-every", "5")):
                if a in conf:
                    conf[conf.index(a) + 1] = b
            log("\nConfirm it at full fidelity before quoting it. A screening rung "
                "is a ranking statistic, not a result.")
            log(f"  estimated cost: {full_h:.1f} A100-hours")
            log("  " + " ".join(shlex.quote(c) for c in conf))
            atomic_json({"trial": best["trial"], "params": best["params"],
                         "objective": best["objective"],
                         "full_fidelity_gpu_h": round(full_h, 2),
                         "confirm_cmd": conf},
                        os.path.join(args.out, "winner.json"))
    log(f"\ntotal measured GPU time this study: {study.spent_gpu_h():.2f} hours")
    log(f"next: python hpo_report.py --study {args.out}")


if __name__ == "__main__":
    main()
