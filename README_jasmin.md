# Running the diffusion training on JASMIN (Orchid GPU)

This guide runs the diffusion training stages of the pipeline (the +60 latent
diffusion model, then the multi-lead CorrDiff model) on **JASMIN**, whose
**Orchid** cluster gives you Nvidia **A100** GPUs instead of the Exeter L4.

Read `README_server.md` first: it covers the advection prior build and the VAE
training, which both run on Exeter, not here. The stage scripts, flags, and
acceptance criteria for those two stages are documented there.

Everything below assumes the data contract and constants in `CLAUDE.md` are
unchanged.

---

## 0. Why move, and what runs where

The L4 (24 GB, ~30 TFLOP FP16) is the bottleneck for diffusion, which trains for
many GPU-days. An A100 (40 or 80 GB, ~300 TFLOP with TF32/BF16) is roughly an
order of magnitude faster and has enough memory to raise the batch size. JASMIN
also lets you request **4 A100s on one node** for data-parallel training later.

JASMIN Orchid A100 is used for diffusion training only: the +60 LDM first
(Phase 1), then the multi-lead CorrDiff model (Phase 2). The advection prior
build (CPU) and the VAE training (L4 GPU) happen on Exeter, see
`README_server.md`; neither runs on JASMIN any more.

JASMIN has an Orchid GPU cluster and a LOTUS CPU cluster plus interactive and
transfer servers, but this guide only uses Orchid, since diffusion is the only
stage that runs here:

| Work                                                    | Machine                       | Why                                       |
| -------------------------------------------------------- | ------------------------------ | ------------------------------------------- |
| Diffusion training (+60 LDM, then multi-lead CorrDiff)   | **Orchid** (Slurm, A100 GPU)   | the GPU stage, trains for many GPU-days   |
| Editing, small tests, `squeue`, env setup                | **sci** server (`sci-vm-*`)    | shared interactive login, has internet     |
| Moving data in/out                                       | **xfer / hpxfer** servers      | the only servers meant for big transfers   |

Key constraint that shapes everything: **compute nodes only have outbound
HTTP(S) via NAT, no outbound SSH.** That is fine for `git clone`/`git pull` and
`pip install` over HTTPS. It just means you cannot SSH out of a batch node to
pull data; transfers are always initiated on an xfer server or from the outside.

---

## 1. One-time access setup

You need two access roles, applied for in order at
[accounts.jasmin.ac.uk](https://accounts.jasmin.ac.uk):

1. **`jasmin-login`** (gives you the login servers, sci servers, and LOTUS).
2. **`orchid`** (gives you the Orchid GPU cluster and the interactive GPU node).
   Apply for this after `jasmin-login` is approved; you will be asked what
   software and workflow you will run (say: PyTorch latent diffusion for radar
   nowcasting, needs 1 to 4 A100 GPUs).

You also need an **SSH key registered with JASMIN** (a dedicated key, not your
GitHub key). Upload the public key in the accounts portal. See
[Present your SSH key](https://help.jasmin.ac.uk/docs/getting-started/present-ssh-key/)
for the Windows/PuTTY or OpenSSH steps.

---

## 2. Logging in (two hops)

JASMIN is a bastion setup: you land on a login server, then hop to a sci server.
Agent-forwarding (`-A`) carries your key across the hop so you are never asked
for a password on the second machine.

```bash
# from your laptop (add the key to your agent first: ssh-add ~/.ssh/id_jasmin)
ssh -A <jasmin_user>@login.jasmin.ac.uk      # bastion
ssh <jasmin_user>@sci-vm-01.jasmin.ac.uk     # onward to a sci server (or sci-vm-02..05)
```

If the second hop asks for a password, your key is not being forwarded (fix your
agent, `ssh-add -l` should list the key). From a sci server you can reach LOTUS,
Orchid, and the interactive GPU node. Do all interactive work, env setup, and
job submission from a sci server. Do **not** run heavy transfers on login or sci
servers, that is what the xfer servers are for.

---

## 3. Software environment (once)

JASMIN removed the shared anaconda defaults, so install your own **miniforge** in
your home directory and rebuild the `nowcast` env from the repo. Home is 100 GB
and backed up, which is fine for a conda env.

```bash
# on a sci server
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh          # accept ~/miniforge3, DECLINE "conda init"
source ~/miniforge3/bin/activate

# recreate the env used for packing latents and diffusion training (same environment.yml as Exeter)
mamba env create -f ~/dissertation/environment.yml
conda activate nowcast

# GPU stages: add PyTorch. The wheel bundles its own CUDA runtime, so cu124
# works on Orchid's A100s regardless of the system CUDA.
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify PyTorch sees a GPU, but do it **on a GPU**, not on the sci server (sci
servers have no GPU, so `cuda.is_available()` is `False` there, which is
expected). Test it in the interactive GPU session in section 6.2.

Note on activation inside Slurm jobs: `conda activate` needs the conda shell
functions, so batch scripts use `source ~/miniforge3/bin/activate nowcast`
rather than bare `conda activate`.

---

## 4. Storage layout on JASMIN

Two tiers, each with a different job. Map the Exeter `/scratch/dv321/...`
layout onto JASMIN like this:

| Artifact                                                                                    | Exeter path                          | JASMIN path                                      | Notes                                                                              |
| ---------------------------------------------------------------------------------------------| --------------------------------------| ---------------------------------------------------| -------------------------------------------------------------------------------------|
| Code (this repo)                                                                            | `~/`                                 | `~/dissertation` (git clone)                     | home, 100 GB, backed up                                                            |
| Prior cache (`.npz`, copied from Exeter), plus the packed latents `pack_latents.py` writes  | `/scratch/dv321/dissertation/prior/` | `/work/scratch-nopw2/<jasmin_user>/dissertation/`  | large, parallel-write volume; the latents are produced here on JASMIN, not copied |
| Checkpoints, logs, figures                                                                  | `~/dissertation_outputs/`            | `~/dissertation_outputs/`                        | home, backed up, small                                                             |

Two warnings that will bite you if ignored:

- **Scratch is wiped after 28 days without access.** A multi-week project can
  lose its prior cache mid-run. Either keep it warm (the training loop reads it,
  which counts as access) or, better, **request a small Group Workspace (GWS)**
  for the project and put the persistent caches there (`/gws/...`, not wiped).
  If you have no GWS, at minimum copy the packed latents and every checkpoint
  back to `~/dissertation_outputs/` (backed up) as they are produced.
- **Home is 100 GB.** Large packed or latent caches do not go in home. Only
  code, checkpoints, logs, and figures.

Check usage with `pdu -sh ~` (home) and `lfs quota -u $USER /work/scratch-nopw2`
(scratch).

---

## 5. Getting code and data onto JASMIN

### 5.1 Code (fast, do this first)

```bash
# on a sci server
git clone <your-repo-url> ~/dissertation
cd ~/dissertation
```

If the repo is not reachable from JASMIN, `rsync` it from your laptop to an xfer
server (see 5.3), or just `scp` the handful of `.py` files as in `README_server.md`.

### 5.2 Which caches to move for the diffusion stage

For the +60 core run you are training the diffusion model right away, so copy the
existing Exeter cache rather than rebuilding from S3. The data flow is
`npz prior cache + vae_best.pt -> pack_latents.py -> latent memmaps ->
train_ldm.py`, so the two things you must have on JASMIN are:

- the **npz prior cache**: `prior/train/` and `prior/val/` (skip `test/` until the
  final evaluation),
- the **VAE checkpoint** `vae_best.pt`.

You do NOT need the raw frame cache (`frames/`, only used to build priors) or the
VAE packed memmaps (`packed/`, only used to train the VAE, which is already done).

**Pack the latents on JASMIN, not on Exeter.** `pack_latents.py` records the
absolute npz paths it read into the index json, and `train_ldm.py`'s sampled
diagnostics read those npz files back for the pixel-space advection baseline. If
you pack on Exeter and copy only the latents, those paths point at Exeter, do not
exist on JASMIN, and the in-training baseline comparison silently switches off.
Packing on JASMIN keeps the paths valid.

Phase 2 (the multi-lead CorrDiff model) copies the multi-lead prior (`prior_ml/`)
and the retrained `vae_best.pt` from Exeter the same way as the +60 cache above;
see section 5.3.

### 5.3 Copying the cache from Exeter

Transfers go through an xfer server. Because JASMIN cannot SSH outward, push from
the Exeter side (which can reach the internet). Run these **on the Exeter server**:

```bash
JUSER=<jasmin_user>
DST=$JUSER@xfer-vm-01.jasmin.ac.uk
SCR=/work/scratch-nopw2/$JUSER/dissertation

# the npz prior cache (train + val). --mkpath creates the target dirs (rsync >= 3.2.3)
rsync -avP --mkpath /scratch/dv321/dissertation/prior/train/ $DST:$SCR/prior/train/
rsync -avP --mkpath /scratch/dv321/dissertation/prior/val/   $DST:$SCR/prior/val/

# the frozen VAE codec (small)
rsync -avP --mkpath ~/dissertation_outputs/vae_v2/vae_best.pt $DST:dissertation_outputs/vae_v2/
```

If your rsync predates 3.2.3 (no `--mkpath`), create the dirs first:
`ssh $DST "mkdir -p $SCR/prior/train $SCR/prior/val dissertation_outputs/vae_v2"`.
Use `hpxfer3`/`hpxfer4` (physical, faster) for the prior cache if it is large. For
hundreds of GB, JASMIN recommends **Globus** over rsync.

Phase 2 (multi-lead CorrDiff): once the multi-lead prior is built on Exeter
(`README_server.md` section 5b) and the VAE v2 is retrained on that data (see
`PLAN_to_Aug28.md`), copy the cache the same way, into its own `prior_ml/`
path so it never collides with the +60 cache above. Run this **on the Exeter
server** too:

```bash
JUSER=<jasmin_user>
DST=$JUSER@xfer-vm-01.jasmin.ac.uk
SCR=/work/scratch-nopw2/$JUSER/dissertation

# the multi-lead npz prior cache (train + val)
rsync -avP --mkpath /scratch/dv321/dissertation/prior_ml/train/ $DST:$SCR/prior_ml/train/
rsync -avP --mkpath /scratch/dv321/dissertation/prior_ml/val/   $DST:$SCR/prior_ml/val/

# the retrained VAE codec (small). Use a distinct --out when retraining on Exeter
# (train_vae_v2.py defaults --out to ~/dissertation_outputs/vae_v2) so this does
# not overwrite the +60 vae_best.pt above; the name below is a suggestion, not
# yet fixed in the code.
rsync -avP --mkpath ~/dissertation_outputs/vae_v2_ml/vae_best.pt $DST:dissertation_outputs/vae_v2_ml/
```

This second copy is what CorrDiff and the multi-lead LDM train from. It is a
separate codec from the +60 `vae_best.pt` above (see `PLAN_to_Aug28.md`,
"VAE consistency") and does not overwrite it.

Then on JASMIN (section 6): `export DISS_SCRATCH=$SCR`, run `pack_latents.py`
(the A100 encodes the latents), then `train_ldm.py` for the +60 LDM (the
CorrDiff trainer is separate and not yet written, see section 6).

---

## 6. Diffusion stage on Orchid (latent pack + EDM training)

Prerequisites: the prior cache and `vae_best.pt` have been copied from Exeter
(section 5). This section covers both the +60 LDM (Phase 1: the +60 `prior/`
cache and the existing `vae_best.pt`) and the multi-lead CorrDiff model (Phase 2:
`prior_ml/` and the retrained `vae_best.pt`; the CorrDiff trainer is a separate
script and is not written yet, see `PLAN_to_Aug28.md`). `LDM.md` documents
the +60 LDM's algorithm; this section is only the operational side.

Both new scripts honour the `DISS_SCRATCH` environment variable and default to
the Exeter convention (`/scratch/$USER/dissertation`) when it is unset. On
JASMIN, set it once per job (or in `~/.bashrc`). The authoritative source for
the exact path is the `vae_ckpt` field and the row-provenance index entries
inside the pack metas (e.g. `latents_ml_ep17/train_latents_L60_meta.json`),
which record absolute paths at pack time; if `DISS_SCRATCH` does not match
those paths, the pixel-space advection baseline in the sampled diagnostics
silently switches off instead of erroring, so a stale value here is easy to miss:

```bash
export DISS_SCRATCH=/work/scratch-nopw2/$USER/dissertation
```

The stage scripts live under `Code/<stage>/` in the repo, so run them by that path
from `~/dissertation`, and keep your `.sbatch` files in `~/dissertation` too (the
self-resubmit line below refers to `diff.sbatch` by name). The templates use
`Code/2 - VAE Stage/pack_latents.py` and `Code/3 - Diffusion stage/train_ldm.py`;
where a one-off example shows a bare `python train_ldm.py`, prepend the same
`Code/3 - Diffusion stage/` path. The VAE checkpoint is expected at
`~/dissertation_outputs/vae_v2/vae_best.pt` (move it there, or pass `--vae <path>`).

### 6.1 Pack the latents (one-off, GPU, roughly 1-2 h)

Encodes every crop (4 past frames + A + y) with the frozen VAE into
`$DISS_SCRATCH/latents/{train,val}_latents.npy`, and measures `sigma_data`,
the EDM noise constant the trainer needs.

`pack_latents.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=packlat
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

source ~/miniforge3/bin/activate nowcast
cd ~/dissertation
export DISS_SCRATCH=/work/scratch-nopw2/$USER/dissertation

python "Code/2 - VAE Stage/pack_latents.py" --vae ~/dissertation_outputs/vae_v2/vae_best.pt
```

```bash
sbatch pack_latents.sbatch
```

Expected in the log: a `sigma_data (std of z_y - z_A) = 0.xxx` line per split
and `spot check: 8/8 rows verified against source npz`. The value of
`sigma_data` should be well below 1 (the target is a residual); the trainer
refuses to start if it looks wrong. Re-running skips complete packs, and a pack
is automatically invalidated (repacked) if the VAE checkpoint hash changes,
so retraining the VAE and re-running this script is always safe.

Memory note: do not raise `--workers` above about 4. The GPU encode caps
throughput near 65 crops/s, so extra loaders cannot go faster, they only enlarge
the prefetch queue. In-flight work is now hard-bounded at `--prefetch x --batch`
crops (default 4 x 16, roughly 96 MiB) and the output mapping is msync'd every
`--flush-rows` rows, which together fixed the OOM described in
`docs/Latent_Packing_Memory_Analysis.md`. Each progress line prints peak RSS, so
growth is visible in the log.

Multi-lead (Phase 2): the multi-lead cache (`prior_ml/`) is packed as **one shard
per lead**, `{split}_latents_L15.npy` and so on, roughly 28 GiB each. Run one job
per lead so each fits the wall comfortably, and they can queue in parallel:

```bash
for L in 15 30 45 60; do
    sbatch --export=ALL,LEAD=$L pack_latents.sbatch     # script passes --lead $LEAD
done
```

Add `--lead $LEAD --root $DISS_SCRATCH/prior_ml --vae <retrained vae_best.pt>` to the
python line for those jobs. Omitting `--lead` packs every lead found sequentially in
one job, which is simpler but likely exceeds the 3 h wall for four leads. The legacy
+60 cache has untagged filenames and still produces the original
`{split}_latents.npy`, so the Phase 1 run above is unaffected.

Each shard's meta stores its own `sigma_data` plus raw `delta_moments`, because the
residual grows with lead time. A lead-conditioned model needs a single **pooled**
`sigma_data`, which can be computed exactly from those stored moments without
re-encoding.

### 6.2 Smoke test (~10 min, interactive GPU)

```bash
srun --partition=orchid --account=orchid --qos=orchid --gres=gpu:1 \
     --cpus-per-task=8 --mem=64G --time=00:30:00 --pty /bin/bash

source ~/miniforge3/bin/activate nowcast
cd ~/dissertation
export DISS_SCRATCH=/work/scratch-nopw2/$USER/dissertation
python "Code/3 - Diffusion stage/train_ldm.py" --limit 4000 --epochs 2 \
    --warmup 20 --sample-every 1 --sample-members 4
```

Confirms: the latent pack and its meta are found, `sigma_data` is read and
printed, the A100 is used, the weighted val loss starts near 1.0, and
`samples_ep001.png`, `curves.png`, `config.json`, `train_log.json` appear in
`~/dissertation_outputs/diffusion/`. Delete that output directory (or point
`--out` elsewhere) before the real run so the smoke `DONE`/checkpoints do not
interfere.

### 6.3 Full training run (batch, self-resubmitting chain)

The resubmit line must be queued BEFORE training starts, not appended at the
bottom of the script. Slurm kills the whole batch script on TIMEOUT, so a
resubmit line placed after the training command never runs, because the
process that would run it is dead too. That is exactly the bug that cost the
multi-lead run: run 1 stopped at epoch 43 of 50, seven epochs short, because
its resubmit line was at the bottom and the job hit the wall before writing
it. The pattern below queues the successor first, then cancels it if the
current job finishes on its own.

`diff.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=diff
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:30:00
#SBATCH --output=%x-%j.out
set -eu

# fail-fast guards: do not queue a successor, or spend the allocation, for a
# job that cannot possibly do useful work
if [ -f ~/dissertation_outputs/diffusion/DONE ]; then
    echo "DONE already present, nothing to do"
    exit 0
fi
test -f /work/scratch-nopw2/$USER/dissertation/latents/train_latents.npy

source ~/miniforge3/bin/activate nowcast
cd ~/dissertation
export DISS_SCRATCH=/work/scratch-nopw2/$USER/dissertation

# queue the successor before training starts, so a successor exists no matter
# how this job ends (TIMEOUT included); cancelled below if it turns out not
# to be needed
NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID diff.sbatch)

python "Code/3 - Diffusion stage/train_ldm.py" --epochs 100 --batch 64 --resume auto

# reached only if python returned on its own (training did not hit TIMEOUT):
# the pre-queued successor is not needed
scancel "$NEXT"
```

```bash
sbatch diff.sbatch
```

How the chain behaves: `--resume auto` starts fresh on the first job and picks
up `diff_last.pt` (model + EMA + optimizer + epoch) on every subsequent one;
`diff_last.pt` is written atomically every epoch, so the 24 h SIGKILL loses at
most the partial epoch in flight. When training completes (target epochs or
early stop) the trainer writes `DONE` and the chain stops; a chained job that
starts after `DONE` exists exits immediately without touching anything.

### 6.4 What to watch in the log

- The **weighted val loss starts near 1.0 by construction** (the denoiser's
  output layer is zero-initialised, so at step 0 the model is exactly the
  do-nothing baseline). Learning means it falls below ~0.9 and keeps falling; a
  model stuck at 1.0 is learning nothing.
- Every `--sample-every` epochs a `sampled:` line scores an ensemble on fixed
  val crops **next to advection on the same crops**: MAE, CSI@1, CSI@8, CRPS,
  and the 2-8 km PSD ratio. Realistic targets (see `LDM.md` section 6): MAE
  near parity with advection, CRPS clearly below advection's, PSD ratio
  climbing toward 1. CSI@8 will be small and noisy; do not panic over it.
- `samples_epXXX.png`: members should look like rainfall, not noise; the
  ensemble mean being smoother than individual members is expected, not a bug.
- Non-finite val loss aborts the run with a message; resume from
  `diff_last.pt` with a lower `--lr`.

### 6.5 Interrupt / resume / outputs

```bash
# clean stop: Ctrl-C (interactive) or scancel (batch); diff_last.pt is per-epoch
python train_ldm.py --epochs 100 --resume auto            # continue
python train_ldm.py --epochs 100 --resume ~/dissertation_outputs/diffusion/diff_last.pt
```

On resume, the architecture and training-regime flags (`--width`, `--mults`,
`--attn`, `--cond-mode`, `--dropout`, `--cond-drop`, `--ema-decay`,
`--p-mean`, `--p-std`) are restored from the checkpoint (with a printed
notice), so a mistyped or omitted flag cannot silently change the run mid-way.
`--lr`, `--batch`, `--warmup`, `--lr-schedule` and `--weight-decay` stay
CLI-adjustable on purpose (e.g. resuming with a lower `--lr` after a
divergence). Every start and finish is appended to `runs.jsonl` with its full
config, so the history survives even if flags change between chunks.

| Output (in `~/dissertation_outputs/diffusion/`) | What it is |
| --- | --- |
| `diff_last.pt`   | full resume checkpoint (model + EMA + optimizer), every epoch |
| `diff_best.pt`   | compact EMA weights at the best val loss (used for sampling/eval) |
| `train_log.json` | per-epoch record: losses, lr, grad norm, sampled metrics, throughput |
| `curves.png`     | live loss + sampled-metric curves |
| `samples_epXXX.png` | obs / advection / ensemble-mean / members montage |
| `config.json`, `runs.jsonl` | full run config (+ git hash) and the append-only runs table |
| `DONE`           | completion marker (stops the sbatch chain) |

Hyperparameter runs (Phase 1 grid in `LDM.md` section 7) reuse the same sbatch
with a distinct `--out` per configuration, e.g.
`--out ~/dissertation_outputs/diffusion_lr2e4_b64 --lr 2e-4`; each run appends
to its own `runs.jsonl`, and every sampled line already carries the advection
baseline for the comparison the supervisor asked for.

### 6.6 Sampling and evaluation (after training)

Two scripts, both reading the EMA weights in `diff_best.pt`. Neither needs the
trainer to be running.

`sample_diffusion.py` draws ensembles for a few crops and writes the montage
figure that goes in the report (obs, persistence, advection, ensemble mean,
members):

```bash
python "Code/3 - Diffusion stage/sample_diffusion.py" --crops 8 --members 8
```

`evaluate_diffusion.py` is the full scorecard, streamed so memory does not grow
with the split. It scores **four** forecasts on identical crops: the ensemble
mean, the average single member, advection, and persistence.

```bash
# smoke run first (minutes), then the full split
python "Code/3 - Diffusion stage/evaluate_diffusion.py" --limit 256
python "Code/3 - Diffusion stage/evaluate_diffusion.py"
```

Cost: each crop needs `members x (2 x steps - 1)` UNet passes, so the defaults (8
members, 25 steps) are about 392 forward passes per crop. A full validation split
takes hours on an A100, so run it as a batch job with `--time` set generously, and
always do the `--limit 256` pass first. `--members` and `--steps` trade cost
against ensemble quality; both are recorded in the output JSON.

Outputs land in `~/dissertation_outputs/diffusion/eval/`:
`diffusion_eval.json` (the runs-table record), `diffusion_eval.md`
(paste-ready tables) and `diffusion_eval.png` (rain-rate histogram, PSD, CSI vs
threshold, reliability diagram, rank histogram, MAE comparison). Use `--tag` to
keep several configurations side by side, and `--split test` only once, at the end.

Why both `model_mean` and `model_member` are reported: averaging an ensemble damps
peaks, so ensemble-mean CSI at high thresholds largely measures how often the
*mean* exceeds the threshold, which understates a generative model. Members do
produce heavy rain. See `docs/Diffusion_Run1_Results.md` section 4.

### 6.7 The 24 h wall and long jobs (matters for diffusion)

Orchid `orchid` QoS is capped at **24 h** wall time. Diffusion training will
exceed that, so it must checkpoint and resume across job chunks, exactly like the
VAE tooling on Exeter already does. Two options:

- Apply to the JASMIN helpdesk for **`orchid48`** QoS (48 h, granted case by
  case, time-limited to two months). Use `--qos=orchid48`.
- Chain jobs. Have the sbatch script queue its own successor before training
  starts, and cancel that successor if training finishes on its own. Putting
  the resubmit line at the bottom instead does not work: Slurm kills the whole
  batch script on TIMEOUT, so a line after the training command never runs.
  See section 6.3 for the full `diff.sbatch` template; the shape is:

  ```bash
  # queue the successor first, cancel it if this job does not hit TIMEOUT
  NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID <stage>.sbatch)
  python ...
  scancel "$NEXT"
  ```

  The trainer writes `DONE` when it converges or hits the target epoch.
  `train_ldm.py` implements exactly this: it writes the `DONE` marker on
  completion or early stop, refuses to retrain past it (unless `--ignore-done`),
  and `--resume auto` picks up `diff_last.pt` if one exists. See section 6.3.

### 6.8 Multiple GPUs (optional, for diffusion)

An Orchid node has 4 A100s. For the diffusion stage you can request all four and
use PyTorch DDP to cut wall time roughly 4x:

```bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
srun torchrun --standalone --nproc_per_node=4 train_ldm.py ...
```

Worth it for either the +60 LDM or (once written) CorrDiff, if a single A100 run
is too slow.

---

## 7. Monitoring jobs

```bash
squeue -u $USER                       # your queued/running jobs
squeue -u $USER --start               # estimated start time while pending
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqTRES%40   # after/while running
scancel <jobid>                       # kill a job
sinfo -p orchid                       # Orchid node availability
tail -f diff-<jobid>.out              # live training log

# GPU utilisation on the node your job holds:
srun --jobid <jobid> --pty nvidia-smi        # attach to a running job's node
```

Pull figures back to your laptop through an xfer server (run on the laptop):

```bash
rsync -avP <jasmin_user>@xfer-vm-01.jasmin.ac.uk:dissertation_outputs/diffusion/curves.png .
rsync -avP <jasmin_user>@xfer-vm-01.jasmin.ac.uk:'dissertation_outputs/diffusion/samples_ep*.png' .
```

---

## 8. The CorrDiff arm (regression, mu pack, residual diffusion)

This is the second, more recent diffusion arm, added alongside the +60 LDM
of section 6. A deterministic regression model predicts the residual mean
(`mu`) per lead time; that prediction is packed alongside the latents; then
`train_corrdiff.py` trains a diffusion model that learns only the remaining
stochastic part, conditioned on `mu`. It runs across four lead times (15,
30, 45, 60 min), unlike the single +60 lead in section 6. `train_ldm.py` is
run alongside it as the control.

Every GPU job below goes through a wrapper that asserts CUDA works before
touching the allocation, so a job cannot silently fall back to CPU (see the
`gpuhost007` entry under Gotchas). Create it once on the sci VM:

```bash
cat > ~/dissertation/gpu.sbatch << 'EOF'
#!/bin/bash
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=gpuhost007
#SBATCH --output=%x-%j.out
set -eu
source ~/miniforge3/bin/activate nowcast
cd ~/dissertation
export DISS_SCRATCH=/work/scratch-nopw2/$USER/dissertation
python -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "FATAL: no CUDA on $(hostname); not spending the allocation"; exit 1; }
echo "host $(hostname) | CUDA ok | $(date)"
exec "$@"
EOF
```

Used as: `sbatch --job-name=NAME --time=HH:MM:SS ~/dissertation/gpu.sbatch python "Code/3 - Diffusion stage/SCRIPT.py" --flags`

Session setup on the sci VM, which every command below assumes:

```bash
export DISS_SCRATCH=/work/scratch-nopw2/$USER/dissertation
export VAE_EP17="$(python -c "import json,os;print(json.load(open(os.environ['DISS_SCRATCH']+'/latents_ml_ep17/train_latents_L60_meta.json'))['vae_ckpt'])")"
```

The stage order, one purpose line each, command below. Job names and `--time`
values are sized from the estimates given here plus margin, adjust as you get
real data points; flags not shown here are covered by `--help` on the script.

1. Stage 0/0b, laptop or sci VM, seconds, zero cost. Eight static check
   classes over the nine diffusion-stage scripts. Must pass before any job
   is queued.

   ```bash
   python "Code/3 - Diffusion stage/check_contract.py"
   ```

2. Stage 0d, sci VM, no GPU and no Slurm allocation, about 1 CPU-hour. The
   ridge gate: the closed-form linear lower bound on explained variance.

   ```bash
   python "Code/3 - Diffusion stage/latent_ridge_gate.py" \
       --latents-dir $DISS_SCRATCH/latents_ml_ep17 --leads 15,30,45,60 \
       --kernel 3 --fit-rows 20000 --eval-rows 5000 --features full \
       --out ~/dissertation_outputs/regression/ridge_gate.json
   ```

3. Stage 1, Phase-0 overfit, about 17 min. Pass condition: train MSE below
   0.02 * sigma_data^2 = 0.01097, equivalently train EV above 0.98. Use
   `--lr-schedule cosine`, not constant: a constant 3e-4 run diverged at
   epoch 110 when the gradient norm jumped from 0.30 to 6.56 in one step and
   never recovered, while cosine annealing reached 0.9820 cleanly. `--out`
   points at scratch, not home, because this run rewrites a 1.05 GB
   checkpoint every epoch.

   ```bash
   sbatch --job-name=regovfit --time=00:45:00 ~/dissertation/gpu.sbatch \
       python "Code/3 - Diffusion stage/train_regression.py" \
       --leads 15,30,45,60 --limit 512 --epochs 200 --batch 32 --warmup 20 \
       --lr 3e-4 --lr-schedule cosine --diag-every 0 --workers 2 \
       --latents-dir $DISS_SCRATCH/latents_ml_ep17 --out $DISS_SCRATCH/regression_overfit
   ```

4. Stage 2, oracle test, about 2 min. Checks the advection column reproduces
   the published single60 numbers and prints the codec floor and ceiling.

   ```bash
   sbatch --job-name=detoracle --time=00:15:00 ~/dissertation/gpu.sbatch \
       python "Code/3 - Diffusion stage/evaluate_deterministic.py" \
       --field advection --lead 60 --limit 512 \
       --latents-dir $DISS_SCRATCH/latents_ml_ep17 --vae "$VAE_EP17" \
       --out ~/dissertation_outputs/regression/eval --tag _sanity_L60
   ```

5. Stage 3a, regression smoke, about 1 min. Same as stage 1 but a short
   limit and epoch count with diagnostics turned on.

   ```bash
   sbatch --job-name=regsmoke --time=00:15:00 ~/dissertation/gpu.sbatch \
       python "Code/3 - Diffusion stage/train_regression.py" \
       --leads 15,30,45,60 --limit 4000 --epochs 2 --batch 32 --warmup 20 \
       --lr 3e-4 --lr-schedule cosine --diag-every 1 --diag-crops 16 \
       --psd-crops 16 --workers 2 --latents-dir $DISS_SCRATCH/latents_ml_ep17 \
       --vae "$VAE_EP17" --out $DISS_SCRATCH/regression_smoke
   ```

6. Stage 4a and 4b, the two regression variants, 4.2 h each, run
   concurrently. 4a predicts a delta on top of `z_A`; 4b predicts `z_y`
   directly from the past frames only, so it has no `--ridge-gate` (its
   target is not an increment) and pauses for a few minutes at startup
   streaming the `z_y` second moments, cached afterwards to `zy_moments.json`
   in its output directory.

   ```bash
   sbatch --job-name=reg-delta --time=05:00:00 ~/dissertation/gpu.sbatch \
       python "Code/3 - Diffusion stage/train_regression.py" \
       --leads 15,30,45,60 --epochs 8 --target delta --cond-mode full \
       --batch 64 --lr 1e-4 --warmup 1000 --lr-schedule cosine \
       --ema-decay 0.999 --diag-every 2 --diag-crops 64 --psd-crops 64 \
       --seed 0 --latents-dir $DISS_SCRATCH/latents_ml_ep17 --vae "$VAE_EP17" \
       --ridge-gate ~/dissertation_outputs/regression/ridge_gate.json \
       --out ~/dissertation_outputs/regression_delta_ep17 --resume auto

   sbatch --job-name=reg-zy --time=05:00:00 ~/dissertation/gpu.sbatch \
       python "Code/3 - Diffusion stage/train_regression.py" \
       --leads 15,30,45,60 --epochs 8 --target z_y --cond-mode x-only \
       --batch 64 --lr 1e-4 --warmup 1000 --lr-schedule cosine \
       --ema-decay 0.999 --diag-every 2 --diag-crops 64 --psd-crops 64 \
       --seed 0 --latents-dir $DISS_SCRATCH/latents_ml_ep17 --vae "$VAE_EP17" \
       --out ~/dissertation_outputs/regression_zy_ep17 --resume auto
   ```

7. Stage 5, deterministic scorecards, four leads per variant, about 0.3 h
   each. Variant (a) (`regression_delta_ep17`) uses `--anchor zA`; variant
   (b) (`regression_zy_ep17`) uses `--anchor none`, because its mean is
   `z_y` directly rather than an increment on `z_A`.

   ```bash
   for L in 15 30 45 60; do
       sbatch --job-name=det-a-L$L --time=00:45:00 ~/dissertation/gpu.sbatch \
           python "Code/3 - Diffusion stage/evaluate_deterministic.py" \
           --field regression --reg ~/dissertation_outputs/regression_delta_ep17/reg_best.pt \
           --anchor zA --oracle --lead $L --split val --batch 32 --seed 0 \
           --out ~/dissertation_outputs/regression_delta_ep17/eval --tag _L$L

       sbatch --job-name=det-b-L$L --time=00:45:00 ~/dissertation/gpu.sbatch \
           python "Code/3 - Diffusion stage/evaluate_deterministic.py" \
           --field regression --reg ~/dissertation_outputs/regression_zy_ep17/reg_best.pt \
           --anchor none --oracle --lead $L --split val --batch 32 --seed 0 \
           --out ~/dissertation_outputs/regression_zy_ep17/eval --tag _L$L
   done
   ```

8. Stage 6, pack_mu, val first then train, about 21.9 GB written in total.
   No runtime benchmark yet, `--time` below is a generous placeholder,
   tighten it once you have a data point. Cross-check with `--field
   mu-pack`, which must reproduce the stage 5 numbers.

   ```bash
   sbatch --job-name=packmu-val --time=02:00:00 ~/dissertation/gpu.sbatch \
       python "Code/3 - Diffusion stage/pack_mu.py" \
       --reg ~/dissertation_outputs/regression_delta_ep17/reg_best.pt \
       --latents-dir $DISS_SCRATCH/latents_ml_ep17 \
       --out $DISS_SCRATCH/latents_ml_ep17_mu_delta --splits val \
       --leads 15,30,45,60 --batch 256

   sbatch --job-name=packmu-all --time=02:00:00 ~/dissertation/gpu.sbatch \
       python "Code/3 - Diffusion stage/pack_mu.py" \
       --reg ~/dissertation_outputs/regression_delta_ep17/reg_best.pt \
       --latents-dir $DISS_SCRATCH/latents_ml_ep17 \
       --out $DISS_SCRATCH/latents_ml_ep17_mu_delta --splits train,val \
       --leads 15,30,45,60 --batch 256

   sbatch --job-name=det-mupack --time=00:15:00 ~/dissertation/gpu.sbatch \
       python "Code/3 - Diffusion stage/evaluate_deterministic.py" \
       --field mu-pack --mu-dir $DISS_SCRATCH/latents_ml_ep17_mu_delta \
       --lead 60 --limit 512
   ```

9. Stage 3b, the two diffusion smokes. The CorrDiff smoke banner must print
   `in_ch=28` and a `sigma_data` equal to the pooled `sigma_data_resid`
   rather than 0.7407; the control (`train_ldm.py`) must reproduce the
   previous behaviour at `in_ch=24` and `sigma_data` 0.7407.

   ```bash
   sbatch --job-name=corrdiff-sm --time=00:30:00 ~/dissertation/gpu.sbatch \
       python "Code/3 - Diffusion stage/train_corrdiff.py" \
       --limit 4000 --epochs 2 --warmup 20 --sample-every 1 \
       --mu-dir $DISS_SCRATCH/latents_ml_ep17_mu_delta --hr-mean-cond on

   sbatch --job-name=ldm-sm --time=00:30:00 ~/dissertation/gpu.sbatch \
       python "Code/3 - Diffusion stage/train_ldm.py" \
       --limit 4000 --epochs 2 --warmup 20 --sample-every 1
   ```

10. Stage 7, the 26.2 h CorrDiff retrain, as a self-resubmitting chain.
    Uses the working resubmit pattern from section 6.3 (queue the successor
    before training starts, cancel it if python returns on its own), the
    same `--exclude=gpuhost007` and CUDA assert as the wrapper above, and
    guards on the mu pack existing and on `DONE`. `REGSHA` identifies the
    regression checkpoint the mu pack was produced from. `pack_mu.py` writes
    it as `reg_sha256` into `{split}_mu{_LNN}_meta.json`, which is why the
    line below reads `train_mu_L60_meta.json`. Passing it as
    `train_corrdiff.py --reg-sha` makes the trainer assert it against every
    shard of both splits, fatally on mismatch, which is what stops a CorrDiff
    run being trained against the wrong frozen mean.

    `corrdiff.sbatch`:

    ```bash
    #!/bin/bash
    #SBATCH --job-name=corrdiff
    #SBATCH --partition=orchid
    #SBATCH --account=orchid
    #SBATCH --qos=orchid
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=64G
    #SBATCH --exclude=gpuhost007
    #SBATCH --time=23:30:00
    #SBATCH --output=%x-%j.out
    set -eu

    # fail-fast guards: do not queue a successor, or spend the allocation, on a
    # job that cannot possibly do useful work
    if [ -f ~/dissertation_outputs/diffusion_corrdiff_v1/DONE ]; then
        echo "DONE already present, nothing to do"
        exit 0
    fi
    MU=/work/scratch-nopw2/$USER/dissertation/latents_ml_ep17_mu_delta
    test -d "$MU"

    source ~/miniforge3/bin/activate nowcast
    cd ~/dissertation
    export DISS_SCRATCH=/work/scratch-nopw2/$USER/dissertation

    python -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" \
        || { echo "FATAL: no CUDA on $(hostname); not spending the allocation"; exit 1; }

    VAE="$(python -c "import json,os;print(json.load(open(os.environ['DISS_SCRATCH']+'/latents_ml_ep17/train_latents_L60_meta.json'))['vae_ckpt'])")"
    REGSHA="$(python -c "import json,glob;print(json.load(open(glob.glob('$MU/*_meta.json')[0]))['reg_sha'])")"
    echo "VAE=$VAE REGSHA=$REGSHA"

    # queue the successor before training starts, cancel it below if this job
    # finishes on its own; see section 6.3 for why the old bottom-of-script
    # resubmit line does not work
    NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID corrdiff.sbatch)

    python "Code/3 - Diffusion stage/train_corrdiff.py" --leads 15,30,45,60 --epochs 50 \
        --batch 64 --lr 1e-4 --warmup 1000 --cond-mode full --cond-drop 0.1 \
        --p-mean -1.2 --p-std 1.2 --ema-decay 0.999 --patience 0 \
        --mu-dir "$MU" --hr-mean-cond on --reg-sha "$REGSHA" \
        --sample-every 5 --sample-crops 16 --sample-members 8 --psd-crops 16 \
        --seed 0 --out ~/dissertation_outputs/diffusion_corrdiff_v1 --resume auto

    scancel "$NEXT"
    ```

    ```bash
    sbatch corrdiff.sbatch
    ```

11. Stage 8a and 8b, full-split evaluations, four independent single-GPU
    jobs each, 3.26 h per lead. 8a (CorrDiff) adds `--mu-dir
    $DISS_SCRATCH/latents_ml_ep17_mu_delta`; 8b (the ml_v2 LDM comparator,
    trained the same way as section 6 but on the multi-lead cache) does
    not. Both pin `--batch 16 --seed 0 --members 8 --steps 25` and name an
    explicit `ckpt_ep050.pt`, never `diff_best.pt`: the evaluator seeds per
    crop-batch as `args.seed + 7919 * s0`, so changing `--batch` changes the
    noise and breaks the paired comparison, and `diff_best.pt` is selected
    on val EDM loss, which is anti-correlated with small-scale power in
    every run of this project. The flag is `--ckpt`, and it defaults to
    `diff_best.pt`, so naming an explicit `ckpt_epNNN.pt` is not optional.

    ```bash
    for L in 15 30 45 60; do
        sbatch --job-name=cdeval-L$L --time=04:00:00 ~/dissertation/gpu.sbatch \
            python "Code/3 - Diffusion stage/evaluate_diffusion.py" \
            --ckpt ~/dissertation_outputs/diffusion_corrdiff_v1/ckpt_ep050.pt \
            --mu-dir $DISS_SCRATCH/latents_ml_ep17_mu_delta \
            --lead $L --batch 16 --seed 0 --members 8 --steps 25

        sbatch --job-name=ldmeval-L$L --time=04:00:00 ~/dissertation/gpu.sbatch \
            python "Code/3 - Diffusion stage/evaluate_diffusion.py" \
            --ckpt ~/dissertation_outputs/diffusion_ml_v2/ckpt_ep050.pt \
            --lead $L --batch 16 --seed 0 --members 8 --steps 25
    done
    ```

12. Stage 9, pooling, sci VM, seconds, no GPU.

    ```bash
    python "Code/3 - Diffusion stage/pool_scorecards.py" --eval-dir <dir> --pattern "diffusion_eval*_L*.json"
    python "Code/3 - Diffusion stage/pool_scorecards.py" --eval-dir <dir> --pattern "det_eval*_L*.json"
    ```

---

## 9. Getting results out

Checkpoints and logs live in `~/dissertation_outputs/` (backed up). To archive
them off JASMIN or bring them to your laptop, again go via an xfer server:

```bash
# on your laptop:
rsync -avP <jasmin_user>@xfer-vm-01.jasmin.ac.uk:dissertation_outputs/ ./jasmin_outputs/
```

---

## 10. Cheat sheet

```bash
# login
ssh -A <jasmin_user>@login.jasmin.ac.uk
ssh <jasmin_user>@sci-vm-01.jasmin.ac.uk

# env (add the export to ~/.bashrc, or to every sbatch script)
source ~/miniforge3/bin/activate nowcast
export DISS_SCRATCH=/work/scratch-nopw2/$USER/dissertation

# GPU smoke test (interactive A100, 1 h)
srun --partition=orchid --account=orchid --qos=orchid --gres=gpu:1 \
     --cpus-per-task=8 --mem=64G --time=01:00:00 --pty /bin/bash

# diffusion stage (batch): latent pack, then the self-resubmitting trainer
sbatch pack_latents.sbatch
sbatch diff.sbatch          # chains itself until DONE appears

# monitor / cancel
squeue -u $USER ; sacct -j <jobid> ; scancel <jobid>

# transfer (always via xfer, never login/sci)
rsync -avP <src> <jasmin_user>@xfer-vm-01.jasmin.ac.uk:<dst>
```

---

## 11. Gotchas, in one place

- Compute nodes: **outbound HTTPS only, no outbound SSH.** boto3 S3 works;
  `git clone`/`pip` over HTTPS works; you cannot SSH out of a batch node.
- The advection prior build and the VAE training happen on **Exeter**, not
  JASMIN, see `README_server.md`.
- Orchid = **24 h** wall (`orchid`) or 48 h (`orchid48`, on request). Long runs
  must checkpoint and resume or self-resubmit. `orchid48` was NOT applied for
  on this project; it relies on the chained self-resubmit pattern instead
  (section 6.3), which is a deliberate decision, not an oversight.
- Orchid QoS concurrency: `MaxJobsPU 8`, no submit cap, checked with
  `sacctmgr show qos orchid format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU`. Up
  to eight jobs can run at once, so every four-way lead fan-out in this plan
  (stage 5, stage 8a/8b in section 8) fits without queuing behind itself.
- **Scratch is wiped after 28 days** without access. Persist anything you cannot
  regenerate to home or a GWS.
- Activate conda in batch jobs with `source ~/miniforge3/bin/activate nowcast`,
  not bare `conda activate`.
- Sci servers have **no GPU**; `cuda.is_available()` is `False` there, test on
  Orchid.
- Accounts differ from Exeter: use `useraccounts` to find your Slurm
  `--account` and QoS. Orchid uses `--account=orchid --qos=orchid`.
- `train_ldm.py`, `train_corrdiff.py`, `train_regression.py`, `pack_mu.py`,
  `evaluate_deterministic.py`, and `latent_ridge_gate.py` honour
  `DISS_SCRATCH`; the older stage scripts (`build_advection_prior.py`,
  `pack_vae_data.py`, `train_vae_v2.py`) do not, so pass their
  `--root`/`--out`/`--packed-dir` flags explicitly on JASMIN.
- Retraining the VAE invalidates the latent pack: `pack_latents.py` detects the
  changed checkpoint hash and repacks; the diffusion trainer refuses mixed
  train/val packs. Never mix latents from two different codecs.
- **`gpuhost007` allocates a GPU but cannot use it.** On 2026-08-09 a job on
  `gpuhost007` got `CUDA_VISIBLE_DEVICES=0`, `nvidia-smi -L` listed an
  A100-SXM4-40GB, and `torch.cuda.device_count()` returned 1, but
  `torch.cuda.is_available()` returned `False` with "CUDA driver
  initialization failed". The two disagree because `device_count()` answers
  from NVML, which only asks the driver what hardware exists, while
  `is_available()` actually initialises a CUDA context, which the driver on
  that node refuses to do. `gpuhost009` and `gpuhost015` are confirmed fine
  (driver 610.43.02, CUDA UMD 13.3). Consequence: a `train_*` script prints
  `WARNING: no GPU found, running on CPU (very slow)` and carries on instead
  of failing, and one Phase-0 run burned about ten hours at 3 img/s instead
  of about 2 seconds per epoch at 287 img/s before anyone noticed. Workaround:
  `--exclude=gpuhost007` on every job (already in `gpu.sbatch`, section 8).
  The general rule: every GPU job must assert `torch.cuda.is_available()`
  before doing any real work, never trust `device_count()` or a successful
  `--gres=gpu:1` allocation alone.

## 12. JASMIN reference docs

- Orchid GPU cluster: https://help.jasmin.ac.uk/docs/batch-computing/orchid-gpu-cluster/
- How to submit a job: https://help.jasmin.ac.uk/docs/batch-computing/how-to-submit-a-job/
- Slurm queues (QoS limits): https://help.jasmin.ac.uk/docs/batch-computing/slurm-queues/
- Logging in: https://help.jasmin.ac.uk/docs/getting-started/how-to-login/
- Storage overview: https://help.jasmin.ac.uk/docs/getting-started/storage/
- Transfer servers: https://help.jasmin.ac.uk/docs/interactive-computing/transfer-servers/
- Miniforge environments: https://help.jasmin.ac.uk/docs/software-on-jasmin/creating-and-using-miniforge-environments/
