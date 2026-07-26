# Running the advection-prior stage on the Exeter compute servers

This builds the semi-Lagrangian **advection prior** `A` and the **residual target**
`r = y - A` for the full training dataset (2024-11-21 -> 2025-12-31), on the
undergraduate compute server (`mcrugcomp02.ex.ac.uk`).

The work runs on the **64-core CPU** (pysteps does not use the GPU). The NVIDIA L4 is for the diffusion stage later. Everything heavy lives on fast **/scratch**.

---

## 0. Before you start

- Be **on campus or on the university VPN** (the servers are behind the firewall).
- Pick the least-busy server from the Grafana dashboard (it suggests one). Because
  `/scratch` is **per-server and not shared**, do _all_ steps on the **same** server.
- `/scratch` is **not backed up and can be wiped** -> treat it as temporary. The
  small outputs (manifest + baseline) are also copied to your home dir.

## 1. Connect

```bash
ssh dv321@mcrugcomp02.ex.ac.uk
```

## 2. Get the code onto the server

Either clone your repo, or copy files from your laptop, for instance:

```bash
# from your laptop (PowerShell / terminal):
scp build_advection_prior.py dv321@mcrugcomp02.ex.ac.uk:~/
```

## 3. Create the conda environment (once)

```bash
conda env create -f environment.yml
conda activate nowcast
```

## 4. Check the server can reach AWS (30 seconds)

```bash
python build_advection_prior.py --check
```

- **OK** -> proceed.
- **FAILED** -> the server has no outbound internet. Download the frames on your
  laptop instead, `rsync` them to `/scratch/<user>/dissertation/frames/`, then run
  step 5 with `--skip-download`.

> Layout note: the stage scripts are organised under `Code/<stage>/`
> (`Code/1 - Advection stage/build_advection_prior.py`,
> `Code/2 - VAE Stage/{pack_vae_data,train_vae_v2,pack_latents}.py`). Run them by that
> path from `~/dissertation`, for example
> `python "Code/1 - Advection stage/build_advection_prior.py" ...`, or `cd` into the
> stage folder first. The commands below use the bare script name for brevity.

## 5. Run the full build (in tmux, so it survives disconnects)

```bash
tmux new -s prior
conda activate nowcast
python build_advection_prior.py --start 2024-11-21 --end 2025-12-31
#   detach:   Ctrl-b  then  d
#   reattach: tmux attach -t prior
```

It is **idempotent** -- if it dies, just re-run the same command and it skips what is already done.

## 5b. Multi-lead build (+15/30/45/60, for the lead-conditioned model)

The same script builds all four lead times in one pass, since the advection
extrapolation already produces the intermediate frames. It writes one `.npz` per
(crop, lead) with an added `lead_min` field, and auto-routes to a **separate**
`prior_ml/` dir so it never touches the `+60` `prior/` cache:

```bash
python build_advection_prior.py --start 2024-11-21 --end 2025-12-31 --leads 15,30,45,60
```

Only the new +15/30/45 target frames are downloaded (the rest are reused from
`frames/`). Expect roughly the same wall time as the +60 build and about 4x the
storage. The manifest reports a per-lead advection baseline (the skill vs lead-time
curve the model must beat). The `+60` core run is unaffected, it keeps using `prior/`.

## 6. Outputs

| What                                         | Where                                                          |
| -------------------------------------------- | -------------------------------------------------------------- |
| Advection-prior cache (`.npz` per crop)      | `/scratch/dv321/dissertation/prior/<split>/<date>/`            |
| Raw frame cache (`.h5`)                      | `/scratch/dv321/dissertation/frames/`                          |
| Manifest + advection-only baseline (CSI/MAE) | `~/dissertation_outputs/manifest.json` (also in the prior dir) |

The baseline (CSI per threshold + MAE) is printed at the end and saved in the
manifest.json, that is the number the full diffusion model must beat.

## Useful options

```bash
python build_advection_prior.py --help
python build_advection_prior.py --start 2024-11-21 --end 2024-11-23   # tiny test slice
python build_advection_prior.py --workers 48 --io-workers 48          # tune parallelism
python build_advection_prior.py --skip-download                       # offline server
```

---

# Stage 2: VAE v2 training (GPU, with full monitoring)

Trains the improved codec (`train_vae_v2.py`, design in
`docs/VAE_architecture_change.md`). Runs on the **L4 GPU**. Do everything on the
same server that holds the prior cache (`mcrugcomp02`).

## 2.0 One-time setup

```bash
# from your laptop, in the project folder:
scp pack_vae_data.py train_vae_v2.py dv321@mcrugcomp02.ex.ac.uk:~/

# on the server (PyTorch only needed once, ~2.5 GB download):
conda activate nowcast
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect: <ver> True
```

## 2.1 Pack the training data (once, ~10-25 min, CPU)

This removes the npz-decompression bottleneck that limited v1 to 52 img/s.

```bash
tmux new -s pack
conda activate nowcast
python pack_vae_data.py
#   expect: "[train] packing 153487 files -> 306974 rows (40.2 GB)" then a
#   spot-check line "5/5 rows verified against source npz" per split.
```

Output: `/scratch/dv321/dissertation/packed/{train,val}_fields.npy` (+ meta and
index json). Re-running skips existing packs; `--check-only` re-verifies.

## 2.2 Smoke test (~5 min)

```bash
python train_vae_v2.py --limit 4000 --epochs 2 --disc-start 60 --verify-every 1
```

Confirms: packed data found, GPU used, the discriminator switches on (g_adv /
d_loss / lam appear in the step lines), and `verify_ep01.png`,
`training_curves.png`, `train_log.json` appear in `~/dissertation_outputs/vae_v2/`.
Note: the smoke run ends with "no vae_best.pt was saved", which is expected
(best-model selection only starts after the GAN grace period); the full run
saves it from epoch 4 onward.

## 2.3 Full training run (in tmux)

```bash
tmux new -s vae2
conda activate nowcast
python train_vae_v2.py --epochs 50
#   detach: Ctrl-b d      reattach: tmux attach -t vae2
```

Defaults: batch 32, lr 1e-4, discriminator starts after 2 epochs, best-model
selection from epoch 4, early stop after 8 epochs without improvement. If you
hit CUDA OOM, rerun with `--batch 24`.

## 2.4 Monitoring while it runs

```bash
tmux attach -t vae2                      # live step/epoch lines (Ctrl-b d to leave)
watch -n 2 nvidia-smi                    # GPU util + memory (Ctrl-C to quit)
python -c "import json;print(json.dumps(json.load(open('/home/links/dv321/dissertation_outputs/vae_v2/train_log.json'))[-1],indent=1))"   # last epoch record
```

What to watch (full guide: `docs/VAE_architecture_change.md`, section 5):

- `val rec` (plain) should head toward <= 0.022; a small bump when the
  discriminator starts (epoch 3) is normal.
- `tail16 / tail32` should climb toward 1.0 (v1 was well below); above ~1.2
  means hallucination, stop and lower `--disc-weight`.
- `psd2-8km` should climb toward >= 0.8 (v1 was ~0.1-0.3).
- `d 0.000` sustained means the discriminator collapsed, restart from
  `vae_last.pt` with `--disc-weight 0.25`.

Pull the figures to your laptop to inspect (run from the laptop):

```bash
scp dv321@mcrugcomp02.ex.ac.uk:~/dissertation_outputs/vae_v2/training_curves.png .
scp dv321@mcrugcomp02.ex.ac.uk:~/dissertation_outputs/vae_v2/verify_ep*.png .
scp dv321@mcrugcomp02.ex.ac.uk:~/dissertation_outputs/vae_v2/recon_ep*.png .
scp dv321@mcrugcomp02.ex.ac.uk:~/dissertation_outputs/vae_v2/vae_verify.png .
```

## 2.5 Interrupt / resume / re-verify

```bash
# stop cleanly: Ctrl-C inside tmux (vae_last.pt is saved every epoch)
python train_vae_v2.py --epochs 50 --resume ~/dissertation_outputs/vae_v2/vae_last.pt
python train_vae_v2.py --verify-only          # regenerate vae_verify.png from vae_best.pt
```

## 2.6 Acceptance before the diffusion stage

Compare the final `vae_verify.png` (generated from `vae_best.pt`) and the
`train_log.json` record of the BEST epoch, not the last one (the final summary
line prints which epoch `vae_best.pt` came from), against the targets in
`docs/Analysis with VAE v1.md`, section 6 (histogram ratio >= 0.8 to 32 mm/h,
PSD ratio >= 0.8 to 4 km, csi_8 >= 0.95, tail ratios in [0.8, 1.2], plain val
recon <= 0.022). If met, the diffusion stage builds on
`~/dissertation_outputs/vae_v2/vae_best.pt`.
