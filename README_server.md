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

## 5. Run the full build (in tmux, so it survives disconnects)

```bash
tmux new -s prior
conda activate nowcast
python build_advection_prior.py --start 2024-11-21 --end 2025-12-31
#   detach:   Ctrl-b  then  d
#   reattach: tmux attach -t prior
```

It is **idempotent** -- if it dies, just re-run the same command and it skips what is already done.

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
