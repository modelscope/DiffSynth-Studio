# Continuation Prompt for Next Agent

Copy the prompt below and give it to the next coding agent as the initial message.

---

## Prompt

I'm working on a research project for a NeurIPS paper: SPAD (Single-Photon Avalanche Diode) binary sensor to RGB image reconstruction using FLUX.1-dev (12B rectified-flow transformer) with ControlNet + LoRA.

**Read these files first** to understand the full project state:
- `agent/HANDOFF.md` — complete project status, what's done, what's running, what's next
- `agent/TODO.md` — prioritized task list
- `agent/DECISIONS.md` — key technical decisions you should respect
- `agent/TEST_STATUS.md` — all experiments and their results
- `EXPERIMENTS.md` — detailed results tables and reproduction commands
- `~/.cursor/plans/spad_neurips_full_plan_1cbbff23.plan.md` — master 666-line research plan

**Current state**:
- The git repo is at `/home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD` on branch `main`
- 14 modified files are uncommitted (see `agent/git_diff.patch`)
- OD filter training is running in tmux session `od-training` (OD03 fine-tune at epoch 17/20, then OD07 and OD03-scratch queued). Check with `tmux attach -t od-training`
- GPU: RTX 5090 32GB, conda env: `diffsynth`
- All conda commands need: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate diffsynth`

**What is already done** (do not redo):
1. Dataset audit + scene-aware stratified train-test split (zero data leakage)
2. FLUX RAW baseline training (40 epochs, best=epoch-15)
3. K=10 multi-seed validation with aggregated metrics (PSNR 18.0±0.09)
4. Linear probing of DiT activations — spatial depth R²=0.64, spatial bit density R²=0.99 (AC3D-style figures at `probing_results/probes/`)
5. Physics ablation with latent-space DPS (eta sweep 0.01-1.0, best=1.0 gives +0.16 PSNR)
6. Frame-count ablation (1-1000 accumulated frames)
7. Consistency training (30 epochs, marginal degradation)
8. Variance analysis (776 images) and calibration (ECE=0.269)
9. CFID metric implementation with numerical stability

**What to do next** (in priority order):
1. Wait for OD training to finish (~24-36h), then run `bash run_od_ablation.sh`
2. Commit the 14 modified + ~10 new files
3. Run SD1.5 re-evaluation: `cd spad-diffusion && bash run_sd15_scene_aware_eval.sh` (needs `conda activate control2`)
4. Frame-vs-seed variance decomposition (Phase 2b)
5. Downstream task stability analysis (Phase 2c — segmentation entropy, depth variance across seeds)
6. Begin paper writing using linear probing figures and results tables

**Critical gotchas**:
- SPAD images are 16-bit PNGs — use `load_spad_image()` from `diffsynth/core/data/operators.py`, NOT PIL's `convert('RGB')`
- All metrics are computed in grayscale
- The latent-space DPS is an approximation (pixel-space DPS caused OOM)
- `probing_results/activations/` is 42GB — do not delete

Please check the tmux session status first, then continue with the next available task.
