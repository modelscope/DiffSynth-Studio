# TODO: SPAD-to-RGB Project

**Updated**: 2026-03-29

---

## P0: Linear Probing (TOP PRIORITY)

### 1. Re-run spatial streaming with `spatial_crossframe_variance`
- **Status**: CODE FIXED, needs re-run
- **Bug**: `linear_probing.py` had a hardcoded whitelist that excluded `spatial_crossframe_variance` (line 502). Fixed now.
- **What**: Re-extract spatial streaming for the main model (allblocks) to get spatial crossframe variance probes
- **Run**:
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate diffsynth
  cd /home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD

  # Re-extract with spatial streaming (GPU, ~30 min)
  python linear_probing.py \
      --extract \
      --lora_checkpoint models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors \
      --output-dir probing_results_allblocks \
      --hook-controlnet \
      --spatial-streaming \
      --pca-dim 0 --ridge-lambda 0.1 \
      --metadata_csv /home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv \
      --dataset_base /home/jw/engsci/thesis/spad/spad_dataset \
      --max_samples 776 --steps 28

  # Re-train to merge results
  python linear_probing.py --train --output-dir probing_results_allblocks \
      --pca-dim 0 --ridge-lambda 0.1 --max_samples 776
  ```
- **After**: Update Appendix E in `agent/reports/probing_report_final.md` with spatial crossframe variance results (run `python update_probing_report.py`)
- **Why it matters**: Missing piece of the cross-frame variance analysis. Spatial probing shows WHERE in the image the model encodes frame sensitivity.

### 2. SD1.5 Linear Probing (NEW EXPERIMENT)
- **Status**: NOT STARTED
- **Why**: Cross-architecture comparison (UNet vs DiT) is a strong paper contribution. Answers: "Does model capacity determine semantic information flow?"
- **What**: Adapt the FLUX linear probing pipeline for SD1.5's UNet + ControlNet architecture
- **Existing infrastructure**:
  - Trained SD1.5 ControlNet: `/home/jw/engsci/thesis/spad/spad-diffusion/lightning_logs/spad_controlnet/two_stage_best/best-epoch=14-val_loss=0.1057.ckpt` (8.6 GB)
  - Inference script: `/home/jw/engsci/thesis/spad/spad-diffusion/sim_inference.py`
  - UNet code: `/home/jw/engsci/thesis/spad/spad-diffusion/ldm/modules/diffusionmodules/openaimodel.py` (class `UNetModel`, line 412)
  - ControlNet code: `/home/jw/engsci/thesis/spad/spad-diffusion/cldm/cldm.py` (class `ControlNet`, `ControlledUnetModel`)
  - Same dataset/split as FLUX (77/20 scene-aware)
- **Implementation plan**:
  1. Write `sd15_linear_probing.py` adapting the FLUX probing pipeline for UNet blocks
     - Hook UNet `down_blocks`, `mid_block`, `up_blocks` (ResBlocks, not transformer blocks)
     - Hook ControlNet encoder blocks + middle block
     - Extract `[B, C, H, W]` spatial activations → pool to match regression format
     - Variable spatial resolution per stage (64x64 → 32x32 → 16x16 → 8x8)
  2. Prepare targets (reuse FLUX's `targets.json` — same val set, same targets)
  3. Extract activations on full val set (776 images, 7 timesteps)
  4. Train ridge probes, generate R² heatmaps
  5. Compare FLUX vs SD1.5: depth R², bit density R², variance R², object recognition
- **Expected results**: SD1.5 likely lower R² across the board (1.2B UNet vs 12B DiT), but should still preserve bit density (>0.99). Depth encoding likely ~0.4-0.5 vs FLUX's 0.685.
- **Paper narrative**: "Diffusion model capacity determines semantic information flow — FLUX's 12B DiT encodes richer geometry than SD1.5's 1.2B UNet"
- **Conda env**: `control2` (separate from `diffsynth`)
- **Effort**: ~4-5h implementation + ~2h GPU extraction + ~2h analysis

### 3. Update probing report with spatial crossframe results
- **Status**: BLOCKED on #1
- **What**: After #1 completes, run `python update_probing_report.py` (path fixed to `agent/reports/`)
- **Also**: Add spatial crossframe variance section to Appendix E (E.3.2 currently missing)

---

## P1: Missing Overnight Pipeline Experiments

### 4. Consistency Epoch Sweep
- **Status**: NEVER RAN (pipeline died at Step 5)
- **Root cause**: Pipeline `set -euo pipefail` + `update_probing_report.py` looked for `probing_report_final.md` in repo root (moved to `agent/reports/`). Path now fixed.
- **Checkpoints available**: epochs 0-29 in `models/train/FLUX-SPAD-LoRA-Consistency/`
- **Sweep targets**: epochs 5, 10, 15, 20, 25, 29
- **Run**:
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate diffsynth
  cd /home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD

  for epoch in 5 10 15 20 25 29; do
      ckpt="models/train/FLUX-SPAD-LoRA-Consistency/epoch-${epoch}.safetensors"
      out_dir="./validation_outputs_consistency_epoch${epoch}"
      [ ! -f "$ckpt" ] && echo "SKIP epoch $epoch" && continue

      python validate_lora.py \
          --lora_checkpoint "$ckpt" \
          --lora_target controlnet \
          --metadata_csv /home/jw/engsci/thesis/spad/spad_dataset/metadata_val.csv \
          --output_dir "$out_dir" \
          --steps 28 --max_samples 776 --seed 42

      python run_metrics.py "$out_dir" --save
  done
  ```
- **After**: Update `agent/reports/dps_consistency_report.md` with epoch sweep results + EXPERIMENTS.md
- **GPU time**: ~20 min/epoch x 6 epochs = ~2h

### 5. img2img Ablation (Training + Sweep)
- **Status**: SCRIPTS CREATED, training not started
- **Scripts**: `train_img2img_ablation.sh`, `validate_img2img.py`, `run_img2img_ablation.sh`
- **What**: FLUX img2img + LoRA-on-DiT, NO ControlNet. SPAD as `input_image`, sweep `denoising_strength` {0.3, 0.5, 0.7, 0.8, 0.9, 1.0}
- **Run**:
  ```bash
  # Train (~10h, rank-32 LoRA on DiT, 20 epochs)
  bash train_img2img_ablation.sh

  # Sweep denoising strength
  bash run_img2img_ablation.sh
  ```
- **GPU time**: ~10h training + ~2h sweep

---

## P2: Other Experiments

### 6. OD Training + Evaluation
- **Status**: Training may still be running (check `tmux attach -t od-training`)
- **Models**: OD03-FT, OD07-FT, OD03-Scratch
- **After training**: `bash run_od_ablation.sh`
- **Update**: EXPERIMENTS.md with results

### 7. SD1.5 Baseline Re-evaluation
- **Status**: Script ready, not run
- **Run**:
  ```bash
  cd /home/jw/engsci/thesis/spad/spad-diffusion
  conda activate control2
  bash run_sd15_scene_aware_eval.sh
  ```
- **Purpose**: Fair SD1.5 vs FLUX comparison on leak-free val set

### 8. Higher LoRA Rank (64 or 128)
- Modify `--lora_rank 64` in `train_scene_aware_raw.sh`, retrain
- Low risk, minimal code change, ~17h training

### 9. ControlNet + LoRA on Both (ControlNet AND DiT)
- Set dual LoRA targets, may need code changes
- Higher VRAM, likely needs cluster

---

## Completed (for reference)

- [x] Cross-frame generation (7 frames x 776 images)
- [x] Cross-frame variance targets computed → targets.json (all 3 conditions)
- [x] Global probing retrained with crossframe_variance (all 3 conditions)
- [x] Spatial streaming re-extraction (Steps 1-4 of overnight pipeline)
- [x] Appendix E populated in probing_report_final.md (global results only)
- [x] Best-of-K NLL reranking analysis → dps_consistency_report.md Sections 6.7-6.8
- [x] Whitelist fix in linear_probing.py (added spatial_crossframe_variance)
- [x] Path fix in update_probing_report.py (agent/reports/)
