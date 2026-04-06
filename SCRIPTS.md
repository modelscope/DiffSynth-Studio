# Script Index

All scripts live under `scripts/`. Run from the repo root:
```bash
python scripts/validation/validate_lora.py --help
bash scripts/experiments/run_multiseed_validation.sh
```

---

## `scripts/training/` — Model Training

| Script | Description |
|--------|-------------|
| `train_lora.py` | FLUX ControlNet LoRA training loop (main training script) |
| `train_lora.sh` | Shell wrapper for `train_lora.py` with default args |
| `train_scene_aware_raw.sh` | Train RAW model with scene-aware train/val split (no leakage) |
| `train_consistency.py` | Training with per-frame consistency loss |
| `train_consistency.sh` | Shell wrapper for consistency training |
| `train_controlnet.sh` | ControlNet LoRA training (DiT target, FP8) |
| `train_od03_finetune.sh` | Fine-tune RAW checkpoint on OD0.3 data |
| `train_od03_scratch.sh` | Train on OD0.3 from scratch |
| `train_od07_finetune.sh` | Fine-tune RAW checkpoint on OD0.7 data |
| `train_img2img_ablation.sh` | Train img2img LoRA (DiT only, no ControlNet) |
| `paired_spad_dataset.py` | Dataset class: two different binary frames per scene |

## `scripts/validation/` — Inference & Validation

| Script | Description |
|--------|-------------|
| `validate_lora.py` | Primary validation: LoRA + ControlNet inference on val set |
| `validate_lora.sh` | Shell wrapper for single validation run |
| `validate_dps.py` | Pixel-space DPS: physics-consistent measurement-guided sampling |
| `validate_flow_dps.py` | Latent-space FlowDPS with VAE pre-encoding |
| `validate_img2img.py` | img2img ablation: LoRA on DiT, no ControlNet |
| `validate_crossframe.py` | Cross-frame variance: different binary realizations of same scene |
| `best_of_k.py` | Best-of-K reranking: pick lowest-NLL seed per image |
| `inference_lora.py` | Single-image inference (low VRAM) |
| `inference_lora.sh` | Shell wrapper for single-image inference |

## `scripts/metrics/` — Evaluation Metrics

| Script | Description |
|--------|-------------|
| `metrics.py` | Core metrics: MSE, PSNR, SSIM, LPIPS, FID, CFID |
| `run_metrics.py` | Compute metrics on a validation output directory |
| `aggregate_metrics.py` | Aggregate metrics across K seeds, generate LaTeX tables |

## `scripts/experiments/` — Experiment Orchestration

Shell scripts that chain training, validation, and metrics for specific experiments.

| Script | Description |
|--------|-------------|
| `run_multiseed_validation.sh` | Generate K=10 seeds, compute per-seed metrics, aggregate |
| `run_frame_ablation.sh` | Frame-count ablation: 4, 16, 64, 256, 1000 frames |
| `run_img2img_ablation.sh` | Denoising strength sweep (0.3 to 1.0) |
| `run_nolora_ablation.sh` | No-LoRA ablation: 5 ControlNet processor modes (gray/lq/canny/tile/depth) |
| `run_od_ablation.sh` | OD filter ablation matrix: 4 models x 4 SPAD inputs |
| `run_od_single_image_v2.sh` | Single-image OD ablation (fast visual comparison) |
| `run_physics_ablation.sh` | Baseline + DPS variants + consistency + combined |
| `run_probing_comprehensive.sh` | Full probing: main + control + no-CN + spatial streaming |
| `run_probing_experiments.sh` | Basic probing pipeline (prepare + extract + train) |
| `run_segmentation_probing.sh` | SAM3 segmentation probing targets |
| `run_overnight_pipeline_v2.sh` | Overnight: consistency epoch sweep + OD ablation + spatial crossframe |
| `run_all_experiments.sh` | Master pipeline: img2img + consistency + probing |
| `run_all_remaining.sh` | Post-multiseed phases: aggregate + variance + calibration + probing |

## `scripts/analysis/` — Analysis & Probing

| Script | Description |
|--------|-------------|
| `linear_probing.py` | Full linear probing pipeline: prepare targets, extract activations, train ridge probes |
| `probing_analysis.py` | Generate publication figures from probing results (heatmaps, comparisons) |
| `compute_crossframe_targets.py` | Compute cross-frame variance targets for probing |
| `compute_variance_maps.py` | Per-pixel RGB variance across K seeds |
| `frame_vs_seed_variance.py` | Decompose variance: measurement vs seed components |
| `calibration_analysis.py` | Empirical confidence intervals from K seeds |
| `downstream_stability.py` | Stability analysis via segmentation + depth on K seeds |
| `analyze_vae_domain_gap.py` | SPAD vs GT RGB through frozen FLUX VAE |
| `audit_vae_roundtrip.py` | Quick VAE roundtrip audit on binary frames |
| `save_intermediate_latents.py` | Save decoded images at intermediate denoising steps |
| `sweep_pixel_dps.py` | Pixel-space DPS hyperparameter sweep (eta, schedule) |
| `test_dps_physics.py` | Unit tests for SPAD physics DPS |
| `update_probing_report.py` | Auto-populate cross-frame results into probing report |

## `scripts/visualization/` — Figure Generation

| Script | Description |
|--------|-------------|
| `generate_thesis_figures.py` | All thesis/paper figures (hero, pairwise, ablation, etc.) |
| `generate_montages.py` | Side-by-side comparison montages |
| `generate_photon_sweep_video.py` | Photon-sweep video: OD0.7 to 1000 frames (boomerang) |
| `generate_seg_targets.py` | Generate SAM3 segmentation targets for probing |
| `plot_probing_heatmap_4panel.py` | 4-panel probing heatmap (bit density, depth, variance, cross-frame) |

## `scripts/probing_viz/` — Presentation Figures

| Script | Description |
|--------|-------------|
| `generate_depth_slide.py` | Slide-quality depth probe figure (2 rows: indoor + outdoor) |
| `generate_probing_visualizations.py` | Spatial depth/variance predictions + scorecard table |
| `extract_single_image_spatial.py` | Extract spatial activations for a single image (hooks specific DiT blocks) |

## `scripts/data_prep/` — Dataset Preparation

| Script | Description |
|--------|-------------|
| `prepare_dataset.py` | Generate metadata.csv with train/val split |
| `prepare_dataset.sh` | Shell wrapper for dataset preparation |
| `prepare_controlnet_metadata.sh` | Update CSV: input_image to controlnet_image column |

---

## Output Directories (gitignored)

| Directory | Contents |
|-----------|----------|
| `validation_outputs_scene_aware/` | Baseline reconstructions (seed_42, 776 images) |
| `validation_outputs_multiseed/` | 10-seed reconstructions (6.1 GB) |
| `validation_outputs_frame_ablation/` | Frame-count ablation (3.5 GB) |
| `validation_outputs_img2img_ablation/` | img2img strength sweep (3.8 GB) |
| `validation_outputs_physics_ablation/` | DPS + consistency variants (3.7 GB) |
| `validation_outputs_crossframe/` | 7 frame realizations x 776 images |
| `validation_outputs_nolora_*/` | 5 no-LoRA processor modes |
| `validation_outputs_od_single_image/` | Single-image OD filter comparison |
| `probing_results_allblocks/` | Main model activations + probing results |
| `probing_results_control/` | No-LoRA baseline activations |
| `probing_results_no_cn/` | No-ControlNet ablation activations |
| `probing_results/` | Legacy probing (100 spatial files) |
| `probing_analysis_output/` | Publication figures from probing |
| `thesis_figures/` | All thesis/paper figures |
| `variance_analysis/` | Per-pixel variance maps |

## Documentation (in `agent/`)

| File | Contents |
|------|----------|
| `agent/THESIS_CONTEXT.md` | Comprehensive thesis context (architecture, experiments, results) |
| `agent/INDEX.md` | Repository index |
| `agent/EXPERIMENTS.md` | Experiment documentation |
| `agent/reports/probing_report_final.md` | Complete probing analysis report |
| `agent/reports/ALL_METRICS.md` | Consolidated metrics across all experiments |
| `agent/reports/dps_consistency_report.md` | DPS and consistency training analysis |
| `agent/reports/literature_gqir_vae.md` | gQIR / VAE literature review |
