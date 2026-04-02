# Agent Directory Index

**Updated**: 2026-03-29

This directory contains all project context, reports, plans, audits, and logs for AI agent continuation.

---

## Quick Start for New Agents

1. **Read first**: `THESIS_CONTEXT.md` — ~1170-line master document covering everything
2. **What to do next**: `TODO.md` — prioritized task list
3. **Key decisions**: `DECISIONS.md` — what not to change casually
4. **Figures**: `FIGURES.md` — complete figure inventory with regeneration commands
5. **Presentation**: `PRESENTATION_STRATEGY.md` — Heilmeier-based talk outline
6. **This document**: `INDEX.md` — navigating the agent folder

---

## Directory Structure

```
agent/
├── THESIS_CONTEXT.md          # ★ MASTER DOCUMENT — read this first (~1170 lines)
├── TODO.md                    # Prioritized task list with exact commands
├── DECISIONS.md               # Key technical decisions (14 items)
├── HANDOFF.md                 # Full project state for handoff
├── LAST_PROMPT.md             # Ready-to-use continuation prompt
├── TEST_STATUS.md             # All experiments and their results
├── INDEX.md                   # THIS FILE
│
├── reports/                   # Analysis reports and experiment results
│   ├── probing_report_final.md    # ★ ~1170-line linear probing analysis (FLUX + SD1.5)
│   │                              #   Appendices A-D: FLUX full tables
│   │                              #   Appendix E: Cross-frame variance
│   │                              #   Appendix F: SD1.5 cross-architecture comparison
│   ├── dps_consistency_report.md  # DPS + consistency training analysis
│   ├── probing_report_interim.md  # Earlier probing report (superseded)
│   ├── probing_report.md          # Initial probing report (superseded)
│   ├── literature_gqir_vae.md     # ★ gQIR paper analysis + VAE role in our pipeline
│   ├── experiment_audit.json      # Full experiment audit data
│   └── EXECUTION_ORDER.md         # Experiment execution sequence
│
├── audits/                    # Formal code and methodology audits
│   ├── AUDIT_IMG2IMG_ABLATION.md          # ★ img2img ablation: code trace, methodology, bugs
│   ├── AUDIT_DPS_CONSISTENCY_2026-03-23.md # DPS + consistency code audit (3 external audits)
│   └── antigravity_spad_audit_report.md    # Antigravity agent: project-wide audit
│
├── external_plans/            # Plans from other AI agents
│   ├── antigravity_brainstorm.md         # Antigravity: initial brainstorm
│   ├── antigravity_thesis_analysis.md    # Antigravity: thesis framing
│   ├── antigravity_dps_consistency_review.md  # Antigravity: DPS review
│   ├── antigravity_linear_probing_review.md   # Antigravity: probing review
│   └── cursor_spad_neurips_full_plan.md  # Cursor: original NeurIPS plan
│
├── chat_exports/              # Conversation logs
│   └── conversation_export_full.md  # Full conversation history (13K lines)
│
├── logs/                      # Training and experiment logs
│   ├── train_scene_aware_raw.log      # RAW baseline training
│   ├── consistency_train.log          # Consistency training
│   ├── od03_ft_log.txt                # OD03 fine-tune training
│   ├── od07_ft_log.txt                # OD07 fine-tune training
│   ├── od03_scratch_log.txt           # OD03 scratch training
│   ├── multiseed_validation.log       # Multi-seed inference
│   ├── physics_ablation.log           # DPS physics sweep
│   ├── frame_ablation.log             # Frame-count ablation
│   ├── probing_allblocks_train_log.txt  # Probing training
│   ├── overnight_pipeline.log         # Overnight pipeline run
│   └── ...                            # Other logs
│
├── FIGURES.md                 # ★ FIGURE INVENTORY — what each figure shows, how to regenerate
├── PRESENTATION_STRATEGY.md   # Heilmeier catechism talk outline and slide plan
├── git_status.txt             # Git status snapshot
├── git_diff.patch             # Git diff snapshot
└── recent_commits.txt         # Recent git commits
```

---

## Reports Summary

| Report | Lines | Content | Status |
|--------|-------|---------|--------|
| `probing_report_final.md` | ~1170 | Full FLUX probing (3 conditions) + SD1.5 cross-architecture (App F) | **Current** |
| `dps_consistency_report.md` | ~900 | DPS physics guidance + consistency training + best-of-K | **Current** |
| `AUDIT_IMG2IMG_ABLATION.md` | ~300 | img2img ablation: end-to-end code trace, methodology audit, bug report | **Current** |
| `AUDIT_DPS_CONSISTENCY_2026-03-23.md` | ~500 | Code audit from 3 external agents (Cursor, Gemini, Codex) | **Current** |
| `literature_gqir_vae.md` | ~170 | gQIR paper analysis, VAE domain gap, ControlNet justification | **Current** |
| `ALL_METRICS.md` | ~200 | **Consolidated metrics** — all experiments in one file | **Current** |
| `parameter_count_audit.md` | ~250 | CN LoRA (40.3M) vs full CN (3.30B) vs DiT LoRA (153.2M) — verified against model | **Current** |

---

## Figures Inventory

**Full reference**: See `FIGURES.md` for complete details, data sources, and regeneration commands.

All publication-quality figures in `thesis_figures/publication/` (16 figures, PDF + PNG each):

### Reconstruction Comparison (8 figures)
| File | Description | Use |
|------|-------------|-----|
| `fig_hero_spad_to_rgb` | 6-column SPAD -> Reconstruction -> GT | **Slide 1 / Fig 1** |
| `fig_master_comparison_grid` | 6x7 grid: all methods + variance | Paper main figure |
| `fig_pairwise_baseline_vs_dps` | Baseline vs DPS side-by-side | DPS section |
| `fig_pairwise_baseline_vs_consistency` | Baseline vs Consistency | Consistency section |
| `fig_pairwise_baseline_vs_consist_dps` | Baseline vs Consist+DPS | Combined method |
| `fig_pairwise_dps_vs_consistency` | DPS vs Consistency | Method comparison |
| `fig_frame_ablation_strip` | N=1..1000 Input/Output + GT | Frame ablation visual |
| `fig_variance_overlay` | Reconstruction + seed variance | Uncertainty section |

### Quantitative (2 figures)
| File | Description | Use |
|------|-------------|-----|
| `fig_frame_ablation_curves` | PSNR/LPIPS/CFID vs frame count | Frame ablation results |
| `fig_dps_ablation_bars` | DPS eta sweep bar charts | DPS results |

### Probing Figures (`probing_analysis_output/`, 7 figures)
| File | Description | Use |
|------|-------------|-----|
| `fig1_main_heatmap` | 3-panel R^2 heatmap across blocks/timesteps | **Key probing figure** |
| `fig2_main_vs_control` | Main vs Control (no LoRA) | LoRA effect |
| `fig3_delta_heatmap` | R^2(Main) - R^2(Control) delta | LoRA contribution |
| `fig4_best_timestep_lineplot` | R^2 vs block index (AC3D-style) | Information flow |
| `fig5_dit_vs_cn` | DiT vs ControlNet blocks | Architecture analysis |
| `fig6_global_vs_spatial` | Global vs spatial probing best-R^2 | Methodology |
| `fig7_object_probing` | 24-object balanced accuracy | Semantic understanding |

### SD1.5 Probing Heatmaps (`spad-diffusion/probing_results_sd15/probes/`)
| File | Description | Use |
|------|-------------|-----|
| `heatmap_{target}.png` | 38-block heatmaps (bit_density, depth, variance, crossframe_variance) | Cross-architecture comparison |
| `heatmap_spatial_{target}.png` | Spatial streaming heatmaps (4 targets) | Spatial probing |
| `heatmap_obj_{object}.png` | 24 object presence heatmaps | Object recognition |

---

## Key Metrics Reference

### Reconstruction Quality
| Experiment | PSNR | SSIM | LPIPS | FID | CFID |
|------------|------|------|-------|-----|------|
| Baseline (seed 42) | 17.89 | 0.596 | 0.415 | 66.84 | 151.94 |
| Baseline (10-seed mean) | 17.99 +/- 0.09 | 0.596 | 0.415 | 66.29 | 152.04 |
| DPS eta=1.0 | 18.05 | 0.597 | 0.413 | 65.97 | 151.35 |
| Consistency epoch-0 | 17.72 | 0.589 | 0.422 | 66.51 | 154.99 |
| Best-of-10 (NLL) | 15.30 | 0.585 | 0.433 | 66.46 | 156.96 |
| img2img (best) | 7.59 | 0.026 | 1.055 | 283.86 | 351.22 |
| Frame N=256 | 14.12 | 0.605 | 0.339 | 70.66 | 110.11 |

### Linear Probing (Best R^2)
| Target | FLUX Global | FLUX Spatial | SD1.5 Global | SD1.5 Spatial |
|--------|------------|-------------|-------------|--------------|
| Bit density | 0.998 | 0.959 | 0.993 | 0.974 |
| Depth | 0.437 | 0.685 | 0.375 | 0.727 |
| Variance (seed) | 0.424 | 0.506 | 0.472 | 0.493 |
| Cross-frame var | 0.292 | **0.359** | 0.293 | 0.279 |
