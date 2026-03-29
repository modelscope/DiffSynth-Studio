# Publication Figures Reference

**Updated**: 2026-03-29
**Location**: `thesis_figures/publication/`
**Generation Script**: `generate_thesis_figures.py`

---

## How to Regenerate

```bash
cd /home/jw/engsci/thesis/spad/DiffSynth-Studio-SPAD

# All figures (31 total, organized into subfolders)
python generate_thesis_figures.py --all

# Individual experiment groups
python generate_thesis_figures.py --hero
python generate_thesis_figures.py --pairwise
python generate_thesis_figures.py --frame-strip
python generate_thesis_figures.py --master-grid
python generate_thesis_figures.py --metric-curves
python generate_thesis_figures.py --dps-ablation
python generate_thesis_figures.py --variance
python generate_thesis_figures.py --depth
python generate_thesis_figures.py --seg
python generate_thesis_figures.py --probing
python generate_thesis_figures.py --leakage
python generate_thesis_figures.py --baseline-table
python generate_thesis_figures.py --contribution-table
```

---

## Directory Structure

```
thesis_figures/publication/
├── 01_hero/
│   ├── hero_6col.{pdf,png}          # 6-column wide hero (best for paper)
│   ├── hero_4col.{pdf,png}          # 4-column (good for slides)
│   └── hero_3col.{pdf,png}          # 3-column compact (slides)
│
├── 02_pairwise/
│   ├── baseline_vs_dps_4row.{pdf,png}
│   ├── baseline_vs_dps_2row.{pdf,png}
│   ├── baseline_vs_consistency_4row.{pdf,png}
│   ├── baseline_vs_consistency_2row.{pdf,png}
│   ├── baseline_vs_consist_dps_4row.{pdf,png}
│   ├── baseline_vs_consist_dps_2row.{pdf,png}
│   ├── dps_vs_consistency_4row.{pdf,png}
│   └── dps_vs_consistency_2row.{pdf,png}
│
├── 03_frame_ablation/
│   ├── strip_3row.{pdf,png}          # 3 scenes × (N=1..1000 + GT λ)
│   └── strip_2row.{pdf,png}          # 2 scenes (compact)
│
├── 04_master_grid/
│   ├── grid_6row.{pdf,png}           # 6 scenes × 7 methods
│   └── grid_4row.{pdf,png}           # 4 scenes (compact)
│
├── 05_metrics/
│   └── frame_ablation_curves.{pdf,png}  # PSNR/LPIPS/CFID vs N
│
├── 06_dps_ablation/
│   └── dps_eta_sweep.{pdf,png}       # η sweep bar charts
│
├── 07_variance/
│   ├── variance_6col.{pdf,png}       # 6 examples
│   └── variance_4col.{pdf,png}       # 4 examples (compact)
│
├── 08_depth/
│   ├── depth_4row.{pdf,png}          # 4 depth comparisons
│   └── depth_2row.{pdf,png}          # 2 depth comparisons (compact)
│
├── 09_segmentation/
│   ├── seg_compact_4row.{pdf,png}    # 3-col: GT | GT Seg | Gen Seg
│   ├── seg_compact_2row.{pdf,png}    # Same, 2 rows
│   ├── seg_wide_4row.{pdf,png}       # 5-col: SPAD | GT | Gen | GT Seg | Gen Seg
│   └── seg_pairs_4row.{pdf,png}      # 4-col: GT+Seg vs Gen+Seg
│
├── 10_probing/
│   ├── summary_panel.{pdf,png}       # R² heatmap + delta + global-vs-spatial
│   ├── variance_heatmaps.{pdf,png}   # Cross-seed + cross-frame variance R²
│   ├── object_recognition.{pdf,png}  # Per-category object R² (DiT vs CN)
│   ├── fig1_main_heatmap.{png,pdf}   # [copied] Main R² heatmap
│   ├── fig2_main_vs_control.{png,pdf}# [copied] LoRA vs no-LoRA
│   ├── fig3_delta_heatmap.{png,pdf}  # [copied] LoRA delta
│   ├── fig4_best_timestep_lineplot.{png,pdf}  # [copied] AC3D-style flow
│   ├── fig5_dit_vs_cn.{png,pdf}      # [copied] DiT vs ControlNet
│   ├── fig6_global_vs_spatial.{png,pdf} # [copied] Global vs spatial
│   └── fig7_object_probing.{png,pdf} # [copied] 24-object accuracy
│
├── 11_leakage/
│   └── leakage_fix.{pdf,png}         # Before/after split bar chart
│
└── 12_tables/
    ├── baseline_summary.{pdf,png}     # Color-coded results table
    └── contribution_boundary.{pdf,png}# Who did what table
```

---

## Key Design Decisions

### Scene-ID Mapping (Critical Fix)
The frame ablation strip GT column now uses **scene-ID-based mapping** instead of positional indexing. The validation CSV (`spad_dataset/bits/metadata_val.csv`) maps each validation index (0-775) to a scene ID like `0724-dgp-297`, which is then looked up in the monochrome folder. This ensures the correct long-exposure lambda image is shown for each scene.

### Segmentation Layout Options
Three layouts are provided because the original SAM3 montages include legends that don't scale well:
- **compact**: 3 columns, drops input/generated RGB, focuses on GT vs segmentation
- **wide**: 5 columns, includes everything including SPAD input
- **pairs**: 4 columns, GT RGB+Seg side-by-side with Gen RGB+Seg

### Probing Variance Types
Two types of variance probing are shown:
- **Cross-seed variance**: How much the model's output varies across different random seeds (same input)
- **Cross-frame variance** (ControlNet): How much the ControlNet activations vary across different binary frames of the same scene

---

## Style Reference

- **Font**: Liberation Serif (paper-appropriate Times-like serif)
- **Color Palette**: Dark blue (#1a5276), Crimson (#c0392b), Forest green (#196f3d), Teal (#117a65), Purple (#6c3483), Gold (#b7950b)
- **DPI**: 300 for saved figures, 200 for display
- **Format**: Both PDF (vector, for LaTeX) and PNG (raster, for slides)
- **Background**: White
- **Borders**: Light grey (#d0d0d0), 0.4pt

---

## Figures Still Needed

1. **Architecture diagram** — FLUX+ControlNet+LoRA with probing hooks (manual TikZ or draw.io)
2. **Spatial depth prediction** — predicted depth map from probing vs GT (from probing spatial outputs)
3. **OD filter ablation** — once OD training completes
4. **SD1.5 comparison** — if SD1.5 re-evaluation on scene-aware split is run
