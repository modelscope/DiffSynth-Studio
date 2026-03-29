# Key Technical Decisions

**Updated**: 2026-03-23

## Decisions Already Made — Do Not Change Casually

### 1. LoRA-on-ControlNet (not on DiT)
**Decision**: LoRA adapters are placed on the ControlNet module, not the DiT backbone.
**Why**: Empirically better results. The ControlNet processes the SPAD conditioning signal; adapting it makes the model better at interpreting SPAD inputs. LoRA-on-DiT changes the generative prior itself, which is already strong.
**Where**: All training scripts set `--lora_base_model "controlnet"`.
**Alternatives considered**: LoRA-on-DiT, LoRA-on-both. LoRA-on-DiT was tested (`FLUX-SPAD-LoRA_no_conditioning/`) and performed worse.

### 2. Scene-Aware Stratified Train-Test Split
**Decision**: Split by physical location (session), stratified by indoor/outdoor. Zero location leakage.
**Why**: The original random split had 94/101 locations appearing in both train and val — severe data leakage inflating all metrics.
**Where**: `spad_dataset/prepare_dataset_scene_aware.py`, outputs at `spad_dataset/metadata_{train,val}.csv`.
**Result**: 77 train locations (1850 views), 20 val locations (776 views), 14 indoor + 6 outdoor in val.
**DO NOT**: Revert to the old `prepare_dataset.py` random split. The old split is backed up at `spad_dataset/old_random_split/`.

### 3. epoch-15 as Best RAW Checkpoint
**Decision**: `epoch-15.safetensors` selected as best checkpoint based on lowest validation loss (0.3083).
**Caveat**: User previously used epoch-39. This decision could be revisited with an epoch sweep validation, but all current experiments and OD fine-tuning are based on epoch-15. Changing this would require re-running OD fine-tuning.
**Where**: Referenced in `train_od03_finetune.sh`, `train_od07_finetune.sh`, `validate_lora.py` defaults.

### 4. 16-bit SPAD Image Loading via `load_spad_image()`
**Decision**: Custom loading function that normalizes uint16 `[0, 65535]` → uint8 `[0, 255]` instead of PIL's default clamping.
**Why**: PIL's `Image.open().convert('RGB')` clamps values >255 to 255 for 16-bit images, destroying multi-frame SPAD data (which has intermediate values like 16384).
**Where**: `diffsynth/core/data/operators.py` (the canonical implementation), also duplicated in `validate_lora.py`, `validate_dps.py`, `linear_probing.py`, `save_intermediate_latents.py`, `inference_lora.py`.
**DO NOT**: Remove or bypass this function. Single-frame images (0 or 65535 only) work fine either way, but multi-frame images break without it.

### 5. Latent-Space DPS (not Pixel-Space)
**Decision**: Physics guidance operates in the VAE's latent space using L2 gradient, not full pixel-space Bernoulli likelihood.
**Why**: Pixel-space DPS requires backpropagating through the VAE decoder at every denoising step, causing CUDA OOM on 32GB GPU. Latent-space DPS avoids this by comparing pre-encoded SPAD measurements to the current latent estimate.
**Where**: `diffsynth/diffusion/latent_dps.py`, `validate_dps.py`.
**Trade-off**: Latent-space guidance is an approximation — it doesn't directly enforce the Bernoulli likelihood. But it's tractable and shows modest improvement (+0.16 PSNR, -0.87 FID).
**Alternative**: Could try pixel-space DPS on a GPU with more VRAM (48GB+), or with aggressive gradient checkpointing.

### 6. Metrics in Grayscale
**Decision**: All image metrics (PSNR, SSIM, LPIPS, FID, CFID) are computed on grayscale-converted images.
**Why**: SPAD input is monochrome (single-channel binary). The model produces color outputs, but grayscale comparison removes color hallucination from the evaluation, focusing on structural accuracy.
**Where**: `run_metrics.py` (default behavior, `--color` flag to override).

### 7. CFID with Ridge Regularization and float64
**Decision**: CFID computation uses `float64` internally and adds Ridge regularization (`reg=1e-6`) to the conditional covariance matrix.
**Why**: With N=776 samples and D=2048 Inception features, the covariance matrix is rank-deficient. Without regularization, eigendecomposition produces negative eigenvalues, leading to negative (invalid) CFID.
**Where**: `metrics.py`, function `compute_cfid()`.
**DO NOT**: Remove the regularization or switch to float32 without testing.

### 8. Probing Architecture: Joint vs Single Blocks
**Decision**: Hook into both joint blocks (`dit.blocks[i]`, output `[0]` = image tokens) and single blocks (`dit.single_blocks[i]`, slice `output[0][:, txt_len:]` to get image tokens).
**Why**: FLUX DiT has two types of transformer blocks. Joint blocks process image and text tokens separately with cross-attention. Single blocks concatenate them. Extracting image-only tokens requires different slicing for each type.
**Where**: `linear_probing.py`, class `ActivationExtractor`.
**Probed blocks**: Joint {0,4,9,14,18}, Single {0,9,19,28,37} at timesteps {0,4,9,14,19,24,27}.

### 9. Adaptive Ridge Regularization for Probing
**Decision**: Ridge regression lambda is scaled by `trace(XTX) / D` instead of using a fixed constant.
**Why**: With global probing (80 train samples, 3072 features), a fixed lambda either under-regularizes (negative R²) or over-regularizes (zero predictions). Adaptive scaling based on feature magnitude provides stable results.
**Where**: `linear_probing.py`, function `_ridge_regression()`.

### 10. Consistency Training with Stop-Gradient
**Decision**: The second frame's noise prediction is computed with `torch.no_grad()` (stop-gradient).
**Why**: Two full forward passes through FLUX + ControlNet would exceed 32GB VRAM. Stop-gradient on F2 halves the memory requirement at the cost of making the consistency loss asymmetric.
**Where**: `diffsynth/diffusion/consistency_loss.py`.
**Result**: Consistency training didn't improve over baseline. Consider trying without stop-gradient on a larger GPU, or with much lower lambda.

---

## Decisions That Could Be Revisited

| Decision | Current | Alternative | Risk |
|----------|---------|-------------|------|
| Best checkpoint epoch | 15 | 39 (user's prior choice) | Would require re-running OD fine-tuning |
| Consistency weight | 0.1 | 0.01, 0.001 | Lower weight may help |
| DPS guidance schedule | constant eta | linear_decay, cosine | Minor — current shows modest improvement |
| Number of probing blocks | 5 joint + 5 single | All blocks | Would increase activation storage dramatically |
| Spatial probing resolution | 32×32 (1024 tokens) | Higher with interpolation | Memory trade-off |
| Grayscale metrics | Default | Color metrics | Would need re-running all metrics |
