# Action Conditioning Framework — Design Document

## Goal

Modify DiffSynth-Studio so that any DiT backbone (Wan, CogVideo, …) can be
conditioned on:

1. **Observation image** — the first frame / current world state
2. **Action sequence** — encoded robot actions
3. **Masked image sequence** — trajectory visualisations produced by `ActionMapper`
4. **History** — last N generated frames (autoregressive)

Every conditioning stream has an independently switchable injection strategy
so we can run ablation experiments **without changing code** — only config.

---

## End-to-End Data Flow

```
[Raw action sequence]  (B, T, action_dim)
        │
  ActionMapper.map()          ← your rendering code; outside the model graph
        │
[Masked image sequence]  (B, 3, T, H, W)   trajectory overlays in [-1,1]
        │
        ├─── VAE.encode() ──────────────────────────────────┐
        │                                                    │
[Masked latents]  (B, C, T, Hl, Wl)                        │
                                                            ▼
══════════════════════════════════════════════════════════════════════
ActionConditionedDiT.forward()

  Inputs assembled from four streams:
  ┌─────────────────────────────────────────────────────────────────┐
  │  obs_latent      (B, C, T, Hl, Wl)   always input_concat       │
  │  masked_latents  (B, C, T, Hl, Wl)   input_concat OR cross_attn│
  │  history_latents (B, C*N, T, Hl, Wl) input_concat OR cross_attn│
  │  actions         (B, T, action_dim)   cross_attn  OR adaln      │
  └─────────────────────────────────────────────────────────────────┘
        │
  Backbone DiT  (WanModel / CogDiT)
        │
  [Predicted noise]  (B, C, T, Hl, Wl)
══════════════════════════════════════════════════════════════════════
        │
   VAE.decode()
        │
[Generated frames]  (B, 3, T, H, W)  →  next chunk's history
```

### Autoregressive loop

```python
history = [obs_latent] * history_len          # initialise with obs

for chunk_actions in sliding_window(all_actions, chunk_size):
    masked_images  = action_mapper.map(chunk_actions)   # render
    masked_latents = vae.encode(masked_images)
    history_latents = stack(history[-history_len:])     # (B, C*N, ...)

    generated = dit(obs_latent, chunk_actions,
                    masked_latents, history_latents)

    history.append(vae.encode(generated))
```

---

## Ablation Experiment Matrix

```
                  cross_attn    input_concat    adaln
────────────────────────────────────────────────────
action tokens        ✓                           ✓
masked images        ✓               ✓
history              ✓               ✓
obs image            fixed: input_concat  (anchor, not ablated)
```

Each cell that is marked ✓ is a valid `injection_type` for that stream.
The config dataclass encodes a single experiment:

```yaml
# exp_B.yaml  — EVAC-style recommended baseline
backbone: wan
backbone_dim: 5120
action:
  injection_type: cross_attn
  encoder_type:   perceiver
masked:
  injection_type: input_concat
  encoder_type:   vae
history:
  injection_type: input_concat
  encoder_type:   vae
```

---

## Repository Changes

### New files

```
diffsynth/
├── models/
│   └── action_conditioning/          ← NEW package
│       ├── __init__.py
│       ├── config.py                 config dataclasses
│       ├── encoders.py               ActionEncoder, Perceiver, MLP
│       ├── injectors.py              CrossAttn / InputConcat / AdaLN injectors
│       ├── action_mapper.py          ActionMapper ABC + placeholder
│       └── dit_wrapper.py            ActionConditionedDiT
└── pipelines/
    └── action_video_pipeline.py      ← NEW autoregressive pipeline
```

### Modified files

```
diffsynth/models/wan_video_dit.py     WanModel.forward()  — 4 line change
diffsynth/models/cog_dit.py           CogDiT.forward()    — 5 line change
```

---

## Detailed File Descriptions

### `config.py`

Two dataclasses:

```python
@dataclass
class ConditionStreamConfig:
    injection_type : "cross_attn" | "input_concat" | "adaln"
    encoder_type   : "perceiver" | "mlp" | "vae" | "identity"
    embed_dim      : int = 1024
    enabled        : bool = True

@dataclass
class ActionConditioningConfig:
    backbone        : "wan" | "cogvideo"
    backbone_dim    : int          # 5120 (Wan-14B), 1536 (Wan-1.3B), 3072 (CogVideo)
    text_dim        : int = 4096
    latent_channels : int = 16
    action_dim      : int = 14     # delta action size
    history_len     : int = 4

    action  : ConditionStreamConfig
    masked  : ConditionStreamConfig
    history : ConditionStreamConfig

    # Perceiver hyper-params
    perceiver_num_queries : int = 16
    perceiver_depth       : int = 4
    perceiver_num_heads   : int = 8
    perceiver_ff_mult     : int = 4
```

---

### `encoders.py`

```
ActionEncoder  (ABC)
  └─ forward(actions: (B,T,D)) -> (B,N,embed_dim)

PerceiverActionEncoder
  input_proj : Linear(action_dim, embed_dim)
  latents    : Parameter(1, num_queries, embed_dim)   ← learned queries
  layers     : N × PerceiverLayer                     ← cross-attend to actions
  norm       : LayerNorm
  output     : (B, num_queries, embed_dim)

MLPActionEncoder
  mlp    : Linear → GELU → [Linear → GELU] × (depth-1)
  norm   : LayerNorm
  output : (B, T, embed_dim)   ← sequence length preserved
```

---

### `injectors.py`

```
CrossAttnInjector
  encoder : ActionEncoder
  proj    : Linear(embed_dim, backbone_dim)
  ─────────────────────────────────────────
  get_context_tokens(condition)
    → encoder(condition)          (B, N, embed_dim)
    → proj(...)                   (B, N, backbone_dim)
    concatenated into context sequence before cross-attn

InputConcatInjector
  n_channels : int
  ─────────────────────────────────────────
  get_extra_channels(latent)
    → latent as-is                (B, C, T, H, W)
    concatenated along channel dim before patchify
    (backbone patch_embedding extended to absorb extra channels)

AdaLNInjector
  encoder : ActionEncoder
  proj    : Linear → SiLU → Linear   output: (B, backbone_dim)
  ─────────────────────────────────────────
  get_time_delta(condition)
    → encoder(condition).mean(dim=1)  (B, embed_dim)  pool over queries
    → proj(...)                       (B, backbone_dim)
    added to time embedding t before time_projection inside backbone
```

---

### `action_mapper.py`

```
ActionMapper  (ABC)
  map(actions: (B,T,D), **kwargs) -> images: (B,3,T,H,W) in [-1,1]

IdentityActionMapper
  Returns zeros.  Replace with your rendering implementation.
```

---

### `dit_wrapper.py` — `ActionConditionedDiT`

#### Initialization

```
__init__(backbone, config)
  1. Build injectors per stream based on config.injection_type
  2. _extend_patch_embed()
       compute extra_ch = latent_ch + 1(mask)
                        + latent_ch         if masked  is input_concat
                        + latent_ch*N       if history is input_concat
       replace backbone.patch_embedding (Wan)  or
               backbone.patchify.proj   (Cog)
       with new Conv3d(old_in + extra_ch, ...)
       original weights copied, new channels zero-initialised
       set backbone.has_image_input = False  (Wan only)
```

#### Forward

```
forward(noisy_latent, timestep, text_context, obs_latent,
        actions=None, masked_latents=None, history_latents=None, mask=None)

Step 1 — build_input()   (channel concat)
  [noisy_latent | obs_latent | mask
   | masked_latents   (if input_concat)
   | history_latents  (if input_concat)]   → (B, total_C, T, H, W)

Step 2 — build_context()  (sequence concat)
  backbone.text_embedding(text_context)    → (B, S, backbone_dim)
  action_injector.get_context_tokens()     → (B, N_a, backbone_dim)  if cross_attn
  masked_proj(flatten(masked_latents))     → (B, N_m, backbone_dim)  if cross_attn
  history_proj(flatten(history_latents))   → (B, N_h, backbone_dim)  if cross_attn
  cat all along dim=1                      → (B, S+N, backbone_dim)

Step 3 — adaln_extra
  if action.injection_type == "adaln":
    adaln_extra = action_injector.get_time_delta(actions)  → (B, backbone_dim)

Step 4 — call backbone
  backbone(x, timestep, context,
           adaln_extra=adaln_extra,
           context_is_projected=True)   ← skips internal text projection
```

---

### Backbone patches

#### `wan_video_dit.py` — `WanModel.forward()`

Add two keyword arguments and three lines:

```diff
 def forward(self, x, timestep, context,
             clip_feature=None, y=None,
             use_gradient_checkpointing=False,
             use_gradient_checkpointing_offload=False,
+            adaln_extra: Optional[torch.Tensor] = None,
+            context_is_projected: bool = False,
             **kwargs):
     t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
+    if adaln_extra is not None:
+        t = t + adaln_extra
     t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
-    context = self.text_embedding(context)
+    if not context_is_projected:
+        context = self.text_embedding(context)
```

#### `cog_dit.py` — `CogDiT.forward()`

Add `from typing import Optional` and the same two-kwarg / three-line pattern:

```diff
+from typing import Optional
 ...
 def forward(self, hidden_states, timestep, prompt_emb,
             image_rotary_emb=None, tiled=False,
             tile_size=90, tile_stride=30,
-            use_gradient_checkpointing=False):
+            use_gradient_checkpointing=False,
+            adaln_extra: Optional[torch.Tensor] = None,
+            context_is_projected: bool = False):
     ...
     time_emb = self.time_embedder(timestep, dtype=hidden_states.dtype)
+    if adaln_extra is not None:
+        time_emb = time_emb + adaln_extra
-    prompt_emb = self.context_embedder(prompt_emb)
+    if not context_is_projected:
+        prompt_emb = self.context_embedder(prompt_emb)
```

---

### `action_video_pipeline.py`

```
ActionVideoPipeline
  vae            : VAE model
  dit            : ActionConditionedDiT
  scheduler      : FlowMatchScheduler or similar
  action_mapper  : ActionMapper
  config         : ActionConditioningConfig

reset_history(obs_latent)
  fill history_buffer with [obs_latent] * history_len

generate_chunk(obs_image, actions, num_inference_steps, guidance_scale)
  1. vae.encode(obs_image)             → obs_latent
  2. action_mapper.map(actions)        → masked_images
     vae.encode(masked_images)         → masked_latents
  3. stack history_buffer              → history_latents
  4. randn noise
  5. denoising loop
       dit.forward(noisy_latent, t, text_ctx=zeros,
                   obs_latent, actions, masked_latents, history_latents)
       scheduler.step()
  6. vae.decode(latents)               → frames
  7. update history_buffer (rolling last N latent frames)
  return frames

generate(obs_image, all_actions, chunk_size=16)
  reset_history(obs_image)
  for each chunk:
      frames = generate_chunk(...)
  return cat(all_frames, dim=2)
```

---

## Tensor Shape Reference

| Variable | Shape | Notes |
|---|---|---|
| `noisy_latent` | `(B, C, T, Hl, Wl)` | diffusion input |
| `obs_latent` | `(B, C, 1, Hl, Wl)` | auto-broadcast to T |
| `masked_latents` | `(B, C, T, Hl, Wl)` | input_concat path |
| `masked_latents` | `(B, C, Tm, Hm, Wm)` | cross_attn path (flattened) |
| `history_latents` | `(B, C*N, T, Hl, Wl)` | input_concat path |
| `history_latents` | `(B, C, N, Hl, Wl)` | cross_attn path (flattened) |
| `actions` | `(B, T, action_dim)` | raw delta actions |
| `text_context` | `(B, S, text_dim)` | pre-projection (4096 for Wan/Cog) |
| `x` (after build_input) | `(B, total_C, T, Hl, Wl)` | fed to patch_embed |
| `context` (after build_context) | `(B, S+N, backbone_dim)` | fed to cross-attn |

`total_C` for the default Exp-B config (Wan-14B, masked+history=input_concat):
```
total_C = 16 (noisy)
        + 16 (obs)
        +  1 (mask)
        + 16 (masked_latents)
        + 64 (history, 4 frames × 16)
        = 113
```
So `patch_embedding` becomes `Conv3d(113, 5120, ...)`.

---

## File Status

| File | Status |
|---|---|
| `diffsynth/models/action_conditioning/__init__.py` | ✅ pushed |
| `diffsynth/models/action_conditioning/config.py` | ✅ pushed |
| `diffsynth/models/action_conditioning/encoders.py` | ✅ pushed |
| `diffsynth/models/action_conditioning/injectors.py` | ✅ pushed |
| `diffsynth/models/action_conditioning/action_mapper.py` | ✅ pushed |
| `diffsynth/models/action_conditioning/dit_wrapper.py` | ✅ pushed |
| `diffsynth/pipelines/action_video_pipeline.py` | ⏳ pending |
| `diffsynth/models/wan_video_dit.py` | ⏳ pending patch |
| `diffsynth/models/cog_dit.py` | ⏳ pending patch |
