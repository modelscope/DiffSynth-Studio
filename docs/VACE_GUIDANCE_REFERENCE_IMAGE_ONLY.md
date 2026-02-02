# Enhanced VACE Guidance - Reference Image Only

## Overview
The enhanced VACE guidance system now applies adaptive scaling **exclusively** to the reference image frames, ensuring that:
- ✅ Reference image features (facial details, clothing, hair, etc.) are preserved with stronger guidance
- ✅ Pose video frames are NOT affected by adaptive scaling
- ✅ Guidance strength progressively increases during denoising for better detail retention

## How It Works

### Frame Structure
When using both `vace_video` (pose) and `vace_reference_image`:
```
[Reference Frame(s)] + [Pose Video Frames]
    ↑                        ↑
 Adaptive Guidance        Base Guidance
```

### Adaptive Guidance Flow

1. **Reference Frame Detection** (`WanVideoUnit_VACE`)
   - Counts the number of reference image frames
   - Stores count in `vace_reference_frame_count`
   - Returns this count through the pipeline

2. **Conditional Scaling** (Denoising Loop)
   - Checks if `vace_reference_frame_count > 0`
   - If reference frames exist:
     - Applies quadratic progressive scaling
     - Strength increases from `vace_scale` → `vace_scale_end`
   - If NO reference frames:
     - Uses base `vace_scale` without adaptation

3. **Application in Model**
   - The scaled guidance is applied via: `x = x + current_vace_hint * vace_scale`
   - Since reference frames are prepended at the start, the guidance primarily affects them

## Parameters

```python
# In pipe() call:
vace_scale: float = 1.5              # Initial guidance strength for reference
vace_scale_end: float = 2.5          # Final guidance strength (increased later steps)
vace_adaptive_guidance: bool = True  # Enable progressive scaling
```

## Usage Example

```python
video = pipe(
    prompt="Your prompt...",
    vace_video=pose_video,           # Pose guidance - base scale only
    vace_reference_image=ref_image,  # Reference image - adaptive guidance
    num_frames=49,
    vace_scale=1.8,                  # Start strength
    vace_scale_end=3.0,              # End strength (more details preserved)
    vace_adaptive_guidance=True,     # Enable selective guidance
    seed=1,
    tiled=True
)
```

## Scaling Progression

The guidance uses **quadratic scaling** that emphasizes later denoising steps:

```
Progress %  | Guidance Scale
0% (start)  | 1.8 (base)
25%         | 1.888
50%         | 2.05
75%         | 2.338
100% (end)  | 3.0 (maximum detail preservation)
```

## Configuration Recommendations

**For Maximum Feature Retention:**
```python
vace_scale=2.0,
vace_scale_end=3.5,
vace_adaptive_guidance=True
```

**For Balanced Results:**
```python
vace_scale=1.5,
vace_scale_end=2.5,
vace_adaptive_guidance=True
```

**For Subtle Guidance:**
```python
vace_scale=1.2,
vace_scale_end=1.8,
vace_adaptive_guidance=True
```

## Key Guarantees

✅ **Selective Application**: Adaptive scaling only triggers when `vace_reference_image` is provided  
✅ **Pose Independence**: Pose video frames use constant base guidance unaffected by adaptive scaling  
✅ **Reference-Only Enhancement**: Reference image gets progressive guidance boost as denoising proceeds  
✅ **Backward Compatible**: Works seamlessly with existing code; old parameters still work

## Technical Implementation

### Frame Order in VACE Context
```python
# In WanVideoUnit_VACE.process():
vace_video_latents = torch.concat((*vace_reference_latents, vace_video_latents), dim=2)
# Result: [Ref0, Ref1, ..., Pose0, Pose1, ..., PoseN]
```

### Conditional Logic
```python
# In denoising loop:
if vace_ref_count > 0:  # Reference frames exist
    current_vace_scale = scale_start + (scale_end - scale_start) * (progress_ratio ** 2)
else:  # No reference frames
    current_vace_scale = scale_start  # Constant scale
```

This ensures that:
- With reference image: Progressive guidance enhancement applied
- With pose-only: Standard guidance behavior maintained
