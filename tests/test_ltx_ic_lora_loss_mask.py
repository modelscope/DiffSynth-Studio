"""CPU contract: IC-LoRA reference tokens condition the DiT but never enter the loss.

The DiT boundary is captured (no weights, no GPU). Native noise, first-frame
conditioning, in-context reference positions (as built by
LTX2AudioVideoUnit_InContextVideoEmbedder), model_fn_ltx2 concatenation and
FlowMatchSFTAudioVideoLoss run unchanged.
"""
import torch
from diffsynth.diffusion.loss import FlowMatchSFTAudioVideoLoss
from diffsynth.models.ltx2_common import VideoLatentShape, get_pixel_coords, VIDEO_SCALE_FACTORS
from diffsynth.pipelines.ltx2_audio_video import (
    LTX2AudioVideoPipeline, LTX2AudioVideoUnit_NoiseInitializer,
    LTX2AudioVideoUnit_InputImagesEmbedder,
)

pipe = LTX2AudioVideoPipeline(device="cpu", torch_dtype=torch.float32)
pipe.is_ltx25 = True
factor, frame_rate = 2, 24
noise = LTX2AudioVideoUnit_NoiseInitializer().process_stage(
    pipe, height=64, width=64, num_frames=17, seed=1, rand_device="cpu", frame_rate=frame_rate
)
video, audio = noise["video_noise"], noise["audio_noise"]
torch.manual_seed(0)
target = torch.randn_like(video)
condition, mask = LTX2AudioVideoUnit_InputImagesEmbedder().apply_input_images_to_latents(
    video, [target[:, :, :1].clone()], [0], input_strength=1.0
)

# Reference latents at 1/factor resolution with positions scaled exactly as
# the in-context unit does, so they land on the target's pixel grid.
ref = torch.randn(1, video.shape[1], video.shape[2], video.shape[3] // factor, video.shape[4] // factor)
coords = pipe.video_patchifier.get_patch_grid_bounds(output_shape=VideoLatentShape.from_torch_shape(ref.shape), device="cpu")
ref_positions = get_pixel_coords(coords, VIDEO_SCALE_FACTORS, True).float()
ref_positions[:, 0, ...] = ref_positions[:, 0, ...] / frame_rate
ref_positions[:, 1, ...] *= factor
ref_positions[:, 2, ...] *= factor
assert ref_positions[:, 1].max() <= noise["video_positions"][:, 1].max()
assert ref_positions[:, 2].max() <= noise["video_positions"][:, 2].max()

n_target = pipe.video_patchifier.patchify(video).shape[1]
n_ref = pipe.video_patchifier.patchify(ref).shape[1]


class CaptureDiT(torch.nn.Module):
    # Predictions over the full concatenated sequence (target + reference),
    # as the real DiT returns them.
    def __init__(self):
        super().__init__()
        self.video_pred = torch.nn.Parameter(torch.zeros(1, n_target + n_ref, video.shape[1]))
        self.audio_pred = torch.nn.Parameter(torch.zeros_like(pipe.audio_patchifier.patchify(audio)))
        self.last = None

    def forward(self, **kwargs):
        self.last = kwargs
        assert kwargs["video_latents"].shape[1] == n_target + n_ref
        return self.video_pred, self.audio_pred


pipe.dit = CaptureDiT()
pipe.scheduler.set_timesteps(10, training=True)
inputs = dict(noise, input_latents=target, audio_input_latents=audio,
              input_latents_video=condition, denoise_mask_video=mask,
              in_context_video_latents=ref, in_context_video_positions=ref_positions,
              video_patchifier=pipe.video_patchifier, audio_patchifier=pipe.audio_patchifier)


def loss(**override):
    torch.manual_seed(17)
    return FlowMatchSFTAudioVideoLoss(pipe, **{**inputs, **override})


base = loss()
seen = pipe.dit.last
ref_slice = slice(n_target, n_target + n_ref)
# Reference tokens enter the DiT clean, at timestep 0, at the scaled positions.
assert torch.equal(seen["video_latents"][:, ref_slice], pipe.video_patchifier.patchify(ref))
assert (seen["video_timesteps"][:, ref_slice] == 0).all()
assert torch.equal(seen["video_positions"][:, :, ref_slice], ref_positions)
generated = pipe.video_patchifier.patchify(mask).squeeze(-1)[0] > 0
assert (seen["video_timesteps"][0, :n_target][generated] > 0).all()

# 1. Perturb only reference tokens: their predictions and their latents.
with torch.no_grad():
    pipe.dit.video_pred[:, ref_slice] += 7
ref_pred_loss = loss().item()
with torch.no_grad():
    pipe.dit.video_pred[:, ref_slice] -= 7
ref_latent_loss = loss(in_context_video_latents=ref + 7).item()
assert torch.equal(pipe.dit.last["video_latents"][:, ref_slice], pipe.video_patchifier.patchify(ref + 7))
assert ref_pred_loss == base.item() and ref_latent_loss == base.item()
loss().backward()
assert (pipe.dit.video_pred.grad[:, ref_slice] == 0).all()
print(f"REF_PERTURB_LOSS_IDENTICAL=PASS base={base.item():.6f} ref_pred={ref_pred_loss:.6f} ref_latent={ref_latent_loss:.6f} ref_grad_abs_sum=0")

# 2. Perturb target tokens (generated, not the clean first frame): loss moves.
pipe.dit.video_pred.grad = None
target_idx = torch.nonzero(generated).squeeze(-1)
with torch.no_grad():
    pipe.dit.video_pred[:, target_idx] += 7
target_pred_loss = loss().item()
with torch.no_grad():
    pipe.dit.video_pred[:, target_idx] -= 7
target_latent_loss = loss(input_latents=target + 7).item()
assert target_pred_loss != base.item() and target_latent_loss != base.item()
loss().backward()
grad = pipe.dit.video_pred.grad[0]
assert (grad[target_idx] != 0).any()
assert (grad[:n_target][~generated] == 0).all()
print(f"TARGET_PERTURB_LOSS_CHANGES=PASS base={base.item():.6f} target_pred={target_pred_loss:.6f} target_latent={target_latent_loss:.6f} "
      f"target_grad_abs_sum={grad[target_idx].abs().sum().item():.6f}")
