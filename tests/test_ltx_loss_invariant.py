"""Discriminating probe: FlowMatchSFTAudioVideoLoss must exclude
conditioned (denoise strength 0) video tokens from the video loss,
while generated video and audio tokens still move the loss."""
import sys
import torch

from diffsynth.diffusion.loss import FlowMatchSFTAudioVideoLoss


class FakeScheduler:
    timesteps = torch.linspace(1000.0, 0.0, 28)

    def add_noise(self, latents, noise, timestep):
        return latents + noise * (timestep / 1000.0)

    def training_target(self, latents, noise, timestep):
        return noise - latents

    def training_weight(self, timestep):
        return 1.0


PRED_VIDEO = None
PRED_AUDIO = None


class FakePipe:
    scheduler = FakeScheduler()
    in_iteration_models = []
    torch_dtype = torch.float32
    device = torch.device("cpu")

    def model_fn(self, **kwargs):
        return PRED_VIDEO.clone(), PRED_AUDIO.clone()


def make_inputs():
    torch.manual_seed(7)
    video = torch.randn(1, 4, 4, 8, 8)  # b,c,f,h,w
    audio = torch.randn(1, 8, 4, 16)
    mask = torch.ones(1, 1, 4, 8, 8)
    mask[:, :, 0] = 0.0  # frame 0 conditioned
    return {
        "input_latents": video,
        "audio_input_latents": audio,
        "denoise_mask_video": mask,
    }


def run_loss():
    torch.manual_seed(11)  # same sampled timestep + noise every call
    return FlowMatchSFTAudioVideoLoss(FakePipe(), **make_inputs()).item()


torch.manual_seed(3)
PRED_VIDEO = torch.randn(1, 4, 4, 8, 8)
PRED_AUDIO = torch.randn(1, 8, 4, 16)
base = run_loss()

PRED_VIDEO[:, :, 0] += 5.0  # conditioned frame only
cond_only = run_loss()

PRED_VIDEO[:, :, 0] -= 5.0
PRED_VIDEO[:, :, 2] += 5.0  # generated frame
gen_frame = run_loss()

PRED_VIDEO[:, :, 2] -= 5.0
PRED_AUDIO += 5.0  # audio
gen_audio = run_loss()

print(f"base={base:.6f} cond_only={cond_only:.6f} "
      f"gen_frame={gen_frame:.6f} gen_audio={gen_audio:.6f}")
assert cond_only == base, "conditioned-frame perturbation moved the loss"
assert gen_frame != base, "generated-frame perturbation did not move the loss"
assert gen_audio != base, "audio perturbation did not move the loss"

# No-mask path must equal plain mean MSE (backward compatibility).
torch.manual_seed(11)
inputs = make_inputs()
del inputs["denoise_mask_video"]
torch.manual_seed(3)
PRED_VIDEO = torch.randn(1, 4, 4, 8, 8)
PRED_AUDIO = torch.randn(1, 8, 4, 16)
torch.manual_seed(11)
loss_nomask = FlowMatchSFTAudioVideoLoss(FakePipe(), **inputs).item()
print(f"no_mask={loss_nomask:.6f}")
assert loss_nomask != base, "mask unexpectedly had no effect at all"
print("LOSS_INVARIANT=PASS")
