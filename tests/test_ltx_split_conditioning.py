"""CPU contract: native split cache -> training transfer -> model_fn -> AV loss.

No weights or GPU required; the DiT boundary is captured rather than simulated
conditioning. Native patchifiers, conditioning, split filtering and loss run.
"""
import importlib.util
import io
from pathlib import Path
import torch
from diffsynth.diffusion.loss import FlowMatchSFTAudioVideoLoss
from diffsynth.pipelines.ltx2_audio_video import (
    LTX2AudioVideoPipeline, LTX2AudioVideoUnit_NoiseInitializer,
    LTX2AudioVideoUnit_InputImagesEmbedder,
)

spec = importlib.util.spec_from_file_location(
    "ltx_training", Path(__file__).resolve().parents[1] / "examples/ltx2/model_training/train.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
trainer = module.LTX2TrainingModule.__new__(module.LTX2TrainingModule)
torch.nn.Module.__init__(trainer)
pipe = LTX2AudioVideoPipeline(device="cpu", torch_dtype=torch.bfloat16)
pipe.is_ltx25 = True
trainer.pipe = pipe
noise = LTX2AudioVideoUnit_NoiseInitializer().process_stage(
    pipe, height=64, width=64, num_frames=17, seed=1, rand_device="cpu", frame_rate=24
)
video = noise["video_noise"]
audio = noise["audio_noise"]
clean_frame = torch.full_like(video[:, :, :1], 0.375)
condition, mask = LTX2AudioVideoUnit_InputImagesEmbedder().apply_input_images_to_latents(
    video, [clean_frame], [0], input_strength=1.0
)
shared = dict(noise, input_latents=video, audio_input_latents=audio,
              input_latents_video=condition, denoise_mask_video=mask,
              video_patchifier=pipe.video_patchifier, audio_patchifier=pipe.audio_patchifier,
              video_latents=video, audio_latents=audio, obsolete_field="discard")
trainer.split_pipeline_units(
    "sft:data_process", pipe, lora_base_model="dit", remove_unnecessary_params=True,
    force_remove_params_shared=("audio_latents", "video_latents"),
    force_remove_params_nega=("audio_context", "video_context"),
)
# Use the exact final native cache filter produced by the training split.
cached = pipe.units[-1].process(pipe, shared, {}, {})
assert "denoise_mask_video" in cached[0]
assert "input_latents_video" in cached[0]
assert "video_keyframes_mask" in cached[0]
assert "video_latents" not in cached[0] and "obsolete_field" not in cached[0]
stream = io.BytesIO()
torch.save(cached, stream)
stream.seek(0)
cached = torch.load(stream, map_location="cpu", weights_only=False)
transferred = trainer.transfer_data_to_device(cached, "cpu", torch.bfloat16)[0]
for key in ("video_positions", "audio_positions"):
    assert transferred[key].dtype == torch.float32, (key, transferred[key].dtype)
    assert torch.equal(transferred[key], noise[key]), key
assert not torch.equal(noise["video_positions"], noise["video_positions"].bfloat16().float())

class CaptureDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.video_pred = torch.nn.Parameter(torch.zeros_like(pipe.video_patchifier.patchify(video)))
        self.audio_pred = torch.nn.Parameter(torch.zeros_like(pipe.audio_patchifier.patchify(audio)))
        self.last = None

    def forward(self, **kwargs):
        self.last = kwargs
        return self.video_pred, self.audio_pred

pipe.dit = CaptureDiT()
pipe.scheduler.set_timesteps(10, training=True)
def loss():
    torch.manual_seed(17)
    return FlowMatchSFTAudioVideoLoss(pipe, **transferred)

base = loss()
seen = pipe.dit.last
first = pipe.video_patchifier.patchify(mask).squeeze(-1) == 0
assert torch.equal(seen["video_latents"][first], pipe.video_patchifier.patchify(condition)[first])
assert (seen["video_timesteps"][first] == 0).all()
assert (seen["video_timesteps"][~first] > 0).all()
assert (seen["audio_timesteps"] > 0).all()
assert seen["video_timesteps"].dtype == torch.float32
assert seen["audio_timesteps"].dtype == torch.float32
assert (seen["video_keyframes_mask"].squeeze(-1)[first] == 1).all()
assert (seen["video_keyframes_mask"].squeeze(-1)[~first] == 0).all()
with torch.no_grad():
    pipe.dit.video_pred[first] += 5
assert loss().item() == base.item()
with torch.no_grad():
    pipe.dit.video_pred[first] -= 5
    pipe.dit.video_pred[~first] += 5
generated = loss().item()
assert generated != base.item()
with torch.no_grad():
    pipe.dit.video_pred[~first] -= 5
    pipe.dit.audio_pred += 5
audio_changed = loss().item()
assert audio_changed != base.item()
loss().backward()
assert (pipe.dit.video_pred.grad[first] == 0).all()
assert (pipe.dit.video_pred.grad[~first] != 0).any()
assert (pipe.dit.audio_pred.grad != 0).any()
print(f"base={base.item():.6f} generated={generated:.6f} audio={audio_changed:.6f}")
print("LTX25_SPLIT_CACHE_CONDITIONING=PASS")
print("LTX25_FP32_COORDINATES=PASS")
print("LTX25_FRAME0_LOSS_INVARIANT=PASS (CPU contract; not 22B runtime)")
