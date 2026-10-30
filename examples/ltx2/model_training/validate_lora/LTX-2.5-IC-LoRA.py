import argparse
import faulthandler
import json
import os

import numpy as np
import torch
from PIL import Image

from diffsynth.core.device.vram_guard import VRAMGuard
from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig
from diffsynth.utils.data import VideoData
from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2


def read_adapter_factor(lora_path):
    from safetensors import safe_open

    with safe_open(lora_path, framework="pt") as f:
        factor = f.metadata().get("reference_downscale_factor")
    return int(factor) if factor is not None else None


def all_finite(value):
    if torch.is_tensor(value):
        return value.is_floating_point() is False or bool(torch.isfinite(value).all())
    if isinstance(value, (list, tuple)):
        return all(all_finite(v) for v in value)
    if isinstance(value, dict):
        return all(all_finite(v) for v in value.values())
    return True


class FiniteTrace:
    # Ordered finite/non-finite log over pipeline stages. The first
    # non-finite event names the stage that produced the NaN.
    def __init__(self):
        self.events = []
        self.dit_calls = 0

    def record(self, stage, ok):
        self.events.append((stage, ok))
        print(f"FINITE {stage} {'ok' if ok else 'NONFINITE'}", flush=True)

    def first_nonfinite(self):
        return next((stage for stage, ok in self.events if not ok), None)

    def wrap(self, obj, method, stage):
        # Indexed per call: vae_encode[1] is the first frame, [2] the reference.
        original = getattr(obj, method)
        calls = [0]
        def checked(*args, **kwargs):
            out = original(*args, **kwargs)
            calls[0] += 1
            self.record(f"{stage}[{calls[0]}]", all_finite(out))
            return out
        setattr(obj, method, checked)

    def attach_text_encoder(self, text_encoder, post_modules):
        # Per prompt: which Gemma hidden-state layers are non-finite, split
        # into valid and padded token positions (left padding).
        def hook(_module, _args, kwargs, output):
            mask = kwargs["attention_mask"].bool()[0]
            bad = []
            for i, h in enumerate(output.hidden_states):
                finite = torch.isfinite(h[0]).all(dim=-1)
                if not finite.all():
                    bad.append(f"L{i}:valid={int((~finite[mask]).sum())},pad={int((~finite[~mask]).sum())}")
            self.record(f"text_encoder[valid_tokens={int(mask.sum())}]{'' if not bad else ' ' + ' '.join(bad[:6])}", not bad)
        text_encoder.register_forward_hook(hook, with_kwargs=True)
        self.wrap(post_modules, "process_hidden_states", "text_post_modules")

    def attach_dit(self, dit):
        def pre_hook(_module, _args, kwargs):
            self.dit_calls += 1
            names = ("video_latents", "video_positions", "video_timesteps", "video_context", "audio_latents", "audio_context")
            bad = [k for k in names if not all_finite(kwargs.get(k))]
            self.record(f"dit_input[{self.dit_calls}]{'' if not bad else ' ' + ','.join(bad)}", not bad)
        def hook(_module, _args, _kwargs, output):
            self.record(f"dit_output[{self.dit_calls}]", all_finite(output))
        dit.register_forward_pre_hook(pre_hook, with_kwargs=True)
        dit.register_forward_hook(hook, with_kwargs=True)


def lora_fold_finite(dit):
    tensors = []
    for module in dit.modules():
        tensors += list(getattr(module, "lora_A_weights", [])) + list(getattr(module, "lora_B_weights", []))
    return len(tensors), all_finite(tensors)


def main():
    parser = argparse.ArgumentParser(description="Validate an LTX-2.5 IC-LoRA adapter with one I2AV inference.")
    parser.add_argument("--lora_path", type=str, default=None, help="Adapter to load. Omit to validate the base model.")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="Strength passed to pipe.load_lora.")
    parser.add_argument("--ref_video", type=str, default="data/example_video_dataset/ltx2/depth_video.mp4")
    parser.add_argument("--first_frame", type=str, default="data/example_video_dataset/ltx2/first_frame.png")
    parser.add_argument("--prompt", type=str, default="Two cute orange cats, wearing boxing gloves, stand in a boxing ring and fight each other. They are punching each other fast and yelling: 'I will win!'")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--frame_rate", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--ref_scale_factor", type=int, default=None, help="Defaults to the adapter's reference_downscale_factor metadata.")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--vram_limit", type=float, required=True, help="Residency threshold in GB of device-wide used VRAM (layers stay on GPU below it). Not a cap.")
    parser.add_argument("--abort_vram_gb", type=float, default=29.5, help="Hard line on card sysfs mem_info_vram_used: the process is capped below it and killed if it is crossed.")
    parser.add_argument("--offload", choices=["cpu", "disk"], default="cpu", help="Where frozen weights live between uses.")
    parser.add_argument("--tiled", dest="tiled", action="store_true", default=True)
    parser.add_argument("--no_tiled", dest="tiled", action="store_false", help="Disable tiled VAE decode (needed for tiny latents where tiles degenerate).")
    parser.add_argument("--output_path", type=str, default="ltx2.5_ic_lora_i2av.mp4")
    parser.add_argument("--frames_npy", type=str, default=None, help="Also save decoded RGB frames (uint8) for pixel comparisons.")
    args = parser.parse_args()

    # A GPU memory fault aborts the process from C++; dump the Python stack
    # so the launching line is named (exact with AMD_SERIALIZE_KERNEL=3).
    faulthandler.enable()
    guard = VRAMGuard(args.abort_vram_gb)
    trace = FiniteTrace()
    store = "disk" if args.offload == "disk" else torch.bfloat16
    vram_config = {
        "offload_dtype": store,
        "offload_device": args.offload,
        "onload_dtype": store,
        "onload_device": args.offload,
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    pipe = LTX2AudioVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        tokenizer_config=ModelConfig(path="./models/DiffSynth-Studio/LTX-2.5-Repackage/tokenizer"),
        model_configs=[
            ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.5-Repackage", origin_file_pattern="text_encoder_post_modules.safetensors", **vram_config),
            ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors", **vram_config),
            ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-video-vae-bf16.safetensors", **vram_config),
            ModelConfig(model_id="Lightricks/LTX-2.5", origin_file_pattern="vae/ltx-2.5-audio-vae-bf16.safetensors", **vram_config),
        ],
        vram_limit=args.vram_limit,
    )
    guard.phase("loaded")

    ref_scale_factor = args.ref_scale_factor
    if args.lora_path is not None:
        pipe.load_lora(pipe.dit, args.lora_path, alpha=args.lora_alpha)
        count, ok = lora_fold_finite(pipe.dit)
        trace.record(f"lora_fold[alpha={args.lora_alpha},tensors={count}]", ok)
        if ref_scale_factor is None:
            ref_scale_factor = read_adapter_factor(args.lora_path)
    if ref_scale_factor is None:
        ref_scale_factor = 2
    print(f"reference_downscale_factor={ref_scale_factor} lora={args.lora_path} alpha={args.lora_alpha}")

    trace.wrap(pipe.video_vae_encoder, "encode", "vae_encode")
    trace.wrap(pipe.video_vae_decoder, "decode", "vae_decode")
    trace.attach_dit(pipe.dit)
    trace.attach_text_encoder(pipe.text_encoder, pipe.text_encoder_post_modules)

    first_frame = Image.open(args.first_frame).convert("RGB").resize((args.width, args.height))
    ref_video = VideoData(args.ref_video, height=args.height // ref_scale_factor, width=args.width // ref_scale_factor)
    ref_video = ref_video.raw_data()
    video, audio = pipe(
        prompt=args.prompt,
        negative_prompt=pipe.default_negative_prompt["LTX-2.5"],
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        tiled=args.tiled,
        input_images=[first_frame],
        input_images_indexes=[0],
        input_images_strength=1.0,
        in_context_videos=[ref_video],
        in_context_downsample_factor=ref_scale_factor,
    )
    guard.phase("denoised_and_decoded")

    frames = np.stack([np.asarray(f.convert("RGB")) for f in video])
    trace.record("audio_decode", all_finite(audio))
    first_bad = trace.first_nonfinite()
    print(f"FIRST_NONFINITE_STAGE={first_bad or 'NONE'} dit_calls={trace.dit_calls}")
    stats = {
        "frames": int(frames.shape[0]), "rgb_mean": round(float(frames.mean()), 2), "rgb_max": int(frames.max()),
        "rgb_std": round(float(frames.std()), 2),
        "interframe_mae": round(float(np.abs(frames[1:].astype(np.float64) - frames[0]).mean()), 3),
    }
    print("FRAME_STATS " + json.dumps(stats))
    if args.frames_npy:
        np.save(args.frames_npy, frames)
    healthy = guard.report()
    guard.close()

    # Never write or pass a NaN collapse: non-finite anywhere, or the black
    # frames a NaN decode turns into, fails before any file is produced.
    if first_bad is not None:
        raise SystemExit(f"IC_LORA_I2AV_INFERENCE=FAIL non-finite at {first_bad}")
    if frames.max() <= 2 or frames.std() < 1.0:
        raise SystemExit(f"IC_LORA_I2AV_INFERENCE=FAIL black or flat output {stats}")
    if not healthy:
        raise SystemExit("IC_LORA_I2AV_INFERENCE=FAIL amdgpu faults during run")
    write_video_audio_ltx2(
        video=video,
        audio=audio,
        output_path=args.output_path,
        fps=args.frame_rate,
        audio_sample_rate=pipe.audio_vocoder.output_sampling_rate,
    )
    print(f"IC_LORA_I2AV_INFERENCE=PASS output={os.path.abspath(args.output_path)} fps={args.frame_rate} "
          f"frames={len(video)} decode_finite=True rgb_max={stats['rgb_max']} rgb_std={stats['rgb_std']}")


if __name__ == "__main__":
    main()
