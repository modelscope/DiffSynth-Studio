"""
Extract spatial activations for a single image.
Only hooks the specific blocks needed for the presentation figures:
  - single_9 @ t14  (depth probe)
  - single_28 @ t4   (variance probe)
Saves to probing_analysis_output/extra_spatial_{idx:04d}.pt
"""
import sys, torch, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from diffsynth.pipelines.flux_image import FluxImagePipeline, ModelConfig
from diffsynth.utils.controlnet import ControlNetInput
from diffsynth.utils.lora.flux import FluxLoRALoader
from diffsynth.core import load_state_dict
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from linear_probing import ActivationExtractor, load_spad_image

DATASET_BASE = Path("/home/jw/engsci/thesis/spad/spad_dataset")
METADATA_CSV = DATASET_BASE / "metadata_val.csv"
LORA_CKPT = ROOT / "models/train/FLUX-SPAD-LoRA-SceneAware-RAW/epoch-15.safetensors"
OUT_DIR = ROOT / "probing_analysis_output"

# Only hook the blocks we need
SINGLE_IDS = [9, 28]
JOINT_IDS = []
TIMESTEP_IDS = [0, 4, 9, 14, 19, 24, 27]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--idx", type=int, required=True, help="Image index in metadata_val.csv")
    args = p.parse_args()
    idx = args.idx

    out_path = OUT_DIR / f"extra_spatial_{idx:04d}.pt"
    if out_path.exists():
        print(f"Already exists: {out_path}")
        d = torch.load(out_path, map_location="cpu", weights_only=True)
        for k, v in d.items():
            print(f"  {k}: {v.shape}")
        return

    # Load metadata
    with open(METADATA_CSV) as f:
        samples = list(csv.DictReader(f))
    sample = samples[idx]
    spad_path = str(DATASET_BASE / sample["controlnet_image"])
    print(f"Image {idx}: {sample['image']}")
    print(f"SPAD:      {spad_path}")

    # Load pipeline
    print("Loading FLUX pipeline...")
    vc = dict(
        offload_dtype=torch.float8_e4m3fn, offload_device="cpu",
        onload_dtype=torch.float8_e4m3fn, onload_device="cpu",
        preparing_dtype=torch.float8_e4m3fn, preparing_device="cuda",
        computation_dtype=torch.bfloat16, computation_device="cuda",
    )
    vram = torch.cuda.mem_get_info()[1] / (1024**3) - 0.5
    mc = [
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev",
                    origin_file_pattern="flux1-dev.safetensors", **vc),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev",
                    origin_file_pattern="text_encoder/model.safetensors", **vc),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev",
                    origin_file_pattern="text_encoder_2/*.safetensors", **vc),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev",
                    origin_file_pattern="ae.safetensors", **vc),
        ModelConfig(model_id="InstantX/FLUX.1-dev-Controlnet-Union-alpha",
                    origin_file_pattern="diffusion_pytorch_model.safetensors"),
    ]
    pipe = FluxImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=mc, vram_limit=vram,
    )

    # Fuse LoRA
    print(f"Loading LoRA: {LORA_CKPT}")
    sd = load_state_dict(str(LORA_CKPT), torch_dtype=pipe.torch_dtype, device=pipe.device)
    FluxLoRALoader(torch_dtype=pipe.torch_dtype, device=pipe.device).fuse_lora_to_base_model(
        pipe.controlnet, sd, alpha=1.0
    )

    # Hook
    extractor = ActivationExtractor(pipe.dit, JOINT_IDS, SINGLE_IDS, TIMESTEP_IDS)

    # Load SPAD
    ctrl_img = load_spad_image(spad_path)

    # Run inference
    print("Running inference (28 steps)...")
    extractor.clear()
    pipe.scheduler.set_timesteps(28, denoising_strength=1.0)
    cn_inputs = [ControlNetInput(image=ctrl_img, processor_id="gray", scale=1.0)]

    inp_shared = {
        "cfg_scale": 1.0, "embedded_guidance": 3.5, "t5_sequence_length": 512,
        "input_image": None, "denoising_strength": 1.0,
        "height": 512, "width": 512,
        "seed": 42 + idx, "rand_device": "cpu",
        "sigma_shift": None, "num_inference_steps": 28,
        "multidiffusion_prompts": (), "multidiffusion_masks": (),
        "multidiffusion_scales": (),
        "kontext_images": None, "controlnet_inputs": cn_inputs,
        "ipadapter_images": None, "ipadapter_scale": 1.0,
        "eligen_entity_prompts": None, "eligen_entity_masks": None,
        "eligen_enable_on_negative": False, "eligen_enable_inpaint": False,
        "infinityou_id_image": None, "infinityou_guidance": 1.0,
        "flex_inpaint_image": None, "flex_inpaint_mask": None,
        "flex_control_image": None, "flex_control_strength": 0.5,
        "flex_control_stop": 0.5, "value_controller_inputs": None,
        "step1x_reference_image": None, "nexus_gen_reference_image": None,
        "lora_encoder_inputs": None, "lora_encoder_scale": 1.0,
        "tea_cache_l1_thresh": None,
        "tiled": False, "tile_size": 128, "tile_stride": 64,
        "progress_bar_cmd": lambda x: x,
    }
    inp_posi = {"prompt": ""}
    inp_nega = {"negative_prompt": ""}

    for unit in pipe.units:
        inp_shared, inp_posi, inp_nega = pipe.unit_runner(
            unit, pipe, inp_shared, inp_posi, inp_nega
        )
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {n: getattr(pipe, n) for n in pipe.in_iteration_models}

    with torch.no_grad():
        for pid, ts in enumerate(pipe.scheduler.timesteps):
            extractor.set_step(pid)
            ts_t = ts.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            np_ = pipe.cfg_guided_model_fn(
                pipe.model_fn, 1.0,
                inp_shared, inp_posi, inp_nega,
                **models, timestep=ts_t, progress_id=pid,
            )
            inp_shared["latents"] = pipe.step(
                pipe.scheduler, progress_id=pid,
                noise_pred=np_, **inp_shared,
            )

    # Save spatial features
    sf = {k: v.half() for k, v in extractor.spatial_features().items()}
    OUT_DIR.mkdir(exist_ok=True)
    torch.save(sf, out_path)
    print(f"\nSaved {len(sf)} keys to {out_path}")
    for k, v in sorted(sf.items()):
        print(f"  {k}: {v.shape}")

    extractor.remove()
    print("Done.")


if __name__ == "__main__":
    main()
