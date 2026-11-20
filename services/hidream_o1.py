from diffsynth.pipelines.hidream_o1_image import HiDreamO1ImagePipeline, ModelConfig
from diffsynth.diffusion import HiDreamO1FlashScheduler, FlowMatchScheduler
from diffsynth.core.quant import QuantizeConfig
from PIL import Image
import torch
from typing import List, Tuple
from .base import ModelService


class HiDreamO1Image(ModelService):
    def example_inputs(self):
        return [
            dict(
                prompt="medium shot, eye-level, front view. A woman is seated in an ornate bedroom, illuminated by candlelight, with a calm and composed expression. The subject is a young woman with fair skin, light brown hair styled in an updo with loose tendrils framing her face, and blue eyes. She wears a cream-colored satin robe with delicate floral embroidery and lace trim along the neckline. Her ears are adorned with pearl drop earrings. She is seated on a bed with a dark, intricately carved wooden headboard. To her left, a wooden nightstand holds three lit white candles and a candelabra with multiple lit candles in the background. The bed is covered with patterned pillows and a dark, textured blanket. The walls are paneled with dark wood and feature a large, ornate tapestry with muted earth tones. The lighting creates soft highlights on her face and robe, with warm shadows cast across the room.",
                negative_prompt=" ",
                seed=42,
                cfg_scale=4,
                num_inference_steps=50,
                height=2048,
                width=2048,
            ),
            dict(
                prompt="change her clothes to blue",
                negative_prompt=" ",
                seed=43,
                cfg_scale=4,
                num_inference_steps=50,
                height=2048,
                width=2048,
                edit_image=[Image.open("data/diffsynth_service_tests/assets/hidream_o1/image_1.png")],
            ),
        ]

    def load_models(self, device="cuda", deploy_mode=0):
        if deploy_mode in [0, 1, 2]:
            offload_config = {
                "offload_dtype": torch.bfloat16, "offload_device": "cpu",
                "onload_dtype": torch.bfloat16, "onload_device": "cpu",
                "preparing_dtype": torch.bfloat16, "preparing_device": "cpu",
                "computation_dtype": torch.bfloat16, "computation_device": device,
            }
            inside_loop_config = {} if deploy_mode in [0] else {"quantize": QuantizeConfig(method="entropack_lossy_quant_int8_raw", backend_config_kwargs={"use_convrot": True, "convrot_groupsize": 256})}
            # HiDream-O1-Image has no outside-loop models, so the offload deploy mode is applied to the DiT itself.
            if deploy_mode in [2]: inside_loop_config = {**inside_loop_config, **offload_config}
            pipe = HiDreamO1ImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=[
                    ModelConfig(model_id="HiDream-ai/HiDream-O1-Image", origin_file_pattern="model-*.safetensors", **inside_loop_config),
                ],
                processor_config=ModelConfig(model_id="HiDream-ai/HiDream-O1-Image", origin_file_pattern="./"),
            )
            if deploy_mode in [1, 2]: pipe.compile_pipeline()
        else:
            raise NotImplementedError(f"`deploy_mode` {deploy_mode} is not implemented.")
        self.pipe = pipe

    def infer(
        self,
        prompt: str,
        seed: int,
        num_inference_steps: int,
        height: int,
        width: int,
        negative_prompt: str = " ",
        cfg_scale: float = 4.0,
        loras: List[Tuple[str, float, str]] = None,
        edit_image: List[Image.Image] = None,
        model_type: str = "full",
        scheduler_type: str = "Euler",
    ) -> Image.Image:
        # Inputs
        inputs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "cfg_scale": cfg_scale,
            "num_inference_steps": num_inference_steps,
            "height": height if height % 32 == 0 else int(round(height / 32) * 32),
            "width": width if width % 32 == 0 else int(round(width / 32) * 32),
            "rand_device": "cpu",
            "model_type": model_type,
            "keep_original_aspect": False,
            "noise_scale": {"full": 8.0, "dev": 7.5}[model_type],
        }
        # Scheduler
        if scheduler_type == "Euler":
            self.pipe.scheduler = FlowMatchScheduler("HiDream-O1-Image")
        elif scheduler_type == "UniPC":
            self.pipe.scheduler = HiDreamO1FlashScheduler(noise_scale_start=7.5, noise_scale_end=7.5, noise_clip_std=2.5)
        else:
            raise ValueError(f"Unsupported `scheduler_type`: {scheduler_type}")
        # LoRAs
        self.pipe.clear_lora()
        if loras is not None:
            for lora_path, lora_scale, lora_keyword in loras:
                self.pipe.load_lora(self.pipe.dit, lora_path, alpha=lora_scale)
                if lora_keyword != "":
                    prompt = f"{lora_keyword} {prompt}"
            inputs.update({"prompt": prompt})
        # Edit image
        if edit_image is not None:
            if isinstance(edit_image, Image.Image):
                edit_image = [edit_image]
            edit_image = [i.convert("RGB") for i in edit_image]
            inputs.update({"edit_image": edit_image})
        # Inference
        image = self.pipe(**inputs)
        image = image.resize((width, height))
        # Clear loaded LoRAs
        self.pipe.clear_lora()
        return image
