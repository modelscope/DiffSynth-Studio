from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
from diffsynth.core.data.operators import ImageCropAndResize
from PIL import Image
import torch
from typing import List, Tuple
from .base import ModelService


class Flux2KleinService(ModelService):
    dit_model_id = "black-forest-labs/FLUX.2-klein-4B"
    reference_model_id = "black-forest-labs/FLUX.2-klein-4B"

    def load_models(self, device="cuda", deploy_mode=0):
        if deploy_mode in [0, 1, 2]:
            offload_config = {
                "offload_dtype": torch.bfloat16, "offload_device": "cpu",
                "onload_dtype": torch.bfloat16, "onload_device": "cpu",
                "preparing_dtype": torch.bfloat16, "preparing_device": "cpu",
                "computation_dtype": torch.bfloat16, "computation_device": device,
            }
            inside_loop_config = {} if deploy_mode in [0] else {"quantize": QuantizeConfig(method="entropack_lossy_quant_int8_raw", backend_config_kwargs={"use_convrot": True, "convrot_groupsize": 256})}
            outside_loop_config = {} if deploy_mode in [0, 1] else offload_config
            pipe = Flux2ImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=[
                    ModelConfig(model_id=self.dit_model_id, origin_file_pattern="transformer/*.safetensors", **inside_loop_config),
                    ModelConfig(model_id=self.reference_model_id, origin_file_pattern="text_encoder/*.safetensors", **outside_loop_config),
                    ModelConfig(model_id=self.reference_model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **outside_loop_config),
                ],
                tokenizer_config=ModelConfig(model_id=self.reference_model_id, origin_file_pattern="tokenizer/"),
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
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        loras: List[Tuple[str, float, str]] = None,
        edit_image: List[Image.Image] = None,
    ) -> Image.Image:
        # Inputs
        inputs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "cfg_scale": cfg_scale,
            "num_inference_steps": num_inference_steps,
            "height": height if height % 16 == 0 else int(round(height / 16) * 16),
            "width": width if width % 16 == 0 else int(round(width / 16) * 16),
            "rand_device": "cpu",
        }
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
            processor = ImageCropAndResize(max_pixels=2048 * 2048, height_division_factor=16, width_division_factor=16)
            edit_image = [processor(i.convert("RGB")) for i in edit_image]
            inputs.update({"edit_image": edit_image})
        # Inference
        image = self.pipe(**inputs)
        image = image.resize((width, height))
        # Clear loaded LoRAs
        self.pipe.clear_lora()
        return image


class Flux2Klein4B(Flux2KleinService):
    dit_model_id = "black-forest-labs/FLUX.2-klein-4B"
    reference_model_id = "black-forest-labs/FLUX.2-klein-4B"

    def example_inputs(self):
        return [
            dict(
                prompt="Masterpiece, best quality. Anime-style portrait of a woman in a blue dress, underwater, surrounded by colorful bubbles.",
                negative_prompt="",
                seed=0,
                cfg_scale=1,
                num_inference_steps=4,
                height=1024,
                width=1024,
            ),
            dict(
                prompt="change the color of the clothes to red",
                negative_prompt="",
                seed=1,
                cfg_scale=1,
                num_inference_steps=4,
                height=1024,
                width=1024,
                edit_image=[Image.open("data/diffsynth_service_tests/assets/flux2/image_1.png")],
            ),
        ]


class Flux2KleinBase4B(Flux2KleinService):
    dit_model_id = "black-forest-labs/FLUX.2-klein-base-4B"
    reference_model_id = "black-forest-labs/FLUX.2-klein-4B"

    def example_inputs(self):
        return [
            dict(
                prompt="Masterpiece, best quality. Anime-style portrait of a woman in a blue dress, underwater, surrounded by colorful bubbles.",
                negative_prompt="",
                seed=0,
                cfg_scale=4,
                num_inference_steps=50,
                height=1024,
                width=1024,
            ),
            dict(
                prompt="change the color of the clothes to red",
                negative_prompt="",
                seed=1,
                cfg_scale=4,
                num_inference_steps=50,
                height=1024,
                width=1024,
                edit_image=[Image.open("data/diffsynth_service_tests/assets/flux2/image_1.png")],
            ),
        ]


class Flux2Klein9B(Flux2KleinService):
    dit_model_id = "black-forest-labs/FLUX.2-klein-9B"
    reference_model_id = "black-forest-labs/FLUX.2-klein-9B"

    def example_inputs(self):
        return [
            dict(
                prompt="Masterpiece, best quality. Anime-style portrait of a woman in a blue dress, underwater, surrounded by colorful bubbles.",
                negative_prompt="",
                seed=0,
                cfg_scale=1,
                num_inference_steps=4,
                height=1024,
                width=1024,
            ),
            dict(
                prompt="change the color of the clothes to red",
                negative_prompt="",
                seed=1,
                cfg_scale=1,
                num_inference_steps=4,
                height=1024,
                width=1024,
                edit_image=[Image.open("data/diffsynth_service_tests/assets/flux2/image_1.png")],
            ),
        ]


class Flux2KleinBase9B(Flux2KleinService):
    dit_model_id = "black-forest-labs/FLUX.2-klein-base-9B"
    reference_model_id = "black-forest-labs/FLUX.2-klein-9B"

    def example_inputs(self):
        return [
            dict(
                prompt="Masterpiece, best quality. Anime-style portrait of a woman in a blue dress, underwater, surrounded by colorful bubbles.",
                negative_prompt="",
                seed=0,
                cfg_scale=4,
                num_inference_steps=50,
                height=1024,
                width=1024,
            ),
            dict(
                prompt="change the color of the clothes to red",
                negative_prompt="",
                seed=1,
                cfg_scale=4,
                num_inference_steps=50,
                height=1024,
                width=1024,
                edit_image=[Image.open("data/diffsynth_service_tests/assets/flux2/image_1.png")],
            ),
        ]
