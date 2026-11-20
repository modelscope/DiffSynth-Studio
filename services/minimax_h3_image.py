from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
from PIL import Image
import torch
from typing import List, Tuple
from .base import ModelService


class MiniMaxH3Image(ModelService):
    def example_inputs(self):
        prompt = (
            "An overhead photograph of a rustic farm-to-table spread on a weathered wooden table: a round loaf of sourdough "
            "on burlap, wedges of cheese on a dark board, a copper bowl of vegetables, vine tomatoes, olives and walnuts, "
            "warm afternoon light raking across the grain of the wood."
        )
        return [
            dict(
                prompt=prompt,
                seed=42,
                height=4096,
                width=4096,
                num_inference_steps=6,
                cfg_scale=1,
            ),
            dict(
                prompt="Change the color of the tomato in <Picture 1> to green.",
                seed=43,
                height=4096,
                width=4096,
                num_inference_steps=6,
                cfg_scale=1,
                edit_image=[Image.open("data/diffsynth_service_tests/assets/minimax_h3_image/image_1.png")]
            ),
        ]

    def load_models(self, device="cuda", deploy_mode=0):
        if deploy_mode in [0]:
            # This model is too large, only one `deploy_mode` is provided.
            pipe = MiniMaxH3ImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=[
                    ModelConfig(
                        model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack",
                        origin_file_pattern="models/dit_8bit_int8.safetensors",
                        origin_config_file="models/dit_8bit_int8.json",
                    ),
                    ModelConfig(
                        model_id="DiffSynth-Studio/EntroPackPreQuants",
                        origin_file_pattern="models/MiniMax-H3/text_encoder_int8_convrot.safetensors",
                        origin_config_file="models/MiniMax-H3/text_encoder_int8_convrot.json",
                    ),
                    ModelConfig(
                        model_id="MiniMax/MiniMax-H3-Image",
                        origin_file_pattern="pdd_fused_heads.safetensors",
                    ),
                    ModelConfig(
                        model_id="DiffSynth-Studio/MiniMax-H3-Image-EntroPack",
                        origin_file_pattern="models/vae_int8.safetensors",
                    ),
                ],
                processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3-Image", origin_file_pattern="processor/"),
            )
            pipe.compile_pipeline()
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
        edit_image: List[Image.Image] = None,
        loras: List[Tuple[str, float, str]] = None,
        negative_prompt: str = None,
        cfg_scale: float = 1,
    ) -> Image.Image:
        # Inputs
        inputs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "rand_device": "cpu",
            "num_inference_steps": num_inference_steps,
            "height": height if height % 32 == 0 else int(round(height / 32) * 32),
            "width": width if width % 32 == 0 else int(round(width / 32) * 32),
            "edit_image": edit_image,
            "distilled": True,
        }
        # LoRAs
        self.pipe.clear_lora()
        if loras is not None:
            for lora_path, lora_scale, lora_keyword in loras:
                self.pipe.load_lora(self.pipe.dit, lora_path, alpha=lora_scale)
                if lora_keyword != "":
                    prompt = f"{lora_keyword} {prompt}"
                inputs.update({"prompt": prompt})
        # Inference
        image = self.pipe(**inputs)
        image = image.resize((width, height))
        # Clear loaded LoRAs
        self.pipe.clear_lora()
        return image
