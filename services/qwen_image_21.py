from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
from PIL import Image
import torch
from typing import List, Tuple
from .base import ModelService


class QwenImage21(ModelService):
    def example_inputs(self):
        return [
            dict(
                prompt="Flat anime-style illustration, a girl with long black hair, wearing a JK uniform.",
                seed=0,
                num_inference_steps=40,
                height=1024,
                width=1024,
            ),
            dict(
                prompt="Flat anime-style illustration, a sunny and cheerful high school girl.",
                seed=0,
                num_inference_steps=40,
                height=1024,
                width=1024,
            ),
            dict(
                prompt="Generate a group photo of these two characters.",
                seed=1,
                num_inference_steps=40,
                height=1024,
                width=1024,
                edit_image=[Image.open("data/diffsynth_service_tests/assets/qwen_image_21/image_1.png"),
                            Image.open("data/diffsynth_service_tests/assets/qwen_image_21/image_2.png")],
            ),
        ]


    def load_models(self, device="cuda", deploy_mode=0):
        if deploy_mode in [0, 1, 2]:
            offload_config = {
                "offload_dtype": torch.bfloat16, "offload_device": "cpu",
                "onload_dtype": torch.bfloat16, "onload_device": "cpu",
                "preparing_dtype": torch.bfloat16, "preparing_device": "cpu",
                "computation_dtype": torch.bfloat16, "computation_device": "cuda",
            }
            inside_loop_config = {} if deploy_mode in [0] else {"quantize": QuantizeConfig(method="entropack_lossy_quant_int8_raw", backend_config_kwargs={"use_convrot": True, "convrot_groupsize": 256})}
            outside_loop_config = {} if deploy_mode in [0, 1] else offload_config
            pipe = QwenImage21Pipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=[
                    ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="transformer/*.safetensors", **inside_loop_config),
                    ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="text_encoder/*.safetensors", **outside_loop_config),
                    ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="vae/*.safetensors", **outside_loop_config),
                ],
                processor_config=ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="processor/"),
            )
            if deploy_mode in [1, 2]: pipe.compile_pipeline()
        else:
            raise NotImplementedError("`deploy_mode` {deploy_mode} is not implemented.")
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
        is_turbo: bool = False,
    ) -> Image.Image:
        # Inputs
        if negative_prompt is None or negative_prompt == "": negative_prompt = " "
        inputs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "cfg_scale": cfg_scale if not is_turbo else 1,
            "sigmas": [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568] if is_turbo else None,
            "seed": seed,
            "rand_device": "cpu",
            "num_inference_steps": num_inference_steps,
            "height": height if height % 32 == 0 else int(round(height / 32) * 32),
            "width": width if width % 32 == 0 else int(round(width / 32) * 32),
            "edit_image": edit_image,
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


class QwenImage21Turbo(QwenImage21):
    def example_inputs(self):
        return [
            dict(
                prompt="Flat anime-style illustration, a girl with long black hair, wearing a JK uniform.",
                seed=0,
                num_inference_steps=8,
                height=1024,
                width=1024,
                is_turbo=True,
            ),
            dict(
                prompt="Flat anime-style illustration, a sunny and cheerful high school girl.",
                seed=0,
                num_inference_steps=8,
                height=1024,
                width=1024,
                is_turbo=True,
            ),
            dict(
                prompt="Generate a group photo of these two characters.",
                seed=1,
                num_inference_steps=8,
                height=1024,
                width=1024,
                edit_image=[Image.open("data/diffsynth_service_tests/assets/qwen_image_21/image_1.png"),
                            Image.open("data/diffsynth_service_tests/assets/qwen_image_21/image_2.png")],
                is_turbo=True,
            ),
        ]


    def load_models(self, device="cuda", deploy_mode=0):
        if deploy_mode in [0, 1, 2]:
            offload_config = {
                "offload_dtype": torch.bfloat16, "offload_device": "cpu",
                "onload_dtype": torch.bfloat16, "onload_device": "cpu",
                "preparing_dtype": torch.bfloat16, "preparing_device": "cpu",
                "computation_dtype": torch.bfloat16, "computation_device": "cuda",
            }
            inside_loop_config = {} if deploy_mode in [0] else {"quantize": QuantizeConfig(method="entropack_lossy_quant_int8_raw", backend_config_kwargs={"use_convrot": True, "convrot_groupsize": 256})}
            outside_loop_config = {} if deploy_mode in [0, 1] else offload_config
            pipe = QwenImage21Pipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=[
                    ModelConfig(model_id="Qwen/Qwen-Image-2.1-Turbo", origin_file_pattern="transformer/*.safetensors", **inside_loop_config),
                    ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="text_encoder/*.safetensors", **outside_loop_config),
                    ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="vae/*.safetensors", **outside_loop_config),
                ],
                processor_config=ModelConfig(model_id="Qwen/Qwen-Image-2.1", origin_file_pattern="processor/"),
            )
            if deploy_mode in [1, 2]: pipe.compile_pipeline()
        else:
            raise NotImplementedError("`deploy_mode` {deploy_mode} is not implemented.")
        self.pipe = pipe
