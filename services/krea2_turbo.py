from diffsynth.pipelines.krea2 import Krea2Pipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
from PIL import Image
import torch
from typing import List, Tuple
from .base import ModelService


class Krea2Turbo(ModelService):
    def example_inputs(self):
        return [
            dict(
                prompt="Portrait of a woman in a blue dress, underwater, surrounded by colorful bubbles.",
                seed=0,
                height=2048,
                width=2048,
                # The following parameters are fixed for the turbo model.
                cfg_scale=1,
                num_inference_steps=8,
                mu=1.15,
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
            outside_loop_config = {} if deploy_mode in [0, 1] else offload_config
            pipe = Krea2Pipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=[
                    ModelConfig(model_id="krea/Krea-2-Turbo", origin_file_pattern="turbo.safetensors", **inside_loop_config),
                    ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern="*.safetensors", **outside_loop_config),
                    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **outside_loop_config),
                ],
                tokenizer_config=ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern=""),
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
        mu: float = None,
        loras: List[Tuple[str, float, str]] = None,
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
            "mu": mu,
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
