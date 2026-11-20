from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
from PIL import Image
import torch
from typing import List, Tuple
from .base import ModelService


class ZImageTurbo(ModelService):
    def example_inputs(self):
        return [
            dict(
                prompt='Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.',
                seed=42,
                cfg_scale=1,
                num_inference_steps=8,
                height=1024,
                width=1024,
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
            pipe = ZImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=[
                    ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/*.safetensors", **inside_loop_config),
                    ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors", **outside_loop_config),
                    ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **outside_loop_config),
                ],
                tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
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
        # Inference
        image = self.pipe(**inputs)
        image = image.resize((width, height))
        # Clear loaded LoRAs
        self.pipe.clear_lora()
        return image


class ZImage(ZImageTurbo):
    def example_inputs(self):
        return [
            dict(
                prompt='Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.',
                seed=42,
                cfg_scale=4,
                num_inference_steps=50,
                height=1024,
                width=1024,
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
            inside_loop_config = {} if deploy_mode in [0] else {"quantize": QuantizeConfig(
                method="entropack_lossy_quant_int8_raw",
                exclude_modules=["all_final_layer.2-1.linear"],
                backend_config_kwargs={"use_convrot": True, "convrot_groupsize": 256},
            )}
            outside_loop_config = {} if deploy_mode in [0, 1] else offload_config
            pipe = ZImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=[
                    ModelConfig(model_id="Tongyi-MAI/Z-Image", origin_file_pattern="transformer/*.safetensors", **inside_loop_config),
                    ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors", **outside_loop_config),
                    ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **outside_loop_config),
                ],
                tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
            )
            if deploy_mode in [1, 2]: pipe.compile_pipeline()
        else:
            raise NotImplementedError(f"`deploy_mode` {deploy_mode} is not implemented.")
        self.pipe = pipe
