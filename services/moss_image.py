from diffsynth.pipelines.moss_image import MOSSImagePipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
from PIL import Image
import torch
from typing import List, Tuple
from .base import ModelService


class MOSSImageSFT(ModelService):
    def example_inputs(self):
        return [
            dict(
                prompt='A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in Chinese:\n"记得买牛奶和鸡蛋！\n牛奶两盒，鸡蛋一打。\n顺便带一袋面包。"',
                seed=42,
                num_inference_steps=50,
                height=1024,
                width=1024,
                cfg_scale=3.0,
            ),
        ]

    def load_models(self, device="cuda", deploy_mode=0):
        if deploy_mode in [0, 1, 2]:
            offload_config = {
                "offload_dtype": torch.bfloat16, "offload_device": "cpu",
                "onload_dtype": torch.bfloat16, "onload_device": "cpu",
                "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
                "computation_dtype": torch.bfloat16, "computation_device": "cuda",
            }
            inside_loop_config = {} if deploy_mode in [0] else {"quantize": QuantizeConfig(method="entropack_lossy_quant_int8_raw")}
            outside_loop_config = {} if deploy_mode in [0, 1] else offload_config
            vae_config = {**outside_loop_config, "computation_dtype": torch.float32}
            pipe = MOSSImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=[
                    ModelConfig(model_id="openmoss/MOSS-Image-SFT", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **inside_loop_config),
                    ModelConfig(model_id="openmoss/MOSS-Image-SFT", origin_file_pattern="text_encoder/model*.safetensors", **outside_loop_config),
                    ModelConfig(model_id="openmoss/MOSS-Image-SFT", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vae_config),
                ],
                tokenizer_config=ModelConfig(model_id="openmoss/MOSS-Image-SFT", origin_file_pattern="tokenizer/"),
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
        negative_prompt: str = "",
        cfg_scale: float = 3.0,
        loras: List[Tuple[str, float, str]] = None,
    ) -> Image.Image:
        # Inputs
        inputs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "rand_device": "cpu",
            "num_inference_steps": num_inference_steps,
            "height": height if height % 16 == 0 else int(round(height / 16) * 16),
            "width": width if width % 16 == 0 else int(round(width / 16) * 16),
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
