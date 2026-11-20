from diffsynth.pipelines.fibo import FiboPipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
from PIL import Image
import json
import torch
from typing import List, Tuple
from .base import ModelService


class Fibo20Dev(ModelService):
    def example_inputs(self):
        caption = {
            "short_description": "An origami whale swimming through paper clouds.",
            "objects": [{"description": "a blue origami whale, folded from glossy paper", "location": "center"}],
            "background_setting": "a bright sky made of layered white paper clouds",
            "lighting": {"conditions": "soft daylight", "direction": "from above"},
            "style_medium": "paper craft photograph",
        }
        return [
            dict(
                prompt=json.dumps(caption, separators=(",", ":"), ensure_ascii=False),
                seed=42,
                num_inference_steps=50,
                height=1024,
                width=1024,
                cfg_scale=5.0,
            ),
            dict(
                prompt="change the lighting to a warm sunset, keep the origami whale and paper clouds",
                seed=43,
                num_inference_steps=50,
                height=1024,
                width=1024,
                cfg_scale=5.0,
                edit_image=[Image.open("data/diffsynth_service_tests/assets/fibo/image_1.png")],
            ),
            dict(
                prompt="add a red paper crane on the origami whale",
                seed=44,
                num_inference_steps=50,
                height=1024,
                width=1024,
                cfg_scale=5.0,
                edit_image=[Image.open("data/diffsynth_service_tests/assets/fibo/image_1.png")],
                mask=Image.open("data/diffsynth_service_tests/assets/fibo/mask.png"),
            )
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
            pipe = FiboPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=[
                    ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="transformer/*.safetensors", **inside_loop_config),
                    ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="text_encoder/model.safetensors", **outside_loop_config),
                    ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **outside_loop_config),
                ],
                processor_config=ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="processor/"),
                font_config=ModelConfig(model_id="briaai/Fibo-2.0-dev", origin_file_pattern="fonts/"),
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
        edit_image: List[Image.Image] = None,
        mask: Image.Image = None,
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
            "edit_image": edit_image,
            "mask": mask,
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


class Fibo20EditTurbo(Fibo20Dev):
    def example_inputs(self):
        return [
            dict(
                prompt="change the lighting to a warm sunset, keep the origami whale and paper clouds",
                seed=43,
                num_inference_steps=4,
                height=1024,
                width=1024,
                cfg_scale=1.0,
                edit_image=[Image.open("data/diffsynth_service_tests/assets/fibo/image_1.png")],
            ),
            dict(
                prompt="add a red paper crane on the origami whale",
                seed=44,
                num_inference_steps=4,
                height=1024,
                width=1024,
                cfg_scale=1.0,
                edit_image=[Image.open("data/diffsynth_service_tests/assets/fibo/image_1.png")],
                mask=Image.open("data/diffsynth_service_tests/assets/fibo/mask.png"),
            )
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
            pipe = FiboPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=[
                    ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="transformer/*.safetensors", **inside_loop_config),
                    ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="text_encoder/model.safetensors", **outside_loop_config),
                    ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **outside_loop_config),
                ],
                processor_config=ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="processor/"),
                font_config=ModelConfig(model_id="briaai/Fibo-2.0-edit-turbo", origin_file_pattern="fonts/"),
            )
            if deploy_mode in [1, 2]: pipe.compile_pipeline()
        else:
            raise NotImplementedError("`deploy_mode` {deploy_mode} is not implemented.")
        self.pipe = pipe
