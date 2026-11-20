from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
from diffsynth.core.data.operators import ImageCropAndResize
from PIL import Image
import torch
from typing import List, Tuple, Literal
from .base import ModelService


class QwenImageService(ModelService):
    dit_model_id = "Qwen/Qwen-Image"
    controlnet_config = None

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
            text_encoder_config = outside_loop_config if deploy_mode in [0, 1] else {
                "offload_dtype": torch.bfloat16, "offload_device": "cpu",
                "onload_dtype": torch.bfloat16, "onload_device": "cpu",
                "preparing_dtype": torch.bfloat16, "preparing_device": "cpu",
                "computation_dtype": torch.bfloat16, "computation_device": device,
                "quantize": QuantizeConfig(method="entropack_lossy_quant_int8_raw", backend_config_kwargs={"use_convrot": True, "convrot_groupsize": 256})
            }
            pipe = QwenImagePipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=[
                    ModelConfig(model_id=self.dit_model_id, origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **inside_loop_config),
                    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors", **text_encoder_config),
                    ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **outside_loop_config),
                ],
                tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
                processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
            )
            if deploy_mode in [1, 2]: pipe.compile_pipeline()
        else:
            raise NotImplementedError(f"`deploy_mode` {deploy_mode} is not implemented.")
        self.pipe = pipe
        if self.controlnet_config is not None:
            self.controlnet_config.download_if_necessary()

    def infer(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        cfg_scale: float,
        num_inference_steps: int,
        height: int,
        width: int,
        loras: List[Tuple[str, float, str]] = None,
        controlnet_path: str = None,
        controlnet_image: Image.Image = None,
        controlnet_scale: float = 1,
        edit_image: List[Image.Image] = None,
        edit_model_type: Literal["Qwen-Image-Edit", "Qwen-Image-Edit-2509", "Qwen-Image-Edit-2511"] = "Qwen-Image-Edit",
    ) -> Image.Image:
        inputs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "cfg_scale": cfg_scale,
            "num_inference_steps": num_inference_steps,
            "height": height if height % 16 == 0 else int(round(height / 16) * 16),
            "width": width if width % 16 == 0 else int(round(width / 16) * 16)
        }
        # ControlNet and LoRAs
        self.pipe.clear_lora()
        if loras is None:
            loras = []
        if controlnet_path is not None:
            loras.append((controlnet_path, controlnet_scale, "Context_Control."))
            inputs.update({"context_image": controlnet_image.convert("RGB")})
        if loras is not None:
            for lora_path, lora_scale, lora_keyword in loras:
                self.pipe.load_lora(self.pipe.dit, lora_path, alpha=lora_scale)
                if lora_keyword != "":
                    prompt = f"{lora_keyword} {prompt}"
        inputs.update({"prompt": prompt})
        # Edit image
        if edit_image is not None:
            processor = ImageCropAndResize(max_pixels=2048*2048, height_division_factor=16, width_division_factor=16)
            if edit_model_type == "Qwen-Image-Edit":
                edit_image = processor(edit_image[0].convert("RGB"))
            elif edit_model_type == "Qwen-Image-Edit-2509":
                edit_image = [processor(i.convert("RGB")) for i in edit_image]
            elif edit_model_type == "Qwen-Image-Edit-2511":
                edit_image = [processor(i.convert("RGB")) for i in edit_image]
                inputs.update({"zero_cond_t": True})
            inputs.update({"edit_image": edit_image})
        # Inference
        image = self.pipe(**inputs)
        image = image.resize((width, height))
        # Clear loaded LoRAs
        self.pipe.clear_lora()
        return image


class QwenImage(QwenImageService):
    dit_model_id = "Qwen/Qwen-Image"
    controlnet_config = ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-In-Context-Control-Union", origin_file_pattern="model.safetensors")

    def example_inputs(self):
        return [
            dict(
                prompt="精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。",
                seed=0,
                num_inference_steps=40,
                height=1472,
                width=1104,
            ),
            dict(
                prompt="精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。",
                seed=0,
                num_inference_steps=40,
                height=1664,
                width=1664,
            ),
        ]


    def infer(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = None,
        cfg_scale: float = 4,
        num_inference_steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        loras: List[Tuple[str, float, str]] = None,
        controlnet_path: str = None,
        controlnet_image: Image.Image = None,
        controlnet_scale: float = 1,
        edit_image: List[Image.Image] = None,
        edit_model_type: Literal["Qwen-Image-Edit", "Qwen-Image-Edit-2509", "Qwen-Image-Edit-2511"] = "Qwen-Image-Edit",
    ) -> Image.Image:
        if controlnet_image is not None:
            controlnet_path = self.controlnet_config.path
        return super().infer(
            prompt,
            negative_prompt,
            seed,
            cfg_scale,
            num_inference_steps,
            height,
            width,
            loras,
            controlnet_path,
            controlnet_image,
            controlnet_scale,
            edit_image,
            edit_model_type,
        )


class QwenImage2512(QwenImageService):
    dit_model_id = "Qwen/Qwen-Image-2512"
    controlnet_config = None

    def example_inputs(self):
        return [
            dict(
                prompt="精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。",
                seed=0,
                num_inference_steps=40,
                height=1472,
                width=1104,
            ),
            dict(
                prompt="精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。",
                seed=0,
                num_inference_steps=40,
                height=1664,
                width=1664,
            ),
        ]


    def infer(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = None,
        cfg_scale: float = 4,
        num_inference_steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        loras: List[Tuple[str, float, str]] = None,
        controlnet_path: str = None,
        controlnet_image: Image.Image = None,
        controlnet_scale: float = 1,
        edit_image: List[Image.Image] = None,
        edit_model_type: Literal["Qwen-Image-Edit", "Qwen-Image-Edit-2509", "Qwen-Image-Edit-2511"] = "Qwen-Image-Edit",
    ) -> Image.Image:
        return super().infer(
            prompt,
            negative_prompt,
            seed,
            cfg_scale,
            num_inference_steps,
            height,
            width,
            loras,
            controlnet_path,
            controlnet_image,
            controlnet_scale,
            edit_image,
            edit_model_type,
        )


class QwenImageEdit(QwenImageService):
    dit_model_id = "Qwen/Qwen-Image-Edit"
    controlnet_config = None

    def example_inputs(self):
        return [
            dict(
                prompt="将裙子改为粉色",
                negative_prompt="",
                seed=1,
                cfg_scale=4,
                num_inference_steps=40,
                height=1328,
                width=1024,
                edit_image=[Image.open("data/diffsynth_service_tests/assets/qwen_image/underwater.png")],
            ),
        ]


    def infer(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = None,
        cfg_scale: float = 4,
        num_inference_steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        loras: List[Tuple[str, float, str]] = None,
        controlnet_path: str = None,
        controlnet_image: Image.Image = None,
        controlnet_scale: float = 1,
        edit_image: List[Image.Image] = None,
        edit_model_type: Literal["Qwen-Image-Edit", "Qwen-Image-Edit-2509", "Qwen-Image-Edit-2511"] = "Qwen-Image-Edit",
    ) -> Image.Image:
        edit_model_type = "Qwen-Image-Edit"
        return super().infer(
            prompt,
            negative_prompt,
            seed,
            cfg_scale,
            num_inference_steps,
            height,
            width,
            loras,
            controlnet_path,
            controlnet_image,
            controlnet_scale,
            edit_image,
            edit_model_type,
        )


class QwenImageEdit2509(QwenImageService):
    dit_model_id = "Qwen/Qwen-Image-Edit-2509"
    controlnet_config = None

    def example_inputs(self):
        return [
            dict(
                prompt="生成这两个人的合影",
                negative_prompt="",
                seed=1,
                cfg_scale=4,
                num_inference_steps=40,
                height=1328,
                width=1024,
                edit_image=[Image.open("data/diffsynth_service_tests/assets/qwen_image/image_1.jpg"),
                            Image.open("data/diffsynth_service_tests/assets/qwen_image/image_2.jpg")],
            ),
        ]


    def infer(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = None,
        cfg_scale: float = 4,
        num_inference_steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        loras: List[Tuple[str, float, str]] = None,
        controlnet_path: str = None,
        controlnet_image: Image.Image = None,
        controlnet_scale: float = 1,
        edit_image: List[Image.Image] = None,
        edit_model_type: Literal["Qwen-Image-Edit", "Qwen-Image-Edit-2509", "Qwen-Image-Edit-2511"] = "Qwen-Image-Edit",
    ) -> Image.Image:
        edit_model_type = "Qwen-Image-Edit-2509"
        return super().infer(
            prompt,
            negative_prompt,
            seed,
            cfg_scale,
            num_inference_steps,
            height,
            width,
            loras,
            controlnet_path,
            controlnet_image,
            controlnet_scale,
            edit_image,
            edit_model_type,
        )


class QwenImageEdit2511(QwenImageService):
    dit_model_id = "Qwen/Qwen-Image-Edit-2511"
    controlnet_config = None

    def example_inputs(self):
        return [
            dict(
                prompt="生成这两个人的合影",
                negative_prompt="",
                seed=1,
                cfg_scale=4,
                num_inference_steps=40,
                height=1152,
                width=896,
                edit_image=[Image.open("data/diffsynth_service_tests/assets/qwen_image/image_1.jpg"),
                            Image.open("data/diffsynth_service_tests/assets/qwen_image/image_2.jpg")],
            ),
        ]


    def infer(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = None,
        cfg_scale: float = 4,
        num_inference_steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        loras: List[Tuple[str, float, str]] = None,
        controlnet_path: str = None,
        controlnet_image: Image.Image = None,
        controlnet_scale: float = 1,
        edit_image: List[Image.Image] = None,
        edit_model_type: Literal["Qwen-Image-Edit", "Qwen-Image-Edit-2509", "Qwen-Image-Edit-2511"] = "Qwen-Image-Edit",
    ) -> Image.Image:
        edit_model_type = "Qwen-Image-Edit-2511"
        return super().infer(
            prompt,
            negative_prompt,
            seed,
            cfg_scale,
            num_inference_steps,
            height,
            width,
            loras,
            controlnet_path,
            controlnet_image,
            controlnet_scale,
            edit_image,
            edit_model_type,
        )


class FireRedImageEdit(QwenImageEdit2509):
    dit_model_id = "FireRedTeam/FireRed-Image-Edit-1.0"
    controlnet_config = None

    def example_inputs(self):
        return [
            dict(
                prompt="生成这两个人的合影",
                negative_prompt="",
                seed=1,
                cfg_scale=4,
                num_inference_steps=40,
                height=1152,
                width=896,
                edit_image=[Image.open("data/diffsynth_service_tests/assets/qwen_image/image_1.jpg"),
                            Image.open("data/diffsynth_service_tests/assets/qwen_image/image_2.jpg")],
            ),
        ]
