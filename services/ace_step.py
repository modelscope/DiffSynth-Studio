from diffsynth.pipelines.ace_step import AceStepPipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
import torch
from typing import List, Tuple
from .base import ModelService


class AceStep15XLSFT(ModelService):
    def example_inputs(self):
        return [
            dict(
                prompt="An explosive, high-energy pop-rock track with a strong anime theme song feel. The song kicks off with a catchy, synthesized brass fanfare over a driving rock beat with punchy drums and a solid bassline. A powerful, clear male vocal enters with a theatrical and energetic delivery, soaring through the verses and hitting powerful high notes in the chorus. The arrangement is dense and dynamic, featuring rhythmic electric guitar chords, brief instrumental breaks with synth flourishes, and a consistent, danceable groove throughout. The overall mood is triumphant, adventurous, and exhilarating.",
                duration=160,
                lyrics="[Intro - Synth Brass Fanfare]\n\n[Verse 1]\n黑夜里的风吹过耳畔\n甜蜜时光转瞬即万\n脚步飘摇在星光上\n心追节奏心跳狂乱\n耳边传来电吉他呼唤\n手指轻触碰点流点燃\n梦在云端任它蔓延\n疯狂跳跃自由无间\n\n[Chorus]\n心电感应在震动间\n拥抱未来勇敢冒险\n那旋律在心中无限\n世界变得如此耀眼\n\n[Instrumental Break - Synth Brass Melody]\n\n[Verse 2]\n鼓点撞击黑夜的底端\n跳动节拍连接你我俩\n在这里让灵魂发光\n燃尽所有不留遗憾\n\n[Instrumental Break - Synth Brass Melody]\n\n[Bridge]\n光影交错彼此的视线\n霓虹之下夜空的蔚蓝\n月光洒下温热心田\n追逐梦想它不会遥远\n\n[Chorus]\n心电感应在震动间\n拥抱未来勇敢冒险\n那旋律在心中无限\n世界变得如此耀眼\n\n[Outro - Instrumental with Synth Brass Melody]\n[Song ends abruptly]",
                vocal_language="zh",
                keyscale="B minor",
                bpm=100,
                timesignature="4",
                seed=42,
                cfg_scale=4,
                num_inference_steps=50,
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
                # The audio tokenizer in this checkpoint has a 6-input projection
                # that cannot use convrot (it needs a power-of-four group).
                exclude_modules=["tokenizer.quantizer.project_out"],
                backend_config_kwargs={"use_convrot": True, "convrot_groupsize": 256},
            )}
            outside_loop_config = {} if deploy_mode in [0, 1] else offload_config
            pipe = AceStepPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=[
                    ModelConfig(model_id="ACE-Step/acestep-v15-xl-sft", origin_file_pattern="model-*.safetensors", **inside_loop_config),
                    ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="Qwen3-Embedding-0.6B/model.safetensors", **outside_loop_config),
                    ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **outside_loop_config),
                ],
                text_tokenizer_config=ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="Qwen3-Embedding-0.6B/"),
                silence_latent_config=ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="acestep-v15-turbo/silence_latent.pt"),
            )
            if deploy_mode in [1, 2]: pipe.compile_pipeline()
        else:
            raise NotImplementedError(f"`deploy_mode` {deploy_mode} is not implemented.")
        self.pipe = pipe

    def infer(
        self,
        prompt: str,
        duration: float,
        lyrics: str,
        vocal_language: str,
        keyscale: str,
        bpm: float,
        timesignature: str,
        seed: int,
        cfg_scale: float,
        num_inference_steps: int,
        loras: List[Tuple[str, float, str]] = None,
    ):
        inputs = {
            "prompt": prompt,
            "duration": duration,
            "lyrics": lyrics,
            "vocal_language": vocal_language,
            "keyscale": keyscale,
            "bpm": bpm,
            "timesignature": timesignature,
            "seed": seed,
            "cfg_scale": cfg_scale,
            "num_inference_steps": num_inference_steps,
            "tiled": True,
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
        audio = self.pipe(**inputs)
        # Clear loaded LoRAs
        self.pipe.clear_lora()
        return audio
