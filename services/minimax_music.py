from diffsynth.pipelines.minimax_music3 import MiniMaxMusic3Pipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
import torch
from .base import ModelService


class MiniMaxMusic3(ModelService):
    def example_inputs(self):
        return [
            dict(
                prompt='Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. Vocals: soft female lead, close and breathy, light stacked harmonies in the chorus. Arrangement: fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus.',
                lyrics='[verse]\nMorning light filtering through the pine\nEvery quiet street is yours and mine\n[chorus]\nSoftly the world begins to breathe',
                num_inference_steps=30,
                cfg_scale=1.7,
                seed=7,
                max_audio_duration=60.0,
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
            pipe = MiniMaxMusic3Pipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=[
                    ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="language_model/model*.safetensors", **outside_loop_config),
                    ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="rvq_depth_decoder/diffusion_pytorch_model.safetensors", **outside_loop_config),
                    ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **inside_loop_config),
                    ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="condition_encoder/diffusion_pytorch_model.safetensors", **outside_loop_config),
                    ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="vocoder/diffusion_pytorch_model.safetensors", **outside_loop_config),
                ],
                tokenizer_config=ModelConfig(model_id="MiniMax/MiniMax-Music3", origin_file_pattern="tokenizer/"),
            )
            if deploy_mode in [1, 2]: pipe.compile_pipeline()
        else:
            raise NotImplementedError(f"`deploy_mode` {deploy_mode} is not implemented.")
        self.pipe = pipe

    def infer(
        self,
        prompt: str,
        lyrics: str,
        num_inference_steps: int,
        cfg_scale: float,
        seed: int,
        max_audio_duration: float = 300.0,
    ):
        inputs = {
            "prompt": prompt,
            "lyrics": lyrics,
            "max_audio_duration": max_audio_duration,
            "num_inference_steps": num_inference_steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "rand_device": "cpu",
        }
        # Inference
        audio = self.pipe(**inputs)
        return audio
