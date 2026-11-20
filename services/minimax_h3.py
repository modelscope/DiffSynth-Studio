from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.utils.data.audio_video import write_video_audio, read_video_audio, read_audio
from PIL import Image
import torch
from typing import List, Tuple
from .base import ModelService


class MiniMaxH3FL2VA(ModelService):
    dit_variant = "fl2va"
    def example_inputs(self):
        return [
            dict(
                prompt="A girl is very happy, she is speaking in english: “I enjoy working with Diffsynth-Studio, it's a perfect framework.”",
                seed=0,
                num_inference_steps=50,
                height=480,
                width=832,
                num_frames=124,
                output_path="data/diffsynth_service_tests/videos/video_1.mp4",
                first_frame=None,
                last_frame=None,
            ),
            dict(
                prompt='室内家庭争吵短剧场景，竖屏短剧质感，真实真人表演，中式家庭/小饭馆室内环境，暖色灯光，背景有红色装饰和书法字幅，浅景深，情绪强烈，剪辑节奏紧凑。表演要求：真实短剧表演风格，不要夸张舞台腔。男人的语气是愤怒、委屈、急切的反驳，他说“你到底想干什么？”；中老年女性的语气是尖锐、强势、咄咄逼人的质问，她说“你必须赔钱！”。两人之间有强烈对峙感，节奏逐步升级。画面风格：竖屏9:16，手机短剧质感，真人实拍感，浅景深，室内暖光，中近景为主，频繁正反打剪辑，背景保持生活化，不要科幻、不要古装、不要动画感。画面中不要出现任何字幕、文字、平台水印或贴片。 ',
                seed=0,
                num_inference_steps=50,
                height=832,
                width=480,
                num_frames=124,
                output_path="data/diffsynth_service_tests/videos/video_2.mp4",
                first_frame=Image.open("data/diffsynth_service_tests/assets/minimax_h3/first.png"),
                last_frame=Image.open("data/diffsynth_service_tests/assets/minimax_h3/last.png"),
            ),
        ]

    def load_models(self, device="cuda", deploy_mode=0):
        if deploy_mode in [0]:
            vram_config = {
                "offload_dtype": torch.bfloat16,
                "offload_device": "cpu",
                "onload_dtype": torch.bfloat16,
                "onload_device": "cpu",
                "preparing_dtype": torch.bfloat16,
                "preparing_device": "cuda",
                "computation_dtype": torch.bfloat16,
                "computation_device": "cuda",
            }
            pipe = MiniMaxH3Pipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-fl2va-nf4.safetensors" if self.dit_variant == "fl2va" else "minimax-h3-ref2va-nf4.safetensors", **vram_config),
                    ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="minimax-h3-text-encoder-nf4.safetensors", **vram_config),
                    ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="video_vae_nf4.safetensors", **vram_config),
                    ModelConfig(model_id="DiffSynth-Studio/MiniMax-H3-NF4", origin_file_pattern="audio_vae_nf4.safetensors", **vram_config),
                ],
                processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
                vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
            )
        elif deploy_mode in [1]:
            dit_config = {
                "quantize": QuantizeConfig(method="entropack_lossy_quant_int8_raw", backend_config_kwargs={"use_convrot": True, "convrot_groupsize": 256})
            }
            text_encoder_config = {
                "quantize": QuantizeConfig(method="entropack_lossy_quant_int8_raw", backend_config_kwargs={"use_convrot": True, "convrot_groupsize": 256})
            }
            others_config = {}
            pipe = MiniMaxH3Pipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=[
                    ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **text_encoder_config),
                    ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors" if self.dit_variant == "fl2va" else "Ref2VA/transformer/model*.safetensors", **dit_config),
                    ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **others_config),
                    ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **others_config),
                ],
                processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
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
        num_frames: int,
        output_path: str,
        first_frame: Image.Image = None,
        last_frame: Image.Image = None,
        references: dict = None,
        loras: List[Tuple[str, float, str]] = None,
        accelerate_lora: str = None,
    ):
        inputs = {
            "prompt": prompt,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "height": height if height % 32 == 0 else int(round(height / 32) * 32),
            "width": width if width % 32 == 0 else int(round(width / 32) * 32),
            "num_frames": num_frames,
            "rand_device": "cpu",
        }
        # LoRAs
        if accelerate_lora is not None:
            inputs["flow_shift"] = 6.0
            inputs["num_inference_steps"] = 4
            if loras is None:
                loras = []
            loras.append((accelerate_lora, 1, ""))
        self.pipe.clear_lora()
        if loras is not None:
            for lora_path, lora_scale, lora_keyword in loras:
                self.pipe.load_lora(self.pipe.dit, lora_path, alpha=lora_scale)
                if lora_keyword != "":
                    prompt = f"{lora_keyword} {prompt}"
                inputs.update({"prompt": prompt})
        # First & Last Frame
        image_processor = ImageCropAndResize(
            height=height if height % 32 == 0 else int(round(height / 32) * 32),
            width=width if width % 32 == 0 else int(round(width / 32) * 32)
        )
        if first_frame is not None:
            first_frame = image_processor(first_frame)
        if last_frame is not None:
            last_frame = image_processor(last_frame)
        if first_frame is not None and last_frame is not None:
            inputs["keyframes"], inputs["keyframe_indices"] = [first_frame, last_frame], [0, -1]
        elif first_frame is not None:
            inputs["keyframes"], inputs["keyframe_indices"] = [first_frame], [0]
        elif last_frame is not None:
            inputs["keyframes"], inputs["keyframe_indices"] = [last_frame], [-1]
        # References
        if references is not None:
            for reference in references:
                if reference["type"] == "image":
                    reference["image"] = Image.open(reference["image"]).convert("RGB")
                elif reference["type"] == "audio":
                    audio, sample_rate = read_audio(reference["audio"])
                    reference["audio"] = audio
                    reference["sample_rate"] = sample_rate
                elif reference["type"] == "video":
                    video, audio, sample_rate = read_video_audio(reference["video"], height=height, width=width, num_frames=num_frames, audio_sample_rate=32000)
                    reference["video"] = video
                    if audio is not None:
                        reference["audio"] = audio
                        reference["sample_rate"] = sample_rate
                        reference["type"] = "video_audio"
            inputs["references"] = references
        # Inference
        video, audio = self.pipe(**inputs)
        self.pipe.clear_lora()
        # Save
        write_video_audio(
            video=video, audio=audio,
            output_path=output_path, fps=24, audio_sample_rate=32000,
        )


class MiniMaxH3Ref2VA(MiniMaxH3FL2VA):
    dit_variant = "ref2va"
    def example_inputs(self):
        return [
            dict(
                prompt='一个网站页面，网站页面UI设计，网站动效，视频展示了流畅的网页向下滚动效果。一个极具爆发力与动感的产品官网风格产品落地页 UI/UX 演示视频，核心展示主体是该产品图片1。页面采用粗犷有力、倾斜的超大号无衬线字体进行张扬的排版。背景有极具速度感的动态光影、暗色碳纤维或运动透气网眼纹理在交织变换。视频展示了节奏紧凑、充满力量感的网页向下滚动效果，以及鼠标悬停时强烈的视觉放大与颜色反转等 UI 交互动作。',
                seed=42,
                num_inference_steps=50,
                height=480,
                width=832,
                num_frames=124,
                output_path="data/diffsynth_service_tests/videos/video_3.mp4",
                references=[
                    {"type": "image", "image": "data/diffsynth_service_tests/assets/minimax_h3/ref_image.png"},
                ],
            ),
            dict(
                prompt="subject_definitions:\n<Subject 1> is the young man with short wavy blonde hair, wearing a bright pink suit jacket, matching pink trousers, an unbuttoned white shirt, and silver rings, holding a small black lamb in his arms in <Video 1>.\n<Video 1> is the source video for the editing task.\n<Audio 1> is the synchronized audio track of <Video 1>, providing the background music.\n<Audio 2> is the voice timbre reference for <Subject 1>'s voice, containing a spoken male voiceover.\n\nsummary:\n[video editing + audio reference + audio reuse] The target video is an edited version of <Video 1>. <Subject 1>, wearing a bright pink suit and holding a black lamb, stands in a grassy field with other white lambs in the background. The edit animates <Subject 1>'s face to speak the user-provided dialogue. <Audio 1> is partially reused as the continuous background music, while the target references the calm male voice timbre of <Audio 2> for <Subject 1>'s spoken lines.\n\nretention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - the man retains his identity, wavy blonde hair, pink suit, white shirt, accessories, and the black lamb he holds, with his mouth newly animated to speak.\n<Video 1> (source video editing): fully_preserved - the original camera framing, warm golden hour lighting, grassy hill setting, and background white lambs are maintained while the central character is edited.\n<Audio 1>: partially_copy - the atmospheric background music from <Audio 1> is reused in the target video, mixed beneath the newly added spoken dialogue.\n<Audio 2>: reference - the target audio references the male voice timbre from <Audio 2> to generate <Subject 1>'s spoken dialogue.\n\ndetailed_description:\nThe target video is in realistic photographic style.\n[Shot 1] The shot begins from the source <Video 1>, showing <Subject 1>, a young man with short wavy blonde hair, wearing a bright pink suit jacket, matching pink trousers, and a casually unbuttoned white shirt. He stands confidently in a sunlit green pasture, gently holding a small black lamb securely in his arms. The warm, golden hour lighting casts soft shadows across his face and the bright pink fabric of his suit. Behind him, several white lambs stand and graze on the rolling grassy hill against a clear, pale blue sky. The atmospheric background music from <Audio 1> plays continuously throughout the scene. <Subject 1> physically speaks, his mouth movements naturally syncing to the new dialogue, with his voice timbre referencing the calm male delivery from <Audio 2>. Looking thoughtfully forward, <Subject 1> (S1) speaks softly, <d>[English] Follow the wind, live free.</d> As he delivers the line, he subtly shifts his weight, cradling the resting black lamb while the camera slowly pushes in. <Subject 1> (S1) continues his thought, <d>[English] Leave worries behind, enjoy the moment.</d> Exactly as his voice stops, his lips meet in a relaxed, peaceful smile, and his jaw ceases speaking motion. He then turns his gaze slightly away toward the horizon, gently stroking the black lamb's fleece with his fingers as the camera holds on this tranquil, sunlit state through the end of the video.\n\noverall_soundscape:\nThe soundscape consists of the continuous, atmospheric background music from <Audio 1>, overlaid with the clear, calm male dialogue spoken by the main character, referencing the voice timbre of <Audio 2>.\n\nnon_diegetic_music:\nThe atmospheric, sustained background music from <Audio 1> is reused as the continuous score, playing quietly beneath the spoken dialogue.",
                seed=42,
                num_inference_steps=50,
                height=480,
                width=832,
                num_frames=124,
                output_path="data/diffsynth_service_tests/videos/video_4.mp4",
                references=[
                    {"type": "video", "video": "data/diffsynth_service_tests/assets/minimax_h3/ref_video.mp4"},
                    {"type": "audio", "audio": "data/diffsynth_service_tests/assets/minimax_h3/ref_voice.mp3"},
                ],
            ),
        ]
