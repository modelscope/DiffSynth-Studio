from diffsynth import SDVideoPipelineRunner


# Download models
# `models/stable_diffusion/aingdiffusion_v12.safetensors`: [link](https://civitai.com/api/download/models/229575)
# `models/AnimateDiff/mm_sd_v15_v2.ckpt`: [link](https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt)
# `models/ControlNet/control_v11p_sd15_lineart.pth`: [link](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth)
# `models/ControlNet/control_v11f1e_sd15_tile.pth`: [link](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth)
# `models/Annotators/sk_model.pth`: [link](https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth)
# `models/Annotators/sk_model2.pth`: [link](https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth)
# `models/textual_inversion/verybadimagenegative_v1.3.pt`: [link](https://civitai.com/api/download/models/25820?type=Model&format=PickleTensor&size=full&fp=fp16)

# The original video in the example is https://www.bilibili.com/video/BV1iG411a7sQ/.

config = {
    "models": {
        "model_list": [
            "models/stable_diffusion/aingdiffusion_v12.safetensors",
            "models/AnimateDiff/mm_sd_v15_v2.ckpt",
            "models/ControlNet/control_v11f1e_sd15_tile.pth",
            "models/ControlNet/control_v11p_sd15_lineart.pth"
        ],
        "textual_inversion_folder": "models/textual_inversion",
        "device": "cuda",
        "lora_alphas": [],
        "controlnet_units": [
            {
                "processor_id": "tile",
                "model_path": "models/ControlNet/control_v11f1e_sd15_tile.pth",
                "scale": 0.5
            },
            {
                "processor_id": "lineart",
                "model_path": "models/ControlNet/control_v11p_sd15_lineart.pth",
                "scale": 0.5
            }
        ]
    },
    "data": {
        "input_frames": {
            "video_file": "data/examples/diffutoon/input_video.mp4",
            "image_folder": None,
            "height": 1536,
            "width": 1536,
            "start_frame_id": 0,
            "end_frame_id": 30
        },
        "controlnet_frames": [
            {
                "video_file": "data/examples/diffutoon/input_video.mp4",
                "image_folder": None,
                "height": 1536,
                "width": 1536,
                "start_frame_id": 0,
                "end_frame_id": 30
            },
            {
                "video_file": "data/examples/diffutoon/input_video.mp4",
                "image_folder": None,
                "height": 1536,
                "width": 1536,
                "start_frame_id": 0,
                "end_frame_id": 30
            }
        ],
        "output_folder": "data/examples/diffutoon/output",
        "fps": 30
    },
    "pipeline": {
        "seed": 0,
        "pipeline_inputs": {
            "prompt": "best quality, perfect anime illustration, light, a girl is dancing, smile, solo",
            "negative_prompt": "verybadimagenegative_v1.3",
            "cfg_scale": 7.0,
            "clip_skip": 2,
            "denoising_strength": 1.0,
            "num_inference_steps": 10,
            "animatediff_batch_size": 16,
            "animatediff_stride": 8,
            "unet_batch_size": 1,
            "controlnet_batch_size": 1,
            "cross_frame_attention": False,
            # The following parameters will be overwritten. You don't need to modify them.
            "input_frames": [],
            "num_frames": 30,
            "width": 1536,
            "height": 1536,
            "controlnet_frames": []
        }
    }
}

runner = SDVideoPipelineRunner()
runner.run(config)
