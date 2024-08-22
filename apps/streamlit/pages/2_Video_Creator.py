import streamlit as st
st.set_page_config(layout="wide")
from diffsynth import SDVideoPipelineRunner
import os
import numpy as np


def load_model_list(folder):
    file_list = os.listdir(folder)
    file_list = [i for i in file_list if i.endswith(".safetensors") or i.endswith(".pth") or i.endswith(".ckpt")]
    file_list = sorted(file_list)
    return file_list


def match_processor_id(model_name, supported_processor_id_list):
    sorted_processor_id = [i[1] for i in sorted([(-len(i), i) for i in supported_processor_id_list])]
    for processor_id in sorted_processor_id:
        if processor_id in model_name:
            return supported_processor_id_list.index(processor_id) + 1
    return 0


config = {
    "models": {
        "model_list": [],
        "textual_inversion_folder": "models/textual_inversion",
        "device": "cuda",
        "lora_alphas": [],
        "controlnet_units": []
    },
    "data": {
        "input_frames": None,
        "controlnet_frames": [],
        "output_folder": "output",
        "fps": 60
    },
    "pipeline": {
        "seed": 0,
        "pipeline_inputs": {}
    }
}


with st.expander("Model", expanded=True):
    stable_diffusion_ckpt = st.selectbox("Stable Diffusion", ["None"] + load_model_list("models/stable_diffusion"))
    if stable_diffusion_ckpt != "None":
        config["models"]["model_list"].append(os.path.join("models/stable_diffusion", stable_diffusion_ckpt))
    animatediff_ckpt = st.selectbox("AnimateDiff", ["None"] + load_model_list("models/AnimateDiff"))
    if animatediff_ckpt != "None":
        config["models"]["model_list"].append(os.path.join("models/AnimateDiff", animatediff_ckpt))
    column_lora, column_lora_alpha = st.columns([2, 1])
    with column_lora:
        sd_lora_ckpt = st.selectbox("LoRA", ["None"] + load_model_list("models/lora"))
    with column_lora_alpha:
        lora_alpha = st.slider("LoRA Alpha", min_value=-4.0, max_value=4.0, value=1.0, step=0.1)
    if sd_lora_ckpt != "None":
        config["models"]["model_list"].append(os.path.join("models/lora", sd_lora_ckpt))
        config["models"]["lora_alphas"].append(lora_alpha)


with st.expander("Data", expanded=True):
    with st.container(border=True):
        input_video = st.text_input("Input Video File Path (e.g., data/your_video.mp4)", value="")
        column_height, column_width, column_start_frame_index, column_end_frame_index = st.columns([2, 2, 1, 1])
        with column_height:
            height = st.select_slider("Height", options=[256, 512, 768, 1024, 1536, 2048], value=1024)
        with column_width:
            width = st.select_slider("Width", options=[256, 512, 768, 1024, 1536, 2048], value=1024)
        with column_start_frame_index:
            start_frame_id = st.number_input("Start Frame id", value=0)
        with column_end_frame_index:
            end_frame_id = st.number_input("End Frame id", value=16)
        if input_video != "":
            config["data"]["input_frames"] = {
                "video_file": input_video,
                "image_folder": None,
                "height": height,
                "width": width,
                "start_frame_id": start_frame_id,
                "end_frame_id": end_frame_id
            }
    with st.container(border=True):
        output_video = st.text_input("Output Video File Path (e.g., data/a_folder_to_save_something)", value="output")
        fps = st.number_input("FPS", value=60)
        config["data"]["output_folder"] = output_video
        config["data"]["fps"] = fps


with st.expander("ControlNet Units", expanded=True):
    supported_processor_id_list = ["canny", "depth", "softedge", "lineart", "lineart_anime", "openpose", "tile"]
    controlnet_units = st.tabs(["ControlNet Unit 0", "ControlNet Unit 1", "ControlNet Unit 2"])
    for controlnet_id in range(len(controlnet_units)):
        with controlnet_units[controlnet_id]:
            controlnet_ckpt = st.selectbox("ControlNet", ["None"] + load_model_list("models/ControlNet"),
                                        key=f"controlnet_ckpt_{controlnet_id}")
            processor_id = st.selectbox("Processor", ["None"] + supported_processor_id_list,
                                        index=match_processor_id(controlnet_ckpt, supported_processor_id_list),
                                        disabled=controlnet_ckpt == "None", key=f"processor_id_{controlnet_id}")
            controlnet_scale = st.slider("Scale", min_value=0.0, max_value=1.0, step=0.01, value=0.5,
                                        disabled=controlnet_ckpt == "None", key=f"controlnet_scale_{controlnet_id}")
            use_input_video_as_controlnet_input = st.checkbox("Use input video as ControlNet input", value=True,
                                                              disabled=controlnet_ckpt == "None",
                                                              key=f"use_input_video_as_controlnet_input_{controlnet_id}")
            if not use_input_video_as_controlnet_input:
                controlnet_input_video = st.text_input("ControlNet Input Video File Path", value="",
                                            disabled=controlnet_ckpt == "None", key=f"controlnet_input_video_{controlnet_id}")
                column_height, column_width, column_start_frame_index, column_end_frame_index = st.columns([2, 2, 1, 1])
                with column_height:
                    height = st.select_slider("Height", options=[256, 512, 768, 1024, 1536, 2048], value=1024,
                                              disabled=controlnet_ckpt == "None", key=f"controlnet_height_{controlnet_id}")
                with column_width:
                    width = st.select_slider("Width", options=[256, 512, 768, 1024, 1536, 2048], value=1024,
                                              disabled=controlnet_ckpt == "None", key=f"controlnet_width_{controlnet_id}")
                with column_start_frame_index:
                    start_frame_id = st.number_input("Start Frame id", value=0,
                                                     disabled=controlnet_ckpt == "None", key=f"controlnet_start_frame_id_{controlnet_id}")
                with column_end_frame_index:
                    end_frame_id = st.number_input("End Frame id", value=16,
                                                   disabled=controlnet_ckpt == "None", key=f"controlnet_end_frame_id_{controlnet_id}")
                if input_video != "":
                    config["data"]["input_video"] = {
                        "video_file": input_video,
                        "image_folder": None,
                        "height": height,
                        "width": width,
                        "start_frame_id": start_frame_id,
                        "end_frame_id": end_frame_id
                    }
            if controlnet_ckpt != "None":
                config["models"]["model_list"].append(os.path.join("models/ControlNet", controlnet_ckpt))
                config["models"]["controlnet_units"].append({
                    "processor_id": processor_id,
                    "model_path": os.path.join("models/ControlNet", controlnet_ckpt),
                    "scale": controlnet_scale,
                })
                if use_input_video_as_controlnet_input:
                    config["data"]["controlnet_frames"].append(config["data"]["input_frames"])
                else:
                    config["data"]["controlnet_frames"].append({
                        "video_file": input_video,
                        "image_folder": None,
                        "height": height,
                        "width": width,
                        "start_frame_id": start_frame_id,
                        "end_frame_id": end_frame_id
                    })


with st.container(border=True):
    with st.expander("Seed", expanded=True):
        use_fixed_seed = st.checkbox("Use fixed seed", value=False)
        if use_fixed_seed:
            seed = st.number_input("Random seed", min_value=0, max_value=10**9, step=1, value=0)
        else:
            seed = np.random.randint(0, 10**9)
    with st.expander("Textual Guidance", expanded=True):
        prompt = st.text_area("Positive prompt")
        negative_prompt = st.text_area("Negative prompt")
        column_cfg_scale, column_clip_skip = st.columns(2)
        with column_cfg_scale:
            cfg_scale = st.slider("Classifier-free guidance scale", min_value=1.0, max_value=10.0, value=7.0)
        with column_clip_skip:
            clip_skip = st.slider("Clip Skip", min_value=1, max_value=4, value=1)
    with st.expander("Denoising", expanded=True):
        column_num_inference_steps, column_denoising_strength = st.columns(2)
        with column_num_inference_steps:
            num_inference_steps = st.slider("Inference steps", min_value=1, max_value=100, value=10)
        with column_denoising_strength:
            denoising_strength = st.slider("Denoising strength", min_value=0.0, max_value=1.0, value=1.0)
    with st.expander("Efficiency", expanded=False):
        animatediff_batch_size = st.slider("Animatediff batch size (sliding window size)", min_value=1, max_value=32, value=16, step=1)
        animatediff_stride = st.slider("Animatediff stride",
                                       min_value=1,
                                       max_value=max(2, animatediff_batch_size),
                                       value=max(1, animatediff_batch_size // 2),
                                       step=1)
        unet_batch_size = st.slider("UNet batch size", min_value=1, max_value=32, value=1, step=1)
        controlnet_batch_size = st.slider("ControlNet batch size", min_value=1, max_value=32, value=1, step=1)
        cross_frame_attention = st.checkbox("Enable Cross-Frame Attention", value=False)
    config["pipeline"]["seed"] = seed
    config["pipeline"]["pipeline_inputs"] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "cfg_scale": cfg_scale,
        "clip_skip": clip_skip,
        "denoising_strength": denoising_strength,
        "num_inference_steps": num_inference_steps,
        "animatediff_batch_size": animatediff_batch_size,
        "animatediff_stride": animatediff_stride,
        "unet_batch_size": unet_batch_size,
        "controlnet_batch_size": controlnet_batch_size,
        "cross_frame_attention": cross_frame_attention,
    }

run_button = st.button("☢️Run☢️", type="primary")
if run_button:
    SDVideoPipelineRunner(in_streamlit=True).run(config)
