import torch, os, io, json, time
import numpy as np
from PIL import Image
import streamlit as st
st.set_page_config(layout="wide")
from streamlit_drawable_canvas import st_canvas
from diffsynth.models import ModelManager
from diffsynth.pipelines import SDImagePipeline, SDXLImagePipeline, SD3ImagePipeline, HunyuanDiTImagePipeline, FluxImagePipeline
from diffsynth.data.video import crop_and_resize


config = {
    "Stable Diffusion": {
        "model_folder": "models/stable_diffusion",
        "pipeline_class": SDImagePipeline,
        "fixed_parameters": {}
    },
    "Stable Diffusion XL": {
        "model_folder": "models/stable_diffusion_xl",
        "pipeline_class": SDXLImagePipeline,
        "fixed_parameters": {}
    },
    "Stable Diffusion 3": {
        "model_folder": "models/stable_diffusion_3",
        "pipeline_class": SD3ImagePipeline,
        "fixed_parameters": {}
    },
    "Stable Diffusion XL Turbo": {
        "model_folder": "models/stable_diffusion_xl_turbo",
        "pipeline_class": SDXLImagePipeline,
        "fixed_parameters": {
            "negative_prompt": "",
            "cfg_scale": 1.0,
            "num_inference_steps": 1,
            "height": 512,
            "width": 512,
        }
    },
    "Kolors": {
        "model_folder": "models/kolors",
        "pipeline_class": SDXLImagePipeline,
        "fixed_parameters": {}
    },
    "HunyuanDiT": {
        "model_folder": "models/HunyuanDiT",
        "pipeline_class": HunyuanDiTImagePipeline,
        "fixed_parameters": {
            "height": 1024,
            "width": 1024,
        }
    },
    "FLUX": {
        "model_folder": "models/FLUX",
        "pipeline_class": FluxImagePipeline,
        "fixed_parameters": {
            "cfg_scale": 1.0,
        }
    }
}


def load_model_list(model_type):
    folder = config[model_type]["model_folder"]
    file_list = [i for i in os.listdir(folder) if i.endswith(".safetensors")]
    if model_type in ["HunyuanDiT", "Kolors", "FLUX"]:
        file_list += [i for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))]
    file_list = sorted(file_list)
    return file_list


def release_model():
    if "model_manager" in st.session_state:
        st.session_state["model_manager"].to("cpu")
        del st.session_state["loaded_model_path"]
        del st.session_state["model_manager"]
        del st.session_state["pipeline"]
        torch.cuda.empty_cache()


def load_model(model_type, model_path):
    model_manager = ModelManager()
    if model_type == "HunyuanDiT":
        model_manager.load_models([
            os.path.join(model_path, "clip_text_encoder/pytorch_model.bin"),
            os.path.join(model_path, "mt5/pytorch_model.bin"),
            os.path.join(model_path, "model/pytorch_model_ema.pt"),
            os.path.join(model_path, "sdxl-vae-fp16-fix/diffusion_pytorch_model.bin"),
        ])
    elif model_type == "Kolors":
        model_manager.load_models([
            os.path.join(model_path, "text_encoder"),
            os.path.join(model_path, "unet/diffusion_pytorch_model.safetensors"),
            os.path.join(model_path, "vae/diffusion_pytorch_model.safetensors"),
        ])
    elif model_type == "FLUX":
        model_manager.torch_dtype = torch.bfloat16
        file_list = [
            os.path.join(model_path, "text_encoder/model.safetensors"),
            os.path.join(model_path, "text_encoder_2"),
        ]
        for file_name in os.listdir(model_path):
            if file_name.endswith(".safetensors"):
                file_list.append(os.path.join(model_path, file_name))
        model_manager.load_models(file_list)
    else:
        model_manager.load_model(model_path)
    pipeline = config[model_type]["pipeline_class"].from_model_manager(model_manager)
    st.session_state.loaded_model_path = model_path
    st.session_state.model_manager = model_manager
    st.session_state.pipeline = pipeline
    return model_manager, pipeline


def use_output_image_as_input(update=True):
    # Search for input image
    output_image_id = 0
    selected_output_image = None
    while True:
        if f"use_output_as_input_{output_image_id}" not in st.session_state:
            break
        if st.session_state[f"use_output_as_input_{output_image_id}"]:
            selected_output_image = st.session_state["output_images"][output_image_id]
            break
        output_image_id += 1
    if update and selected_output_image is not None:
        st.session_state["input_image"] = selected_output_image
    return selected_output_image is not None


def apply_stroke_to_image(stroke_image, image):
    image = np.array(image.convert("RGB")).astype(np.float32)
    height, width, _ = image.shape

    stroke_image = np.array(Image.fromarray(stroke_image).resize((width, height))).astype(np.float32)
    weight = stroke_image[:, :, -1:] / 255
    stroke_image = stroke_image[:, :, :-1]

    image = stroke_image * weight + image * (1 - weight)
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    return image


@st.cache_data
def image2bits(image):
    image_byte = io.BytesIO()
    image.save(image_byte, format="PNG")
    image_byte = image_byte.getvalue()
    return image_byte


def show_output_image(image):
    st.image(image, use_column_width="always")
    st.button("Use it as input image", key=f"use_output_as_input_{image_id}")
    st.download_button("Download", data=image2bits(image), file_name="image.png", mime="image/png", key=f"download_output_{image_id}")


column_input, column_output = st.columns(2)
with st.sidebar:
    # Select a model
    with st.expander("Model", expanded=True):
        model_type = st.selectbox("Model type", [model_type_ for model_type_ in config])
        fixed_parameters = config[model_type]["fixed_parameters"]
        model_path_list = ["None"] + load_model_list(model_type)
        model_path = st.selectbox("Model path", model_path_list)

        # Load the model
        if model_path == "None":
            # No models are selected. Release VRAM.
            st.markdown("No models are selected.")
            release_model()
        else:
            # A model is selected.
            model_path = os.path.join(config[model_type]["model_folder"], model_path)
            if st.session_state.get("loaded_model_path", "") != model_path:
                # The loaded model is not the selected model. Reload it.
                st.markdown(f"Loading model at {model_path}.")
                st.markdown("Please wait a moment...")
                release_model()
                model_manager, pipeline = load_model(model_type, model_path)
                st.markdown("Done.")
            else:
                # The loaded model is not the selected model. Fetch it from `st.session_state`.
                st.markdown(f"Loading model at {model_path}.")
                st.markdown("Please wait a moment...")
                model_manager, pipeline = st.session_state.model_manager, st.session_state.pipeline
                st.markdown("Done.")

    # Show parameters
    with st.expander("Prompt", expanded=True):
        prompt = st.text_area("Positive prompt")
        if "negative_prompt" in fixed_parameters:
            negative_prompt = fixed_parameters["negative_prompt"]
        else:
            negative_prompt = st.text_area("Negative prompt")
        if "cfg_scale" in fixed_parameters:
            cfg_scale = fixed_parameters["cfg_scale"]
        else:
            cfg_scale = st.slider("Classifier-free guidance scale", min_value=1.0, max_value=10.0, value=7.5)
    with st.expander("Image", expanded=True):
        if "num_inference_steps" in fixed_parameters:
            num_inference_steps = fixed_parameters["num_inference_steps"]
        else:
            num_inference_steps = st.slider("Inference steps", min_value=1, max_value=100, value=20)
        if "height" in fixed_parameters:
            height = fixed_parameters["height"]
        else:
            height = st.select_slider("Height", options=[256, 512, 768, 1024, 2048], value=512)
        if "width" in fixed_parameters:
            width = fixed_parameters["width"]
        else:
            width = st.select_slider("Width", options=[256, 512, 768, 1024, 2048], value=512)
        num_images = st.number_input("Number of images", value=2)
        use_fixed_seed = st.checkbox("Use fixed seed", value=False)
        if use_fixed_seed:
            seed = st.number_input("Random seed", min_value=0, max_value=10**9, step=1, value=0)

    # Other fixed parameters
    denoising_strength = 1.0
    repetition = 1


# Show input image
with column_input:
    with st.expander("Input image (Optional)", expanded=True):
        with st.container(border=True):
            column_white_board, column_upload_image = st.columns([1, 2])
            with column_white_board:
                create_white_board = st.button("Create white board")
                delete_input_image = st.button("Delete input image")
            with column_upload_image:
                upload_image = st.file_uploader("Upload image", type=["png", "jpg"], key="upload_image")

        if upload_image is not None:
            st.session_state["input_image"] = crop_and_resize(Image.open(upload_image), height, width)
        elif create_white_board:
            st.session_state["input_image"] = Image.fromarray(np.ones((height, width, 3), dtype=np.uint8) * 255)
        else:
            use_output_image_as_input()

        if delete_input_image and "input_image" in st.session_state:
            del st.session_state.input_image
        if delete_input_image and "upload_image" in st.session_state:
            del st.session_state.upload_image

        input_image = st.session_state.get("input_image", None)
        if input_image is not None:
            with st.container(border=True):
                column_drawing_mode, column_color_1, column_color_2 = st.columns([4, 1, 1])
                with column_drawing_mode:
                    drawing_mode = st.radio("Drawing tool", ["transform", "freedraw", "line", "rect"], horizontal=True, index=1)
                with column_color_1:
                    stroke_color = st.color_picker("Stroke color")
                with column_color_2:
                    fill_color = st.color_picker("Fill color")
                stroke_width = st.slider("Stroke width", min_value=1, max_value=50, value=10)
            with st.container(border=True):
                denoising_strength = st.slider("Denoising strength", min_value=0.0, max_value=1.0, value=0.7)
                repetition = st.slider("Repetition", min_value=1, max_value=8, value=1)
            with st.container(border=True):
                input_width, input_height = input_image.size
                canvas_result = st_canvas(
                    fill_color=fill_color,
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color="rgba(255, 255, 255, 0)",
                    background_image=input_image,
                    update_streamlit=True,
                    height=int(512 / input_width * input_height),
                    width=512,
                    drawing_mode=drawing_mode,
                    key="canvas"
                )

    num_painter_layer = st.number_input("Number of painter layers", min_value=0, max_value=10, step=1, value=0)
    local_prompts, masks, mask_scales = [], [], []
    white_board = Image.fromarray(np.ones((512, 512, 3), dtype=np.uint8) * 255)
    painter_layers_json_data = []
    for painter_tab_id in range(num_painter_layer):
        with st.expander(f"Painter layer {painter_tab_id}", expanded=True):
            enable_local_prompt = st.checkbox(f"Enable prompt {painter_tab_id}", value=True)
            local_prompt = st.text_area(f"Prompt {painter_tab_id}")
            mask_scale = st.slider(f"Mask scale {painter_tab_id}", min_value=0.0, max_value=3.0, value=1.0)
            stroke_width = st.slider(f"Stroke width {painter_tab_id}", min_value=1, max_value=300, value=100)
            canvas_result_local = st_canvas(
                fill_color="#000000",
                stroke_width=stroke_width,
                stroke_color="#000000",
                background_color="rgba(255, 255, 255, 0)",
                background_image=white_board,
                update_streamlit=True,
                height=512,
                width=512,
                drawing_mode="freedraw",
                key=f"canvas_{painter_tab_id}"
            )
            if canvas_result_local.json_data is not None:
                painter_layers_json_data.append(canvas_result_local.json_data.copy())
                painter_layers_json_data[-1]["prompt"] = local_prompt
            if enable_local_prompt:
                local_prompts.append(local_prompt)
                if canvas_result_local.image_data is not None:
                    mask = apply_stroke_to_image(canvas_result_local.image_data, white_board)
                else:
                    mask = white_board
                mask = Image.fromarray(255 - np.array(mask))
                masks.append(mask)
                mask_scales.append(mask_scale)
    save_painter_layers = st.button("Save painter layers")
    if save_painter_layers:
        os.makedirs("data/painter_layers", exist_ok=True)
        json_file_path = f"data/painter_layers/{time.time_ns()}.json"
        with open(json_file_path, "w") as f:
            json.dump(painter_layers_json_data, f, indent=4)
            st.markdown(f"Painter layers are saved in {json_file_path}.")


with column_output:
    run_button = st.button("Generate image", type="primary")
    auto_update = st.checkbox("Auto update", value=False)
    num_image_columns = st.slider("Columns", min_value=1, max_value=8, value=2)
    image_columns = st.columns(num_image_columns)

    # Run
    if (run_button or auto_update) and model_path != "None":

        if input_image is not None:
            input_image = input_image.resize((width, height))
            if canvas_result.image_data is not None:
                input_image = apply_stroke_to_image(canvas_result.image_data, input_image)

        output_images = []
        for image_id in range(num_images * repetition):
            if use_fixed_seed:
                torch.manual_seed(seed + image_id)
            else:
                torch.manual_seed(np.random.randint(0, 10**9))
            if image_id >= num_images:
                input_image = output_images[image_id - num_images]
            with image_columns[image_id % num_image_columns]:
                progress_bar_st = st.progress(0.0)
                image = pipeline(
                    prompt, negative_prompt=negative_prompt,
                    local_prompts=local_prompts, masks=masks, mask_scales=mask_scales,
                    cfg_scale=cfg_scale, num_inference_steps=num_inference_steps,
                    height=height, width=width,
                    input_image=input_image, denoising_strength=denoising_strength,
                    progress_bar_st=progress_bar_st
                )
                output_images.append(image)
                progress_bar_st.progress(1.0)
                show_output_image(image)
                st.session_state["output_images"] = output_images

    elif "output_images" in st.session_state:
        for image_id in range(len(st.session_state.output_images)):
            with image_columns[image_id % num_image_columns]:
                image = st.session_state.output_images[image_id]
                progress_bar = st.progress(1.0)
                show_output_image(image)
    if "upload_image" in st.session_state and use_output_image_as_input(update=False):
        st.markdown("If you want to use an output image as input image, please delete the uploaded image manually.")
