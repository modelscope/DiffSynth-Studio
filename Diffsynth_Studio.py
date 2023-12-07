import torch, os, io
import numpy as np
from PIL import Image
import streamlit as st
st.set_page_config(layout="wide")
from streamlit_drawable_canvas import st_canvas
from diffsynth.models import ModelManager
from diffsynth.prompts import SDXLPrompter, SDPrompter
from diffsynth.pipelines import SDXLPipeline, SDPipeline


torch.cuda.set_per_process_memory_fraction(0.999, 0)


@st.cache_data
def load_model_list(folder):
    file_list = os.listdir(folder)
    file_list = [i for i in file_list if i.endswith(".safetensors")]
    file_list = sorted(file_list)
    return file_list


def detect_model_path(sd_model_path, sdxl_model_path):
    if sd_model_path != "None":
        model_path = os.path.join("models/stable_diffusion", sd_model_path)
    elif sdxl_model_path != "None":
        model_path = os.path.join("models/stable_diffusion_xl", sdxl_model_path)
    else:
        model_path = None
    return model_path


def load_model(sd_model_path, sdxl_model_path):
    if sd_model_path != "None":
        model_path = os.path.join("models/stable_diffusion", sd_model_path)
        model_manager = ModelManager()
        model_manager.load_from_safetensors(model_path)
        prompter = SDPrompter()
        pipeline = SDPipeline()
    elif sdxl_model_path != "None":
        model_path = os.path.join("models/stable_diffusion_xl", sdxl_model_path)
        model_manager = ModelManager()
        model_manager.load_from_safetensors(model_path)
        prompter = SDXLPrompter()
        pipeline = SDXLPipeline()
    else:
        return None, None, None, None
    return model_path, model_manager, prompter, pipeline


def release_model():
    if "model_manager" in st.session_state:
        st.session_state["model_manager"].to("cpu")
        del st.session_state["loaded_model_path"]
        del st.session_state["model_manager"]
        del st.session_state["prompter"]
        del st.session_state["pipeline"]
        torch.cuda.empty_cache()


def use_output_image_as_input():
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
    if selected_output_image is not None:
        st.session_state["input_image"] = selected_output_image


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

# with column_input:
with st.sidebar:
    # Select a model
    with st.expander("Model", expanded=True):
        sd_model_list = ["None"] + load_model_list("models/stable_diffusion")
        sd_model_path = st.selectbox(
            "Stable Diffusion", sd_model_list
        )
        sdxl_model_list = ["None"] + load_model_list("models/stable_diffusion_xl")
        sdxl_model_path = st.selectbox(
            "Stable Diffusion XL", sdxl_model_list
        )

        # Load the model
        model_path = detect_model_path(sd_model_path, sdxl_model_path)
        if model_path is None:
            st.markdown("No models selected.")
            release_model()
        elif st.session_state.get("loaded_model_path", "") != model_path:
            st.markdown(f"Using model at {model_path}.")
            release_model()
            model_path, model_manager, prompter, pipeline = load_model(sd_model_path, sdxl_model_path)
            st.session_state.loaded_model_path = model_path
            st.session_state.model_manager = model_manager
            st.session_state.prompter = prompter
            st.session_state.pipeline = pipeline
        else:
            st.markdown(f"Using model at {model_path}.")
            model_path, model_manager, prompter, pipeline = (
                st.session_state.loaded_model_path,
                st.session_state.model_manager,
                st.session_state.prompter,
                st.session_state.pipeline,
            )

    # Show parameters
    with st.expander("Prompt", expanded=True):
        column_positive, column_negative = st.columns(2)
        prompt = st.text_area("Positive prompt")
        negative_prompt = st.text_area("Negative prompt")
    with st.expander("Classifier-free guidance", expanded=True):
        use_cfg = st.checkbox("Use classifier-free guidance", value=True)
        if use_cfg:
            cfg_scale = st.slider("Classifier-free guidance scale", min_value=1.0, max_value=10.0, step=0.1, value=7.5)
        else:
            cfg_scale = 1.0
    with st.expander("Inference steps", expanded=True):
        num_inference_steps = st.slider("Inference steps", min_value=1, max_value=100, value=20, label_visibility="hidden")
    with st.expander("Image size", expanded=True):
        height = st.select_slider("Height", options=[256, 512, 768, 1024, 2048], value=512)
        width = st.select_slider("Width", options=[256, 512, 768, 1024, 2048], value=512)
    with st.expander("Seed", expanded=True):
        use_fixed_seed = st.checkbox("Use fixed seed", value=False)
        if use_fixed_seed:
            seed = st.number_input("Random seed", value=0, label_visibility="hidden")
    with st.expander("Number of images", expanded=True):
        num_images = st.number_input("Number of images", value=4, label_visibility="hidden")
    with st.expander("Tile (for high resolution)", expanded=True):
        tiled = st.checkbox("Use tile", value=False)
        tile_size = st.select_slider("Tile size", options=[64, 128], value=64)
        tile_stride = st.select_slider("Tile stride", options=[8, 16, 32, 64], value=32)


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
            st.session_state["input_image"] = Image.open(upload_image)
        elif create_white_board:
            st.session_state["input_image"] = Image.fromarray(np.ones((1024, 1024, 3), dtype=np.uint8) * 255)
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


with column_output:
    run_button = st.button("Generate image", type="primary")
    auto_update = st.checkbox("Auto update", value=False)
    num_image_columns = st.slider("Columns", min_value=1, max_value=8, value=2)
    image_columns = st.columns(num_image_columns)

    # Run
    if (run_button or auto_update) and model_path is not None:

        if not use_fixed_seed:
            torch.manual_seed(np.random.randint(0, 10**9))

        output_images = []
        for image_id in range(num_images):
            if use_fixed_seed:
                torch.manual_seed(seed + image_id)
            if input_image is not None:
                input_image = input_image.resize((width, height))
                if canvas_result.image_data is not None:
                    input_image = apply_stroke_to_image(canvas_result.image_data, input_image)
            else:
                denoising_strength = 1.0
            with image_columns[image_id % num_image_columns]:
                progress_bar = st.progress(0.0)
                image = pipeline(
                    model_manager, prompter,
                    prompt, negative_prompt=negative_prompt, cfg_scale=cfg_scale,
                    num_inference_steps=num_inference_steps,
                    height=height, width=width,
                    init_image=input_image, denoising_strength=denoising_strength,
                    tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
                    progress_bar_st=progress_bar
                )
                output_images.append(image)
                progress_bar.progress(1.0)
                show_output_image(image)
                st.session_state["output_images"] = output_images

    elif "output_images" in st.session_state:
        for image_id in range(len(st.session_state.output_images)):
            with image_columns[image_id % num_image_columns]:
                image = st.session_state.output_images[image_id]
                progress_bar = st.progress(1.0)
                show_output_image(image)
