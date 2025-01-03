import gradio as gr
from diffsynth import ModelManager, FluxImagePipeline, download_customized_models
import os, torch
from PIL import Image
import numpy as np
from PIL import ImageDraw, ImageFont
import random
import json

def save_mask_prompts(masks, mask_prompts, global_prompt, seed=0, random_dir='0000000'):
    save_dir = os.path.join('workdirs/tmp_mask', random_dir)
    os.makedirs(save_dir, exist_ok=True)
    for i, mask in enumerate(masks):
        save_path = os.path.join(save_dir, f'{i}.png')
        mask.save(save_path)
    sample = {
        "global_prompt": global_prompt,
        "mask_prompts": mask_prompts,
        "seed": seed,
    }
    with open(os.path.join(save_dir, f"prompts.json"), 'w') as f:
        json.dump(sample, f, indent=4)

def visualize_masks(image, masks, mask_prompts, font_size=35, use_random_colors=False):
    # Create a blank image for overlays
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    colors = [
        (165, 238, 173, 80),
        (76, 102, 221, 80),
        (221, 160, 77, 80),
        (204, 93, 71, 80),
        (145, 187, 149, 80),
        (134, 141, 172, 80),
        (157, 137, 109, 80),
        (153, 104, 95, 80),
        (165, 238, 173, 80),
        (76, 102, 221, 80),
        (221, 160, 77, 80),
        (204, 93, 71, 80),
        (145, 187, 149, 80),
        (134, 141, 172, 80),
        (157, 137, 109, 80),
        (153, 104, 95, 80),
    ]
    # Generate random colors for each mask
    if use_random_colors:
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 80) for _ in range(len(masks))]
    # Font settings
    try:
        font = ImageFont.truetype("arial", font_size)  # Adjust as needed
    except IOError:
        font = ImageFont.load_default(font_size)
    # Overlay each mask onto the overlay image
    for mask, mask_prompt, color in zip(masks, mask_prompts, colors):
        # Convert mask to RGBA mode
        mask_rgba = mask.convert('RGBA')
        mask_data = mask_rgba.getdata()
        new_data = [(color if item[:3] == (255, 255, 255) else (0, 0, 0, 0)) for item in mask_data]
        mask_rgba.putdata(new_data)
        # Draw the mask prompt text on the mask
        draw = ImageDraw.Draw(mask_rgba)
        mask_bbox = mask.getbbox()  # Get the bounding box of the mask
        text_position = (mask_bbox[0] + 10, mask_bbox[1] + 10)  # Adjust text position based on mask position
        draw.text(text_position, mask_prompt, fill=(255, 255, 255, 255), font=font)
        # Alpha composite the overlay with this mask
        overlay = Image.alpha_composite(overlay, mask_rgba)
    # Composite the overlay onto the original image
    result = Image.alpha_composite(image.convert('RGBA'), overlay)
    return result

config = {
    "model_config": {
        "FLUX": {
            "model_folder": "models/FLUX",
            "pipeline_class": FluxImagePipeline,
            "default_parameters": {
                "cfg_scale": 3.0,
                "embedded_guidance": 3.5,
                "num_inference_steps": 50,
            }
        },
    },
    "max_num_painter_layers": 8,
    "max_num_model_cache": 1,
}


def load_model_list(model_type):
    if model_type is None:
        return []
    folder = config["model_config"][model_type]["model_folder"]
    file_list = [i for i in os.listdir(folder) if i.endswith(".safetensors")]
    if model_type in ["HunyuanDiT", "Kolors", "FLUX"]:
        file_list += [i for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))]
    file_list = sorted(file_list)
    return file_list


model_dict = {}

def load_model(model_type, model_path):
    global model_dict
    model_key = f"{model_type}:{model_path}"
    if model_key in model_dict:
        return model_dict[model_key]
    model_path = os.path.join(config["model_config"][model_type]["model_folder"], model_path)
    model_manager = ModelManager()
    if model_type == "FLUX":
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda", model_id_list=["FLUX.1-dev"])
        model_manager.load_lora(
            download_customized_models(
                model_id="DiffSynth-Studio/Eligen",
                origin_file_path="model_bf16.safetensors",
                local_dir="models/lora/entity_control",
            ),
            lora_alpha=1,
        )
    else:
        model_manager.load_model(model_path)
    pipe = config["model_config"][model_type]["pipeline_class"].from_model_manager(model_manager)
    while len(model_dict) + 1 > config["max_num_model_cache"]:
        key = next(iter(model_dict.keys()))
        model_manager_to_release, _ = model_dict[key]
        model_manager_to_release.to("cpu")
        del model_dict[key]
        torch.cuda.empty_cache()
    model_dict[model_key] = model_manager, pipe
    return model_manager, pipe


with gr.Blocks() as app:
    gr.Markdown("""
    # 实体级控制文生图模型EliGen
    **UI说明**
    1. **点击Load model读取模型**，然后左侧界面为文生图输入参数；右侧Painter为局部控制区域绘制区域，每个局部控制条件由其Local prompt和绘制的mask组成，支持精准控制文生图和Inpainting两种模式。
    2. **精准控制生图模式：** 输入Globalprompt；激活并绘制一个或多个局部控制条件，点击Generate生成图像; Global Prompt推荐包含每个Local Prompt。
    3. **Inpainting模式：** 你可以上传图像，或者将上一步生成的图像设置为Inpaint Input Image，采用类似的方式输入局部控制条件，进行局部重绘。
    4. 尽情创造！
    """)
    gr.Markdown("""
    # Entity-Level Controlled Text-to-Image Model: EliGen
    **UI Instructions**
    1. **Click "Load model" to load the model.** The left interface is for text-to-image input parameters; the right "Painter" is the area for drawing local control regions. Each local control condition consists of its Local Prompt and the drawn mask, supporting both precise control text-to-image and Inpainting modes.
    2. **Precise Control Image Generation Mode:** Enter the Global Prompt; activate and draw one or more local control conditions, then click "Generate" to create the image. It is recommended that the Global Prompt includes all Local Prompts.
    3. **Inpainting Mode:** You can upload an image or set the image generated in the previous step as the "Inpaint Input Image." Use a similar method to input local control conditions for local redrawing.
    4. Enjoy!
    """)
    with gr.Row():
        random_mask_dir = gr.State('')
        with gr.Column(scale=382, min_width=100):
            model_type = gr.State('FLUX')
            model_path = gr.State('FLUX.1-dev')
            with gr.Accordion(label="Model"):
                load_model_button = gr.Button(value="Load model")

            with gr.Accordion(label="Global prompt"):
                prompt = gr.Textbox(label="Prompt", lines=3)
                negative_prompt = gr.Textbox(label="Negative prompt", value="worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,", lines=1)
                cfg_scale = gr.Slider(minimum=1.0, maximum=10.0, value=7.0, step=0.1, interactive=True, label="Classifier-free guidance scale")
                embedded_guidance = gr.Slider(minimum=0.0, maximum=10.0, value=0.0, step=0.1, interactive=True, label="Embedded guidance scale")
            
            with gr.Accordion(label="Inference Options"):
                num_inference_steps = gr.Slider(minimum=1, maximum=100, value=20, step=1, interactive=True, label="Inference steps")
                height = gr.Slider(minimum=64, maximum=2048, value=1024, step=64, interactive=True, label="Height")
                width = gr.Slider(minimum=64, maximum=2048, value=1024, step=64, interactive=True, label="Width")
                return_with_mask = gr.Checkbox(value=True, interactive=True, label="show result with mask painting")
                with gr.Column():
                    use_fixed_seed = gr.Checkbox(value=True, interactive=False, label="Use fixed seed")
                    seed = gr.Number(minimum=0, maximum=10**9, value=0, interactive=True, label="Random seed", show_label=False)
            with gr.Accordion(label="Inpaint Input Image (Testing)"):
                input_image = gr.Image(sources=None, show_label=False, interactive=True, type="pil")
                background_weight = gr.Slider(minimum=0.0, maximum=1000., value=0., step=1, interactive=False, label="background_weight")

                with gr.Column():
                    reset_input_button = gr.Button(value="Reset Inpaint Input")
                    send_input_to_painter = gr.Button(value="Set as painter's background")
                @gr.on(inputs=[input_image], outputs=[input_image], triggers=reset_input_button.click)
                def reset_input_image(input_image):
                    return None
            @gr.on(
                inputs=[model_type, model_path, prompt, negative_prompt, cfg_scale, embedded_guidance, num_inference_steps, height, width, return_with_mask],
                outputs=[prompt, negative_prompt, cfg_scale, embedded_guidance, num_inference_steps, height, width, return_with_mask, load_model_button, random_mask_dir],
                triggers=load_model_button.click
            )
            def model_path_to_default_params(model_type, model_path, prompt, negative_prompt, cfg_scale, embedded_guidance, num_inference_steps, height, width, return_with_mask):
                load_model(model_type, model_path)
                cfg_scale = config["model_config"][model_type]["default_parameters"].get("cfg_scale", cfg_scale)
                embedded_guidance = config["model_config"][model_type]["default_parameters"].get("embedded_guidance", embedded_guidance)
                num_inference_steps = config["model_config"][model_type]["default_parameters"].get("num_inference_steps", num_inference_steps)
                height = config["model_config"][model_type]["default_parameters"].get("height", height)
                width = config["model_config"][model_type]["default_parameters"].get("width", width)
                return_with_mask = config["model_config"][model_type]["default_parameters"].get("return_with_mask", return_with_mask)
                return prompt, negative_prompt, cfg_scale, embedded_guidance, num_inference_steps, height, width, return_with_mask, gr.update(value="Loaded FLUX"), gr.State(f'{random.randint(0, 1000000):08d}')


        with gr.Column(scale=618, min_width=100):
            with gr.Accordion(label="Painter"):
                enable_local_prompt_list = []
                local_prompt_list = []
                mask_scale_list = []
                canvas_list = []
                for painter_layer_id in range(config["max_num_painter_layers"]):
                    with gr.Tab(label=f"Layer {painter_layer_id}"):
                        enable_local_prompt = gr.Checkbox(label="Enable", value=False, key=f"enable_local_prompt_{painter_layer_id}")
                        local_prompt = gr.Textbox(label="Local prompt", key=f"local_prompt_{painter_layer_id}")
                        mask_scale = gr.Slider(minimum=0.0, maximum=5.0, value=1.0, step=0.1, interactive=True, label="Mask scale", key=f"mask_scale_{painter_layer_id}")
                        canvas = gr.ImageEditor(canvas_size=(512, 1), sources=None, layers=False, interactive=True, image_mode="RGBA",
                                                brush=gr.Brush(default_size=30, default_color="#000000", colors=["#000000"]),
                                                label="Painter", key=f"canvas_{painter_layer_id}")
                        @gr.on(inputs=[height, width, canvas], outputs=canvas, triggers=[height.change, width.change, canvas.clear, enable_local_prompt.change], show_progress="hidden")
                        def resize_canvas(height, width, canvas):
                            h, w = canvas["background"].shape[:2]
                            if h != height or width != w:
                                return np.ones((height, width, 3), dtype=np.uint8) * 255
                            else:
                                return canvas

                        enable_local_prompt_list.append(enable_local_prompt)
                        local_prompt_list.append(local_prompt)
                        mask_scale_list.append(mask_scale)
                        canvas_list.append(canvas)
            with gr.Accordion(label="Results"):
                run_button = gr.Button(value="Generate", variant="primary")
                output_image = gr.Image(sources=None, show_label=False, interactive=False, type="pil")
                with gr.Row():
                    with gr.Column():
                        output_to_painter_button = gr.Button(value="Set as painter's background")
                    with gr.Column():
                        output_to_input_button = gr.Button(value="Set as input image")
                real_output = gr.State(None)
                mask_out = gr.State(None)

                @gr.on(
                    inputs=[model_type, model_path, prompt, negative_prompt, cfg_scale, embedded_guidance, num_inference_steps, height, width, return_with_mask, seed, input_image, background_weight, random_mask_dir] + enable_local_prompt_list + local_prompt_list + mask_scale_list + canvas_list,
                    outputs=[output_image, real_output, mask_out],
                    triggers=run_button.click
                )
                def generate_image(model_type, model_path, prompt, negative_prompt, cfg_scale, embedded_guidance, num_inference_steps, height, width, return_with_mask, seed, input_image, background_weight, random_mask_dir, *args, progress=gr.Progress()):
                    _, pipe = load_model(model_type, model_path)
                    input_params = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "cfg_scale": cfg_scale,
                        "num_inference_steps": num_inference_steps,
                        "height": height,
                        "width": width,
                        "progress_bar_cmd": progress.tqdm,
                    }
                    if isinstance(pipe, FluxImagePipeline):
                        input_params["embedded_guidance"] = embedded_guidance
                    if input_image is not None:
                        input_params["input_image"] = input_image.resize((width, height)).convert("RGB")
                        input_params["enable_eligen_inpaint"] = True

                    enable_local_prompt_list, local_prompt_list, mask_scale_list, canvas_list = (
                        args[0 * config["max_num_painter_layers"]: 1 * config["max_num_painter_layers"]],
                        args[1 * config["max_num_painter_layers"]: 2 * config["max_num_painter_layers"]],
                        args[2 * config["max_num_painter_layers"]: 3 * config["max_num_painter_layers"]],
                        args[3 * config["max_num_painter_layers"]: 4 * config["max_num_painter_layers"]]
                    )
                    local_prompts, masks, mask_scales = [], [], []
                    for enable_local_prompt, local_prompt, mask_scale, canvas in zip(
                        enable_local_prompt_list, local_prompt_list, mask_scale_list, canvas_list
                    ):
                        if enable_local_prompt:
                            local_prompts.append(local_prompt)
                            masks.append(Image.fromarray(canvas["layers"][0][:, :, -1]).convert("RGB"))
                            mask_scales.append(mask_scale)
                    entity_masks = None if len(masks) == 0 else masks
                    entity_prompts = None if len(local_prompts) == 0 else local_prompts
                    input_params.update({
                        "eligen_entity_prompts": entity_prompts,
                        "eligen_entity_masks": entity_masks,
                    })
                    torch.manual_seed(seed)
                    image = pipe(**input_params)
                    # visualize masks
                    masks = [mask.resize(image.size) for mask in masks]
                    image_with_mask = visualize_masks(image, masks, local_prompts)
                    # save_mask_prompts(masks, local_prompts, prompt, seed, random_mask_dir.value)

                    real_output = gr.State(image)
                    mask_out = gr.State(image_with_mask)

                    if return_with_mask:
                        return image_with_mask, real_output, mask_out
                    return image, real_output, mask_out
                
                @gr.on(inputs=[input_image] + canvas_list, outputs=canvas_list, triggers=send_input_to_painter.click)
                def send_input_to_painter_background(input_image, *canvas_list):
                    if input_image is None:
                        return tuple(canvas_list)
                    for canvas in canvas_list:
                        h, w = canvas["background"].shape[:2]
                        canvas["background"] = input_image.resize((w, h))
                    return tuple(canvas_list)
                @gr.on(inputs=[real_output] + canvas_list, outputs=canvas_list, triggers=output_to_painter_button.click)
                def send_output_to_painter_background(real_output, *canvas_list):
                    if real_output is None:
                        return tuple(canvas_list)
                    for canvas in canvas_list:
                        h, w = canvas["background"].shape[:2]
                        canvas["background"] = real_output.value.resize((w, h))
                    return tuple(canvas_list)
                @gr.on(inputs=[return_with_mask, real_output, mask_out], outputs=[output_image], triggers=[return_with_mask.change], show_progress="hidden")
                def show_output(return_with_mask, real_output, mask_out):
                    if return_with_mask:
                        return mask_out.value
                    else:
                        return real_output.value
                @gr.on(inputs=[real_output], outputs=[input_image], triggers=output_to_input_button.click)
                def send_output_to_pipe_input(real_output):
                    return real_output.value

app.launch()
