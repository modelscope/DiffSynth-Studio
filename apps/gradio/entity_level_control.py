import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import json
import gradio as gr
from diffsynth import ModelManager, FluxImagePipeline, download_customized_models
from modelscope import dataset_snapshot_download


dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth", local_dir="./", allow_file_pattern=f"data/examples/eligen/entity_control/*")
example_json = 'data/examples/eligen/entity_control/ui_examples.json'
with open(example_json, 'r') as f:
    examples = json.load(f)['examples']

for idx in range(len(examples)):
    example_id = examples[idx]['example_id']
    entity_prompts = examples[idx]['local_prompt_list']
    examples[idx]['mask_lists'] = [Image.open(f"data/examples/eligen/entity_control/example_{example_id}/{i}.png").convert('RGB') for i in range(len(entity_prompts))]

def create_canvas_data(background, masks):
    if background.shape[-1] == 3:
        background = np.dstack([background, np.full(background.shape[:2], 255, dtype=np.uint8)])
    layers = []
    for mask in masks:
        if mask is not None:
            mask_single_channel = mask if mask.ndim == 2 else mask[..., 0]
            layer = np.zeros((mask_single_channel.shape[0], mask_single_channel.shape[1], 4), dtype=np.uint8)
            layer[..., -1] = mask_single_channel
            layers.append(layer)
        else:
            layers.append(np.zeros_like(background))

    composite = background.copy()
    for layer in layers:
        if layer.size > 0:
            composite = np.where(layer[..., -1:] > 0, layer, composite)
    return {
        "background": background,
        "layers": layers,
        "composite": composite,
    }

def load_example(load_example_button):
    example_idx = int(load_example_button.split()[-1]) - 1
    example = examples[example_idx]
    result = [
        50,
        example["global_prompt"],
        example["negative_prompt"],
        example["seed"],
        *example["local_prompt_list"],
    ]
    num_entities = len(example["local_prompt_list"])
    result += [""] * (config["max_num_painter_layers"] - num_entities)
    masks = []
    for mask in example["mask_lists"]:
        mask_single_channel = np.array(mask.convert("L"))
        masks.append(mask_single_channel)
    for _ in range(config["max_num_painter_layers"] - len(masks)):
        blank_mask = np.zeros_like(masks[0]) if masks else np.zeros((512, 512), dtype=np.uint8)
        masks.append(blank_mask)
    background = np.ones((masks[0].shape[0], masks[0].shape[1], 4), dtype=np.uint8) * 255
    canvas_data_list = []
    for mask in masks:
        canvas_data = create_canvas_data(background, [mask])
        canvas_data_list.append(canvas_data)
    result.extend(canvas_data_list)
    return result

def save_mask_prompts(masks, mask_prompts, global_prompt, seed=0, random_dir='0000000'):
    save_dir = os.path.join('workdirs/tmp_mask', random_dir)
    print(f'save to {save_dir}')
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
        if mask is None:
            continue
        # Convert mask to RGBA mode
        mask_rgba = mask.convert('RGBA')
        mask_data = mask_rgba.getdata()
        new_data = [(color if item[:3] == (255, 255, 255) else (0, 0, 0, 0)) for item in mask_data]
        mask_rgba.putdata(new_data)
        # Draw the mask prompt text on the mask
        draw = ImageDraw.Draw(mask_rgba)
        mask_bbox = mask.getbbox()  # Get the bounding box of the mask
        if mask_bbox is None:
            continue
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
                "num_inference_steps": 30,
            }
        },
    },
    "max_num_painter_layers": 8,
    "max_num_model_cache": 1,
}

model_dict = {}

def load_model(model_type='FLUX', model_path='FLUX.1-dev'):
    global model_dict
    model_key = f"{model_type}:{model_path}"
    if model_key in model_dict:
        return model_dict[model_key]
    model_path = os.path.join(config["model_config"][model_type]["model_folder"], model_path)
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda", model_id_list=["FLUX.1-dev"])
    model_manager.load_lora(
        download_customized_models(
            model_id="DiffSynth-Studio/Eligen",
            origin_file_path="model_bf16.safetensors",
            local_dir="models/lora/entity_control",
        ),
        lora_alpha=1,
    )
    pipe = config["model_config"][model_type]["pipeline_class"].from_model_manager(model_manager)
    model_dict[model_key] = model_manager, pipe
    return model_manager, pipe


with gr.Blocks() as app:
    gr.Markdown(
        """## EliGen: Entity-Level Controllable Text-to-Image Model
                1. On the left, input the **global prompt** for the overall image, such as "a person stands by the river."
                2. On the right, input the **local prompt** for each entity, such as "person," and draw the corresponding mask in the **Entity Mask Painter**. Generally, solid rectangular masks yield better results.
                3. Click the **Generate** button to create the image. By selecting different **random seeds**, you can generate diverse images.
                4. **You can directly click the "Load Example" button on any sample at the bottom to load example inputs.**
                """
    )

    loading_status = gr.Textbox(label="Loading Model...", value="Loading model... Please wait...", visible=True)
    main_interface = gr.Column(visible=False)

    def initialize_model():
        try:
            load_model()
            return {
                loading_status: gr.update(value="Model loaded successfully!", visible=False),
                main_interface: gr.update(visible=True),
            }
        except Exception as e:
            print(f'Failed to load model with error: {e}')
            return {
                loading_status: gr.update(value=f"Failed to load model: {str(e)}", visible=True),
                main_interface: gr.update(visible=True),
            }

    app.load(initialize_model, inputs=None, outputs=[loading_status, main_interface])

    with main_interface:
        with gr.Row():
            local_prompt_list = []
            canvas_list = []
            random_mask_dir = gr.State(f'{random.randint(0, 1000000):08d}')
            with gr.Column(scale=382, min_width=100):
                model_type = gr.State('FLUX')
                model_path = gr.State('FLUX.1-dev')
                with gr.Accordion(label="Global prompt"):
                    prompt = gr.Textbox(label="Global Prompt", lines=3)
                    negative_prompt = gr.Textbox(label="Negative prompt", value="worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw, blur,", lines=3)
                with gr.Accordion(label="Inference Options", open=True):
                    seed = gr.Number(minimum=0, maximum=10**9, value=42, interactive=True, label="Random seed", show_label=True)
                    num_inference_steps = gr.Slider(minimum=1, maximum=100, value=30, step=1, interactive=True, label="Inference steps")
                    cfg_scale = gr.Slider(minimum=2.0, maximum=10.0, value=3.0, step=0.1, interactive=True, label="Classifier-free guidance scale")
                    embedded_guidance = gr.Slider(minimum=0.0, maximum=10.0, value=3.5, step=0.1, interactive=True, label="Embedded guidance scale")
                    height = gr.Slider(minimum=64, maximum=2048, value=1024, step=64, interactive=True, label="Height")
                    width = gr.Slider(minimum=64, maximum=2048, value=1024, step=64, interactive=True, label="Width")
                with gr.Accordion(label="Inpaint Input Image", open=False):
                    input_image = gr.Image(sources=None, show_label=False, interactive=True, type="pil")
                    background_weight = gr.Slider(minimum=0.0, maximum=1000., value=0., step=1, interactive=False, label="background_weight", visible=False)

                    with gr.Column():
                        reset_input_button = gr.Button(value="Reset Inpaint Input")
                        send_input_to_painter = gr.Button(value="Set as painter's background")
                    @gr.on(inputs=[input_image], outputs=[input_image], triggers=reset_input_button.click)
                    def reset_input_image(input_image):
                        return None

            with gr.Column(scale=618, min_width=100):
                with gr.Accordion(label="Entity Painter"):
                    for painter_layer_id in range(config["max_num_painter_layers"]):
                        with gr.Tab(label=f"Entity {painter_layer_id}"):
                            local_prompt = gr.Textbox(label="Local prompt", key=f"local_prompt_{painter_layer_id}")
                            canvas = gr.ImageEditor(
                                canvas_size=(512, 512),
                                sources=None,
                                layers=False,
                                interactive=True,
                                image_mode="RGBA",
                                brush=gr.Brush(
                                    default_size=50,
                                    default_color="#000000",
                                    colors=["#000000"],
                                ),
                                label="Entity Mask Painter",
                                key=f"canvas_{painter_layer_id}",
                                width=width,
                                height=height,
                            )
                            @gr.on(inputs=[height, width, canvas], outputs=canvas, triggers=[height.change, width.change, canvas.clear], show_progress="hidden")
                            def resize_canvas(height, width, canvas):
                                h, w = canvas["background"].shape[:2]
                                if h != height or width != w:
                                    return np.ones((height, width, 3), dtype=np.uint8) * 255
                                else:
                                    return canvas
                            local_prompt_list.append(local_prompt)
                            canvas_list.append(canvas)
                with gr.Accordion(label="Results"):
                    run_button = gr.Button(value="Generate", variant="primary")
                    output_image = gr.Image(sources=None, show_label=False, interactive=False, type="pil")
                    with gr.Row():
                        with gr.Column():
                            output_to_painter_button = gr.Button(value="Set as painter's background")
                        with gr.Column():
                            return_with_mask = gr.Checkbox(value=False, interactive=True, label="show result with mask painting")
                            output_to_input_button = gr.Button(value="Set as input image", visible=False, interactive=False)
                    real_output = gr.State(None)
                    mask_out = gr.State(None)

                    @gr.on(
                        inputs=[model_type, model_path, prompt, negative_prompt, cfg_scale, embedded_guidance, num_inference_steps, height, width, return_with_mask, seed, input_image, background_weight, random_mask_dir] + local_prompt_list + canvas_list,
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

                        local_prompt_list, canvas_list = (
                            args[0 * config["max_num_painter_layers"]: 1 * config["max_num_painter_layers"]],
                            args[1 * config["max_num_painter_layers"]: 2 * config["max_num_painter_layers"]],
                        )
                        local_prompts, masks = [], []
                        for local_prompt, canvas in zip(local_prompt_list, canvas_list):
                            if isinstance(local_prompt, str) and len(local_prompt) > 0:
                                local_prompts.append(local_prompt)
                                masks.append(Image.fromarray(canvas["layers"][0][:, :, -1]).convert("RGB"))
                        entity_masks = None if len(masks) == 0 else masks
                        entity_prompts = None if len(local_prompts) == 0 else local_prompts
                        input_params.update({
                            "eligen_entity_prompts": entity_prompts,
                            "eligen_entity_masks": entity_masks,
                        })
                        torch.manual_seed(seed)
                        # save_mask_prompts(masks, local_prompts, prompt, seed, random_mask_dir)
                        image = pipe(**input_params)
                        masks = [mask.resize(image.size) for mask in masks]
                        image_with_mask = visualize_masks(image, masks, local_prompts)

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

        with gr.Column():
            gr.Markdown("## Examples")
            for i in range(0, len(examples), 2):
                with gr.Row():
                    if i < len(examples):
                        example = examples[i]
                        with gr.Column():
                            example_image = gr.Image(
                                value=f"data/examples/eligen/entity_control/example_{example['example_id']}/example_image.png",
                                label=example["description"],
                                interactive=False,
                                width=1024,
                                height=512
                            )
                            load_example_button = gr.Button(value=f"Load Example {example['example_id']}")
                            load_example_button.click(
                                load_example,
                                inputs=[load_example_button],
                                outputs=[num_inference_steps, prompt, negative_prompt, seed] + local_prompt_list + canvas_list
                            )

                    if i + 1 < len(examples):
                        example = examples[i + 1]
                        with gr.Column():
                            example_image = gr.Image(
                                value=f"data/examples/eligen/entity_control/example_{example['example_id']}/example_image.png",
                                label=example["description"],
                                interactive=False,
                                width=1024,
                                height=512
                            )
                            load_example_button = gr.Button(value=f"Load Example {example['example_id']}")
                            load_example_button.click(
                                load_example,
                                inputs=[load_example_button],
                                outputs=[num_inference_steps, prompt, negative_prompt, seed] + local_prompt_list + canvas_list
                            )
app.config["show_progress"] = "hidden"
app.launch()
