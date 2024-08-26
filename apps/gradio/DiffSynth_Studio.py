import gradio as gr
from diffsynth import ModelManager, SDImagePipeline, SDXLImagePipeline, SD3ImagePipeline, HunyuanDiTImagePipeline, FluxImagePipeline
import os, torch
from PIL import Image
import numpy as np


config = {
    "model_config": {
        "Stable Diffusion": {
            "model_folder": "models/stable_diffusion",
            "pipeline_class": SDImagePipeline,
            "default_parameters": {
                "cfg_scale": 7.0,
                "height": 512,
                "width": 512,
            }
        },
        "Stable Diffusion XL": {
            "model_folder": "models/stable_diffusion_xl",
            "pipeline_class": SDXLImagePipeline,
            "default_parameters": {
                "cfg_scale": 7.0,
            }
        },
        "Stable Diffusion 3": {
            "model_folder": "models/stable_diffusion_3",
            "pipeline_class": SD3ImagePipeline,
            "default_parameters": {
                "cfg_scale": 7.0,
            }
        },
        "Stable Diffusion XL Turbo": {
            "model_folder": "models/stable_diffusion_xl_turbo",
            "pipeline_class": SDXLImagePipeline,
            "default_parameters": {
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
            "default_parameters": {
                "cfg_scale": 7.0,
            }
        },
        "HunyuanDiT": {
            "model_folder": "models/HunyuanDiT",
            "pipeline_class": HunyuanDiTImagePipeline,
            "default_parameters": {
                "cfg_scale": 7.0,
            }
        },
        "FLUX": {
            "model_folder": "models/FLUX",
            "pipeline_class": FluxImagePipeline,
            "default_parameters": {
                "cfg_scale": 1.0,
            }
        }
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


def load_model(model_type, model_path):
    global model_dict
    model_key = f"{model_type}:{model_path}"
    if model_key in model_dict:
        return model_dict[model_key]
    model_path = os.path.join(config["model_config"][model_type]["model_folder"], model_path)
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
    pipe = config["model_config"][model_type]["pipeline_class"].from_model_manager(model_manager)
    while len(model_dict) + 1 > config["max_num_model_cache"]:
        key = next(iter(model_dict.keys()))
        model_manager_to_release, _ = model_dict[key]
        model_manager_to_release.to("cpu")
        del model_dict[key]
        torch.cuda.empty_cache()
    model_dict[model_key] = model_manager, pipe
    return model_manager, pipe


model_dict = {}

with gr.Blocks() as app:
    gr.Markdown("# DiffSynth-Studio Painter")
    with gr.Row():
        with gr.Column(scale=382, min_width=100):

            with gr.Accordion(label="Model"):
                model_type = gr.Dropdown(choices=[i for i in config["model_config"]], label="Model type")
                model_path = gr.Dropdown(choices=[], interactive=True, label="Model path")

                @gr.on(inputs=model_type, outputs=model_path, triggers=model_type.change)
                def model_type_to_model_path(model_type):
                    return gr.Dropdown(choices=load_model_list(model_type))
                
            with gr.Accordion(label="Prompt"):
                prompt = gr.Textbox(label="Prompt", lines=3)
                negative_prompt = gr.Textbox(label="Negative prompt", lines=1)
                cfg_scale = gr.Slider(minimum=1.0, maximum=10.0, value=7.0, step=0.1, interactive=True, label="Classifier-free guidance scale")
                embedded_guidance = gr.Slider(minimum=0.0, maximum=10.0, value=0.0, step=0.1, interactive=True, label="Embedded guidance scale (only for FLUX)")
            
            with gr.Accordion(label="Image"):
                num_inference_steps = gr.Slider(minimum=1, maximum=100, value=20, step=1, interactive=True, label="Inference steps")
                height = gr.Slider(minimum=64, maximum=2048, value=1024, step=64, interactive=True, label="Height")
                width = gr.Slider(minimum=64, maximum=2048, value=1024, step=64, interactive=True, label="Width")
                with gr.Column():
                    use_fixed_seed = gr.Checkbox(value=True, interactive=False, label="Use fixed seed")
                    seed = gr.Number(minimum=0, maximum=10**9, value=0, interactive=True, label="Random seed", show_label=False)

            @gr.on(
                inputs=[model_type, model_path, prompt, negative_prompt, cfg_scale, embedded_guidance, num_inference_steps, height, width],
                outputs=[prompt, negative_prompt, cfg_scale, embedded_guidance, num_inference_steps, height, width],
                triggers=model_path.change
            )
            def model_path_to_default_params(model_type, model_path, prompt, negative_prompt, cfg_scale, embedded_guidance, num_inference_steps, height, width):
                load_model(model_type, model_path)
                cfg_scale = config["model_config"][model_type]["default_parameters"].get("cfg_scale", cfg_scale)
                embedded_guidance = config["model_config"][model_type]["default_parameters"].get("embedded_guidance", embedded_guidance)
                num_inference_steps = config["model_config"][model_type]["default_parameters"].get("num_inference_steps", num_inference_steps)
                height = config["model_config"][model_type]["default_parameters"].get("height", height)
                width = config["model_config"][model_type]["default_parameters"].get("width", width)
                return prompt, negative_prompt, cfg_scale, embedded_guidance, num_inference_steps, height, width
                

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
                                                brush=gr.Brush(default_size=100, default_color="#000000", colors=["#000000"]),
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
                painter_background = gr.State(None)
                input_background = gr.State(None)
                @gr.on(
                    inputs=[model_type, model_path, prompt, negative_prompt, cfg_scale, embedded_guidance, num_inference_steps, height, width, seed] + enable_local_prompt_list + local_prompt_list + mask_scale_list + canvas_list,
                    outputs=[output_image],
                    triggers=run_button.click
                )
                def generate_image(model_type, model_path, prompt, negative_prompt, cfg_scale, embedded_guidance, num_inference_steps, height, width, seed, *args, progress=gr.Progress()):
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
                    input_params.update({
                        "local_prompts": local_prompts,
                        "masks": masks,
                        "mask_scales": mask_scales,
                    })
                    torch.manual_seed(seed)
                    image = pipe(**input_params)
                    return image
                
                @gr.on(inputs=[output_image] + canvas_list, outputs=canvas_list, triggers=output_to_painter_button.click)
                def send_output_to_painter_background(output_image, *canvas_list):
                    for canvas in canvas_list:
                        h, w = canvas["background"].shape[:2]
                        canvas["background"] = output_image.resize((w, h))
                    return tuple(canvas_list)
app.launch()
