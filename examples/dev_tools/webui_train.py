import os, importlib.util, argparse, pkgutil, inspect
from dataclasses import dataclass
import streamlit as st
st.set_page_config(layout="wide")

available_data_file_keys = ["animate_face_video", "animate_pose_video", "audio", "blockwise_controlnet_image", "blockwise_controlnet_inpaint_mask", "context_image", "control_video", "controlnet_image", "controlnet_inpaint_mask", "edit_image", "eligen_entity_masks", "image", "in_context_videos", "infinityou_id_image", "input_audio", "ipadapter_images", "kontext_images", "layer_input_image", "nexus_gen_reference_image", "reference_image", "s2v_pose_video", "step1x_reference_image", "vace_reference_image", "vace_video", "vap_video", "video", "wantodance_keyframes", "wantodance_music_path", "wantodance_reference_image"]
available_extra_inputs = ["animate_face_video", "animate_pose_video", "blockwise_controlnet_image", "blockwise_controlnet_inpaint_mask", "camera_control_direction", "camera_control_speed", "cfg_scale", "context_image", "control_video", "controlnet_image", "controlnet_inpaint_mask", "controlnet_processor_id", "edit_image", "eligen_entity_masks", "eligen_entity_prompts", "end_image", "frame_rate", "framewise_decoding", "in_context_downsample_factor", "in_context_videos", "infinityou_guidance", "infinityou_id_image", "input_audio", "input_image", "ipadapter_images", "kontext_images", "layer_input_image", "layer_num", "lora_encoder_inputs", "motion_bucket_id", "nexus_gen_reference_image", "num_inference_steps", "rand_device", "reference_image", "s2v_pose_video", "seed", "step1x_reference_image", "template_inputs", "vace_reference_image", "vace_video", "value_controller_inputs", "vap_video", "wantodance_fps", "wantodance_keyframes", "wantodance_keyframes_mask", "wantodance_music_path", "wantodance_reference_image"]
available_model_components = ["animate_adapter", "audio_dit", "audio_encoder", "audio_vae", "audio_vae_decoder", "audio_vae_encoder", "audio_vocoder", "blockwise_controlnet", "conditioner", "controlnet", "dinov3_image_encoder", "dit", "dit2", "dual_tower_bridge", "image2lora_coarse", "image2lora_fine", "image2lora_style", "image_encoder", "image_proj_model", "infinityou_processor", "ipadapter", "ipadapter_image_encoder", "lora_encoder", "lora_patcher", "motion_controller", "nexus_gen", "nexus_gen_editing_adapter", "nexus_gen_generation_adapter", "qwenvl", "siglip2_image_encoder", "step1x_connector", "text_encoder", "text_encoder_1", "text_encoder_2", "text_encoder_post_modules", "text_encoder_qwen3", "tokenizer_model", "tokenizer_t5xxl", "unet", "upsampler", "vace", "vace2", "vae", "vae_decoder", "vae_encoder", "value_controller", "vap", "video_dit", "video_dit2", "video_vae", "video_vae_decoder", "video_vae_encoder"]

@dataclass
class Parameter:
    name: str = None
    dtype: type = None
    value: any = None
    required: bool = False
    choices: list = None
    help: str = None

def parse_available_pipelines():
    from diffsynth.diffusion.base_pipeline import BasePipeline
    import diffsynth.pipelines as _pipelines_pkg
    available_pipelines = {}
    for _, name, _ in pkgutil.iter_modules(_pipelines_pkg.__path__):
        mod = importlib.import_module(f"diffsynth.pipelines.{name}")
        classes = {
            cls_name: cls for cls_name, cls in inspect.getmembers(mod, inspect.isclass)
            if issubclass(cls, BasePipeline) and cls is not BasePipeline and cls.__module__ == mod.__name__
        }
        available_pipelines.update(classes)
    return available_pipelines

def search_for_options(name):
    files = search_for_files("examples", ".sh")
    params = set()
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if f"--{name}" in line and not line.startswith("#"):
                    line = line.strip()
                    line = line.replace(" \\", "").replace(f"--{name}", "").replace('"', "").replace(" ", "")
                    for param in line.split(","):
                        params.add(param)
    for param in sorted(list(params)):
        print(f'"{param}", ', end="")
    print()

def search_for_available_pipeline_options():
    from diffsynth.diffusion.base_pipeline import BasePipeline
    pipeline_classes = parse_available_pipelines()
    base_attrs = set(vars(BasePipeline()))
    black_list = ["tokenizer", "processor", "tokenizer_1", "tokenizer_2", "audio_processor"]
    options = []
    for pipeline_class in pipeline_classes:
        pipe = pipeline_classes[pipeline_class]()
        members = [attr for attr in vars(pipe) if not attr.startswith("__") and attr not in base_attrs and getattr(pipe, attr) is None]
        members = [attr for attr in members if attr not in black_list]
        options.extend(members)
    options = sorted(list(set(options)))
    for option in sorted(list(options)):
        print(f'"{option}", ', end="")
    print()

def parse_available_training_scripts(path):
    training_scripts = {}
    for folder in os.listdir(path):
        if os.path.isfile(f"{path}/{folder}/model_training/train.py"):
            training_scripts[folder] = f"{path}/{folder}/model_training/train.py"
    return training_scripts

def search_for_files(path, suffix):
    if os.path.isfile(path):
        if path.endswith(suffix): return [path]
        return []
    else:
        files = []
        for sub_path in os.listdir(path):
            files.extend(search_for_files(os.path.join(path, sub_path), suffix))
        return files

def parse_available_examples(path):
    path = os.path.dirname(path)
    examples = search_for_files(path, ".sh")
    return examples

def parse_example(example_path):
    value_dict = {}
    with open(example_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("#"):
                continue
            if not line.startswith("--"):
                continue
            line = line.replace("\\", "").strip()
            if " " in line:
                name, value = line[2:line.index(" ")], line[line.index(" ") + 1:]
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
            else:
                name, value = line[2:], True
            value_dict[name] = value
    return value_dict

def parse_parser(path):
    spec = importlib.util.spec_from_file_location("train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in dir(module):
        if name.endswith("parser") and callable(getattr(module, name)):
            return getattr(module, name)
    return None

def parse_parser_action(action, value=None):
    if isinstance(action, argparse._StoreTrueAction) or isinstance(action, argparse._StoreFalseAction):
        dtype = bool
    else:
        dtype = action.type
    param = Parameter(
        name=action.dest,
        dtype=dtype,
        value=action.default if value is None else dtype(value),
        required=action.required,
        choices=action.choices,
        help=action.help,
    )
    return param

def parse_parser_actions(parser, example_path=None):
    value_dict = {} if example_path is None or example_path == "None" else parse_example(example_path)
    params = []
    for action in parser._actions:
        param = parse_parser_action(action, value=value_dict.get(action.dest))
        if param.name == "help":
            continue
        params.append(param)
    return params

def draw_model_id_with_origin_paths(param, disabled=False):
    with st.container(border=True):
        st.markdown(param.name, help=param.help)
        model_id_with_origin_paths = [] if param.value is None else param.value.split(",")
        num = st.number_input(f"Number of models", min_value=0, max_value=20, value=len(model_id_with_origin_paths), disabled=disabled)
        result = []
        for i in range(num):
            col1, col2 = st.columns(2)
            value = model_id_with_origin_paths[i].split(":") if i < len(model_id_with_origin_paths) else (None, None)
            with col1:
                model_id = st.text_input("model_id", value=value[0], key=f"model_id_{i}", disabled=disabled)
            with col2:
                origin_file_pattern = st.text_input("origin_file_pattern", value=value[1], key=f"origin_file_pattern_{i}", disabled=disabled)
            result.append(f"{model_id}:{origin_file_pattern}")
        result = ",".join(result)
    return result

def draw_parameter(param, check_enable=True, disabled=False):
    if check_enable and param.value is None:
        with st.container(border=True):
            enable_button = st.checkbox(f"Enable {param.name}", value=False, disabled=disabled)
            ui = draw_parameter(param, check_enable=False, disabled=disabled or not enable_button)
            if enable_button:
                return ui
            else:
                return None
    if param.name == "data_file_keys":
        ui = st.multiselect(param.name, options=available_data_file_keys, accept_new_options=True, default=param.value.split(","), disabled=disabled, help=param.help)
        ui = ",".join(ui)
    elif param.name == "model_paths":
        ui = st.text_area(param.name, value=param.value, height=3, disabled=disabled, help=param.help)
    elif param.name == "model_id_with_origin_paths":
        ui = draw_model_id_with_origin_paths(param, disabled=disabled)
    elif param.name == "extra_inputs":
        value = None if param.value is None else param.value.split(",")
        ui = st.multiselect(param.name, options=available_extra_inputs, accept_new_options=True, default=value, disabled=disabled, help=param.help)
        ui = ",".join(ui)
    elif param.name in ["fp8_models", "offload_models", "trainable_models", "lora_base_model", "preset_lora_model"]:
        value = None if param.value is None else param.value.split(",")
        ui = st.multiselect(param.name, options=available_model_components, accept_new_options=True, default=value, disabled=disabled, help=param.help)
        ui = ",".join(ui)
    elif param.name == "learning_rate":
        ui = st.number_input(param.name, value=param.value, format="%0.7f", step=1e-4, disabled=disabled, help=param.help)
    elif param.dtype == str:
        ui = st.text_input(param.name, value=param.value, disabled=disabled, help=param.help)
    elif param.dtype == int:
        ui = st.number_input(param.name, value=param.value, step=1, disabled=disabled, help=param.help)
    elif param.dtype == float:
        ui = st.number_input(param.name, value=param.value, disabled=disabled, help=param.help)
    elif param.dtype == bool:
        ui = st.checkbox(param.name, value=param.value, disabled=disabled, help=param.help)
    else:
        st.markdown(f"(`{param.name}` is not not configurable in WebUI). dtype: `{param.dtype}`.")
        ui = None
    return ui

def draw_dataset_configs(dataset_base_path, dataset_metadata_path, dataset_repeat, dataset_num_workers, data_file_keys):
    dataset_base_path = draw_parameter(dataset_base_path)
    dataset_metadata_path = draw_parameter(dataset_metadata_path)
    col_1, col_2 = st.columns(2)
    with col_1:
        dataset_repeat = draw_parameter(dataset_repeat)
    with col_2:
        dataset_num_workers = draw_parameter(dataset_num_workers)
    data_file_keys = draw_parameter(data_file_keys)
    params = {
        "dataset_base_path": dataset_base_path,
        "dataset_metadata_path": dataset_metadata_path,
        "dataset_repeat": dataset_repeat,
    }
    if dataset_num_workers > 0: params["dataset_num_workers"] = dataset_num_workers
    params["data_file_keys"] = data_file_keys
    return params

def draw_image_size(height, width, max_pixels):
    mode = st.selectbox("Image scaling and cropping", options=["Scale if pixel count exceeds threshold", "Resize to a fixed size and crop"], index=int(max_pixels.value is None))
    use_max_pixel = mode == "Scale if pixel count exceeds threshold"
    use_height_width = mode == "Resize to a fixed size and crop"
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            max_pixels = draw_parameter(max_pixels, disabled=not use_max_pixel)
    with col2:
        with st.container(border=True):
            height = draw_parameter(height, disabled=not use_height_width, check_enable=False)
            width = draw_parameter(width, disabled=not use_height_width, check_enable=False)
    if use_max_pixel:
        return {"max_pixels": max_pixels}
    else:
        return {"height": height, "width": width}

def draw_model_configs(model_paths, model_id_with_origin_paths, extra_inputs, fp8_models, offload_models):
    model_id_with_origin_paths = draw_parameter(model_id_with_origin_paths)
    model_paths = draw_parameter(model_paths)
    extra_inputs = draw_parameter(extra_inputs)
    fp8_models = draw_parameter(fp8_models)
    offload_models = draw_parameter(offload_models)
    params = {}
    if model_paths is not None: params["model_paths"] = model_paths
    if model_id_with_origin_paths is not None: params["model_id_with_origin_paths"] = model_id_with_origin_paths
    if extra_inputs is not None: params["extra_inputs"] = extra_inputs
    if fp8_models is not None: params["fp8_models"] = fp8_models
    if offload_models is not None: params["offload_models"] = offload_models
    return params

def draw_video_size(height, width, max_pixels, num_frames):
    mode = st.selectbox("Video scaling and cropping", options=["Scale if pixel count exceeds threshold", "Resize to a fixed size and crop"], index=int(max_pixels.value is not None))
    use_max_pixel = mode == "Scale if pixel count exceeds threshold"
    use_height_width = mode == "Resize to a fixed size and crop"
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            max_pixels = draw_parameter(max_pixels, disabled=not use_max_pixel)
    with col2:
        with st.container(border=True):
            height = draw_parameter(height, disabled=not use_height_width)
            width = draw_parameter(width, disabled=not use_height_width)
    num_frames = draw_parameter(num_frames)
    if use_max_pixel:
        return {"max_pixels": max_pixels, "num_frames": num_frames}
    else:
        return {"height": height, "width": width, "num_frames": num_frames}

def draw_training_configs(learning_rate, num_epochs, trainable_models, find_unused_parameters, weight_decay, task):
    learning_rate = draw_parameter(learning_rate)
    num_epochs = draw_parameter(num_epochs)
    trainable_models = draw_parameter(trainable_models)
    weight_decay = draw_parameter(weight_decay)
    task = draw_parameter(task)
    find_unused_parameters = draw_parameter(find_unused_parameters)
    params = {
        "task": task,
        "find_unused_parameters": find_unused_parameters,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
    }
    if weight_decay != 0.01: params["weight_decay"] = weight_decay
    if trainable_models is not None: params["trainable_models"] = trainable_models
    return params

def draw_output_configs(output_path, remove_prefix_in_ckpt, save_steps):
    output_path = draw_parameter(output_path)
    remove_prefix_in_ckpt = draw_parameter(remove_prefix_in_ckpt)
    save_steps = draw_parameter(save_steps)
    params = {
        "output_path": output_path,
        "remove_prefix_in_ckpt": remove_prefix_in_ckpt,
    }
    if save_steps is not None: params["save_steps"] = save_steps
    return params

def draw_lora_configs(lora_base_model, lora_target_modules, lora_rank, lora_checkpoint, preset_lora_path, preset_lora_model):
    with st.container(border=True):
        train_lora = st.checkbox("Train LoRA", value=lora_base_model is not None)
        lora_base_model = draw_parameter(lora_base_model, check_enable=False, disabled=not train_lora)
        lora_target_modules = draw_parameter(lora_target_modules, check_enable=False, disabled=not train_lora)
        lora_rank = draw_parameter(lora_rank, check_enable=False, disabled=not train_lora)
        lora_checkpoint = draw_parameter(lora_checkpoint, check_enable=False, disabled=not train_lora)
    preset_lora_path = draw_parameter(preset_lora_path)
    preset_lora_model = draw_parameter(preset_lora_model)
    params = {}
    if train_lora:
        if lora_base_model is not None: params["lora_base_model"] = lora_base_model
        if lora_target_modules is not None: params["lora_target_modules"] = lora_target_modules
        if lora_rank is not None: params["lora_rank"] = lora_rank
        if lora_checkpoint is not None: params["lora_checkpoint"] = lora_checkpoint
    if preset_lora_path is not None: params["preset_lora_path"] = preset_lora_path
    if preset_lora_model is not None: params["preset_lora_model"] = preset_lora_model
    return params

def draw_gradient_configs(use_gradient_checkpointing, use_gradient_checkpointing_offload, gradient_accumulation_steps):
    use_gradient_checkpointing = draw_parameter(use_gradient_checkpointing)
    use_gradient_checkpointing_offload = draw_parameter(use_gradient_checkpointing_offload)
    gradient_accumulation_steps = draw_parameter(gradient_accumulation_steps)
    params = {
        "use_gradient_checkpointing": use_gradient_checkpointing,
        "use_gradient_checkpointing_offload": use_gradient_checkpointing_offload,
    }
    if gradient_accumulation_steps != 1:
        params["gradient_accumulation_steps"] = gradient_accumulation_steps
    return params

def draw_template_model_configs(template_model_id_or_path, enable_lora_hot_loading):
    template_model_id_or_path = draw_parameter(template_model_id_or_path)
    enable_lora_hot_loading = draw_parameter(enable_lora_hot_loading)
    params = {"enable_lora_hot_loading": enable_lora_hot_loading}
    if template_model_id_or_path is not None: params["template_model_id_or_path"] = template_model_id_or_path
    return params

def match_ui_groups(params, ui_groups):
    param_names = [param.name for param in params]
    for ui_group in ui_groups:
        if sum([name in param_names for name in ui_group["params"]]) == len(ui_group["params"]):
            group_params = {param.name: param for param in params if param.name in ui_group["params"]}
            other_params = [param for param in params if param.name not in ui_group["params"]]
            return group_params, other_params, ui_group
    return {}, params, None

def draw_other_params(params):
    results = {}
    for param in params:
        results[param.name] = draw_parameter(param)
    return results

def draw_all_params(params, ui_groups):
    matched_ui_groups = []
    while True:
        group_params, other_params, ui_group = match_ui_groups(params, ui_groups)
        if len(group_params) == 0:
            break
        matched_ui_groups.append((ui_group, group_params))
        params = other_params
    tabs = st.tabs([ui_group["name"] for ui_group, _ in matched_ui_groups] + ["Others"])
    inputs = {}
    for tab, (ui_group, group_params) in zip(tabs, matched_ui_groups):
        with tab:
            inputs.update(ui_group["fn"](**group_params))
    with tabs[-1]:
        inputs.update(draw_other_params(params))
    return inputs

def generate_training_script(script_path, inputs):
    cmd = f"accelerate launch {script_path}"
    for name, value in inputs.items():
        if value is not None:
            if isinstance(value, bool):
                if value == True:
                    cmd = f"{cmd} \\\n  --{name}"
            elif isinstance(value, str):
                cmd = f"{cmd} \\\n  --{name} \"{value}\""
            else:
                cmd = f"{cmd} \\\n  --{name} {value}"
    return cmd

ui_groups = [
    {
        "name": "Dataset",
        "params": ("dataset_base_path", "dataset_metadata_path", "dataset_repeat", "dataset_num_workers", "data_file_keys"),
        "fn": draw_dataset_configs,
    },
    {
        "name": "Video Size",
        "params": ("height", "width", "max_pixels", "num_frames"),
        "fn": draw_video_size,
    },
    {
        "name": "Image Size",
        "params": ("height", "width", "max_pixels"),
        "fn": draw_image_size,
    },
    {
        "name": "Model",
        "params": ("model_paths", "model_id_with_origin_paths", "extra_inputs", "fp8_models", "offload_models"),
        "fn": draw_model_configs,
    },
    {
        "name": "Training",
        "params": ("learning_rate", "num_epochs", "trainable_models", "find_unused_parameters", "weight_decay", "task"),
        "fn": draw_training_configs,
    },
    {
        "name": "Output",
        "params": ("output_path", "remove_prefix_in_ckpt", "save_steps"),
        "fn": draw_output_configs,
    },
    {
        "name": "LoRA",
        "params": ("lora_base_model", "lora_target_modules", "lora_rank", "lora_checkpoint", "preset_lora_path", "preset_lora_model"),
        "fn": draw_lora_configs,
    },
    {
        "name": "Gradient",
        "params": ("use_gradient_checkpointing", "use_gradient_checkpointing_offload", "gradient_accumulation_steps"),
        "fn": draw_gradient_configs,
    },
    {
        "name": "Templates",
        "params": ("template_model_id_or_path", "enable_lora_hot_loading"),
        "fn": draw_template_model_configs,
    },
]

def launch_webui():
    input_col, output_col = st.columns(2)
    with input_col:
        if "available_training_scripts" not in st.session_state:
            st.session_state["available_training_scripts"] = parse_available_training_scripts("examples")
        with st.container(border=True):
            script_path = st.selectbox(label="Script path", options=st.session_state["available_training_scripts"].values(), index=0)
            example_path = st.selectbox(label="Example path (Optional)", options=["None"] + parse_available_examples(script_path), index=0)
        if st.button("Step 1: Parse Training Script", type="primary"):
            st.session_state["script_path"] = script_path
        
        if "script_path" not in st.session_state:
            return
        with st.spinner("Fetching input parameters", show_time=False):
            parser = parse_parser(script_path)
        parser = parser()
        params = parse_parser_actions(parser, example_path)
        inputs = draw_all_params(params, ui_groups)
    with output_col:
        if st.button("Step 2: Generate training script", type="primary"):
            script = generate_training_script(script_path, inputs)
            st.code(script, language="shell")

launch_webui()
