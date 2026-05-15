import importlib, inspect, pkgutil, traceback, torch, os, re, typing, io
from typing import Union, List, Optional, Tuple, Iterable, Dict, Literal
from contextlib import contextmanager
from diffsynth.utils.data import VideoData
import streamlit as st
from diffsynth import ModelConfig
from diffsynth.diffusion.base_pipeline import ControlNetInput
from PIL import Image
from tqdm import tqdm
st.set_page_config(layout="wide")

class StreamlitTqdmWrapper:
    """Wrapper class that combines tqdm and streamlit progress bar"""
    def __init__(self, iterable, st_progress_bar=None):
        self.iterable = iterable
        self.st_progress_bar = st_progress_bar
        self.tqdm_bar = tqdm(iterable)
        self.total = len(iterable) if hasattr(iterable, '__len__') else None
        self.current = 0

    def __iter__(self):
        for item in self.tqdm_bar:
            if self.st_progress_bar is not None and self.total is not None:
                self.current += 1
                self.st_progress_bar.progress(self.current / self.total)
            yield item

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if hasattr(self.tqdm_bar, '__exit__'):
            self.tqdm_bar.__exit__(*args)

@contextmanager
def catch_error(error_value):
    try:
        yield
    except Exception as e:
        error_message = traceback.format_exc()
        print(f"Error {error_value}:\n{error_message}")

def parse_vram_config_from_an_example(path):
    vram_config = {
        "offload_dtype": None,
        "offload_device": None,
        "onload_dtype": None,
        "onload_device": None,
        "preparing_dtype": None,
        "preparing_device": None,
        "computation_dtype": None,
        "computation_device": None,
    }
    with open(path, "r") as f:
        for code in f.readlines():
            code = code.strip()
            for param in vram_config:
                if vram_config[param] is None and f'"{param}":' in code:
                    value = code.split(" ")[-1].replace(",", "").replace('"', "").replace("torch.", "")
                    vram_config[param] = value
    return vram_config

def parse_model_configs_from_an_example(path):
    model_configs = []
    vram_config = parse_vram_config_from_an_example(path)
    with open(path, "r") as f:
        for code in f.readlines():
            code = code.strip()
            if not code.startswith("ModelConfig"):
                continue
            pairs = re.findall(r'(\w+)\s*=\s*["\']([^"\']+)["\']', code)
            config_dict = {k: v for k, v in pairs}
            vram_config_ = vram_config if "**vram_config" in code else {}
            model_configs.append(ModelConfig(model_id=config_dict["model_id"], origin_file_pattern=config_dict["origin_file_pattern"], **vram_config_))
    return model_configs

def list_examples(path, keyword=None):
    examples = []
    if os.path.isdir(path):
        for file_name in os.listdir(path):
            examples.extend(list_examples(os.path.join(path, file_name), keyword=keyword))
    elif path.endswith(".py"):
        with open(path, "r") as f:
            code = f.read()
        if keyword is None or keyword in code:
            examples.extend([path])
    return examples

def parse_available_pipelines():
    from diffsynth.diffusion.base_pipeline import BasePipeline
    import diffsynth.pipelines as _pipelines_pkg
    available_pipelines = {}
    for _, name, _ in pkgutil.iter_modules(_pipelines_pkg.__path__):
        with catch_error(f"Failed: import diffsynth.pipelines.{name}"):
            mod = importlib.import_module(f"diffsynth.pipelines.{name}")
            classes = {
                cls_name: cls for cls_name, cls in inspect.getmembers(mod, inspect.isclass)
                if issubclass(cls, BasePipeline) and cls is not BasePipeline and cls.__module__ == mod.__name__
            }
            available_pipelines.update(classes)
    return available_pipelines

def parse_available_examples(path, available_pipelines):
    available_examples = {}
    for pipeline_name in available_pipelines:
        examples = ["None"] + list_examples(path, keyword=f"{pipeline_name}.from_pretrained")
        available_examples[pipeline_name] = examples
    return available_examples

def draw_selectbox(label, options, option_map, value=None, disabled=False):
    default_index = 0 if value is None else tuple(options).index([option for option in option_map if option_map[option]==value][0])
    option = st.selectbox(label=label, options=tuple(options), index=default_index, disabled=disabled)
    return option_map.get(option)

def parse_params(fn):
    params = []
    for name, param in inspect.signature(fn).parameters.items():
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else None
        default = param.default if param.default is not inspect.Parameter.empty else None
        params.append({"name": name, "dtype": annotation, "value": default})
    return params

def draw_vram_device(label, value=None, key_suffix="", disabled=False):
    option_map = {"None": None, "disk": "disk", "cuda": "cuda", "cpu": "cpu"}
    options = option_map.keys()
    default_index = 0 if value is None else tuple(options).index(value)
    option = st.selectbox(label=label, options=tuple(options), index=default_index, key=label + key_suffix, disabled=disabled)
    return option_map.get(option)

def draw_vram_dtype(label, value=None, key_suffix="", disabled=False):
    option_map = {"None": None, "disk": "disk", "bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16, "float8_e4m3fn": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2}
    options = option_map.keys()
    default_index = 0 if value is None else tuple(options).index(value)
    option = st.selectbox(label=label, options=tuple(options), index=default_index, key=label + key_suffix, disabled=disabled)
    return option_map.get(option)

def draw_model_config(model_config=None, key_suffix="", disabled=False, enable_vram_config=False):
    with st.container(border=True):
        if model_config is None:
            model_config = ModelConfig()
        path = st.text_input(label="path", key="path" + key_suffix, value=model_config.path, disabled=disabled)
        col1, col2 = st.columns(2)
        with col1:
            model_id = st.text_input(label="model_id", key="model_id" + key_suffix, value=model_config.model_id, disabled=disabled)
        with col2:
            origin_file_pattern = st.text_input(label="origin_file_pattern", key="origin_file_pattern" + key_suffix, value=model_config.origin_file_pattern, disabled=disabled)
        if enable_vram_config:
            with st.container(border=True):
                col1, col2 = st.columns(2)
                with col1:
                    offload_device = draw_vram_device(label="offload_device", value=model_config.offload_device, key_suffix=key_suffix, disabled=disabled)
                    onload_device = draw_vram_device(label="onload_device", value=model_config.onload_device, key_suffix=key_suffix, disabled=disabled)
                    preparing_device = draw_vram_device(label="preparing_device", value=model_config.preparing_device, key_suffix=key_suffix, disabled=disabled)
                    computation_device = draw_vram_device(label="computation_device", value=model_config.computation_device, key_suffix=key_suffix, disabled=disabled)
                with col2:
                    offload_dtype = draw_vram_dtype(label="offload_dtype", value=model_config.offload_dtype, key_suffix=key_suffix, disabled=disabled)
                    onload_dtype = draw_vram_dtype(label="onload_dtype", value=model_config.onload_dtype, key_suffix=key_suffix, disabled=disabled)
                    preparing_dtype = draw_vram_dtype(label="preparing_dtype", value=model_config.preparing_dtype, key_suffix=key_suffix, disabled=disabled)
                    computation_dtype = draw_vram_dtype(label="computation_dtype", value=model_config.computation_dtype, key_suffix=key_suffix, disabled=disabled)
                vram_config = {
                    "offload_device": offload_device,
                    "onload_device": onload_device,
                    "preparing_device": preparing_device,
                    "computation_device": computation_device,
                    "offload_dtype": offload_dtype,
                    "onload_dtype": onload_dtype,
                    "preparing_dtype": preparing_dtype,
                    "computation_dtype": computation_dtype,
                }
        else:
            vram_config = {}
        model_config = ModelConfig(
            path=None if path == "" else path,
            model_id=model_id,
            origin_file_pattern=origin_file_pattern,
            **vram_config,
        )
    return model_config

def draw_multi_model_config(name="", value=None, disabled=False, enable_vram_config=False):
    model_configs = []
    with st.container(border=True):
        st.markdown(name)
        num = st.number_input(f"num_{name}", min_value=0, max_value=20, value=0 if value is None else len(value), disabled=disabled)
        for i in range(num):
            model_config = draw_model_config(key_suffix=f"_{name}_{i}", model_config=None if value is None else value[i], disabled=disabled, enable_vram_config=enable_vram_config)
            model_configs.append(model_config)
    return model_configs

def draw_single_model_config(name="", value=None, disabled=False):
    with st.container(border=True):
        st.markdown(name)
        model_config = draw_model_config(value, key_suffix=f"_{name}", disabled=disabled)
    return model_config

def draw_multi_images(name="", value=None, disabled=False):
    images = []
    with st.container(border=True):
        st.markdown(name)
        num = st.number_input(f"num_{name}", min_value=0, max_value=20, value=0 if value is None else len(value), disabled=disabled)
        for i in range(num):
            image = st.file_uploader(name, type=["png", "jpg", "jpeg", "webp"], key=f"{name}_{i}", disabled=disabled)
            if image is not None: images.append(Image.open(image))
    return images

def draw_multi_elements(st_element, name="", value=None, disabled=False, kwargs=None):
    if kwargs is None:
        kwargs = {}
    elements = []
    with st.container(border=True):
        st.markdown(name)
        num = st.number_input(f"num_{name}", min_value=0, max_value=20, value=0 if value is None else len(value), disabled=disabled)
        for i in range(num):
            element = st_element(name, key=f"{name}_{i}", disabled=disabled, value=None if value is None else value[i], **kwargs)
            elements.append(element)
    return elements

def draw_lora_configs(name="", value=None, disabled=False):
    elements = []
    with st.container(border=True):
        st.markdown(name)
        num = st.number_input(f"num_{name}", min_value=0, max_value=20, value=0 if value is None else len(value), disabled=disabled)
        for i in range(num):
            with st.container(border=True):
                lora_base_model = st.text_input(label="LoRA base model", key="LoRA base model" + f"LoRA_{i}")
                lora_scale = st.slider(label="LoRA scale", min_value=-8.0, max_value=8.0, value=1.0, step=0.1, key="LoRA scale" + f"LoRA_{i}")
                lora_config = draw_model_config(key_suffix=f"LoRA_{i}", disabled=disabled)
                element = {"base_model": lora_base_model, "alpha": lora_scale, "lora_config": lora_config}
                elements.append(element)
    return elements

def draw_controlnet_input(name="", value=None, disabled=False):
    with st.container(border=True):
        st.markdown(name)
        controlnet_id = st.number_input("controlnet_id", value=0, min_value=0, max_value=20, step=1, key=f"{name}_controlnet_id")
        scale = st.number_input("scale", value=1.0, min_value=0.0, max_value=10.0, key=f"{name}_scale")
        image = st.file_uploader("image", type=["png", "jpg", "jpeg", "webp"], disabled=disabled, key=f"{name}_image")
        if image is not None: image = Image.open(image)
        inpaint_image = st.file_uploader("inpaint_image", type=["png", "jpg", "jpeg", "webp"], disabled=disabled, key=f"{name}_inpaint_image")
        if inpaint_image is not None: inpaint_image = Image.open(inpaint_image)
        inpaint_mask = st.file_uploader("inpaint_mask", type=["png", "jpg", "jpeg", "webp"], disabled=disabled, key=f"{name}_inpaint_mask")
        if inpaint_mask is not None: inpaint_mask = Image.open(inpaint_mask)
    return ControlNetInput(controlnet_id=controlnet_id, scale=scale, image=image, inpaint_image=inpaint_image, inpaint_mask=inpaint_mask)

def draw_controlnet_inputs(name, value=None, disabled=False):
    controlnet_inputs = []
    with st.container(border=True):
        st.markdown(name)
        num = st.number_input(f"num_{name}", min_value=0, max_value=20, value=0 if value is None else len(value), disabled=disabled)
        for i in range(num):
            controlnet_input = draw_controlnet_input(name=f"{name}_{i}", value=None, disabled=disabled)
            controlnet_inputs.append(controlnet_input)
    return controlnet_inputs

def draw_ui_element(name, dtype, value):
    unsupported_dtype = [
        Dict[str, torch.Tensor],
        torch.Tensor,
    ]
    if dtype in unsupported_dtype:
        return
    if value is None:
        with st.container(border=True):
            enable = st.checkbox(f"Enable {name}", value=False)
            ui = draw_ui_element_safely(name, dtype, value=value, disabled=not enable)
        if enable:
            return ui
        else:
            return None
    else:
        return draw_ui_element_safely(name, dtype, value)

def draw_video(name, value=None, disabled=False):
    ui = st.file_uploader(name, type=["mp4"], disabled=disabled)
    if ui is not None:
        ui = VideoData(ui)
        ui = [ui[i] for i in range(len(ui))]
    return ui

def draw_ui_element_safely(name, dtype, value, disabled=False):
    if dtype == torch.dtype:
        option_map = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}
        ui = draw_selectbox(name, option_map.keys(), option_map, value=value, disabled=disabled)
    elif dtype == Union[str, torch.device]:
        option_map = {"cuda": "cuda", "cpu": "cpu"}
        ui = draw_selectbox(name, option_map.keys(), option_map, value=value, disabled=disabled)
    elif dtype == bool:
        ui = st.checkbox(name, value=value, disabled=disabled)
    elif dtype == ModelConfig:
        ui = draw_single_model_config(name, value=value, disabled=disabled)
    elif dtype in [list[ModelConfig], List[ModelConfig], Union[list[ModelConfig], ModelConfig, str]]:
        if name == "model_configs":
            model_configs = st.session_state.get("model_configs_from_example")
            ui = draw_multi_model_config(name, model_configs, disabled=disabled, enable_vram_config=True)
        else:
            ui = draw_multi_model_config(name, disabled=disabled)
    elif dtype == str:
        if "prompt" in name:
            ui = st.text_area(name, value=value, height=3, disabled=disabled)
        else:
            ui = st.text_input(name, value=value, disabled=disabled)
    elif dtype == float:
        ui = st.number_input(name, value=value, disabled=disabled)
    elif dtype == int:
        ui = st.number_input(name, value=value, step=1, disabled=disabled)
    elif dtype == Image.Image:
        ui = st.file_uploader(name, type=["png", "jpg", "jpeg", "webp"], disabled=disabled)
        if ui is not None: ui = Image.open(ui)
    elif dtype in [List[Image.Image], list[Image.Image], Union[list[Image.Image], Image.Image], Union[List[Image.Image], Image.Image]]:
        if "video" in name:
            ui = draw_video(name, value=value, disabled=disabled)
        else:
            ui = draw_multi_images(name, value=value, disabled=disabled)
    elif dtype in [List[ControlNetInput], list[ControlNetInput]]:
        ui = draw_controlnet_inputs(name, value=value, disabled=disabled)
    elif dtype in [List[str], list[str]]:
        ui = draw_multi_elements(st.text_input, name, value=value, disabled=disabled)
    elif dtype in [List[float], list[float], Union[list[float], float], Union[List[float], float]]:
        ui = draw_multi_elements(st.number_input, name, value=value, disabled=disabled)
    elif dtype in [List[int], list[int]]:
        ui = draw_multi_elements(st.number_input, name, value=value, disabled=disabled, kwargs={"step": 1})
    elif dtype in [List[List[Image.Image]], list[list[Image.Image]]]:
        ui = draw_multi_elements(draw_video, name, value=value, disabled=disabled)
    elif dtype in [tuple[int, int], Tuple[int, int]]:
        with st.container(border=True):
            st.markdown(name)
            ui = (st.text_input(f"{name}_0", value=value[0], disabled=disabled), st.text_input(f"{name}_1", value=value[1], disabled=disabled))
    elif isinstance(dtype, typing._LiteralGenericAlias):
        with st.container(border=True):
            st.markdown(f"{name} ({dtype})")
            ui = st.text_input(name, value=value, disabled=disabled, label_visibility="hidden")
    elif dtype is None:
        if name == "progress_bar_cmd":
            ui = value
    else:
        st.markdown(f"(`{name}` is not not configurable in WebUI). dtype: `{dtype}`.")
        ui = value
    return ui

def flush_example():
    for key in list(st.session_state.keys()):
        if key not in ["available_pipelines", "available_examples"]:
            del st.session_state[key]

def launch_webui():
    input_col, output_col = st.columns(2)
    with input_col:
        if "available_pipelines" not in st.session_state:
            st.session_state["available_pipelines"] = parse_available_pipelines()
        if "available_examples" not in st.session_state:
            st.session_state["available_examples"] = parse_available_examples("./examples", st.session_state["available_pipelines"])

        with st.expander("Pipeline", expanded=True):
            pipeline_class = draw_selectbox("Pipeline Class", st.session_state["available_pipelines"].keys(), st.session_state["available_pipelines"], value=st.session_state["available_pipelines"]["ZImagePipeline"])
            example = st.selectbox("Parse model configs from an example (optional)", st.session_state["available_examples"][pipeline_class.__name__], on_change=flush_example)

        if st.button("Step 1: Parse Pipeline", type="primary"):
            st.session_state["pipeline_class"] = pipeline_class
            if example != "None":
                st.session_state["model_configs_from_example"] = parse_model_configs_from_an_example(example)

        if "pipeline_class" not in st.session_state:
            return
        with st.expander("Model", expanded=True):
            input_params = {}
            params = parse_params(pipeline_class.from_pretrained)
            for param in params:
                input_params[param["name"]] = draw_ui_element(**param)
            lora_configs = draw_lora_configs(name="LoRA")
        if st.button("Step 2: Load Models", type="primary"):
            with st.spinner("Loading models", show_time=True):
                if "pipe" in st.session_state:
                    del st.session_state["pipe"]
                    torch.cuda.empty_cache()
                pipe = pipeline_class.from_pretrained(**input_params)
                for lora_config in lora_configs:
                    pipe.load_lora(pipe.get_module(pipe, lora_config["base_model"]), lora_config=lora_config["lora_config"], alpha=lora_config["alpha"])
                st.session_state["pipe"] = pipe

        if "pipe" not in st.session_state:
            return
        with st.expander("Input", expanded=True):
            pipe = st.session_state["pipe"]
            input_params = {}
            params = parse_params(pipeline_class.__call__)
            for param in params:
                if param["name"] in ["self"]:
                    continue
                input_params[param["name"]] = draw_ui_element(**param)
    
    with output_col:
        if st.button("Step 3: Generate", type="primary"):
            if "progress_bar_cmd" in input_params:
                input_params["progress_bar_cmd"] = lambda iterable: StreamlitTqdmWrapper(iterable, st.progress(0))
            result = pipe(**input_params)
            st.session_state["result"] = result
        
        if "result" in st.session_state:
            result = st.session_state["result"]
            if isinstance(result, Image.Image):
                st.image(result)
                buf = io.BytesIO()
                result.save(buf, format='PNG')
                st.download_button(label="Download", data=buf.getvalue(), file_name="image.png", mime="image/png", type="primary")
            else:
                print(f"unsupported result format: {result}")

launch_webui()
