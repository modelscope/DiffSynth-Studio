import importlib, inspect, pkgutil, traceback, torch, os, re
from typing import Union, List, Optional, Tuple, Iterable, Dict
from contextlib import contextmanager
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

def parse_model_configs_from_an_example(path):
    model_configs = []
    with open(path, "r") as f:
        for code in f.readlines():
            code = code.strip()
            if not code.startswith("ModelConfig"):
                continue
            pairs = re.findall(r'(\w+)\s*=\s*["\']([^"\']+)["\']', code)
            config_dict = {k: v for k, v in pairs}
            model_configs.append(ModelConfig(model_id=config_dict["model_id"], origin_file_pattern=config_dict["origin_file_pattern"]))
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

def draw_model_config(model_config=None, key_suffix="", disabled=False):
    with st.container(border=True):
        if model_config is None:
            model_config = ModelConfig()
        path = st.text_input(label="path", key="path" + key_suffix, value=model_config.path, disabled=disabled)
        col1, col2 = st.columns(2)
        with col1:
            model_id = st.text_input(label="model_id", key="model_id" + key_suffix, value=model_config.model_id, disabled=disabled)
        with col2:
            origin_file_pattern = st.text_input(label="origin_file_pattern", key="origin_file_pattern" + key_suffix, value=model_config.origin_file_pattern, disabled=disabled)
        model_config = ModelConfig(
            path=None if path == "" else path,
            model_id=model_id,
            origin_file_pattern=origin_file_pattern,
        )
    return model_config

def draw_multi_model_config(name="", value=None, disabled=False):
    model_configs = []
    with st.container(border=True):
        st.markdown(name)
        num = st.number_input(f"num_{name}", min_value=0, max_value=20, value=0 if value is None else len(value), disabled=disabled)
        for i in range(num):
            model_config = draw_model_config(key_suffix=f"_{name}_{i}", model_config=None if value is None else value[i], disabled=disabled)
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
            ui = draw_ui_element_safely(name, dtype, value, disabled=not enable)
        if enable:
            return ui
        else:
            return None
    else:
        return draw_ui_element_safely(name, dtype, value)

def draw_ui_element_safely(name, dtype, value, disabled=False):
    if dtype == torch.dtype:
        option_map = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}
        ui = draw_selectbox(name, option_map.keys(), option_map, value=value, disabled=disabled)
    elif dtype == Union[str, torch.device]:
        option_map = {"cuda": "cuda", "cpu": "cpu"}
        ui = draw_selectbox(name, option_map.keys(), option_map, value=value, disabled=disabled)
    elif dtype == bool:
        ui = st.checkbox(name, value, disabled=disabled)
    elif dtype == ModelConfig:
        ui = draw_single_model_config(name, value, disabled=disabled)
    elif dtype == list[ModelConfig]:
        if name == "model_configs" and "model_configs_from_example" in st.session_state:
            model_configs = st.session_state["model_configs_from_example"]
            del st.session_state["model_configs_from_example"]
            ui = draw_multi_model_config(name, model_configs, disabled=disabled)
        else:
            ui = draw_multi_model_config(name, disabled=disabled)
    elif dtype == str:
        if "prompt" in name:
            ui = st.text_area(name, value, height=3, disabled=disabled)
        else:
            ui = st.text_input(name, value, disabled=disabled)
    elif dtype == float:
        ui = st.number_input(name, value, disabled=disabled)
    elif dtype == int:
        ui = st.number_input(name, value, step=1, disabled=disabled)
    elif dtype == Image.Image:
        ui = st.file_uploader(name, type=["png", "jpg", "jpeg", "webp"], disabled=disabled)
        if ui is not None: ui = Image.open(ui)
    elif dtype == List[Image.Image]:
        ui = draw_multi_images(name, value, disabled=disabled)
    elif dtype == List[ControlNetInput]:
        ui = draw_controlnet_inputs(name, value, disabled=disabled)
    elif dtype is None:
        if name == "progress_bar_cmd":
            ui = value
    else:
        st.markdown(f"(`{name}` is not not configurable in WebUI). dtype: `{dtype}`.")
        ui = value
    return ui


def launch_webui():
    input_col, output_col = st.columns(2)
    with input_col:
        if "available_pipelines" not in st.session_state:
            st.session_state["available_pipelines"] = parse_available_pipelines()
        if "available_examples" not in st.session_state:
            st.session_state["available_examples"] = parse_available_examples("./examples", st.session_state["available_pipelines"])

        with st.expander("Pipeline", expanded=True):
            pipeline_class = draw_selectbox("Pipeline Class", st.session_state["available_pipelines"].keys(), st.session_state["available_pipelines"], value=st.session_state["available_pipelines"]["ZImagePipeline"])
            example = st.selectbox("Parse model configs from an example (optional)", st.session_state["available_examples"][pipeline_class.__name__])
            if example != "None":
                st.session_state["model_configs_from_example"] = parse_model_configs_from_an_example(example)
        if st.button("Step 1: Parse Pipeline", type="primary"):
            st.session_state["pipeline_class"] = pipeline_class

        if "pipeline_class" not in st.session_state:
            return
        with st.expander("Model", expanded=True):
            input_params = {}
            params = parse_params(pipeline_class.from_pretrained)
            for param in params:
                input_params[param["name"]] = draw_ui_element(**param)
        if st.button("Step 2: Load Models", type="primary"):
            with st.spinner("Loading models", show_time=True):
                if "pipe" in st.session_state:
                    del st.session_state["pipe"]
                    torch.cuda.empty_cache()
                st.session_state["pipe"] = pipeline_class.from_pretrained(**input_params)

        if "pipe" not in st.session_state:
            return
        with st.expander("Input", expanded=True):
            pipe = st.session_state["pipe"]
            input_params = {}
            params = parse_params(pipe.__call__)
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
            else:
                print(f"unsupported result format: {result}")

launch_webui()
# streamlit run examples/dev_tools/webui.py --server.fileWatcherType none
