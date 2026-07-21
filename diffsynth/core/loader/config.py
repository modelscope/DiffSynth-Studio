import torch, glob, os, sys
from typing import Optional, Union, Dict
from dataclasses import dataclass
from modelscope import snapshot_download
from huggingface_hub import snapshot_download as hf_snapshot_download
from typing import Optional

_download_tips_printed = False

if sys.platform.startswith('win'):
    DOWNLOADING_TIPS = """
┌──────────────────────────────────────────────────────────────────────┐
│ DiffSynth-Studio Model Downloader Configuration:                     │
│                                                                      │
│ [0] Download from https://modelscope.cn/                             │
│     (default behavior)                                               │
│ [1] Download from https://modelscope.ai/                             │
│     (enabled via `$env:MODELSCOPE_ENDPOINT="https://modelscope.ai"`) │
│ [2] Download from https://huggingface.co/                            │
│     (enabled via `$env:DIFFSYNTH_DOWNLOAD_SOURCE="HuggingFace"`)     │
│ [3] Skip download and load only pre-downloaded model files           │
│     (enabled via `$env:DIFFSYNTH_SKIP_DOWNLOAD="True"`)              │
└──────────────────────────────────────────────────────────────────────┘
""".strip()
else:
    DOWNLOADING_TIPS = """
┌──────────────────────────────────────────────────────────────────────┐
│ DiffSynth-Studio Model Downloader Configuration:                     │
│                                                                      │
│ [0] Download from https://modelscope.cn/                             │
│     (default behavior)                                               │
│ [1] Download from https://modelscope.ai/                             │
│     (enabled via `export MODELSCOPE_ENDPOINT=https://modelscope.ai`) │
│ [2] Download from https://huggingface.co/                            │
│     (enabled via `export DIFFSYNTH_DOWNLOAD_SOURCE=HuggingFace`)     │
│ [3] Skip download and load only pre-downloaded model files           │
│     (enabled via `export DIFFSYNTH_SKIP_DOWNLOAD=True`)              │
└──────────────────────────────────────────────────────────────────────┘
""".strip()


@dataclass
class ModelConfig:
    path: Union[str, list[str]] = None
    model_id: str = None
    origin_file_pattern: Union[str, list[str]] = None
    download_source: str = None
    local_model_path: str = None
    skip_download: bool = None
    offload_device: Optional[Union[str, torch.device]] = None
    offload_dtype: Optional[torch.dtype] = None
    onload_device: Optional[Union[str, torch.device]] = None
    onload_dtype: Optional[torch.dtype] = None
    preparing_device: Optional[Union[str, torch.device]] = None
    preparing_dtype: Optional[torch.dtype] = None
    computation_device: Optional[Union[str, torch.device]] = None
    computation_dtype: Optional[torch.dtype] = None
    clear_parameters: bool = False
    state_dict: Dict[str, torch.Tensor] = None
    
    def check_input(self):
        if self.path is None and self.model_id is None:
            raise ValueError(f"""No valid model files. Please use `ModelConfig(path="xxx")` or `ModelConfig(model_id="xxx/yyy", origin_file_pattern="zzz")`. `skip_download=True` only supports the first one.""")
    
    def parse_original_file_pattern(self):
        if self.origin_file_pattern in [None, "", "./"]:
            return "*"
        elif self.origin_file_pattern.endswith("/"):
            return self.origin_file_pattern + "*"
        else:
            return self.origin_file_pattern
        
    def parse_download_source(self):
        if self.download_source is None:
            if os.environ.get('DIFFSYNTH_DOWNLOAD_SOURCE') is not None:
                return os.environ.get('DIFFSYNTH_DOWNLOAD_SOURCE')
            else:
                return "modelscope"
        else:
            return self.download_source
        
    def parse_skip_download(self):
        if self.skip_download is None:
            if os.environ.get('DIFFSYNTH_SKIP_DOWNLOAD') is not None:
                if os.environ.get('DIFFSYNTH_SKIP_DOWNLOAD').lower() == "true":
                    return True
                elif os.environ.get('DIFFSYNTH_SKIP_DOWNLOAD').lower() == "false":
                    return False
            else:
                return False
        else:
            return self.skip_download

    def download(self):
        origin_file_pattern = self.parse_original_file_pattern()
        downloaded_files = glob.glob(origin_file_pattern, root_dir=os.path.join(self.local_model_path, self.model_id))
        download_source = self.parse_download_source()
        if download_source.lower() == "modelscope":
            snapshot_download(
                self.model_id,
                local_dir=os.path.join(self.local_model_path, self.model_id),
                allow_file_pattern=origin_file_pattern,
                ignore_file_pattern=downloaded_files,
                local_files_only=False
            )
        elif download_source.lower() == "huggingface":
            hf_snapshot_download(
                self.model_id,
                local_dir=os.path.join(self.local_model_path, self.model_id),
                allow_patterns=origin_file_pattern,
                ignore_patterns=downloaded_files,
                local_files_only=False
            )
        else:
            raise ValueError("`download_source` should be `modelscope` or `huggingface`.")
        
    def require_downloading(self):
        if self.path is not None:
            return False
        skip_download = self.parse_skip_download()
        return not skip_download
    
    def reset_local_model_path(self):
        if os.environ.get('DIFFSYNTH_MODEL_BASE_PATH') is not None:
            self.local_model_path = os.environ.get('DIFFSYNTH_MODEL_BASE_PATH')
        elif self.local_model_path is None:
            self.local_model_path = "./models"

    def check_download_source(self):
        download_source = self.parse_download_source().lower()
        if os.environ.get('DIFFSYNTH_SKIP_DOWNLOAD', "").lower() == "true":
            behavior = 3
        elif download_source == "modelscope":
            if os.environ.get("MODELSCOPE_ENDPOINT", "") == "https://modelscope.ai" or os.environ.get("MODELSCOPE_DOMAIN", "") == "https://modelscope.ai":
                behavior = 1
            else:
                behavior = 0
        else:
            behavior = 2
        tips = DOWNLOADING_TIPS
        for i in range(4): tips = tips.replace(f"[{i}]", ["[ ]", "[√]"][i==behavior])
        global _download_tips_printed
        if not _download_tips_printed:
            is_main_process = True
            if torch.distributed.is_initialized():
                is_main_process = torch.distributed.get_rank() == 0
            if is_main_process:
                print(tips)
                _download_tips_printed = True

    def download_if_necessary(self):
        self.check_download_source()
        self.check_input()
        self.reset_local_model_path()
        if self.require_downloading():
            self.download()
        if self.path is None:
            if self.origin_file_pattern in [None, "", "./"]:
                self.path = os.path.join(self.local_model_path, self.model_id)
            else:
                self.path = glob.glob(os.path.join(self.local_model_path, self.model_id, self.origin_file_pattern))
        if isinstance(self.path, list) and len(self.path) == 1:
            self.path = self.path[0]

    def vram_config(self):
        return {
            "offload_device": self.offload_device,
            "offload_dtype": self.offload_dtype,
            "onload_device": self.onload_device,
            "onload_dtype": self.onload_dtype,
            "preparing_device": self.preparing_device,
            "preparing_dtype": self.preparing_dtype,
            "computation_device": self.computation_device,
            "computation_dtype": self.computation_dtype,
        }
