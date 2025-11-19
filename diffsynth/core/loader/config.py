import torch, glob, os
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
from huggingface_hub import snapshot_download as hf_snapshot_download
from typing import Optional


@dataclass
class ModelConfig:
    path: Union[str, list[str]] = None
    model_id: str = None
    origin_file_pattern: Union[str, list[str]] = None
    download_resource: str = None
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
    
    def check_input(self):
        if self.path is None and self.model_id is None:
            raise ValueError(f"""No valid model files. Please use `ModelConfig(path="xxx")` or `ModelConfig(model_id="xxx/yyy", origin_file_pattern="zzz")`. `skip_download=True` only supports the first one.""")
    
    def download(self):
        origin_file_pattern = self.origin_file_pattern + ("*" if self.origin_file_pattern.endswith("/") else "")
        downloaded_files = glob.glob(origin_file_pattern, root_dir=os.path.join(self.local_model_path, self.model_id))
        if self.download_resource is None:
            if os.environ.get('DIFFSYNTH_DOWNLOAD_RESOURCE') is not None:
                self.download_resource = os.environ.get('DIFFSYNTH_DOWNLOAD_RESOURCE')
            else:
                self.download_resource = "modelscope"
        if self.download_resource.lower() == "modelscope":
            snapshot_download(
                self.model_id,
                local_dir=os.path.join(self.local_model_path, self.model_id),
                allow_file_pattern=self.origin_file_pattern,
                ignore_file_pattern=downloaded_files,
                local_files_only=False
            )
        elif self.download_resource.lower() == "huggingface":
            hf_snapshot_download(
                self.model_id,
                local_dir=os.path.join(self.local_model_path, self.model_id),
                allow_patterns=self.origin_file_pattern,
                ignore_patterns=downloaded_files,
                local_files_only=False
            )
        else:
            raise ValueError("`download_resource` should be `modelscope` or `huggingface`.")
        
    def require_downloading(self):
        if self.path is not None:
            return False
        if self.skip_download is None:
            if os.environ.get('DIFFSYNTH_SKIP_DOWNLOAD') is not None:
                if os.environ.get('DIFFSYNTH_SKIP_DOWNLOAD') in ["True", "true"]:
                    self.skip_download = True
                elif os.environ.get('DIFFSYNTH_SKIP_DOWNLOAD') in ["False", "false"]:
                    self.skip_download = False
            else:
                self.skip_download = False
        return not self.skip_download
    
    def reset_local_model_path(self):
        if os.environ.get('DIFFSYNTH_MODEL_BASE_PATH') is not None:
            self.local_model_path = os.environ.get('DIFFSYNTH_MODEL_BASE_PATH')
        elif self.local_model_path is None:
            self.local_model_path = "./models"

    def download_if_necessary(self):
        self.check_input()
        self.reset_local_model_path()
        if self.require_downloading():
            self.download()
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
