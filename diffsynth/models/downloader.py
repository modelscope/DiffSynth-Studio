from huggingface_hub import hf_hub_download
from modelscope import snapshot_download
import os


def download_from_modelscope(model_id, origin_file_path, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    if os.path.basename(origin_file_path) in os.listdir(local_dir):
        print(f"{os.path.basename(origin_file_path)} has been already in {local_dir}.")
        return
    else:
        print(f"Start downloading {os.path.join(local_dir, os.path.basename(origin_file_path))}")
    snapshot_download(model_id, allow_file_pattern=origin_file_path, local_dir=local_dir)


def download_from_huggingface(model_id, origin_file_path, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    if os.path.basename(origin_file_path) in os.listdir(local_dir):
        print(f"{os.path.basename(origin_file_path)} has been already in {local_dir}.")
        return
    else:
        print(f"Start downloading {os.path.join(local_dir, os.path.basename(origin_file_path))}")
    hf_hub_download(model_id, origin_file_path, local_dir=local_dir)
