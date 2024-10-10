from huggingface_hub import hf_hub_download
from modelscope import snapshot_download
import os, shutil
from typing_extensions import Literal, TypeAlias
from typing import List
from ..configs.model_config import preset_models_on_huggingface, preset_models_on_modelscope, Preset_model_id


def download_from_modelscope(model_id, origin_file_path, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    file_name = os.path.basename(origin_file_path)
    if file_name in os.listdir(local_dir):
        print(f"    {file_name} has been already in {local_dir}.")
    else:
        print(f"    Start downloading {os.path.join(local_dir, file_name)}")
        snapshot_download(model_id, allow_file_pattern=origin_file_path, local_dir=local_dir)
        downloaded_file_path = os.path.join(local_dir, origin_file_path)
        target_file_path = os.path.join(local_dir, os.path.split(origin_file_path)[-1])
        if downloaded_file_path != target_file_path:
            shutil.move(downloaded_file_path, target_file_path)
            shutil.rmtree(os.path.join(local_dir, origin_file_path.split("/")[0]))


def download_from_huggingface(model_id, origin_file_path, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    file_name = os.path.basename(origin_file_path)
    if file_name in os.listdir(local_dir):
        print(f"    {file_name} has been already in {local_dir}.")
    else:
        print(f"    Start downloading {os.path.join(local_dir, file_name)}")
        hf_hub_download(model_id, origin_file_path, local_dir=local_dir)
        downloaded_file_path = os.path.join(local_dir, origin_file_path)
        target_file_path = os.path.join(local_dir, file_name)
        if downloaded_file_path != target_file_path:
            shutil.move(downloaded_file_path, target_file_path)
            shutil.rmtree(os.path.join(local_dir, origin_file_path.split("/")[0]))


Preset_model_website: TypeAlias = Literal[
    "HuggingFace",
    "ModelScope",
]
website_to_preset_models = {
    "HuggingFace": preset_models_on_huggingface,
    "ModelScope": preset_models_on_modelscope,
}
website_to_download_fn = {
    "HuggingFace": download_from_huggingface,
    "ModelScope": download_from_modelscope,
}


def download_customized_models(
    model_id,
    origin_file_path,
    local_dir,
    downloading_priority: List[Preset_model_website] = ["ModelScope", "HuggingFace"],
):
    downloaded_files = []
    for website in downloading_priority:
        # Check if the file is downloaded.
        file_to_download = os.path.join(local_dir, os.path.basename(origin_file_path))
        if file_to_download in downloaded_files:
            continue
        # Download
        website_to_download_fn[website](model_id, origin_file_path, local_dir)
        if os.path.basename(origin_file_path) in os.listdir(local_dir):
            downloaded_files.append(file_to_download)
    return downloaded_files


def download_models(
    model_id_list: List[Preset_model_id] = [],
    downloading_priority: List[Preset_model_website] = ["ModelScope", "HuggingFace"],
):
    print(f"Downloading models: {model_id_list}")
    downloaded_files = []
    load_files = []

    for model_id in model_id_list:
        for website in downloading_priority:
            if model_id in website_to_preset_models[website]:
                
                # Parse model metadata
                model_metadata = website_to_preset_models[website][model_id]
                if isinstance(model_metadata, list):
                    file_data = model_metadata
                else:
                    file_data = model_metadata.get("file_list", [])

                # Try downloading the model from this website.
                model_files = []
                for model_id, origin_file_path, local_dir in file_data:
                    # Check if the file is downloaded.
                    file_to_download = os.path.join(local_dir, os.path.basename(origin_file_path))
                    if file_to_download in downloaded_files:
                        continue
                    # Download
                    website_to_download_fn[website](model_id, origin_file_path, local_dir)
                    if os.path.basename(origin_file_path) in os.listdir(local_dir):
                        downloaded_files.append(file_to_download)
                        model_files.append(file_to_download)
                
                # If the model is successfully downloaded, break.
                if len(model_files) > 0:
                    if isinstance(model_metadata, dict) and "load_path" in model_metadata:
                        model_files = model_metadata["load_path"]
                    load_files.extend(model_files)
                    break
                
    return load_files
