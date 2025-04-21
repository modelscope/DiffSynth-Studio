import torch, os, dashscope
import pandas as pd
from tqdm import tqdm
from diffsynth import load_state_dict, hash_state_dict_keys


def search_for_model_file(path, allow_file_extensions=(".safetensors",)):
    for file_name in os.listdir(path):
        for file_extension in allow_file_extensions:
            if file_name.endswith(file_extension):
                return os.path.join(path, file_name)
            
            
def search_for_cover_images(path, allow_file_extensions=(".png", ".jpg", ".jpeg")):
    image_files = []
    for file_name in os.listdir(path):
        for file_extension in allow_file_extensions:
            if file_name.endswith(file_extension):
                image_files.append(os.path.join(path, file_name))
                break
    return image_files


def search_for_lora_data(path):
    model_file = search_for_model_file(path)
    if "_cover_images_" not in os.listdir(path):
        return None
    image_files = search_for_cover_images(os.path.join(path, "_cover_images_"))
    if model_file is None or len(image_files) == 0:
        return None
    state_dict = load_state_dict(model_file)
    if hash_state_dict_keys(state_dict, with_shape=False) != "52544ae3076666228978b738fbb8b086":
        return None
    return model_file, image_files


def image_to_text(images=[], prompt="", system_prompt=None):
    dashscope.api_key = "xxxxx" # TODO
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    if not isinstance(images, list):
        images = [images]
    messages.append({"role": "user", "content": [{"text": prompt}] + [{"image": image} for image in images]})
    response = dashscope.MultiModalConversation.call(model="qwen-vl-max-latest", messages=messages)
    response = response["output"]["choices"][0]["message"]["content"][0]["text"]
    return response


qwen_i2t_prompt = '''
You are a professional image captioner. 
Generate a caption according to the image so that another image generation model can generate the image via the caption. Just return the string description, do not return anything else.
'''.strip()


def data_to_csv(model_file_list, image_file_list, text_list, save_path):
    data_df = pd.DataFrame()
    data_df["model_file"] = model_file_list
    data_df["image_file"] = image_file_list
    data_df["text"] = text_list
    data_df.to_csv(save_path, index=False, encoding="utf-8-sig")


base_path = "/data/zhiwen/LoRA-Fusion/models/FLUXLoRA"

model_file_list = []
image_file_list = []
text_list = []

for lora_name in tqdm(os.listdir(base_path)):
    lora_folder_path = os.path.join(base_path, lora_name)
    if os.path.isdir(lora_folder_path):
        data = search_for_lora_data(lora_folder_path)
        if data is not None:
            model_file, image_files = data
            for image_file in image_files:
                try:
                    text = image_to_text(image_file, prompt=qwen_i2t_prompt)
                except:
                    continue
                model_file_list.append(model_file)
                image_file_list.append(image_file)
                text_list.append(text)
                data_to_csv(model_file_list, image_file_list, text_list, "data/loras.csv")
                
