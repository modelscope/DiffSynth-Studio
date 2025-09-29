import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForImageTextToText, Qwen2_5_VLForConditionalGeneration
from InternVL3_utils import load_video
import json
from tqdm import tqdm

def load_SmolVLM():
    model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2"
    ).to("cuda")
    return processor, model

def SmolVLM_generate(processor, model, messages, max_new_tokens):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=128)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return generated_texts[0].split("Assistant:")[-1].strip()

def load_Qwen(size="7B"): 
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        f"Qwen/Qwen2.5-VL-{size}-Instruct",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        # load_in_8bit=True,
        device_map="auto",
        # offload_folder="offload",
        # offload_state_dict=True
    )
    processor = AutoProcessor.from_pretrained(f"Qwen/Qwen2.5-VL-{size}-Instruct")
    return processor, model

def Qwen_generate(processor, model, messages, max_new_tokens):
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=24,
        max_pixels=832 * 480,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]
    
def load_InternVL3():
    # path = "OpenGVLab/InternVL3-78B"
    path = "OpenGVLab/InternVL3-78B-AWQ"
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return tokenizer, model

def InternVL3_generate(tokenizer, model, messages, max_new_tokens):
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=True)
    
    video_path = messages[0]['content'][0]['path']
    pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
    pixel_values = pixel_values.to(dtype=torch.bfloat16).cuda()
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    question = video_prefix + messages[0]['content'][1]['text']
    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=True)
    
    return response
    
if __name__ == "__main__":
    generator_list = ["SmolVLM", "Qwen", "InternVL3"]
    generator = generator_list[1]
    video_key = "path"
    
    if generator == "SmolVLM":
        processor, model = load_SmolVLM()
    
    elif generator == "Qwen":
        processor, model = load_Qwen(size="32B")
        video_key = "video"
        
    elif generator == "InternVL3":
        tokenizer, model = load_InternVL3()
    
    # PROMPT="Describe the video in detail, use the name \"Practice place\" to refer to the scene, specifying the set design and decoration including key props, furniture placement, and color scheme—along with the environment and time of day (location type, lighting style, weather, and ambient sounds). Note the spatial layout by describing camera angles, depth, and distances between objects for dramatic effect, then explain how characters move through the space, manipulate props, and react to environmental cues. Finally, show how these combined elements establish the scene’s mood and narrative significance."
    # PROMPT="Describe the video in detail, use the name Antonina Jarnuszkiewicz to refer to the woman protagonist with long hair, and man or woman refer to other characters focusing on the Antonina Jarnuszkiewicz’s facial expressions, body movements, and interactions with others. Include details about eye contact, gestures, and emotional changes. Also describe the setting, lighting, and background elements, explaining how they contribute to the scene’s mood and story."
    # PROMPT="Describe the video in detail, use the name Piotr Witkowski to refer to the man protagonist with short hair, and man or woman refer to other characters focusing on the Piotr Witkowski’s facial expressions, body movements, and interactions with others. Include details about eye contact, gestures, and emotional changes. Also describe the setting, lighting, and background elements, explaining how they contribute to the scene’s mood and story."
    # PROMPT="Describe the video in detail in less than 150 words, use the name Piotr Witkowski to refer to the male protagonist. Focusing on Piotr Witkowski’s facial expressions, body movements, and interactions with others. Do not include the title of the description, only use sentences."
    # special_prompt = "Use the name Piotr Witkowski to refer to the male protagonist."
    # special_prompt = ", use the name Antonina Jarnuszkiewicz to refer to the female protagonist."
    # special_prompt = ", use the name Practice place to refer to the scene."
    special_prompt = ""
    
    PROMPT=f"Describe the video in **less than 100 words**.{special_prompt} Focusing on the Character’s appearance, expressions, and emotions; the Action they perform and how their movements evolve; the Background setting and environmental elements; the Lighting quality, color temperature, and direction; and the Camera angles, framing, and motion that shape the scene."
    # PROMPT=f"Describe the video in **less than 100 words**.{special_prompt} Focusing on the room’s layout and décor; the Background setting and environmental elements; the Lighting quality, color temperature, and direction; and the Camera angles, framing, and motion that shape the scene."

    max_new_tokens = 256

    # dir_name = "preprocessed_peter"
    dir_name = "video"
    responses = {}
    
    for video_file in tqdm(os.listdir(f"/eva_data0/lynn/VideoGAI/DiffSynth-Studio/styledata/{dir_name}")):
        if not video_file.endswith(".mp4"):
            continue
        messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            video_key: f"/eva_data0/lynn/VideoGAI/DiffSynth-Studio/styledata/{dir_name}/{video_file}"
                        },
                        {
                            "type": "text",
                            "text": PROMPT
                        },
                    ],
                }
            ]
        
        with torch.no_grad():
            if generator == "SmolVLM":
                response = SmolVLM_generate(processor, model, messages, max_new_tokens=max_new_tokens)
            elif generator == "Qwen":
                response = Qwen_generate(processor, model, messages, max_new_tokens=max_new_tokens)
            elif generator == "InternVL3":
                response = InternVL3_generate(tokenizer, model, messages, max_new_tokens=max_new_tokens)

        responses[video_file] = response
        
    out_path = f"/eva_data0/lynn/VideoGAI/DiffSynth-Studio/outputStyle/{dir_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)
