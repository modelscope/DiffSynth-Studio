from diffsynth import ModelManager, FluxImagePipeline, download_customized_models
from modelscope import dataset_snapshot_download
from examples.EntityControl.utils import visualize_masks
from PIL import Image
import torch

def example(pipe, seeds, example_id, global_prompt, entity_prompts):
    dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth", local_dir="./", allow_file_pattern=f"data/examples/eligen/entity_control/example_{example_id}/*.png")
    masks = [Image.open(f"./data/examples/eligen/entity_control/example_{example_id}/{i}.png").convert('RGB') for i in range(len(entity_prompts))]
    negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"
    for seed in seeds:
        # generate image
        image = pipe(
            prompt=global_prompt,
            cfg_scale=3.0,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            embedded_guidance=3.5,
            seed=seed,
            height=1024,
            width=1024,
            eligen_entity_prompts=entity_prompts,
            eligen_entity_masks=masks,
        )
        image.save(f"styled_eligen_example_{example_id}_{seed}.png")
        visualize_masks(image, masks, entity_prompts, f"styled_entity_control_example_{example_id}_mask_{seed}.png")

# download and load model
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda", model_id_list=["FLUX.1-dev"])
model_manager.load_lora(
    download_customized_models(
        model_id="FluxLora/merve-flux-lego-lora-dreambooth",
        origin_file_path="pytorch_lora_weights.safetensors",
        local_dir="models/lora/merve-flux-lego-lora-dreambooth"
    ),
    lora_alpha=1
)
model_manager.load_lora(
    download_customized_models(
        model_id="DiffSynth-Studio/Eligen",
        origin_file_path="model_bf16.safetensors",
        local_dir="models/lora/entity_control"
    ),
    lora_alpha=1
)
pipe = FluxImagePipeline.from_model_manager(model_manager)

# example 1
trigger_word = "lego set in style of TOK, "
global_prompt = "A breathtaking beauty of Raja Ampat by the late-night moonlight , one beautiful woman from behind wearing a pale blue long dress with soft glow, sitting at the top of a cliff looking towards the beach,pastell light colors, a group of small distant birds flying in far sky, a boat sailing on the sea, best quality, realistic, whimsical, fantastic, splash art, intricate detailed, hyperdetailed, maximalist style, photorealistic, concept art, sharp focus, harmony, serenity, tranquility, soft pastell colors,ambient occlusion, cozy ambient lighting, masterpiece, liiv1, linquivera, metix, mentixis, masterpiece, award winning, view from above\n"
global_prompt = trigger_word + global_prompt
entity_prompts = ["cliff", "sea", "moon", "sailing boat", "a seated beautiful woman", "pale blue long dress with soft glow"]
example(pipe, [0], 1, global_prompt, entity_prompts)

# example 2
global_prompt = "samurai girl wearing a kimono, she's holding a sword  glowing with red flame, her long hair is flowing in the wind, she is looking at a small bird perched on the back of her hand. ultra realist style. maximum image detail. maximum realistic render."
global_prompt = trigger_word + global_prompt
entity_prompts = ["flowing hair", "sword glowing with red flame", "A cute bird", "blue belt"]
example(pipe, [0], 2, global_prompt, entity_prompts)

# example 3
global_prompt = "Image of a neverending staircase up to a mysterious palace in the sky, The ancient palace stood majestically atop a mist-shrouded mountain, sunrise, two traditional monk walk in the stair looking at the sunrise, fog,see-through, best quality, whimsical, fantastic, splash art, intricate detailed, hyperdetailed, photorealistic, concept art, harmony, serenity, tranquility, ambient occlusion, halation, cozy ambient lighting, dynamic lighting,masterpiece, liiv1, linquivera, metix, mentixis, masterpiece, award winning,"
global_prompt = trigger_word + global_prompt
entity_prompts = ["ancient palace", "stone staircase with railings", "a traditional monk", "a traditional monk"]
example(pipe, [27], 3, global_prompt, entity_prompts)

# example 4
global_prompt = "A beautiful girl wearing shirt and shorts in the street,  holding a sign 'Entity Control'"
global_prompt = trigger_word + global_prompt
entity_prompts = ["A beautiful girl", "sign 'Entity Control'", "shorts", "shirt"]
example(pipe, [21], 4, global_prompt, entity_prompts)

# example 5
global_prompt = "A captivating, dramatic scene in a painting that exudes mystery and foreboding. A white sky, swirling blue clouds, and a crescent yellow moon illuminate a solitary woman standing near the water's edge. Her long dress flows in the wind, silhouetted against the eerie glow. The water mirrors the fiery sky and moonlight, amplifying the uneasy atmosphere."
global_prompt = trigger_word + global_prompt
entity_prompts = ["crescent yellow moon", "a solitary woman", "water", "swirling blue clouds"]
example(pipe, [0], 5, global_prompt, entity_prompts)

# example 6
global_prompt = "Snow White and the 6 Dwarfs."
global_prompt = trigger_word + global_prompt
entity_prompts = ["Dwarf 1", "Dwarf 2", "Dwarf 3", "Snow White", "Dwarf 4", "Dwarf 5", "Dwarf 6"]
example(pipe, [8], 6, global_prompt, entity_prompts)

# example 7, same prompt with different seeds
seeds = range(5, 9)
global_prompt = "A beautiful woman wearing white dress, holding a mirror, with a warm light background;"
global_prompt = trigger_word + global_prompt
entity_prompts = ["A beautiful woman", "mirror", "necklace", "glasses", "earring", "white dress", "jewelry headpiece"]
example(pipe, seeds, 7, global_prompt, entity_prompts)
