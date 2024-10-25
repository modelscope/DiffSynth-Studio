from diffsynth import ModelManager, FluxImagePipeline, ControlNetConfigUnit, download_models, download_customized_models
import torch
from PIL import Image
import numpy as np



def example_1():
    model_manager = ModelManager(torch_dtype=torch.bfloat16, model_id_list=["FLUX.1-dev", "jasperai/Flux.1-dev-Controlnet-Upscaler"])
    pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
        ControlNetConfigUnit(
            processor_id="tile",
            model_path="models/ControlNet/jasperai/Flux.1-dev-Controlnet-Upscaler/diffusion_pytorch_model.safetensors",
            scale=0.7
        ),
    ])

    image_1 = pipe(
        prompt="a photo of a cat, highly detailed",
        height=768, width=768,
        seed=0
    )
    image_1.save("image_1.jpg")

    image_2 = pipe(
        prompt="a photo of a cat, highly detailed",
        controlnet_image=image_1.resize((2048, 2048)),
        input_image=image_1.resize((2048, 2048)), denoising_strength=0.99,
        height=2048, width=2048, tiled=True,
        seed=1
    )
    image_2.save("image_2.jpg")



def example_2():
    model_manager = ModelManager(torch_dtype=torch.bfloat16, model_id_list=["FLUX.1-dev", "jasperai/Flux.1-dev-Controlnet-Upscaler"])
    pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
        ControlNetConfigUnit(
            processor_id="tile",
            model_path="models/ControlNet/jasperai/Flux.1-dev-Controlnet-Upscaler/diffusion_pytorch_model.safetensors",
            scale=0.7
        ),
    ])

    image_1 = pipe(
        prompt="a beautiful Chinese girl, delicate skin texture",
        height=768, width=768,
        seed=2
    )
    image_1.save("image_3.jpg")

    image_2 = pipe(
        prompt="a beautiful Chinese girl, delicate skin texture",
        controlnet_image=image_1.resize((2048, 2048)),
        input_image=image_1.resize((2048, 2048)), denoising_strength=0.99,
        height=2048, width=2048, tiled=True,
        seed=3
    )
    image_2.save("image_4.jpg")


def example_3():
    model_manager = ModelManager(torch_dtype=torch.bfloat16, model_id_list=["FLUX.1-dev", "InstantX/FLUX.1-dev-Controlnet-Union-alpha"])
    pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
        ControlNetConfigUnit(
            processor_id="canny",
            model_path="models/ControlNet/InstantX/FLUX.1-dev-Controlnet-Union-alpha/diffusion_pytorch_model.safetensors",
            scale=0.3
        ),
        ControlNetConfigUnit(
            processor_id="depth",
            model_path="models/ControlNet/InstantX/FLUX.1-dev-Controlnet-Union-alpha/diffusion_pytorch_model.safetensors",
            scale=0.3
        ),
    ])

    image_1 = pipe(
        prompt="a cat is running",
        height=1024, width=1024,
        seed=4
    )
    image_1.save("image_5.jpg")

    image_2 = pipe(
        prompt="sunshine, a cat is running",
        controlnet_image=image_1,
        height=1024, width=1024,
        seed=5
    )
    image_2.save("image_6.jpg")


def example_4():
    model_manager = ModelManager(torch_dtype=torch.bfloat16, model_id_list=["FLUX.1-dev", "InstantX/FLUX.1-dev-Controlnet-Union-alpha"])
    pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
        ControlNetConfigUnit(
            processor_id="canny",
            model_path="models/ControlNet/InstantX/FLUX.1-dev-Controlnet-Union-alpha/diffusion_pytorch_model.safetensors",
            scale=0.3
        ),
        ControlNetConfigUnit(
            processor_id="depth",
            model_path="models/ControlNet/InstantX/FLUX.1-dev-Controlnet-Union-alpha/diffusion_pytorch_model.safetensors",
            scale=0.3
        ),
    ])

    image_1 = pipe(
        prompt="a beautiful Asian girl, full body, red dress, summer",
        height=1024, width=1024,
        seed=6
    )
    image_1.save("image_7.jpg")

    image_2 = pipe(
        prompt="a beautiful Asian girl, full body, red dress, winter",
        controlnet_image=image_1,
        height=1024, width=1024,
        seed=7
    )
    image_2.save("image_8.jpg")



def example_5():
    model_manager = ModelManager(torch_dtype=torch.bfloat16, model_id_list=["FLUX.1-dev", "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta"])
    pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
        ControlNetConfigUnit(
            processor_id="inpaint",
            model_path="models/ControlNet/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta/diffusion_pytorch_model.safetensors",
            scale=0.9
        ),
    ])

    image_1 = pipe(
        prompt="a cat sitting on a chair",
        height=1024, width=1024,
        seed=8
    )
    image_1.save("image_9.jpg")

    mask = np.zeros((1024, 1024, 3), dtype=np.uint8)
    mask[100:350, 350: -300] = 255
    mask = Image.fromarray(mask)
    mask.save("mask_9.jpg")

    image_2 = pipe(
        prompt="a cat sitting on a chair, wearing sunglasses",
        controlnet_image=image_1, controlnet_inpaint_mask=mask,
        height=1024, width=1024,
        seed=9
    )
    image_2.save("image_10.jpg")



def example_6():
    model_manager = ModelManager(torch_dtype=torch.bfloat16, model_id_list=[
        "FLUX.1-dev",
        "jasperai/Flux.1-dev-Controlnet-Surface-Normals",
        "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta"
    ])
    pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
        ControlNetConfigUnit(
            processor_id="inpaint",
            model_path="models/ControlNet/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta/diffusion_pytorch_model.safetensors",
            scale=0.9
        ),
        ControlNetConfigUnit(
            processor_id="normal",
            model_path="models/ControlNet/jasperai/Flux.1-dev-Controlnet-Surface-Normals/diffusion_pytorch_model.safetensors",
            scale=0.6
        ),
    ])

    image_1 = pipe(
        prompt="a beautiful Asian woman looking at the sky, wearing a blue t-shirt.",
        height=1024, width=1024,
        seed=10
    )
    image_1.save("image_11.jpg")

    mask = np.zeros((1024, 1024, 3), dtype=np.uint8)
    mask[-400:, 10:-40] = 255
    mask = Image.fromarray(mask)
    mask.save("mask_11.jpg")

    image_2 = pipe(
        prompt="a beautiful Asian woman looking at the sky, wearing a yellow t-shirt.",
        controlnet_image=image_1, controlnet_inpaint_mask=mask,
        height=1024, width=1024,
        seed=11
    )
    image_2.save("image_12.jpg")


def example_7():
    model_manager = ModelManager(torch_dtype=torch.bfloat16, model_id_list=[
        "FLUX.1-dev",
        "InstantX/FLUX.1-dev-Controlnet-Union-alpha",
        "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
        "jasperai/Flux.1-dev-Controlnet-Upscaler",
    ])
    pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
        ControlNetConfigUnit(
            processor_id="inpaint",
            model_path="models/ControlNet/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta/diffusion_pytorch_model.safetensors",
            scale=0.9
        ),
        ControlNetConfigUnit(
            processor_id="canny",
            model_path="models/ControlNet/InstantX/FLUX.1-dev-Controlnet-Union-alpha/diffusion_pytorch_model.safetensors",
            scale=0.5
        ),
    ])

    image_1 = pipe(
        prompt="a beautiful Asian woman and a cat on a bed. The woman wears a dress.",
        height=1024, width=1024,
        seed=100
    )
    image_1.save("image_13.jpg")

    mask_global = np.zeros((1024, 1024, 3), dtype=np.uint8)
    mask_global = Image.fromarray(mask_global)
    mask_global.save("mask_13_global.jpg")

    mask_1 = np.zeros((1024, 1024, 3), dtype=np.uint8)
    mask_1[300:-100, 30: 450] = 255
    mask_1 = Image.fromarray(mask_1)
    mask_1.save("mask_13_1.jpg")

    mask_2 = np.zeros((1024, 1024, 3), dtype=np.uint8)
    mask_2[500:-100, -400:] = 255
    mask_2[-200:-100, -500:-400] = 255
    mask_2 = Image.fromarray(mask_2)
    mask_2.save("mask_13_2.jpg")

    image_2 = pipe(
        prompt="a beautiful Asian woman and a cat on a bed. The woman wears a dress.",
        controlnet_image=image_1, controlnet_inpaint_mask=mask_global,
        local_prompts=["an orange cat, highly detailed", "a girl wearing a red camisole"], masks=[mask_1, mask_2], mask_scales=[10.0, 10.0],
        height=1024, width=1024,
        seed=101
    )
    image_2.save("image_14.jpg")

    model_manager.load_lora("models/lora/FLUX-dev-lora-AntiBlur.safetensors", lora_alpha=2)
    image_3 = pipe(
        prompt="a beautiful Asian woman wearing a red camisole and an orange cat on a bed. clear background.",
        negative_prompt="blur, blurry",
        input_image=image_2, denoising_strength=0.7,
        height=1024, width=1024,
        cfg_scale=2.0, num_inference_steps=50,
        seed=102
    )
    image_3.save("image_15.jpg")

    pipe = FluxImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
        ControlNetConfigUnit(
            processor_id="tile",
            model_path="models/ControlNet/jasperai/Flux.1-dev-Controlnet-Upscaler/diffusion_pytorch_model.safetensors",
            scale=0.7
        ),
    ])
    image_4 = pipe(
        prompt="a beautiful Asian woman wearing a red camisole and an orange cat on a bed. highly detailed, delicate skin texture, clear background.",
        controlnet_image=image_3.resize((2048, 2048)),
        input_image=image_3.resize((2048, 2048)), denoising_strength=0.99,
        height=2048, width=2048, tiled=True,
        seed=103
    )
    image_4.save("image_16.jpg")

    image_5 = pipe(
        prompt="a beautiful Asian woman wearing a red camisole and an orange cat on a bed. highly detailed, delicate skin texture, clear background.",
        controlnet_image=image_4.resize((4096, 4096)),
        input_image=image_4.resize((4096, 4096)), denoising_strength=0.99,
        height=4096, width=4096, tiled=True,
        seed=104
    )
    image_5.save("image_17.jpg")



download_models(["Annotators:Depth", "Annotators:Normal"])
download_customized_models(
    model_id="LiblibAI/FLUX.1-dev-LoRA-AntiBlur",
    origin_file_path="FLUX-dev-lora-AntiBlur.safetensors",
    local_dir="models/lora"
)
example_1()
example_2()
example_3()
example_4()
example_5()
example_6()
example_7()
