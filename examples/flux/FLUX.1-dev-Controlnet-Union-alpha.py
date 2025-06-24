import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig, ControlNetInput
from diffsynth.controlnets.processors import Annotator
from diffsynth import download_models



download_models(["Annotators:Depth"])
pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ModelConfig(model_id="InstantX/FLUX.1-dev-Controlnet-Union-alpha", origin_file_pattern="diffusion_pytorch_model.safetensors"),
    ],
)

image_1 = pipe(
    prompt="a beautiful Asian girl, full body, red dress, summer",
    height=1024, width=1024,
    seed=6, rand_device="cuda",
)
image_1.save("image_1.jpg")

image_canny = Annotator("canny")(image_1)
image_depth = Annotator("depth")(image_1)

image_2 = pipe(
    prompt="a beautiful Asian girl, full body, red dress, winter",
    controlnet_inputs=[
        ControlNetInput(image=image_canny, scale=0.3, processor_id="canny"),
        ControlNetInput(image=image_depth, scale=0.3, processor_id="depth"),
    ],
    height=1024, width=1024,
    seed=7, rand_device="cuda",
)
image_2.save("image_2.jpg")
