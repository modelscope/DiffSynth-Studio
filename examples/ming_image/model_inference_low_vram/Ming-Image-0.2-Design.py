from diffsynth.pipelines.ming_image import MingImagePipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig
from PIL import Image
import torch

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
    "quantize": QuantizeConfig(method="entropack_lossy_quant_int8_raw"),
}
disk_vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": "disk",
    "onload_device": "disk",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MingImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", **vram_config),
        ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **disk_vram_config),
        ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="mllm/model*.safetensors", **disk_vram_config),
        ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="connector/model*.safetensors", **disk_vram_config),
        ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="mlp/model.safetensors", **disk_vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="inclusionAI/Ming-Image-0.2-Design", origin_file_pattern="mllm/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

# Text-to-Image with a natural-language prompt
prompt = """Landing page for a productivity app called Flow. Clean minimal design, white background, generous whitespace. Top navigation with the wordmark 'Flow' on the left and links Product, Pricing, Docs on the right. Centered hero with a bold headline 'Focus without the noise' and a gray subheading 'One calm workspace to plan, focus, and finish.' A single purple call-to-action button labeled 'Start free'. Below the hero, three feature cards with small icons, soft rounded corners and soft shadows. Card 1: title 'Plan your day', body text 'Turn scattered tasks into a clear, realistic plan.' Card 2: title 'Focus sessions', body text 'Start a distraction-free session and stay in flow.' Card 3: title 'See progress', body text 'Know what moved, what slipped, and what comes next.' Modern sans-serif typography. Keep every heading and body line short and legible. Render all text exactly as written. Do not invent additional copy."""
image = pipe(prompt=prompt, height=2048, width=2048, seed=0, num_inference_steps=12, cfg_scale=1.0)
image.save("image1.png")

# Text-to-Image with a structured Figma-style JSON prompt
prompt = """{
  "canvas_settings": {
    "aspect_ratio": "1:1, 2048 × 2048 px",
    "ambient_lighting": "Four season-specific lighting conditions unified by one fixed eye-level sun direction, consistent horizon at 48% canvas height, matching atmospheric depth, and natural cinematic exposure.",
    "image_style": "High-resolution cinematic photorealistic environmental concept sheet; four equal edge-to-edge vertical panels with crisp shared boundaries and no gutters, frames, extra title, or decorative inter-panel objects. Fixed 35 mm lens, eye-level three-quarter front-right camera angle, identical perspective and cabin registration in every panel. Each panel uses restrained bold uppercase condensed sans-serif labeling at the bottom."
  },
  "layers": [
    {
      "description": "Leftmost spring panel. The same small one-story rectangular log cabin appears at a fixed local position and scale: horizontal dark-brown logs, pale stone foundation, centered vertical-plank front door, two square four-pane front windows, one matching right-side window, and a charcoal standing-seam gabled roof with identical ridge height and overhang. Fresh pale-green grass, budding branches, white and pink blossoms, rain-darkened soil, shallow puddles, wet roof highlights, fine recent-rain droplets, and soft mist fill the landscape beneath an overcast clearing sky. Render the exact label \"SPRING\" centered near the bottom in bold condensed uppercase white sans-serif with a subtle dark shadow.",
      "coordinates": "cx: 0.125, cy: 0.500, w: 0.250, h: 1",
      "hierarchy_and_relation": "Complete first panel; aligned to the shared horizon and fixed cabin registration, flush with the canvas left edge and directly abutting the second panel without a gutter.",
      "color_specs": ["#A9C7D2", "#D8E4DE", "#7FAF69", "#B7D58B", "#F2D7DE", "#F4F1E8", "#5A3B28", "#2C3032", "#7B7468", "#FFFFFF"]
    },
    {
      "description": "Second summer panel. Repeat the cabin with exactly the same architecture, camera angle, local placement, dimensions, centered plank door, front and side windows, stone foundation, roof pitch, ridge height, and overhang. Surround it with lush meadow grass, dense layered deciduous foliage, full tree canopies, small sunlit shrubs, and dry natural ground; use a bright clear sky, strong warm sunlight from the same direction, crisp leaf highlights, controlled shadows, and mild atmospheric haze. Render the exact label \"SUMMER\" centered near the bottom in bold condensed uppercase white sans-serif with a subtle dark shadow.",
      "coordinates": "cx: 0.375, cy: 0.500, w: 0.250, h: 1",
      "hierarchy_and_relation": "Complete second panel; aligned to the shared horizon and fixed cabin registration, directly abutting the first and third panels without gutters.",
      "color_specs": ["#4FA9D8", "#BDE8F2", "#26733E", "#4F9A45", "#79B84A", "#D8C66A", "#5A3B28", "#2C3032", "#7B7468", "#FFFFFF"]
    },
    {
      "description": "Third autumn panel. Repeat the cabin with exactly the same architecture, camera angle, local placement, dimensions, centered plank door, front and side windows, stone foundation, roof pitch, ridge height, and overhang. Cover the landscape with amber grass, rust-orange and golden deciduous foliage, scattered fallen leaves, and a few leaves drifting naturally in the air. Use warm late-afternoon sunlight from the same direction, long soft shadows, copper rim light on logs and roof, a pale warm sky, and subtle atmospheric depth. Render the exact label \"AUTUMN\" centered near the bottom in bold condensed uppercase white sans-serif with a subtle dark shadow.",
      "coordinates": "cx: 0.625, cy: 0.500, w: 0.250, h: 1",
      "hierarchy_and_relation": "Complete third panel; aligned to the shared horizon and fixed cabin registration, directly abutting the second and fourth panels without gutters.",
      "color_specs": ["#E6B06A", "#F0D6A2", "#D66A2C", "#A84324", "#E2A62B", "#80613B", "#5A3B28", "#2C3032", "#7B7468", "#FFFFFF"]
    },
    {
      "description": "Rightmost winter panel. Repeat the cabin with exactly the same architecture, camera angle, local placement, dimensions, centered plank door, front and side windows, stone foundation, roof pitch, ridge height, and overhang. Blanket the ground and roof with natural snow accumulation while keeping doors and windows readable; add bare deciduous branches, snow-laden low shrubs, faint footprints away from the cabin, and sparse fine snowfall. Use cool blue ambient light, a pale clouded sky, soft low-contrast shadows from the same direction, icy edge highlights, and slight winter haze. Render the exact label \"WINTER\" centered near the bottom in bold condensed uppercase white sans-serif with a subtle dark shadow.",
      "coordinates": "cx: 0.875, cy: 0.500, w: 0.250, h: 1",
      "hierarchy_and_relation": "Complete fourth panel; aligned to the shared horizon and fixed cabin registration, directly abutting the third panel without a gutter and flush with the canvas right edge.",
      "color_specs": ["#AFC8DE", "#DCE8F1", "#F4F7F8", "#C6D8E5", "#8198AA", "#6D5949", "#5A3B28", "#2C3032", "#7B7468", "#FFFFFF"]
    }
  ]
}"""
image = pipe(prompt=prompt, height=2048, width=2048, seed=0, num_inference_steps=12, cfg_scale=1.0)
image.save("image2.png")

# Image Editing
prompt = "Change the app's color scheme from purple to green: the call-to-action button, icons and accents become green."
image = pipe(prompt=prompt, edit_image=Image.open("image1.png"), height=2048, width=2048, seed=1, num_inference_steps=12, cfg_scale=1.0)
image.save("image3.png")
