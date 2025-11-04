VRAM_MANAGEMENT_MODULE_MAPS = {
    "diffsynth.models.qwen_image_dit.QwenImageDiT": {
        "diffsynth.models.qwen_image_dit.RMSNorm": "diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
    },
    "diffsynth.models.qwen_image_text_encoder.QwenImageTextEncoder": {
        "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Embedding": "diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLRotaryEmbedding": "diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2RMSNorm": "diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionPatchEmbed": "diffsynth.core.vram.layers.AutoWrappedModule",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionRotaryEmbedding": "diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "diffsynth.models.qwen_image_vae.QwenImageVAE": {
        "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
        "torch.nn.Conv3d": "diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Conv2d": "diffsynth.core.vram.layers.AutoWrappedModule",
        "diffsynth.models.qwen_image_vae.QwenImageRMS_norm": "diffsynth.core.vram.layers.AutoWrappedModule",
    },
    "diffsynth.models.qwen_image_controlnet.BlockWiseControlBlock": {
        "diffsynth.models.qwen_image_dit.RMSNorm": "diffsynth.core.vram.layers.AutoWrappedModule",
        "torch.nn.Linear": "diffsynth.core.vram.layers.AutoWrappedLinear",
    },
}
