qwen_image_series = [
    {
        # Example: ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors")
        "model_hash": "0319a1cb19835fb510907dd3367c95ff",
        "model_name": "qwen_image_dit",
        "model_class": "diffsynth.models.qwen_image_dit.QwenImageDiT",
    },
    {
        # Example: ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors")
        "model_hash": "8004730443f55db63092006dd9f7110e",
        "model_name": "qwen_image_text_encoder",
        "model_class": "diffsynth.models.qwen_image_text_encoder.QwenImageTextEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.qwen_image_text_encoder.QwenImageTextEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors")
        "model_hash": "ed4ea5824d55ec3107b09815e318123a",
        "model_name": "qwen_image_vae",
        "model_class": "diffsynth.models.qwen_image_vae.QwenImageVAE",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Depth", origin_file_pattern="model.safetensors")
        "model_hash": "073bce9cf969e317e5662cd570c3e79c",
        "model_name": "qwen_image_blockwise_controlnet",
        "model_class": "diffsynth.models.qwen_image_controlnet.QwenImageBlockWiseControlNet",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Inpaint", origin_file_pattern="model.safetensors")
        "model_hash": "a9e54e480a628f0b956a688a81c33bab",
        "model_name": "qwen_image_blockwise_controlnet",
        "model_class": "diffsynth.models.qwen_image_controlnet.QwenImageBlockWiseControlNet",
        "extra_kwargs": {"additional_in_dim": 4},
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/General-Image-Encoders", origin_file_pattern="SigLIP2-G384/model.safetensors")
        "model_hash": "469c78b61e3e31bc9eec0d0af3d3f2f8",
        "model_name": "siglip2_image_encoder",
        "model_class": "diffsynth.models.siglip2_image_encoder.Siglip2ImageEncoder",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/General-Image-Encoders", origin_file_pattern="DINOv3-7B/model.safetensors")
        "model_hash": "5722b5c873720009de96422993b15682",
        "model_name": "dinov3_image_encoder",
        "model_class": "diffsynth.models.dinov3_image_encoder.DINOv3ImageEncoder",
    },
    {
        # Example: 
        "model_hash": "a166c33455cdbd89c0888a3645ca5c0f",
        "model_name": "qwen_image_image2lora_coarse",
        "model_class": "diffsynth.models.qwen_image_image2lora.QwenImageImage2LoRAModel",
    },
    {
        # Example: 
        "model_hash": "a5476e691767a4da6d3a6634a10f7408",
        "model_name": "qwen_image_image2lora_fine",
        "model_class": "diffsynth.models.qwen_image_image2lora.QwenImageImage2LoRAModel",
        "extra_kwargs": {"residual_length": 37*37+7, "residual_mid_dim": 64}
    },
    {
        # Example: 
        "model_hash": "0aad514690602ecaff932c701cb4b0bb",
        "model_name": "qwen_image_image2lora_style",
        "model_class": "diffsynth.models.qwen_image_image2lora.QwenImageImage2LoRAModel",
        "extra_kwargs": {"compress_dim": 64, "use_residual": False}
    },
    {
        # Example: ModelConfig(model_id="Qwen/Qwen-Image-Layered", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors")
        "model_hash": "8dc8cda05de16c73afa755e2c1ce2839",
        "model_name": "qwen_image_dit",
        "model_class": "diffsynth.models.qwen_image_dit.QwenImageDiT",
        "extra_kwargs": {"use_layer3d_rope": True, "use_additional_t_cond": True}
    },
    {
        # Example: ModelConfig(model_id="Qwen/Qwen-Image-Layered", origin_file_pattern="vae/diffusion_pytorch_model.safetensors")
        "model_hash": "44b39ddc499e027cfb24f7878d7416b9",
        "model_name": "qwen_image_vae",
        "model_class": "diffsynth.models.qwen_image_vae.QwenImageVAE",
        "extra_kwargs": {"image_channels": 4}
    },
]

wan_series = [
    {
        # Example: ModelConfig(model_id="krea/krea-realtime-video", origin_file_pattern="krea-realtime-video-14b.safetensors")
        "model_hash": "5ec04e02b42d2580483ad69f4e76346a",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
        "model_class": "diffsynth.models.wan_video_text_encoder.WanTextEncoder",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth")
        "model_hash": "ccc42284ea13e1ad04693284c7a09be6",
        "model_name": "wan_video_vae",
        "model_class": "diffsynth.models.wan_video_vae.WanVideoVAE",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vae.WanVideoVAEStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="meituan-longcat/LongCat-Video", origin_file_pattern="dit/diffusion_pytorch_model*.safetensors")
        "model_hash": "8b27900f680d7251ce44e2dc8ae1ffef",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.longcat_video_dit.LongCatVideoTransformer3DModel",
    },
    {
        # Example: ModelConfig(model_id="ByteDance/Video-As-Prompt-Wan2.1-14B", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors")
        "model_hash": "5f90e66a0672219f12d9a626c8c21f61",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTFromDiffusers"
    },
    {
        # Example: ModelConfig(model_id="ByteDance/Video-As-Prompt-Wan2.1-14B", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors")
        "model_hash": "5f90e66a0672219f12d9a626c8c21f61",
        "model_name": "wan_video_vap",
        "model_class": "diffsynth.models.wan_video_mot.MotWanModel",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_mot.WanVideoMotStateDictConverter"
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
        "model_hash": "5941c53e207d62f20f9025686193c40b",
        "model_name": "wan_video_image_encoder",
        "model_class": "diffsynth.models.wan_video_image_encoder.WanImageEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_image_encoder.WanImageEncoderStateDictConverter"
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1", origin_file_pattern="model.safetensors")
        "model_hash": "dbd5ec76bbf977983f972c151d545389",
        "model_name": "wan_video_motion_controller",
        "model_class": "diffsynth.models.wan_video_motion_controller.WanMotionControllerModel",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "9269f8db9040a9d860eaca435be61814",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06}
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-FLF2V-14B-720P", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "3ef3b1f8e1dab83d5b71fd7b617f859f",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'has_image_pos_emb': True}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-1.3B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "349723183fc063b2bfc10bb2835cf677",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 48, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-1.3B-InP", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "6d6ccde6845b95ad9114ab993d917893",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-14B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "efa44cddf936c70abd0ea28b6cbe946c",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 48, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-14B-InP", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "6bfcfb3b342cb286ce886889d519a77e",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "ac6a5aa74f4a0aab6f64eb9a72f19901",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 32, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06, 'has_ref_conv': False, 'add_control_adapter': True, 'in_dim_control_adapter': 24}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-1.3B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "70ddad9d3a133785da5ea371aae09504",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 48, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06, 'has_ref_conv': True}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control-Camera", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "b61c605c2adbd23124d152ed28e049ae",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 32, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'has_ref_conv': False, 'add_control_adapter': True, 'in_dim_control_adapter': 24}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "26bde73488a92e64cc20b0a7485b9e5b",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 48, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'has_ref_conv': True}
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "aafcfd9672c3a2456dc46e1cb6e52c70",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06}
    },
    {
        # Example: ModelConfig(model_id="iic/VACE-Wan2.1-1.3B-Preview", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "a61453409b67cd3246cf0c3bebad47ba",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 1536, 'ffn_dim': 8960, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 12, 'num_layers': 30, 'eps': 1e-06},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="iic/VACE-Wan2.1-1.3B-Preview", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "a61453409b67cd3246cf0c3bebad47ba",
        "model_name": "wan_video_vace",
        "model_class": "diffsynth.models.wan_video_vace.VaceWanModel",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vace.VaceWanModelDictConverter"
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "7a513e1f257a861512b1afd387a8ecd9",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 16, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "7a513e1f257a861512b1afd387a8ecd9",
        "model_name": "wan_video_vace",
        "model_class": "diffsynth.models.wan_video_vace.VaceWanModel",
        "extra_kwargs": {'vace_layers': (0, 5, 10, 15, 20, 25, 30, 35), 'vace_in_dim': 96, 'patch_size': (1, 2, 2), 'has_image_input': False, 'dim': 5120, 'num_heads': 40, 'ffn_dim': 13824, 'eps': 1e-06},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vace.VaceWanModelDictConverter"
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "31fa352acb8a1b1d33cd8764273d80a2",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': True, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_dit.WanVideoDiTStateDictConverter"
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-Animate-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "31fa352acb8a1b1d33cd8764273d80a2",
        "model_name": "wan_video_animate_adapter",
        "model_class": "diffsynth.models.wan_video_animate_adapter.WanAnimateAdapter",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_animate_adapter.WanAnimateAdapterStateDictConverter"
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control-Camera", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors")
        "model_hash": "47dbeab5e560db3180adf51dc0232fb1",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'has_ref_conv': False, 'add_control_adapter': True, 'in_dim_control_adapter': 24, 'require_clip_embedding': False}
    },
    {
        # Example: ModelConfig(model_id="PAI/Wan2.2-Fun-A14B-Control", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors")
        "model_hash": "2267d489f0ceb9f21836532952852ee5",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 52, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'has_ref_conv': True, 'require_clip_embedding': False},
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors")
        "model_hash": "5b013604280dd715f8457c6ed6d6a626",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 36, 'dim': 5120, 'ffn_dim': 13824, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 16, 'num_heads': 40, 'num_layers': 40, 'eps': 1e-06, 'require_clip_embedding': False}
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "966cffdcc52f9c46c391768b27637614",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit_s2v.WanS2VModel",
        "extra_kwargs": {'dim': 5120, 'in_dim': 16, 'ffn_dim': 13824, 'out_dim': 16, 'text_dim': 4096, 'freq_dim': 256, 'eps': 1e-06, 'patch_size': (1, 2, 2), 'num_heads': 40, 'num_layers': 40, 'cond_dim': 16, 'audio_dim': 1024, 'num_audio_token': 4}
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "1f5ab7703c6fc803fdded85ff040c316",
        "model_name": "wan_video_dit",
        "model_class": "diffsynth.models.wan_video_dit.WanModel",
        "extra_kwargs": {'has_image_input': False, 'patch_size': [1, 2, 2], 'in_dim': 48, 'dim': 3072, 'ffn_dim': 14336, 'freq_dim': 256, 'text_dim': 4096, 'out_dim': 48, 'num_heads': 24, 'num_layers': 30, 'eps': 1e-06, 'seperated_timestep': True, 'require_clip_embedding': False, 'require_vae_embedding': False, 'fuse_vae_embedding_in_latents': True}
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth")
        "model_hash": "e1de6c02cdac79f8b739f4d3698cd216",
        "model_name": "wan_video_vae",
        "model_class": "diffsynth.models.wan_video_vae.WanVideoVAE38",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wan_video_vae.WanVideoVAEStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/model.safetensors")
        "model_hash": "06be60f3a4526586d8431cd038a71486",
        "model_name": "wans2v_audio_encoder",
        "model_class": "diffsynth.models.wav2vec.WanS2VAudioEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.wans2v_audio_encoder.WanS2VAudioEncoderStateDictConverter",
    },
]

flux_series = [
    {
        # Example: ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors")
        "model_hash": "a29710fea6dddb0314663ee823598e50",
        "model_name": "flux_dit",
        "model_class": "diffsynth.models.flux_dit.FluxDiT",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_dit.FluxDiTStateDictConverter",
    },
    {
        # Supported due to historical reasons.
        "model_hash": "605c56eab23e9e2af863ad8f0813a25d",
        "model_name": "flux_dit",
        "model_class": "diffsynth.models.flux_dit.FluxDiT",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_dit.FluxDiTStateDictConverterFromDiffusers",
    },
    {
        # Example: ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors")
        "model_hash": "94eefa3dac9cec93cb1ebaf1747d7b78",
        "model_name": "flux_text_encoder_clip",
        "model_class": "diffsynth.models.flux_text_encoder_clip.FluxTextEncoderClip",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_text_encoder_clip.FluxTextEncoderClipStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/*.safetensors")
        "model_hash": "22540b49eaedbc2f2784b2091a234c7c",
        "model_name": "flux_text_encoder_t5",
        "model_class": "diffsynth.models.flux_text_encoder_t5.FluxTextEncoderT5",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_text_encoder_t5.FluxTextEncoderT5StateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors")
        "model_hash": "21ea55f476dfc4fd135587abb59dfe5d",
        "model_name": "flux_vae_encoder",
        "model_class": "diffsynth.models.flux_vae.FluxVAEEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_vae.FluxVAEEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors")
        "model_hash": "21ea55f476dfc4fd135587abb59dfe5d",
        "model_name": "flux_vae_decoder",
        "model_class": "diffsynth.models.flux_vae.FluxVAEDecoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_vae.FluxVAEDecoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="ostris/Flex.2-preview", origin_file_pattern="Flex.2-preview.safetensors")
        "model_hash": "d02f41c13549fa5093d3521f62a5570a",
        "model_name": "flux_dit",
        "model_class": "diffsynth.models.flux_dit.FluxDiT",
        "extra_kwargs": {'input_dim': 196, 'num_blocks': 8},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_dit.FluxDiTStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/AttriCtrl-FLUX.1-Dev", origin_file_pattern="models/brightness.safetensors")
        "model_hash": "0629116fce1472503a66992f96f3eb1a",
        "model_name": "flux_value_controller",
        "model_class": "diffsynth.models.flux_value_control.SingleValueEncoder",
    },
    {
        # Example: ModelConfig(model_id="alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", origin_file_pattern="diffusion_pytorch_model.safetensors")
        "model_hash": "52357cb26250681367488a8954c271e8",
        "model_name": "flux_controlnet",
        "model_class": "diffsynth.models.flux_controlnet.FluxControlNet",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_controlnet.FluxControlNetStateDictConverter",
        "extra_kwargs": {"num_joint_blocks": 6, "num_single_blocks": 0, "additional_input_dim": 4},
    },
    {
        # Example: ModelConfig(model_id="InstantX/FLUX.1-dev-Controlnet-Union-alpha", origin_file_pattern="diffusion_pytorch_model.safetensors")
        "model_hash": "78d18b9101345ff695f312e7e62538c0",
        "model_name": "flux_controlnet",
        "model_class": "diffsynth.models.flux_controlnet.FluxControlNet",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_controlnet.FluxControlNetStateDictConverter",
        "extra_kwargs": {"num_mode": 10, "mode_dict": {"canny": 0, "tile": 1, "depth": 2, "blur": 3, "pose": 4, "gray": 5, "lq": 6}},
    },
    {
        # Example: ModelConfig(model_id="jasperai/Flux.1-dev-Controlnet-Upscaler", origin_file_pattern="diffusion_pytorch_model.safetensors")
        "model_hash": "b001c89139b5f053c715fe772362dd2a",
        "model_name": "flux_controlnet",
        "model_class": "diffsynth.models.flux_controlnet.FluxControlNet",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_controlnet.FluxControlNetStateDictConverter",
        "extra_kwargs": {"num_single_blocks": 0},
    },
    {
        # Example: ModelConfig(model_id="ByteDance/InfiniteYou", origin_file_pattern="infu_flux_v1.0/aes_stage2/image_proj_model.bin")
        "model_hash": "c07c0f04f5ff55e86b4e937c7a40d481",
        "model_name": "infiniteyou_image_projector",
        "model_class": "diffsynth.models.flux_infiniteyou.InfiniteYouImageProjector",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_infiniteyou.FluxInfiniteYouImageProjectorStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="ByteDance/InfiniteYou", origin_file_pattern="infu_flux_v1.0/aes_stage2/InfuseNetModel/*.safetensors")
        "model_hash": "7f9583eb8ba86642abb9a21a4b2c9e16",
        "model_name": "flux_controlnet",
        "model_class": "diffsynth.models.flux_controlnet.FluxControlNet",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_controlnet.FluxControlNetStateDictConverter",
        "extra_kwargs": {"num_joint_blocks": 4, "num_single_blocks": 10},
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/LoRA-Encoder-FLUX.1-Dev", origin_file_pattern="model.safetensors")
        "model_hash": "77c2e4dd2440269eb33bfaa0d004f6ab",
        "model_name": "flux_lora_encoder",
        "model_class": "diffsynth.models.flux_lora_encoder.FluxLoRAEncoder",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/LoRAFusion-preview-FLUX.1-dev", origin_file_pattern="model.safetensors")
        "model_hash": "30143afb2dea73d1ac580e0787628f8c",
        "model_name": "flux_lora_patcher",
        "model_class": "diffsynth.models.flux_lora_patcher.FluxLoraPatcher",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="model*.safetensors")
        "model_hash": "2bd19e845116e4f875a0a048e27fc219",
        "model_name": "nexus_gen_llm",
        "model_class": "diffsynth.models.nexus_gen.NexusGenAutoregressiveModel",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.nexus_gen.NexusGenAutoregressiveModelStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="edit_decoder.bin")
        "model_hash": "63c969fd37cce769a90aa781fbff5f81",
        "model_name": "nexus_gen_editing_adapter",
        "model_class": "diffsynth.models.nexus_gen_projector.NexusGenImageEmbeddingMerger",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.nexus_gen_projector.NexusGenMergerStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="edit_decoder.bin")
        "model_hash": "63c969fd37cce769a90aa781fbff5f81",
        "model_name": "flux_dit",
        "model_class": "diffsynth.models.flux_dit.FluxDiT",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_dit.FluxDiTStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="generation_decoder.bin")
        "model_hash": "3e6c61b0f9471135fc9c6d6a98e98b6d",
        "model_name": "nexus_gen_generation_adapter",
        "model_class": "diffsynth.models.nexus_gen_projector.NexusGenAdapter",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.nexus_gen_projector.NexusGenAdapterStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="generation_decoder.bin")
        "model_hash": "3e6c61b0f9471135fc9c6d6a98e98b6d",
        "model_name": "flux_dit",
        "model_class": "diffsynth.models.flux_dit.FluxDiT",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_dit.FluxDiTStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="InstantX/FLUX.1-dev-IP-Adapter", origin_file_pattern="ip-adapter.bin")
        "model_hash": "4daaa66cc656a8fe369908693dad0a35",
        "model_name": "flux_ipadapter",
        "model_class": "diffsynth.models.flux_ipadapter.FluxIpAdapter",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_ipadapter.FluxIpAdapterStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="google/siglip-so400m-patch14-384", origin_file_pattern="model.safetensors")
        "model_hash": "04d8c1e20a1f1b25f7434f111992a33f",
        "model_name": "siglip_vision_model",
        "model_class": "diffsynth.models.flux_ipadapter.SiglipVisionModelSO400M",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_ipadapter.SiglipStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="stepfun-ai/Step1X-Edit", origin_file_pattern="step1x-edit-i1258.safetensors"),
        "model_hash": "d30fb9e02b1dbf4e509142f05cf7dd50",
        "model_name": "step1x_connector",
        "model_class": "diffsynth.models.step1x_connector.Qwen2Connector",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.step1x_connector.Qwen2ConnectorStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="stepfun-ai/Step1X-Edit", origin_file_pattern="step1x-edit-i1258.safetensors"),
        "model_hash": "d30fb9e02b1dbf4e509142f05cf7dd50",
        "model_name": "flux_dit",
        "model_class": "diffsynth.models.flux_dit.FluxDiT",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_dit.FluxDiTStateDictConverter",
        "extra_kwargs": {"disable_guidance_embedder": True},
    },
    {
        # Example: ModelConfig(model_id="MAILAND/majicflus_v1", origin_file_pattern="majicflus_v134.safetensors")
        "model_hash": "3394f306c4cbf04334b712bf5aaed95f",
        "model_name": "flux_dit",
        "model_class": "diffsynth.models.flux_dit.FluxDiT",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_dit.FluxDiTStateDictConverter",
    },
]

flux2_series = [
    {
        # Example: ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="text_encoder/*.safetensors")
        "model_hash": "28fca3d8e5bf2a2d1271748a773f6757",
        "model_name": "flux2_text_encoder",
        "model_class": "diffsynth.models.flux2_text_encoder.Flux2TextEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux2_text_encoder.Flux2TextEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="transformer/*.safetensors")
        "model_hash": "d38e1d5c5aec3b0a11e79327ac6e3b0f",
        "model_name": "flux2_dit",
        "model_class": "diffsynth.models.flux2_dit.Flux2DiT",
    },
    {
        # Example: ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="vae/diffusion_pytorch_model.safetensors")
        "model_hash": "c54288e3ee12ca215898840682337b95",
        "model_name": "flux2_vae",
        "model_class": "diffsynth.models.flux2_vae.Flux2VAE",
    },
    {
        # Example: ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="transformer/*.safetensors")
        "model_hash": "3bde7b817fec8143028b6825a63180df",
        "model_name": "flux2_dit",
        "model_class": "diffsynth.models.flux2_dit.Flux2DiT",
        "extra_kwargs": {"guidance_embeds": False, "joint_attention_dim": 7680, "num_attention_heads": 24, "num_layers": 5, "num_single_layers": 20}
    },
    {
        # Example: ModelConfig(model_id="black-forest-labs/FLUX.2-klein-9B", origin_file_pattern="text_encoder/*.safetensors")
        "model_hash": "9195f3ea256fcd0ae6d929c203470754",
        "model_name": "z_image_text_encoder",
        "model_class": "diffsynth.models.z_image_text_encoder.ZImageTextEncoder",
        "extra_kwargs": {"model_size": "8B"},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.z_image_text_encoder.ZImageTextEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="black-forest-labs/FLUX.2-klein-9B", origin_file_pattern="transformer/*.safetensors")
        "model_hash": "39c6fc48f07bebecedbbaa971ff466c8",
        "model_name": "flux2_dit",
        "model_class": "diffsynth.models.flux2_dit.Flux2DiT",
        "extra_kwargs": {"guidance_embeds": False, "joint_attention_dim": 12288, "num_attention_heads": 32, "num_layers": 8, "num_single_layers": 24}
    },
]

z_image_series = [
    {
        # Example: ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/*.safetensors")
        "model_hash": "fc3a8a1247fe185ce116ccbe0e426c28",
        "model_name": "z_image_dit",
        "model_class": "diffsynth.models.z_image_dit.ZImageDiT",
    },
    {
        # Example: ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors")
        "model_hash": "0f050f62a88876fea6eae0a18dac5a2e",
        "model_name": "z_image_text_encoder",
        "model_class": "diffsynth.models.z_image_text_encoder.ZImageTextEncoder",
    },
    {
        # Example: ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/vae/diffusion_pytorch_model.safetensors")
        "model_hash": "1aafa3cc91716fb6b300cc1cd51b85a3",
        "model_name": "flux_vae_encoder",
        "model_class": "diffsynth.models.flux_vae.FluxVAEEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_vae.FluxVAEEncoderStateDictConverterDiffusers",
        "extra_kwargs": {"use_conv_attention": False},
    },
    {
        # Example: ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/vae/diffusion_pytorch_model.safetensors")
        "model_hash": "1aafa3cc91716fb6b300cc1cd51b85a3",
        "model_name": "flux_vae_decoder",
        "model_class": "diffsynth.models.flux_vae.FluxVAEDecoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.flux_vae.FluxVAEDecoderStateDictConverterDiffusers",
        "extra_kwargs": {"use_conv_attention": False},
    },
    {
        # Example: ModelConfig(model_id="Tongyi-MAI/Z-Image-Omni-Base", origin_file_pattern="transformer/*.safetensors")
        "model_hash": "aa3563718e5c3ecde3dfbb020ca61180",
        "model_name": "z_image_dit",
        "model_class": "diffsynth.models.z_image_dit.ZImageDiT",
        "extra_kwargs": {"siglip_feat_dim": 1152},
    },
    {
        # Example: ModelConfig(model_id="Tongyi-MAI/Z-Image-Omni-Base", origin_file_pattern="siglip/model.safetensors")
        "model_hash": "89d48e420f45cff95115a9f3e698d44a",
        "model_name": "siglip_vision_model_428m",
        "model_class": "diffsynth.models.siglip2_image_encoder.Siglip2ImageEncoder428M",
    },
    {
        # Example: ModelConfig(model_id="PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1", origin_file_pattern="Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors")
        "model_hash": "1677708d40029ab380a95f6c731a57d7",
        "model_name": "z_image_controlnet",
        "model_class": "diffsynth.models.z_image_controlnet.ZImageControlNet",
    },
    {
        # Example: ???
        "model_hash": "9510cb8cd1dd34ee0e4f111c24905510",
        "model_name": "z_image_image2lora_style",
        "model_class": "diffsynth.models.z_image_image2lora.ZImageImage2LoRAModel",
        "extra_kwargs": {"compress_dim": 128},
    },
    {
        # Example: ModelConfig(model_id="Qwen/Qwen3-0.6B", origin_file_pattern="model.safetensors")
        "model_hash": "1392adecee344136041e70553f875f31",
        "model_name": "z_image_text_encoder",
        "model_class": "diffsynth.models.z_image_text_encoder.ZImageTextEncoder",
        "extra_kwargs": {"model_size": "0.6B"},
        "state_dict_converter": "diffsynth.utils.state_dict_converters.z_image_text_encoder.ZImageTextEncoderStateDictConverter",
    },
]

ltx2_series = [
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors")
        "model_hash": "aca7b0bbf8415e9c98360750268915fc",
        "model_name": "ltx2_dit",
        "model_class": "diffsynth.models.ltx2_dit.LTXModel",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_dit.LTXModelStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors")
        "model_hash": "aca7b0bbf8415e9c98360750268915fc",
        "model_name": "ltx2_video_vae_encoder",
        "model_class": "diffsynth.models.ltx2_video_vae.LTX2VideoEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_video_vae.LTX2VideoEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors")
        "model_hash": "aca7b0bbf8415e9c98360750268915fc",
        "model_name": "ltx2_video_vae_decoder",
        "model_class": "diffsynth.models.ltx2_video_vae.LTX2VideoDecoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_video_vae.LTX2VideoDecoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors")
        "model_hash": "aca7b0bbf8415e9c98360750268915fc",
        "model_name": "ltx2_audio_vae_decoder",
        "model_class": "diffsynth.models.ltx2_audio_vae.LTX2AudioDecoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_audio_vae.LTX2AudioDecoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors")
        "model_hash": "aca7b0bbf8415e9c98360750268915fc",
        "model_name": "ltx2_audio_vocoder",
        "model_class": "diffsynth.models.ltx2_audio_vae.LTX2Vocoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_audio_vae.LTX2VocoderStateDictConverter",
    },
    # { # not used currently
    #     # Example: ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors")
    #     "model_hash": "aca7b0bbf8415e9c98360750268915fc",
    #     "model_name": "ltx2_audio_vae_encoder",
    #     "model_class": "diffsynth.models.ltx2_audio_vae.LTX2AudioEncoder",
    #     "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_audio_vae.LTX2AudioEncoderStateDictConverter",
    # },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors")
        "model_hash": "aca7b0bbf8415e9c98360750268915fc",
        "model_name": "ltx2_text_encoder_post_modules",
        "model_class": "diffsynth.models.ltx2_text_encoder.LTX2TextEncoderPostModules",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_text_encoder.LTX2TextEncoderPostModulesStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized", origin_file_pattern="model-*.safetensors")
        "model_hash": "33917f31c4a79196171154cca39f165e",
        "model_name": "ltx2_text_encoder",
        "model_class": "diffsynth.models.ltx2_text_encoder.LTX2TextEncoder",
        "state_dict_converter": "diffsynth.utils.state_dict_converters.ltx2_text_encoder.LTX2TextEncoderStateDictConverter",
    },
    {
        # Example: ModelConfig(model_id="Lightricks/LTX-2", origin_file_pattern="ltx-2-19b-dev.safetensors")
        "model_hash": "c79c458c6e99e0e14d47e676761732d2",
        "model_name": "ltx2_latent_upsampler",
        "model_class": "diffsynth.models.ltx2_upsampler.LTX2LatentUpsampler",
    },
]
MODEL_CONFIGS = qwen_image_series + wan_series + flux_series + flux2_series + z_image_series + ltx2_series
