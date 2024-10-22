# Pipelines

DiffSynth-Studio includes multiple pipelines, categorized into two types: image generation and video generation.

## Image Pipelines

| Pipeline                   | Models                                                     |
|----------------------------|----------------------------------------------------------------|
| SDImagePipeline             | text_encoder: SDTextEncoder<br>unet: SDUNet<br>vae_decoder: SDVAEDecoder<br>vae_encoder: SDVAEEncoder<br>controlnet: MultiControlNetManager<br>ipadapter_image_encoder: IpAdapterCLIPImageEmbedder<br>ipadapter: SDIpAdapter |
| SDXLImagePipeline           | text_encoder: SDXLTextEncoder<br>text_encoder_2: SDXLTextEncoder2<br>text_encoder_kolors: ChatGLMModel<br>unet: SDXLUNet<br>vae_decoder: SDXLVAEDecoder<br>vae_encoder: SDXLVAEEncoder<br>controlnet: MultiControlNetManager<br>ipadapter_image_encoder: IpAdapterXLCLIPImageEmbedder<br>ipadapter: SDXLIpAdapter |
| SD3ImagePipeline            | text_encoder_1: SD3TextEncoder1<br>text_encoder_2: SD3TextEncoder2<br>text_encoder_3: SD3TextEncoder3<br>dit: SD3DiT<br>vae_decoder: SD3VAEDecoder<br>vae_encoder: SD3VAEEncoder |
| HunyuanDiTImagePipeline     | text_encoder: HunyuanDiTCLIPTextEncoder<br>text_encoder_t5: HunyuanDiTT5TextEncoder<br>dit: HunyuanDiT<br>vae_decoder: SDVAEDecoder<br>vae_encoder: SDVAEEncoder |
| FluxImagePipeline     | text_encoder_1: FluxTextEncoder1<br>text_encoder_2: FluxTextEncoder2<br>dit: FluxDiT<br>vae_decoder: FluxVAEDecoder<br>vae_encoder: FluxVAEEncoder |

## Video Pipelines

| Pipeline                   | Models                                                     |
|----------------------------|----------------------------------------------------------------|
| SDVideoPipeline            | text_encoder: SDTextEncoder<br>unet: SDUNet<br>vae_decoder: SDVAEDecoder<br>vae_encoder: SDVAEEncoder<br>controlnet: MultiControlNetManager<br>ipadapter_image_encoder: IpAdapterCLIPImageEmbedder<br>ipadapter: SDIpAdapter<br>motion_modules: SDMotionModel |
| SDXLVideoPipeline          | text_encoder: SDXLTextEncoder<br>text_encoder_2: SDXLTextEncoder2<br>text_encoder_kolors: ChatGLMModel<br>unet: SDXLUNet<br>vae_decoder: SDXLVAEDecoder<br>vae_encoder: SDXLVAEEncoder<br>ipadapter_image_encoder: IpAdapterXLCLIPImageEmbedder<br>ipadapter: SDXLIpAdapter<br>motion_modules: SDXLMotionModel |
| SVDVideoPipeline           | image_encoder: SVDImageEncoder<br>unet: SVDUNet<br>vae_encoder: SVDVAEEncoder<br>vae_decoder: SVDVAEDecoder |
| CogVideoPipeline           | text_encoder: FluxTextEncoder2<br>dit: CogDiT<br>vae_encoder: CogVAEEncoder<br>vae_decoder: CogVAEDecoder |
