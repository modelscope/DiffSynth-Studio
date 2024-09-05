# Pipelines

So far, the following table lists our pipelines and the models supported by each pipeline.

## Image Pipelines

Pipelines for generating images from text descriptions. Each pipeline relies on specific encoder and decoder models.

| Pipeline                   | Models                                                     |
|----------------------------|----------------------------------------------------------------|
| HunyuanDiTImagePipeline     | text_encoder: HunyuanDiTCLIPTextEncoder<br>text_encoder_t5: HunyuanDiTT5TextEncoder<br>dit: HunyuanDiT<br>vae_decoder: SDVAEDecoder<br>vae_encoder: SDVAEEncoder |
| SDImagePipeline             | text_encoder: SDTextEncoder<br>unet: SDUNet<br>vae_decoder: SDVAEDecoder<br>vae_encoder: SDVAEEncoder<br>controlnet: MultiControlNetManager<br>ipadapter_image_encoder: IpAdapterCLIPImageEmbedder<br>ipadapter: SDIpAdapter |
| SD3ImagePipeline            | text_encoder_1: SD3TextEncoder1<br>text_encoder_2: SD3TextEncoder2<br>text_encoder_3: SD3TextEncoder3<br>dit: SD3DiT<br>vae_decoder: SD3VAEDecoder<br>vae_encoder: SD3VAEEncoder |
| SDXLImagePipeline           | text_encoder: SDXLTextEncoder<br>text_encoder_2: SDXLTextEncoder2<br>text_encoder_kolors: ChatGLMModel<br>unet: SDXLUNet<br>vae_decoder: SDXLVAEDecoder<br>vae_encoder: SDXLVAEEncoder<br>controlnet: MultiControlNetManager<br>ipadapter_image_encoder: IpAdapterXLCLIPImageEmbedder<br>ipadapter: SDXLIpAdapter |

## Video Pipelines

Pipelines for generating videos from text descriptions. In addition to the models required for image generation, they include models for handling motion modules.

| Pipeline                   | Models                                                     |
|----------------------------|----------------------------------------------------------------|
| SDVideoPipeline            | text_encoder: SDTextEncoder<br>unet: SDUNet<br>vae_decoder: SDVAEDecoder<br>vae_encoder: SDVAEEncoder<br>controlnet: MultiControlNetManager<br>ipadapter_image_encoder: IpAdapterCLIPImageEmbedder<br>ipadapter: SDIpAdapter<br>motion_modules: SDMotionModel |
| SDXLVideoPipeline          | text_encoder: SDXLTextEncoder<br>text_encoder_2: SDXLTextEncoder2<br>text_encoder_kolors: ChatGLMModel<br>unet: SDXLUNet<br>vae_decoder: SDXLVAEDecoder<br>vae_encoder: SDXLVAEEncoder<br>ipadapter_image_encoder: IpAdapterXLCLIPImageEmbedder<br>ipadapter: SDXLIpAdapter<br>motion_modules: SDXLMotionModel |
| SVDVideoPipeline           | image_encoder: SVDImageEncoder<br>unet: SVDUNet<br>vae_encoder: SVDVAEEncoder<br>vae_decoder: SVDVAEDecoder |



