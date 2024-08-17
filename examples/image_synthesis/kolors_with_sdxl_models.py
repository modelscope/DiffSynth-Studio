from diffsynth import ModelManager, SDXLImagePipeline, download_models, ControlNetConfigUnit
import torch



def run_kolors_with_controlnet():
    download_models(["Kolors", "ControlNet_union_sdxl_promax"])
    model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                                file_path_list=[
                                    "models/kolors/Kolors/text_encoder",
                                    "models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors",
                                    "models/kolors/Kolors/vae/diffusion_pytorch_model.safetensors",
                                    "models/ControlNet/controlnet_union/diffusion_pytorch_model_promax.safetensors",
                                ])
    pipe = SDXLImagePipeline.from_model_manager(model_manager, controlnet_config_units=[
        ControlNetConfigUnit("depth", "models/ControlNet/controlnet_union/diffusion_pytorch_model_promax.safetensors", 0.6)
    ])
    negative_prompt = "半身，苍白的肤色，蜡黄的肤色，尸体，错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，错误的手指，口红，腮红"

    prompt = "一幅充满诗意美感的全身画，泛红的肤色，画中一位银色长发、蓝色眼睛、肤色红润、身穿蓝色吊带连衣裙的少女漂浮在水下，面向镜头，周围是光彩的气泡，和煦的阳光透过水面折射进水下"
    torch.manual_seed(7)
    image = pipe(
        prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=50, cfg_scale=4,
    )
    image.save("image.jpg")

    prompt = "一幅充满诗意美感的全身画，泛红的肤色，画中一位银色长发、黑色眼睛、肤色红润、身穿蓝色吊带连衣裙的少女，面向镜头，周围是绚烂的火焰"
    torch.manual_seed(0)
    image_controlnet = pipe(
        prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=50, cfg_scale=4,
        controlnet_image=image,
    )
    image_controlnet.save("image_depth_1.jpg")

    prompt = "一幅充满诗意美感的全身画，画中一位皮肤白皙、黑色长发、黑色眼睛、身穿金色吊带连衣裙的少女，周围是闪电，画面明亮"
    torch.manual_seed(1)
    image_controlnet = pipe(
        prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=50, cfg_scale=4,
        controlnet_image=image,
    )
    image_controlnet.save("image_depth_2.jpg")



def run_kolors_with_lora():
    download_models(["Kolors", "SDXL_lora_zyd232_ChineseInkStyle_SDXL_v1_0"])
    model_manager = ModelManager(torch_dtype=torch.float16, device="cuda",
                                file_path_list=[
                                    "models/kolors/Kolors/text_encoder",
                                    "models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors",
                                    "models/kolors/Kolors/vae/diffusion_pytorch_model.safetensors"
                                ])
    model_manager.load_lora("models/lora/zyd232_ChineseInkStyle_SDXL_v1_0.safetensors", lora_alpha=1.5)
    pipe = SDXLImagePipeline.from_model_manager(model_manager)

    prompt = "一幅充满诗意美感的全身画，泛红的肤色，画中一位银色长发、蓝色眼睛、肤色红润、身穿蓝色吊带连衣裙的少女漂浮在水下，面向镜头，周围是光彩的气泡，和煦的阳光透过水面折射进水下"
    negative_prompt = "半身，苍白的肤色，蜡黄的肤色，尸体，错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，错误的手指，口红，腮红"

    torch.manual_seed(7)
    image = pipe(
        prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=50, cfg_scale=4,
    )
    image.save("image_lora.jpg")



run_kolors_with_controlnet()
run_kolors_with_lora()
