from dataclasses import dataclass
import importlib
import inspect


@dataclass(frozen=True)
class PipelineMeta:
    type_name: str
    display_name: str
    pipeline_class_path: str
    output_type: str


PIPELINE_REGISTRY = {
    item.type_name: item for item in [
        PipelineMeta("ZImage", "Z-Image", "diffsynth.pipelines.z_image.ZImagePipeline", "image"),
        PipelineMeta("Flux", "FLUX", "diffsynth.pipelines.flux_image.FluxImagePipeline", "image"),
        PipelineMeta("Flux2", "FLUX.2", "diffsynth.pipelines.flux2_image.Flux2ImagePipeline", "image"),
        PipelineMeta("QwenImage", "Qwen Image", "diffsynth.pipelines.qwen_image.QwenImagePipeline", "image"),
        PipelineMeta("ErnieImage", "ERNIE Image", "diffsynth.pipelines.ernie_image.ErnieImagePipeline", "image"),
        PipelineMeta("StableDiffusion", "Stable Diffusion", "diffsynth.pipelines.stable_diffusion.StableDiffusionPipeline", "image"),
        PipelineMeta("StableDiffusionXL", "Stable Diffusion XL", "diffsynth.pipelines.stable_diffusion_xl.StableDiffusionXLPipeline", "image"),
        PipelineMeta("AnimaImage", "Anima", "diffsynth.pipelines.anima_image.AnimaImagePipeline", "image"),
        PipelineMeta("BooguImage", "Boogu Image", "diffsynth.pipelines.boogu_image.BooguImagePipeline", "image"),
        PipelineMeta("JoyAIImage", "JoyAI Image", "diffsynth.pipelines.joyai_image.JoyAIImagePipeline", "image"),
        PipelineMeta("Krea2", "Krea 2", "diffsynth.pipelines.krea2.Krea2Pipeline", "image"),
        PipelineMeta("Ideogram4", "Ideogram 4", "diffsynth.pipelines.ideogram4.Ideogram4Pipeline", "image"),
        PipelineMeta("HiDreamO1", "HiDream O1", "diffsynth.pipelines.hidream_o1_image.HiDreamO1ImagePipeline", "image"),
        PipelineMeta("WanVideo", "Wan Video", "diffsynth.pipelines.wan_video.WanVideoPipeline", "video"),
        PipelineMeta("QwenVideoEdit", "Qwen Video Edit", "diffsynth.pipelines.qwen_video_edit.QwenVideoEditPipeline", "video"),
        PipelineMeta("LingBotVideo", "LingBot Video", "diffsynth.pipelines.lingbot_video.LingBotVideoPipeline", "video"),
        PipelineMeta("LTX2AudioVideo", "LTX-2 Audio Video", "diffsynth.pipelines.ltx2_audio_video.LTX2AudioVideoPipeline", "audio_video"),
        PipelineMeta("MiniMaxH3", "MiniMax H3", "diffsynth.pipelines.minimax_h3_audio_video.MiniMaxH3Pipeline", "audio_video"),
        PipelineMeta("MiniMaxMusic3", "MiniMax Music 3", "diffsynth.pipelines.minimax_music3.MiniMaxMusic3Pipeline", "audio"),
        PipelineMeta("MovaAudioVideo", "MOVA Audio Video", "diffsynth.pipelines.mova_audio_video.MovaAudioVideoPipeline", "audio_video"),
        PipelineMeta("AceStep", "ACE-Step", "diffsynth.pipelines.ace_step.AceStepPipeline", "audio"),
    ]
}


def get_pipeline_class(type_name):
    meta = PIPELINE_REGISTRY[type_name]
    module_name, class_name = meta.pipeline_class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def get_from_pretrained_signature(type_name):
    return inspect.signature(get_pipeline_class(type_name).from_pretrained)


def get_from_pretrained_extra_params(type_name):
    """Return [(name, type_str, default), ...] for extra params in from_pretrained.

    Excludes the common params: self/cls, torch_dtype, device, model_configs,
    vram_limit.  Type strings are: "MODEL_CONFIG", "BOOLEAN", "FLOAT", "STRING".
    """
    sig = get_from_pretrained_signature(type_name)
    common = {"self", "cls", "torch_dtype", "device", "model_configs", "vram_limit"}
    result = []
    for name, param in sig.parameters.items():
        if name in common:
            continue
        ann = param.annotation
        if name.endswith("_config") or "ModelConfig" in str(ann):
            result.append((name, "MODEL_CONFIG", param.default))
        elif ann is bool:
            result.append((name, "BOOLEAN", param.default))
        elif ann in (float, int):
            result.append((name, "FLOAT", param.default))
        else:
            result.append((name, "STRING", param.default))
    return result


def get_module_attr(type_name):
    return PIPELINE_REGISTRY[type_name].module_attr
