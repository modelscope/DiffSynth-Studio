from diffsynth.pipelines.ace_step import AceStepPipeline, ModelConfig
from diffsynth.utils.data.audio import save_audio
from diffsynth import load_state_dict
import torch


pipe = AceStepPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="ACE-Step/acestep-v15-turbo-shift1", origin_file_pattern="model.safetensors"),
        ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="Qwen3-Embedding-0.6B/model.safetensors"),
        ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    text_tokenizer_config=ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="Qwen3-Embedding-0.6B/"),
    silence_latent_config=ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="acestep-v15-turbo/silence_latent.pt"),
)
state_dict = load_state_dict("models/train/acestep-v15-turbo-shift1_full/epoch-1.safetensors")
pipe.dit.load_state_dict(state_dict)

prompt = "An explosive, high-energy pop-rock track with a strong anime theme song feel. The song kicks off with a catchy, synthesized brass fanfare over a driving rock beat with punchy drums and a solid bassline. A powerful, clear male vocal enters with a theatrical and energetic delivery, soaring through the verses and hitting powerful high notes in the chorus. The arrangement is dense and dynamic, featuring rhythmic electric guitar chords, brief instrumental breaks with synth flourishes, and a consistent, danceable groove throughout. The overall mood is triumphant, adventurous, and exhilarating."
lyrics = '[Intro - Synth Brass Fanfare]\n\n[Verse 1]\n黑夜里的风吹过耳畔\n甜蜜时光转瞬即万\n脚步飘摇在星光上\n心追节奏心跳狂乱\n耳边传来电吉他呼唤\n手指轻触碰点流点燃\n梦在云端任它蔓延\n疯狂跳跃自由无间\n\n[Chorus]\n心电感应在震动间\n拥抱未来勇敢冒险\n那旋律在心中无限\n世界变得如此耀眼\n\n[Instrumental Break - Synth Brass Melody]\n\n[Verse 2]\n鼓点撞击黑夜的底端\n跳动节拍连接你我俩\n在这里让灵魂发光\n燃尽所有不留遗憾\n\n[Instrumental Break - Synth Brass Melody]\n\n[Bridge]\n光影交错彼此的视线\n霓虹之下夜空的蔚蓝\n月光洒下温热心田\n追逐梦想它不会遥远\n\n[Chorus]\n心电感应在震动间\n拥抱未来勇敢冒险\n那旋律在心中无限\n世界变得如此耀眼\n\n[Outro - Instrumental with Synth Brass Melody]\n[Song ends abruptly]'
audio = pipe(
    prompt=prompt,
    lyrics=lyrics,
    duration=160,
    bpm=100,
    keyscale="B minor",
    timesignature="4",
    vocal_language="zh",
    seed=1,
    num_inference_steps=30,
    cfg_scale=4.0,
    shift=1,
)
save_audio(audio, pipe.vae.sampling_rate, "acestep-v15-turbo-shift1_full.wav")
