import torch
from PIL import Image
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio, read_audio, read_video_audio
from modelscope import dataset_snapshot_download

vram_config = {
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/transformer/model*.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config),
        ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
)
dataset_snapshot_download(dataset_id="DiffSynth-Studio/diffsynth_example_dataset", local_dir="data/diffsynth_example_dataset", allow_file_pattern="minimax_h3/MiniMax-H3-Retake/*")

num_frames = 124
# Audio -> Video + Audio
prompt = "integrated_multimodal_description: [Shot 1] Live-action, cinematic, the shot opens on a medium close-up that frames a young woman from the chest up, her head and shoulders filling most of the frame and her face turned directly toward the lens, standing in front of a softly blurred blooming cherry tree. Soft, even frontal daylight falls across her face so that her eyes, mouth and chin stay clearly readable, and her dark hair is tied back so that nothing crosses her face. The camera holds a static shot on her face for the entire take. The young woman with a bright, airy singing voice (S1) sings: <d>[English] Mummy don't know daddy's getting hot. At the body shop</d> Her lips, jaw and teeth follow every syllable of the sung line, her mouth stays fully visible and unobstructed from the first frame to the last, and her eyebrows and eyes move with the phrasing. As she sings, she sways her shoulders in time, tilts her head, and lifts one hand to chest height and lowers it again without ever passing it in front of her mouth, drawing a visible breath between phrases. A few petals drift past her shoulders in the blurred background.\n\noverall_soundscape: A light breeze moves through the blossoms behind her with a soft, continuous rustle. Her clothing shifts quietly as she sways, and a short intake of breath is audible between phrases.\n\nnon_diegetic_music: N/A"
source_audio, sample_rate = read_audio("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Retake/source_audio.mp3")
video, audio = pipe(
    prompt=prompt,
    height=832, width=480, num_frames=num_frames, num_inference_steps=20, seed=0,
    retake_audio=source_audio,
    retake_audio_sample_rate=sample_rate,
)
write_video_audio(
    video=video, audio=audio,
    output_path="retake_a2va.mp4", fps=24, audio_sample_rate=32000,
)

# Audio + First frame -> Video + Audio
prompt = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\nintegrated_multimodal_description: [Shot 1] Live-action, cinematic, a square frame holds the young woman shown in <Picture 1> submerged just below the water surface, preserving her face, her pale skin, her long dark hair fanning out to both side edges of the frame, the pale blue chiffon dress with its draped sleeves, the rippling surface running across the top edge and the teal water behind her. She is framed from the chest up, her head near the upper centre of the square and her face turned toward the lens. Shifting sunlight caustics slide across her forehead, cheeks and lips, and fine bubbles drift upward past her shoulders. The camera holds a static shot on her face for the entire take, keeping her head in the same position within the square frame. The young woman with a bright, airy singing voice (S1) sings: <d>[English] Mummy don't know daddy's getting hot. At the body shop</d> Her lips, jaw and teeth follow every syllable of the sung line and her mouth stays fully visible and unobstructed, with only a thin thread of tiny bubbles slipping from one corner of her mouth and rising clear of her face. As she sings, her hair drifts slowly in the water without ever crossing her face, her shoulders rise and settle as the water lifts her, her sleeves billow around her arms, and her eyes stay on the lens while she drifts slightly closer to the surface by the end of the shot.\n\noverall_soundscape: A low, muffled water ambience sits far under the voice, with the faint trickle of bubbles rising past her. Fabric and water swirl quietly around her arms.\n\nnon_diegetic_music: N/A"
source_audio, sample_rate = read_audio("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Retake/source_audio.mp3")
first_frame = Image.open("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Retake/first.jpg").convert("RGB").resize((640, 640))
video, audio = pipe(
    prompt=prompt,
    height=640, width=640, num_frames=num_frames, num_inference_steps=20, seed=0,
    keyframes=[first_frame],
    keyframe_indices=[0],
    retake_audio=source_audio,
    retake_audio_sample_rate=sample_rate,
)
write_video_audio(
    video=video, audio=audio,
    output_path="retake_a2va_firstframe.mp4", fps=24, audio_sample_rate=32000,
)


# Video -> Video + Audio
prompt = "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a handheld medium shot follows two swordsmen exchanging blows on a stone courtyard at dusk. The fighter on the left drives forward and swings a straight steel sword downward; the fighter on the right raises his blade and parries it so that the two edges meet edge-on. He turns the parry aside, steps in and cuts across at waist height, and the second blade catches it so that the edges grind along each other before separating. Both fighters break apart with their boots scraping over grit, then close again: the left fighter thrusts, the right fighter beats the point aside with a short flick of his blade, drives his shoulder into the other's chest, and both stagger back. The exchange ends with a heavy overhead clash that locks the two blades together above their heads while their arms shake against the pressure. Their mouths open in short bursts of effort but no words are spoken.\n\noverall_soundscape: Steel rings sharply on steel at each blade contact, with a long scraping rasp where the edges grind together and a duller, heavier ring on the final locked clash. Boots scrape and pivot over grit, leather and cloth creak with each swing, the shoulder impact lands as a blunt thud, and both fighters breathe hard with short grunts of effort throughout. A low evening wind moves through the courtyard behind them.\n\nnon_diegetic_music: A distorted electric guitar riff at a fast tempo over a double-kick drum pattern and a sustained synth bass drone. The riff drops out for a beat just before the final impact, then returns with crash cymbals that ring out to the end."
video, _, _ = read_video_audio("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Retake/source_video.mp4", height=480, width=832, num_frames=124, fps=24)
video, audio = pipe(
    prompt=prompt,
    height=480, width=832, num_frames=num_frames, num_inference_steps=20, seed=0,
    retake_video=video,
)
write_video_audio(
    video=video, audio=audio,
    output_path="retake_v2va.mp4", fps=24, audio_sample_rate=32000,
)

# Everything outside the requested ranges is kept from the source. The two tracks use different
# units, and both are half-open [start, end):
#   frame_regions_to_retake -> FRAME IDS, counted from 0
#   seconds_regions_to_retake -> SECONDS
#
# For MiniMax-H3 the 17 frames of a VAE clip are coupled in latent space: retaking any frame of
# a clip retakes the whole clip. The pipeline widens ranges itself; align_to_clips below mirrors
# it so the effective range is visible.
def align_to_clips(start, end, total_frames, clip_frames=17):
    """Widen the half-open frame range [start, end) to whole clips. Frames count from 0."""
    first_clip, last_clip = start // clip_frames, (end - 1) // clip_frames
    return first_clip * clip_frames, min((last_clip + 1) * clip_frames, total_frames)

# Video + Audio Retake by region
prompt = "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a locked-off wide shot looks across an arid desert valley from a high vantage point, with a long rocky ridge running low across the frame and a cloudless deep blue sky filling the space above it. Harsh midday sunlight falls straight down, bleaching the pale rock and pinning short hard shadows under every outcrop. The camera holds a static shot for the entire take. Heat shimmer rises continuously off the sunlit rock so that the far ridge line wavers and breaks up, thin veils of dust stream off the crest and dissolve in the wind, and large waist-high dry scrub bushes cling among the foreground rocks, their stiff branches thrashing hard and springing back under the wind throughout. Then a single grey fighter jet enters the sky from the left edge of the frame, crosses above the ridge in level flight at very high speed as a compact hard-edged silhouette, and exits past the right edge of the frame. It leaves no contrail, no vapour trail and no smoke behind it, the sky closes back to the same even blue the moment it is gone, and the heat shimmer, the streaming dust, the thrashing scrub, the shadows and the exposure carry on exactly as before, with no trace of the aircraft anywhere in the frame.\n\noverall_soundscape: A strong dry wind drives steadily across the valley for the whole take, surging and easing in long gusts, with a deep broad roar rolling along the ridge, a hard whistle where it cuts past the camera, and a loud dry clatter and hiss from the thrashing scrub. A distant jet whine rises out of this wind while the sky is still empty and builds rapidly into a hard roar as the aircraft closes in, the pitch of the roar drops sharply the instant it passes overhead, and the sound then stretches out behind it into a long descending rumble that thins away and leaves only the wind.\n\nnon_diegetic_music: A quiet sustained synthesizer pad holds one low chord at a slow, even level from start to finish, with no rhythm, no percussion and no melodic movement, sitting far beneath the wind."
# video_regions and audio_regions support an arbitrary number of entries.
num_frames = 175
video_regions = [align_to_clips(68, 136, num_frames)]
audio_regions = [(68 / 24, num_frames / 24)]
source_video, source_audio, sample_rate = read_video_audio("data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Retake/source_video1.mp4", height=480, width=832, num_frames=175, fps=24)
video, audio = pipe(
    prompt=prompt,
    height=480, width=832, num_frames=num_frames, num_inference_steps=20, seed=1,
    retake_video=source_video, frame_regions_to_retake=video_regions,                                      # frame ids, already clip-aligned
    retake_audio=source_audio, retake_audio_sample_rate=sample_rate, seconds_regions_to_retake=audio_regions,    # seconds
)
write_video_audio(
    video=video, audio=audio,
    output_path="retake_video_audio_with_regions.mp4", fps=24, audio_sample_rate=32000,
)
