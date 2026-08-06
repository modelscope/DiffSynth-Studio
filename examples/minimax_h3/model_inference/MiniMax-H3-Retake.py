import torch
from PIL import Image
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio, read_audio, VideoData, read_video_audio

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
prompt = "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a locked-off wide shot looks across a city skyline at night from a high vantage point, with the rooftops and towers low in the frame and a thick unbroken cloud layer filling the sky above them. Thousands of small windows glow warm yellow and white across the buildings, a few red aircraft-warning lights sit steady on the tallest roofs, and the streets far below hold long stationary streaks of reflected light. The camera holds a static shot for the entire take. The scene begins dark and still, with every lit window steady and no vehicle moving anywhere. Then a forked bolt of lightning snaps across the sky above the towers in a single bright instant, throwing hard white light across the cloud layer and flashing the concrete and glass faces of the buildings into visibility, and a fainter afterglow pulses once deep inside the clouds before it dies, so that the frame falls straight back to the same darkness. The bolt does not reach the buildings, nothing catches fire, no window changes state, and no trace of the flash is left behind: before and after it the skyline, the pattern of lit windows, the cloud layer and the exposure are identical to the opening.\n\noverall_soundscape: A low steady wind passes across the rooftops for the whole take, with a distant flat hum of the city far below. A sharp loud crack arrives just after the bolt and breaks into a heavy rolling boom that tumbles between the buildings and decays slowly into the distance. Once it has faded, only the wind and the city hum remain.\n\nnon_diegetic_music: N/A"
# video_regions and audio_regions support an arbitrary number of entries.
num_frames = 175
video_regions = [align_to_clips(68, 102, num_frames)]   # -> (68, 102)   2.833s .. 4.250s
audio_regions = [(68 / 24, num_frames / 24)]            # -> 2.833s .. 7.292s
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
