import math
import numpy as np
import torch
import torch.nn.functional as F


def get_sample_indices(original_fps, total_frames, target_fps, num_sample, fixed_start=None):
    required_duration = num_sample / target_fps
    required_origin_frames = int(np.ceil(required_duration * original_fps))
    if required_duration > total_frames / original_fps:
        raise ValueError("required_duration must be less than video length")

    if not fixed_start is None and fixed_start >= 0:
        start_frame = fixed_start
    else:
        max_start = total_frames - required_origin_frames
        if max_start < 0:
            raise ValueError("video length is too short")
        start_frame = np.random.randint(0, max_start + 1)
    start_time = start_frame / original_fps

    end_time = start_time + required_duration
    time_points = np.linspace(start_time, end_time, num_sample, endpoint=False)

    frame_indices = np.round(np.array(time_points) * original_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    return frame_indices


def linear_interpolation(features, input_fps, output_fps, output_len=None):
    """
    features: shape=[1, T, 512]
    input_fps: fps for audio, f_a
    output_fps: fps for video, f_m
    output_len: video length
    """
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features, size=output_len, align_corners=True, mode='linear')  # [1, 512, output_len]
    return output_features.transpose(1, 2)


class WanS2VAudioEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
        config = {
            "_name_or_path": "facebook/wav2vec2-large-xlsr-53",
            "activation_dropout": 0.05,
            "apply_spec_augment": True,
            "architectures": ["Wav2Vec2ForCTC"],
            "attention_dropout": 0.1,
            "bos_token_id": 1,
            "conv_bias": True,
            "conv_dim": [512, 512, 512, 512, 512, 512, 512],
            "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
            "conv_stride": [5, 2, 2, 2, 2, 2, 2],
            "ctc_loss_reduction": "mean",
            "ctc_zero_infinity": True,
            "do_stable_layer_norm": True,
            "eos_token_id": 2,
            "feat_extract_activation": "gelu",
            "feat_extract_dropout": 0.0,
            "feat_extract_norm": "layer",
            "feat_proj_dropout": 0.05,
            "final_dropout": 0.0,
            "hidden_act": "gelu",
            "hidden_dropout": 0.05,
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "layer_norm_eps": 1e-05,
            "layerdrop": 0.05,
            "mask_channel_length": 10,
            "mask_channel_min_space": 1,
            "mask_channel_other": 0.0,
            "mask_channel_prob": 0.0,
            "mask_channel_selection": "static",
            "mask_feature_length": 10,
            "mask_feature_prob": 0.0,
            "mask_time_length": 10,
            "mask_time_min_space": 1,
            "mask_time_other": 0.0,
            "mask_time_prob": 0.05,
            "mask_time_selection": "static",
            "model_type": "wav2vec2",
            "num_attention_heads": 16,
            "num_conv_pos_embedding_groups": 16,
            "num_conv_pos_embeddings": 128,
            "num_feat_extract_layers": 7,
            "num_hidden_layers": 24,
            "pad_token_id": 0,
            "transformers_version": "4.7.0.dev0",
            "vocab_size": 33
        }
        self.model = Wav2Vec2ForCTC(Wav2Vec2Config(**config))
        self.video_rate = 30

    def extract_audio_feat(self, input_audio, sample_rate, processor, return_all_layers=False, dtype=torch.float32, device='cpu'):
        input_values = processor(input_audio, sampling_rate=sample_rate, return_tensors="pt").input_values.to(dtype=dtype, device=device)

        # retrieve logits & take argmax
        res = self.model(input_values, output_hidden_states=True)
        if return_all_layers:
            feat = torch.cat(res.hidden_states)
        else:
            feat = res.hidden_states[-1]
        feat = linear_interpolation(feat, input_fps=50, output_fps=self.video_rate)
        return feat

    def get_audio_embed_bucket(self, audio_embed, stride=2, batch_frames=12, m=2):
        num_layers, audio_frame_num, audio_dim = audio_embed.shape

        if num_layers > 1:
            return_all_layers = True
        else:
            return_all_layers = False

        min_batch_num = int(audio_frame_num / (batch_frames * stride)) + 1

        bucket_num = min_batch_num * batch_frames
        batch_idx = [stride * i for i in range(bucket_num)]
        batch_audio_eb = []
        for bi in batch_idx:
            if bi < audio_frame_num:
                audio_sample_stride = 2
                chosen_idx = list(range(bi - m * audio_sample_stride, bi + (m + 1) * audio_sample_stride, audio_sample_stride))
                chosen_idx = [0 if c < 0 else c for c in chosen_idx]
                chosen_idx = [audio_frame_num - 1 if c >= audio_frame_num else c for c in chosen_idx]

                if return_all_layers:
                    frame_audio_embed = audio_embed[:, chosen_idx].flatten(start_dim=-2, end_dim=-1)
                else:
                    frame_audio_embed = audio_embed[0][chosen_idx].flatten()
            else:
                frame_audio_embed = \
                torch.zeros([audio_dim * (2 * m + 1)], device=audio_embed.device) if not return_all_layers \
                    else torch.zeros([num_layers, audio_dim * (2 * m + 1)], device=audio_embed.device)
            batch_audio_eb.append(frame_audio_embed)
        batch_audio_eb = torch.cat([c.unsqueeze(0) for c in batch_audio_eb], dim=0)

        return batch_audio_eb, min_batch_num

    def get_audio_embed_bucket_fps(self, audio_embed, fps=16, batch_frames=81, m=0):
        num_layers, audio_frame_num, audio_dim = audio_embed.shape

        if num_layers > 1:
            return_all_layers = True
        else:
            return_all_layers = False

        scale = self.video_rate / fps

        min_batch_num = int(audio_frame_num / (batch_frames * scale)) + 1

        bucket_num = min_batch_num * batch_frames
        padd_audio_num = math.ceil(min_batch_num * batch_frames / fps * self.video_rate) - audio_frame_num
        batch_idx = get_sample_indices(
            original_fps=self.video_rate, total_frames=audio_frame_num + padd_audio_num, target_fps=fps, num_sample=bucket_num, fixed_start=0
        )
        batch_audio_eb = []
        audio_sample_stride = int(self.video_rate / fps)
        for bi in batch_idx:
            if bi < audio_frame_num:

                chosen_idx = list(range(bi - m * audio_sample_stride, bi + (m + 1) * audio_sample_stride, audio_sample_stride))
                chosen_idx = [0 if c < 0 else c for c in chosen_idx]
                chosen_idx = [audio_frame_num - 1 if c >= audio_frame_num else c for c in chosen_idx]

                if return_all_layers:
                    frame_audio_embed = audio_embed[:, chosen_idx].flatten(start_dim=-2, end_dim=-1)
                else:
                    frame_audio_embed = audio_embed[0][chosen_idx].flatten()
            else:
                frame_audio_embed = \
                torch.zeros([audio_dim * (2 * m + 1)], device=audio_embed.device) if not return_all_layers \
                    else torch.zeros([num_layers, audio_dim * (2 * m + 1)], device=audio_embed.device)
            batch_audio_eb.append(frame_audio_embed)
        batch_audio_eb = torch.cat([c.unsqueeze(0) for c in batch_audio_eb], dim=0)

        return batch_audio_eb, min_batch_num

    def get_audio_feats_per_inference(self, input_audio, sample_rate, processor, fps=16, batch_frames=80, m=0, dtype=torch.float32, device='cpu'):
        audio_feat = self.extract_audio_feat(input_audio, sample_rate, processor, return_all_layers=True, dtype=dtype, device=device)
        audio_embed_bucket, min_batch_num = self.get_audio_embed_bucket_fps(audio_feat, fps=fps, batch_frames=batch_frames, m=m)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0).permute(0, 2, 3, 1).to(device, dtype)
        audio_embeds = [audio_embed_bucket[..., i * batch_frames:(i + 1) * batch_frames] for i in range(min_batch_num)]
        return audio_embeds
