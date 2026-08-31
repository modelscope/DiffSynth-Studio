import torch, torch.nn as nn
from einops import rearrange

from .qwen_image_dit import QwenEmbedRope


class QwenVideoEditRope(QwenEmbedRope):
    """Grid-aware RoPE for Qwen-Video-Edit"""

    def _expand_pos_freqs_if_needed(self, video_fhw, txt_seq_lens):
        if isinstance(video_fhw, list) and video_fhw and isinstance(video_fhw[0], dict):
            video_fhw = (max(x["frame"] + 1 for x in video_fhw),
                         max(x["full_height"] for x in video_fhw),
                         max(x["full_width"] for x in video_fhw))
        super()._expand_pos_freqs_if_needed(video_fhw, txt_seq_lens)

    def forward(self, video_fhw, txt_seq_lens, device):
        if not video_fhw or not isinstance(video_fhw[0], dict):
            return super().forward(video_fhw, txt_seq_lens, device)
        self._expand_pos_freqs_if_needed(video_fhw, txt_seq_lens)
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        values = []
        max_index = 0
        for item in video_fhw:
            frame, h, w = item["frame"], item["height"], item["width"]
            full_h, full_w = item["full_height"], item["full_width"]
            key = "grid_" + "_".join(str(item[x]) for x in
                                      ("frame", "height", "width", "h_off", "w_off", "full_height", "full_width"))
            if key not in self.rope_cache:
                axis_h = freqs_pos[1][:full_h]
                axis_w = freqs_pos[2][:full_w]
                if self.scale_rope:
                    axis_h = torch.cat([freqs_neg[1][-(full_h - full_h // 2):], freqs_pos[1][:full_h // 2]])
                    axis_w = torch.cat([freqs_neg[2][-(full_w - full_w // 2):], freqs_pos[2][:full_w // 2]])
                frame_freq = freqs_pos[0][frame:frame + 1].view(1, 1, 1, -1).expand(1, h, w, -1)
                height_freq = axis_h[item["h_off"]:item["h_off"] + h].view(1, h, 1, -1).expand(1, h, w, -1)
                width_freq = axis_w[item["w_off"]:item["w_off"] + w].view(1, 1, w, -1).expand(1, h, w, -1)
                self.rope_cache[key] = torch.cat([frame_freq, height_freq, width_freq], dim=-1).reshape(h * w, -1).contiguous()
            values.append(self.rope_cache[key])
            if self.scale_rope:
                max_index = max(full_h // 2, full_w // 2, max_index)
            else:
                max_index = max(full_h, full_w, max_index)
        return torch.cat(values, dim=0), self.pos_freqs[max_index:max_index + max(txt_seq_lens)]


class WanToQwenProjection(nn.Module):
    def __init__(self, in_channels=16, inner_dim=3072):
        super().__init__()
        self.group = 1
        self.proj = nn.Conv3d(in_channels, inner_dim, (1, 2, 2), stride=(1, 2, 2))

    @torch.no_grad()
    def init_from_qwen_dit(self, dit):
        self.proj.weight.copy_(dit.img_in.weight.view(self.proj.out_channels, self.proj.in_channels, 2, 2).unsqueeze(2))
        self.proj.bias.copy_(dit.img_in.bias)

    def forward(self, x):
        return rearrange(self.proj(x), "B D T H W -> B (T H W) D")


class QwenToWanProjection(nn.Module):
    def __init__(self, out_channels=16, inner_dim=3072):
        super().__init__()
        self.group = 1
        self.proj = nn.Linear(inner_dim, out_channels * 4)

    @torch.no_grad()
    def init_from_qwen_dit(self, dit):
        self.proj.load_state_dict(dit.proj_out.state_dict())

    def forward(self, x, num_frames, tokens_h, tokens_w):
        return rearrange(self.proj(x), "B (T H W) (C P Q) -> B C T (H P) (W Q)", T=num_frames, H=tokens_h, W=tokens_w, P=2, Q=2)


class QwenVideoEditAdapter(nn.Module):
    def __init__(self, inner_dim=3072, in_channels=16, out_channels=16):
        super().__init__()
        self.in_proj = WanToQwenProjection(in_channels=in_channels, inner_dim=inner_dim)
        self.out_proj = QwenToWanProjection(out_channels=out_channels, inner_dim=inner_dim)

    @torch.no_grad()
    def init_from_qwen_dit(self, dit):
        self.in_proj.init_from_qwen_dit(dit)
        self.out_proj.init_from_qwen_dit(dit)