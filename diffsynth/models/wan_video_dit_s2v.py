import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .utils import hash_state_dict_keys
from .wan_video_dit import rearrange, precompute_freqs_cis_3d, DiTBlock, Head, CrossAttention, modulate, sinusoidal_embedding_1d


def torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [], []
    current_name = parent_name if parent_name else 'root'
    module_names.append(current_name)
    modules.append(model)

    for name, child in model.named_children():
        if parent_name:
            child_name = f'{parent_name}.{name}'
        else:
            child_name = name
        child_modules, child_names = torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


def rope_precompute(x, grid_sizes, freqs, start=None):
    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2

    # split freqs
    if type(freqs) is list:
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = torch.view_as_complex(x.detach().reshape(b, s, n, -1, 2).to(torch.float64))
    seq_bucket = [0]
    if not type(grid_sizes) is list:
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not type(g) is list:
            g = [torch.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]

            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len > 0:
                if t_f > 0:
                    factor_f, factor_h, factor_w = (t_f / seq_f).item(), (t_h / seq_h).item(), (t_w / seq_w).item()
                    # Generate a list of seq_f integers starting from f_o and ending at math.ceil(factor_f * seq_f.item() + f_o.item())
                    if f_o >= 0:
                        f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1, seq_f).astype(int).tolist()
                    else:
                        f_sam = np.linspace(-f_o.item(), (-t_f - f_o).item() + 1, seq_f).astype(int).tolist()
                    h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1, seq_h).astype(int).tolist()
                    w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1, seq_w).astype(int).tolist()

                    assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                    freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][f_sam].conj()
                    freqs_0 = freqs_0.view(seq_f, 1, 1, -1)

                    freqs_i = torch.cat(
                        [
                            freqs_0.expand(seq_f, seq_h, seq_w, -1),
                            freqs[1][h_sam].view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1),
                            freqs[2][w_sam].view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1),
                        ],
                        dim=-1
                    ).reshape(seq_len, 1, -1)
                elif t_f < 0:
                    freqs_i = trainable_freqs.unsqueeze(1)
                # apply rotary embedding
                output[i, seq_bucket[-1]:seq_bucket[-1] + seq_len] = freqs_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output


class CausalConv1d(nn.Module):

    def __init__(self, chan_in, chan_out, kernel_size=3, stride=1, dilation=1, pad_mode='replicate', **kwargs):
        super().__init__()

        self.pad_mode = pad_mode
        padding = (kernel_size - 1, 0)  # T
        self.time_causal_padding = padding

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class MotionEncoder_tc(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, num_heads=int, need_global=True, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.num_heads = num_heads
        self.need_global = need_global
        self.conv1_local = CausalConv1d(in_dim, hidden_dim // 4 * num_heads, 3, stride=1)
        if need_global:
            self.conv1_global = CausalConv1d(in_dim, hidden_dim // 4, 3, stride=1)
        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2)

        if need_global:
            self.final_linear = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)

        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        x = rearrange(x, 'b t c -> b c t')
        x_ori = x.clone()
        b, c, t = x.shape
        x = self.conv1_local(x)
        x = rearrange(x, 'b (n c) t -> (b n) t c', n=self.num_heads)
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv2(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv3(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm3(x)
        x = self.act(x)
        x = rearrange(x, '(b n) t c -> b t n c', b=b)
        padding = self.padding_tokens.repeat(b, x.shape[1], 1, 1).to(device=x.device, dtype=x.dtype)
        x = torch.cat([x, padding], dim=-2)
        x_local = x.clone()

        if not self.need_global:
            return x_local

        x = self.conv1_global(x_ori)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv2(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv3(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm3(x)
        x = self.act(x)
        x = self.final_linear(x)
        x = rearrange(x, '(b n) t c -> b t n c', b=b)

        return x, x_local


class FramePackMotioner(nn.Module):

    def __init__(self, inner_dim=1024, num_heads=16, zip_frame_buckets=[1, 2, 16], drop_mode="drop", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
        self.zip_frame_buckets = torch.tensor(zip_frame_buckets, dtype=torch.long)

        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.freqs = torch.cat(precompute_freqs_cis_3d(inner_dim // num_heads), dim=1)
        self.drop_mode = drop_mode

    def forward(self, motion_latents, add_last_motion=2):
        motion_frames = motion_latents[0].shape[1]
        mot = []
        mot_remb = []
        for m in motion_latents:
            lat_height, lat_width = m.shape[2], m.shape[3]
            padd_lat = torch.zeros(16, self.zip_frame_buckets.sum(), lat_height, lat_width).to(device=m.device, dtype=m.dtype)
            overlap_frame = min(padd_lat.shape[1], m.shape[1])
            if overlap_frame > 0:
                padd_lat[:, -overlap_frame:] = m[:, -overlap_frame:]

            if add_last_motion < 2 and self.drop_mode != "drop":
                zero_end_frame = self.zip_frame_buckets[:self.zip_frame_buckets.__len__() - add_last_motion - 1].sum()
                padd_lat[:, -zero_end_frame:] = 0

            padd_lat = padd_lat.unsqueeze(0)
            clean_latents_4x, clean_latents_2x, clean_latents_post = padd_lat[:, :, -self.zip_frame_buckets.sum():, :, :].split(
                list(self.zip_frame_buckets)[::-1], dim=2
            )  # 16, 2 ,1

            # patchfy
            clean_latents_post = self.proj(clean_latents_post).flatten(2).transpose(1, 2)
            clean_latents_2x = self.proj_2x(clean_latents_2x).flatten(2).transpose(1, 2)
            clean_latents_4x = self.proj_4x(clean_latents_4x).flatten(2).transpose(1, 2)

            if add_last_motion < 2 and self.drop_mode == "drop":
                clean_latents_post = clean_latents_post[:, :0] if add_last_motion < 2 else clean_latents_post
                clean_latents_2x = clean_latents_2x[:, :0] if add_last_motion < 1 else clean_latents_2x

            motion_lat = torch.cat([clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)

            # rope
            start_time_id = -(self.zip_frame_buckets[:1].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[0]
            grid_sizes = [] if add_last_motion < 2 and self.drop_mode == "drop" else \
                        [
                            [torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                            torch.tensor([end_time_id, lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
                            torch.tensor([self.zip_frame_buckets[0], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1), ]
                        ]

            start_time_id = -(self.zip_frame_buckets[:2].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[1] // 2
            grid_sizes_2x = [] if add_last_motion < 1 and self.drop_mode == "drop" else \
            [
                [torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                torch.tensor([end_time_id, lat_height // 4, lat_width // 4]).unsqueeze(0).repeat(1, 1),
                torch.tensor([self.zip_frame_buckets[1], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1), ]
            ]

            start_time_id = -(self.zip_frame_buckets[:3].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[2] // 4
            grid_sizes_4x = [
                [
                    torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([end_time_id, lat_height // 8, lat_width // 8]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([self.zip_frame_buckets[2], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
                ]
            ]

            grid_sizes = grid_sizes + grid_sizes_2x + grid_sizes_4x

            motion_rope_emb = rope_precompute(
                motion_lat.detach().view(1, motion_lat.shape[1], self.num_heads, self.inner_dim // self.num_heads),
                grid_sizes,
                self.freqs,
                start=None
            )

            mot.append(motion_lat)
            mot_remb.append(motion_rope_emb)
        return mot, mot_remb


class AdaLayerNorm(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, elementwise_affine=False)

    def forward(self, x, temb):
        temb = self.linear(F.silu(temb))
        shift, scale = temb.chunk(2, dim=1)
        shift = shift[:, None, :]
        scale = scale[:, None, :]
        x = self.norm(x) * (1 + scale) + shift
        return x


class AudioInjector_WAN(nn.Module):

    def __init__(
        self,
        all_modules,
        all_modules_names,
        dim=2048,
        num_heads=32,
        inject_layer=[0, 27],
        enable_adain=False,
        adain_dim=2048,
    ):
        super().__init__()
        self.injected_block_id = {}
        audio_injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, DiTBlock):
                for inject_id in inject_layer:
                    if f'transformer_blocks.{inject_id}' in mod_name:
                        self.injected_block_id[inject_id] = audio_injector_id
                        audio_injector_id += 1

        self.injector = nn.ModuleList([CrossAttention(
            dim=dim,
            num_heads=num_heads,
        ) for _ in range(audio_injector_id)])
        self.injector_pre_norm_feat = nn.ModuleList([nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        ) for _ in range(audio_injector_id)])
        self.injector_pre_norm_vec = nn.ModuleList([nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        ) for _ in range(audio_injector_id)])
        if enable_adain:
            self.injector_adain_layers = nn.ModuleList([AdaLayerNorm(output_dim=dim * 2, embedding_dim=adain_dim) for _ in range(audio_injector_id)])


class CausalAudioEncoder(nn.Module):

    def __init__(self, dim=5120, num_layers=25, out_dim=2048, num_token=4, need_global=False):
        super().__init__()
        self.encoder = MotionEncoder_tc(in_dim=dim, hidden_dim=out_dim, num_heads=num_token, need_global=need_global)
        weight = torch.ones((1, num_layers, 1, 1)) * 0.01

        self.weights = torch.nn.Parameter(weight)
        self.act = torch.nn.SiLU()

    def forward(self, features):
        # features B * num_layers * dim * video_length
        weights = self.act(self.weights.to(device=features.device, dtype=features.dtype))
        weights_sum = weights.sum(dim=1, keepdims=True)
        weighted_feat = ((features * weights) / weights_sum).sum(dim=1)  # b dim f
        weighted_feat = weighted_feat.permute(0, 2, 1)  # b f dim
        res = self.encoder(weighted_feat)  # b f n dim
        return res  # b f n dim


class WanS2VDiTBlock(DiTBlock):

    def forward(self, x, context, t_mod, seq_len_x, freqs):
        t_mod = (self.modulation.unsqueeze(2).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        # t_mod[:, :, 0] for x, t_mod[:, :, 1] for other like ref, motion, etc.
        t_mod = [
            torch.cat([element[:, :, 0].expand(1, seq_len_x, x.shape[-1]), element[:, :, 1].expand(1, x.shape[1] - seq_len_x, x.shape[-1])], dim=1)
            for element in t_mod
        ]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = t_mod
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class WanS2VModel(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        cond_dim: int,
        audio_dim: int,
        num_audio_token: int,
        enable_adain: bool = True,
        audio_inject_layers: list = [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39],
        zero_timestep: bool = True,
        add_last_motion: bool = True,
        framepack_drop_mode: str = "padd",
        fuse_vae_embedding_in_latents: bool = True,
        require_vae_embedding: bool = False,
        seperated_timestep: bool = False,
        require_clip_embedding: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.enbale_adain = enable_adain
        self.add_last_motion = add_last_motion
        self.zero_timestep = zero_timestep
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.require_vae_embedding = require_vae_embedding
        self.seperated_timestep = seperated_timestep
        self.require_clip_embedding = require_clip_embedding

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.blocks = nn.ModuleList([WanS2VDiTBlock(False, dim, num_heads, ffn_dim, eps) for _ in range(num_layers)])
        self.head = Head(dim, out_dim, patch_size, eps)
        self.freqs = torch.cat(precompute_freqs_cis_3d(dim // num_heads), dim=1)

        self.cond_encoder = nn.Conv3d(cond_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.casual_audio_encoder = CausalAudioEncoder(dim=audio_dim, out_dim=dim, num_token=num_audio_token, need_global=enable_adain)
        all_modules, all_modules_names = torch_dfs(self.blocks, parent_name="root.transformer_blocks")
        # TODO: refactor dfs
        self.audio_injector = AudioInjector_WAN(
            all_modules,
            all_modules_names,
            dim=dim,
            num_heads=num_heads,
            inject_layer=audio_inject_layers,
            enable_adain=enable_adain,
            adain_dim=dim,
        )
        self.trainable_cond_mask = nn.Embedding(3, dim)
        self.frame_packer = FramePackMotioner(inner_dim=dim, num_heads=num_heads, zip_frame_buckets=[1, 2, 16], drop_mode=framepack_drop_mode)

    def patchify(self, x: torch.Tensor):
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2]
        )

    def process_motion_frame_pack(self, motion_latents, drop_motion_frames=False, add_last_motion=2):
        flattern_mot, mot_remb = self.frame_packer(motion_latents, add_last_motion)
        if drop_motion_frames:
            return [m[:, :0] for m in flattern_mot], [m[:, :0] for m in mot_remb]
        else:
            return flattern_mot, mot_remb

    def inject_motion(self, x, rope_embs, mask_input, motion_latents, drop_motion_frames=True, add_last_motion=2):
        # inject the motion frames token to the hidden states
        # TODO: check drop_motion_frames = False
        mot, mot_remb = self.process_motion_frame_pack(motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=add_last_motion)
        if len(mot) > 0:
            x = torch.cat([x, mot[0]], dim=1)
            rope_embs = torch.cat([rope_embs, mot_remb[0]], dim=1)
            mask_input = torch.cat(
                [mask_input, 2 * torch.ones([1, x.shape[1] - mask_input.shape[1]], device=mask_input.device, dtype=mask_input.dtype)], dim=1
            )
        return x, rope_embs, mask_input

    def after_transformer_block(self, block_idx, hidden_states, audio_emb_global, audio_emb, original_seq_len):
        if block_idx in self.audio_injector.injected_block_id.keys():
            audio_attn_id = self.audio_injector.injected_block_id[block_idx]
            num_frames = audio_emb.shape[1]

            input_hidden_states = hidden_states[:, :original_seq_len].clone()  # b (f h w) c
            input_hidden_states = rearrange(input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)

            audio_emb_global = rearrange(audio_emb_global, "b t n c -> (b t) n c")
            adain_hidden_states = self.audio_injector.injector_adain_layers[audio_attn_id](input_hidden_states, temb=audio_emb_global[:, 0])
            attn_hidden_states = adain_hidden_states

            audio_emb = rearrange(audio_emb, "b t n c -> (b t) n c", t=num_frames)
            attn_audio_emb = audio_emb
            residual_out = self.audio_injector.injector[audio_attn_id](attn_hidden_states, attn_audio_emb)
            residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
            hidden_states[:, :original_seq_len] = hidden_states[:, :original_seq_len] + residual_out

        return hidden_states

    def cal_audio_emb(self, audio_input, motion_frames=[73, 19]):
        audio_input = torch.cat([audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input], dim=-1)
        audio_emb_global, audio_emb = self.casual_audio_encoder(audio_input)
        audio_emb_global = audio_emb_global[:, motion_frames[1]:].clone()
        merged_audio_emb = audio_emb[:, motion_frames[1]:, :]
        return audio_emb_global, merged_audio_emb

    def get_grid_sizes(self, grid_size_x, grid_size_ref):
        f, h, w = grid_size_x
        rf, rh, rw = grid_size_ref
        grid_sizes_x = torch.tensor([f, h, w], dtype=torch.long).unsqueeze(0)
        grid_sizes_x = [[torch.zeros_like(grid_sizes_x), grid_sizes_x, grid_sizes_x]]
        grid_sizes_ref = [[
            torch.tensor([30, 0, 0]).unsqueeze(0),
            torch.tensor([31, rh, rw]).unsqueeze(0),
            torch.tensor([1, rh, rw]).unsqueeze(0),
        ]]
        return grid_sizes_x + grid_sizes_ref

    def forward(
        self,
        latents,
        timestep,
        context,
        audio_input,
        motion_latents,
        pose_cond,
        use_gradient_checkpointing_offload=False,
        use_gradient_checkpointing=False
    ):
        origin_ref_latents = latents[:, :, 0:1]
        x = latents[:, :, 1:]

        # context embedding
        context = self.text_embedding(context)

        # audio encode
        audio_emb_global, merged_audio_emb = self.cal_audio_emb(audio_input)

        # x and pose_cond
        pose_cond = torch.zeros_like(x) if pose_cond is None else pose_cond
        x, (f, h, w) = self.patchify(self.patch_embedding(x) + self.cond_encoder(pose_cond))  # torch.Size([1, 29120, 5120])
        seq_len_x = x.shape[1]

        # reference image
        ref_latents, (rf, rh, rw) = self.patchify(self.patch_embedding(origin_ref_latents))  # torch.Size([1, 1456, 5120])
        grid_sizes = self.get_grid_sizes((f, h, w), (rf, rh, rw))
        x = torch.cat([x, ref_latents], dim=1)
        # mask
        mask = torch.cat([torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1).to(torch.long).to(x.device)
        # freqs
        pre_compute_freqs = rope_precompute(
            x.detach().view(1, x.size(1), self.num_heads, self.dim // self.num_heads), grid_sizes, self.freqs, start=None
        )
        # motion
        x, pre_compute_freqs, mask = self.inject_motion(x, pre_compute_freqs, mask, motion_latents, add_last_motion=2)

        x = x + self.trainable_cond_mask(mask).to(x.dtype)

        # t_mod
        timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)])
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim)).unsqueeze(2).transpose(0, 2)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block_id, block in enumerate(self.blocks):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        seq_len_x,
                        pre_compute_freqs,
                        use_reentrant=False,
                    )
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(lambda x: self.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                        x,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    context,
                    t_mod,
                    seq_len_x,
                    pre_compute_freqs,
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(lambda x: self.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, seq_len_x, pre_compute_freqs)
                x = self.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)

        x = x[:, :seq_len_x]
        x = self.head(x, t[:-1])
        x = self.unpatchify(x, (f, h, w))
        # make compatible with wan video
        x = torch.cat([origin_ref_latents, x], dim=2)
        return x

    @staticmethod
    def state_dict_converter():
        return WanS2VModelStateDictConverter()


class WanS2VModelStateDictConverter:

    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        config = {}
        if hash_state_dict_keys(state_dict) == "966cffdcc52f9c46c391768b27637614":
            config = {
                "dim": 5120,
                "in_dim": 16,
                "ffn_dim": 13824,
                "out_dim": 16,
                "text_dim": 4096,
                "freq_dim": 256,
                "eps": 1e-06,
                "patch_size": (1, 2, 2),
                "num_heads": 40,
                "num_layers": 40,
                "cond_dim": 16,
                "audio_dim": 1024,
                "num_audio_token": 4,
            }
        return state_dict, config
