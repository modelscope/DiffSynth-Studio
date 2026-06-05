import math
import re
from dataclasses import dataclass, field

import torch
from einops import rearrange
from torch import Tensor, nn


LATENT_SHIFT = (
    0.01984364, 0.10149707, 0.29689495, 0.27188619, -0.21445648, -0.15979549,
    0.05021099, -0.15083604, -0.15360136, -0.20131799, 0.01922352, 0.0622626,
    0.10140969, -0.06739428, 0.3758261, -0.233712, 0.35164491, -0.02590912,
    -0.0271935, -0.10833897, -0.1476848, -0.01130957, -0.2298372, 0.23526423,
    -0.10893522, 0.11957631, 0.04047799, 0.3134589, -0.17225064, -0.18646109,
    -0.34691978, -0.03571246, 0.02583857, 0.10190072, 0.28402294, 0.26952152,
    -0.21634675, -0.17938656, 0.04358909, -0.15007621, -0.1548502, -0.18971131,
    0.02710861, 0.05609494, 0.10697846, -0.06854968, 0.38167698, -0.24269937,
    0.35705471, -0.03063305, -0.02946109, -0.11244286, -0.14336038, -0.01362137,
    -0.21863696, 0.23228983, -0.11739769, 0.11693044, 0.02563311, 0.31356594,
    -0.17420591, -0.19006285, -0.34905377, -0.04025005, 0.01924137, 0.07652984,
    0.2995608, 0.2628057, -0.22011674, -0.12715361, 0.04879879, -0.14075719,
    -0.15935895, -0.2123584, 0.01974813, 0.05523547, 0.10011992, -0.06428964,
    0.37781868, -0.21491644, 0.34254215, -0.03153528, -0.0310082, -0.10761415,
    -0.14730405, -0.02475182, -0.2285588, 0.2515081, -0.10445128, 0.12446,
    0.07062869, 0.30880162, -0.18016875, -0.18869164, -0.34533499, -0.0129177,
    0.02578168, 0.07993659, 0.28642181, 0.26038408, -0.22459419, -0.14820155,
    0.04059549, -0.14043529, -0.16111187, -0.2020305, 0.02602069, 0.04852717,
    0.10432153, -0.06309942, 0.38402443, -0.22397003, 0.34814481, -0.03774432,
    -0.03381438, -0.11245691, -0.14128767, -0.02853208, -0.21752016, 0.24872463,
    -0.11399775, 0.1222687, 0.05620835, 0.309178, -0.18065738, -0.19401479,
    -0.34495114, -0.01760592,
)

LATENT_SCALE = (
    1.63933691, 1.70204478, 1.73642566, 1.90004803, 1.6675316, 1.69059584,
    1.56853198, 1.62314944, 1.89106626, 1.58086668, 1.60822129, 1.60962993,
    1.63322129, 1.56074359, 1.73419528, 1.7919265, 1.64040632, 1.66802808,
    1.60390303, 1.75480492, 1.63187587, 1.64334594, 1.61722884, 1.60146046,
    1.63459219, 1.55291476, 1.68771497, 1.68415657, 1.78966054, 1.66631641,
    1.65626686, 1.65976433, 1.63487607, 1.69513249, 1.72933756, 1.91310663,
    1.67035057, 1.72286863, 1.56719251, 1.61934825, 1.88628859, 1.56911539,
    1.59455129, 1.60829869, 1.62470611, 1.56052853, 1.73677003, 1.77563606,
    1.63732541, 1.66370527, 1.59508952, 1.75153949, 1.63029275, 1.64517667,
    1.61659342, 1.59722044, 1.64103121, 1.5408531, 1.68610394, 1.67772755,
    1.78998563, 1.66621713, 1.65458955, 1.66041308, 1.64710857, 1.68163503,
    1.74000294, 1.92784786, 1.67411194, 1.67395548, 1.57406532, 1.62199356,
    1.87618195, 1.5584375, 1.57438785, 1.61711053, 1.63094305, 1.55644029,
    1.73124302, 1.80666627, 1.6463621, 1.65932006, 1.60816188, 1.75682671,
    1.64695873, 1.63121722, 1.61380832, 1.60478651, 1.63396035, 1.53505068,
    1.65534289, 1.67132281, 1.80317197, 1.6767314, 1.65700938, 1.68426259,
    1.65339716, 1.67540638, 1.73298504, 1.94067348, 1.67893609, 1.70635117,
    1.5730906, 1.61928553, 1.87148809, 1.56244866, 1.56697152, 1.61584394,
    1.62759496, 1.55480378, 1.73484107, 1.79055143, 1.64688773, 1.66121492,
    1.60135887, 1.75254572, 1.64798332, 1.62989921, 1.61381592, 1.60792883,
    1.63939668, 1.53075757, 1.65371318, 1.66801185, 1.80029087, 1.67591476,
    1.65655173, 1.68533454,
)


def get_latent_norm(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    shift = torch.tensor(LATENT_SHIFT, dtype=torch.float32, device=device)
    scale = torch.tensor(LATENT_SCALE, dtype=torch.float32, device=device)
    return shift, scale


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(
            block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)

        upscale_dtype = next(self.up.parameters()).dtype

        h = self.conv_in(z)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = h.to(upscale_dtype)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Ideogram4VAEEncoder(nn.Module):
    def __init__(
        self,
        resolution: int = 256,
        in_channels: int = 3,
        ch: int = 128,
        ch_mult: list[int] = None,
        num_res_blocks: int = 2,
        z_channels: int = 32,
        **kwargs,
    ):
        super().__init__()
        if ch_mult is None:
            ch_mult = [1, 2, 4, 4]
        self.encoder = Encoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
        )
        self.ae_scale_factor = 8
        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def encode(self, image: Tensor, grid_h: int, grid_w: int, patch_size: int) -> Tensor:
        latents = self.encoder(image)

        ae_channels = latents.shape[1]
        latents = latents.view(1, ae_channels, grid_h, patch_size, grid_w, patch_size)
        latents = latents.permute(0, 2, 4, 3, 5, 1).contiguous()
        latents = latents.view(1, grid_h * grid_w, patch_size * patch_size * ae_channels)

        latent_shift, latent_scale = get_latent_norm(latents.device)
        latents = (latents - latent_shift) / latent_scale
        return latents


class Ideogram4VAEDecoder(nn.Module):
    def __init__(
        self,
        resolution: int = 256,
        in_channels: int = 3,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: list[int] = None,
        num_res_blocks: int = 2,
        z_channels: int = 32,
        **kwargs,
    ):
        super().__init__()
        if ch_mult is None:
            ch_mult = [1, 2, 4, 4]
        self.ae_scale_factor = 8
        self.decoder = Decoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def decode(self, latents: Tensor, grid_h: int, grid_w: int, patch_size: int, torch_dtype: torch.dtype) -> Tensor:
        latent_shift, latent_scale = get_latent_norm(latents.device)
        latents = latents * latent_scale + latent_shift

        ae_channels = latents.shape[-1] // (patch_size * patch_size)
        latents = latents.view(1, grid_h, grid_w, patch_size, patch_size, ae_channels)
        latents = latents.permute(0, 5, 1, 3, 2, 4).contiguous()
        latents = latents.view(1, ae_channels, grid_h * patch_size, grid_w * patch_size)

        latents = latents.to(torch_dtype)
        decoded = self.decoder(latents)
        decoded = decoded.float().clamp(-1.0, 1.0)
        return decoded


_NUM_RESOLUTIONS = 4


def _rewrite_diffusers_key(key: str) -> str | None:
    if key.startswith("bn."):
        return key

    if key.startswith("quant_conv."):
        return key.replace("quant_conv.", "encoder.quant_conv.", 1)
    if key.startswith("post_quant_conv."):
        return key.replace("post_quant_conv.", "decoder.post_quant_conv.", 1)

    if key == "encoder.conv_norm_out.weight":
        return "encoder.norm_out.weight"
    if key == "encoder.conv_norm_out.bias":
        return "encoder.norm_out.bias"
    if key == "decoder.conv_norm_out.weight":
        return "decoder.norm_out.weight"
    if key == "decoder.conv_norm_out.bias":
        return "decoder.norm_out.bias"

    m = re.match(r"^(encoder|decoder)\.mid_block\.resnets\.(\d+)\.(.+)$", key)
    if m:
        side, idx, rest = m.group(1), int(m.group(2)), m.group(3)
        rest = rest.replace("conv_shortcut", "nin_shortcut")
        return f"{side}.mid.block_{idx + 1}.{rest}"
    m = re.match(r"^(encoder|decoder)\.mid_block\.attentions\.0\.(.+)$", key)
    if m:
        side, rest = m.group(1), m.group(2)
        rest = (
            rest.replace("group_norm.", "norm.")
            .replace("to_q.", "q.")
            .replace("to_k.", "k.")
            .replace("to_v.", "v.")
            .replace("to_out.0.", "proj_out.")
        )
        return f"{side}.mid.attn_1.{rest}"

    m = re.match(r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
    if m:
        level, res_idx, rest = m.group(1), m.group(2), m.group(3)
        rest = rest.replace("conv_shortcut", "nin_shortcut")
        return f"encoder.down.{level}.block.{res_idx}.{rest}"
    m = re.match(r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.conv\.(.+)$", key)
    if m:
        return f"encoder.down.{m.group(1)}.downsample.conv.{m.group(2)}"

    m = re.match(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
    if m:
        diffusers_idx = int(m.group(1))
        res_idx = m.group(2)
        rest = m.group(3).replace("conv_shortcut", "nin_shortcut")
        return f"decoder.up.{_NUM_RESOLUTIONS - 1 - diffusers_idx}.block.{res_idx}.{rest}"
    m = re.match(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv\.(.+)$", key)
    if m:
        diffusers_idx = int(m.group(1))
        return (
            f"decoder.up.{_NUM_RESOLUTIONS - 1 - diffusers_idx}.upsample.conv.{m.group(2)}"
        )

    if key.startswith(
        ("encoder.conv_in.", "encoder.conv_out.", "decoder.conv_in.", "decoder.conv_out.")
    ):
        return key

    return None


def Ideogram4VAEEncoderStateDictConverter(state_dict) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    attn_substrings = (".mid.attn_1.",)
    for src_key in state_dict:
        if not src_key.startswith("encoder.") and not src_key.startswith("quant_conv.") and not src_key.startswith("bn."):
            continue
        tensor = state_dict[src_key]
        dst_key = _rewrite_diffusers_key(src_key)
        if dst_key is None:
            continue
        if (
            any(s in dst_key for s in attn_substrings)
            and dst_key.endswith(".weight")
            and tensor.ndim == 2
        ):
            tensor = tensor.unsqueeze(-1).unsqueeze(-1)
        out[dst_key] = tensor
    return out


def Ideogram4VAEDecoderStateDictConverter(state_dict) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    attn_substrings = (".mid.attn_1.",)
    for src_key in state_dict:
        if not src_key.startswith("decoder.") and not src_key.startswith("post_quant_conv."):
            continue
        tensor = state_dict[src_key]
        dst_key = _rewrite_diffusers_key(src_key)
        if dst_key is None:
            continue
        if (
            any(s in dst_key for s in attn_substrings)
            and dst_key.endswith(".weight")
            and tensor.ndim == 2
        ):
            tensor = tensor.unsqueeze(-1).unsqueeze(-1)
        out[dst_key] = tensor
    return out
