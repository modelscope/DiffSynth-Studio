import torch
import torch.nn.functional as F
from einops import rearrange


class Flux2VAEResnetBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels=None, num_groups=32, eps=1e-6):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = torch.nn.GroupNorm(num_groups, in_channels, eps=eps)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = torch.nn.GroupNorm(num_groups, out_channels, eps=eps)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.nonlinearity = torch.nn.SiLU()
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.conv1(self.nonlinearity(self.norm1(hidden_states)))
        hidden_states = self.conv2(self.nonlinearity(self.norm2(hidden_states)))
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class Flux2VAEAttention(torch.nn.Module):

    def __init__(self, channels, num_groups=32, eps=1e-6):
        super().__init__()
        self.group_norm = torch.nn.GroupNorm(num_groups, channels, eps=eps)
        self.to_q = torch.nn.Linear(channels, channels)
        self.to_k = torch.nn.Linear(channels, channels)
        self.to_v = torch.nn.Linear(channels, channels)
        self.to_out = torch.nn.Sequential(torch.nn.Linear(channels, channels), torch.nn.Dropout(0.0))

    def forward(self, hidden_states):
        batch_size, channels, height, width = hidden_states.shape
        residual = hidden_states

        # (B, C, H, W) -> (B, H * W, C)
        hidden_states = self.group_norm(hidden_states).view(batch_size, channels, height * width).transpose(1, 2)
        query = self.to_q(hidden_states).unsqueeze(1)
        key = self.to_k(hidden_states).unsqueeze(1)
        value = self.to_v(hidden_states).unsqueeze(1)
        hidden_states = F.scaled_dot_product_attention(query, key, value)
        hidden_states = self.to_out(hidden_states.squeeze(1))

        # (B, H * W, C) -> (B, C, H, W)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, channels, height, width)
        return residual + hidden_states


class Flux2VAEUpsampler(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, hidden_states):
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        return self.conv(hidden_states)


class Flux2VAEDownsampler(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=2)

    def forward(self, hidden_states):
        hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(hidden_states)


class Flux2VAEMidBlock(torch.nn.Module):

    def __init__(self, channels, num_layers=1, num_groups=32, eps=1e-6, add_attention=True):
        super().__init__()
        self.resnets = torch.nn.ModuleList([
            Flux2VAEResnetBlock(channels, num_groups=num_groups, eps=eps) for _ in range(num_layers + 1)
        ])
        self.attentions = torch.nn.ModuleList([
            Flux2VAEAttention(channels, num_groups=num_groups, eps=eps) if add_attention else None
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states):
        hidden_states = self.resnets[0](hidden_states)
        for attention, resnet in zip(self.attentions, self.resnets[1:]):
            if attention is not None:
                hidden_states = attention(hidden_states)
            hidden_states = resnet(hidden_states)
        return hidden_states


class Flux2VAEDownBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, num_layers, num_groups=32, eps=1e-6, add_downsample=True):
        super().__init__()
        self.resnets = torch.nn.ModuleList([
            Flux2VAEResnetBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                num_groups=num_groups,
                eps=eps,
            )
            for i in range(num_layers)
        ])
        self.downsamplers = None
        if add_downsample:
            self.downsamplers = torch.nn.ModuleList([Flux2VAEDownsampler(out_channels)])

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class Flux2VAEUpBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, num_layers, num_groups=32, eps=1e-6, add_upsample=True):
        super().__init__()
        self.resnets = torch.nn.ModuleList([
            Flux2VAEResnetBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                num_groups=num_groups,
                eps=eps,
            )
            for i in range(num_layers)
        ])
        self.upsamplers = None
        if add_upsample:
            self.upsamplers = torch.nn.ModuleList([Flux2VAEUpsampler(out_channels)])

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class Flux2VAEEncoder(torch.nn.Module):

    def __init__(
        self,
        in_channels=3,
        latent_channels=32,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        num_groups=32,
        add_attention=True,
    ):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.down_blocks = torch.nn.ModuleList([])
        for i, out_channels in enumerate(block_out_channels):
            in_channels_ = block_out_channels[0] if i == 0 else block_out_channels[i - 1]
            self.down_blocks.append(Flux2VAEDownBlock(
                in_channels_,
                out_channels,
                num_layers=layers_per_block,
                num_groups=num_groups,
                add_downsample=i < len(block_out_channels) - 1,
            ))
        self.mid_block = Flux2VAEMidBlock(
            block_out_channels[-1],
            num_groups=num_groups,
            add_attention=add_attention,
        )
        self.conv_norm_out = torch.nn.GroupNorm(num_groups, block_out_channels[-1], eps=1e-6)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(block_out_channels[-1], 2 * latent_channels, kernel_size=3, padding=1)

    def forward(self, hidden_states):
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)
        hidden_states = self.mid_block(hidden_states)
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        return self.conv_out(hidden_states)


class Flux2VAEDecoder(torch.nn.Module):

    def __init__(
        self,
        out_channels=3,
        latent_channels=32,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        num_groups=32,
        add_attention=True,
    ):
        super().__init__()
        block_out_channels = list(reversed(block_out_channels))
        self.conv_in = torch.nn.Conv2d(latent_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.mid_block = Flux2VAEMidBlock(
            block_out_channels[0],
            num_groups=num_groups,
            add_attention=add_attention,
        )
        self.up_blocks = torch.nn.ModuleList([])
        for i, out_channels_ in enumerate(block_out_channels):
            in_channels_ = block_out_channels[0] if i == 0 else block_out_channels[i - 1]
            self.up_blocks.append(Flux2VAEUpBlock(
                in_channels_,
                out_channels_,
                num_layers=layers_per_block + 1,
                num_groups=num_groups,
                add_upsample=i < len(block_out_channels) - 1,
            ))
        self.conv_norm_out = torch.nn.GroupNorm(num_groups, block_out_channels[-1], eps=1e-6)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(block_out_channels[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, hidden_states):
        hidden_states = self.conv_in(hidden_states)
        hidden_states = self.mid_block(hidden_states)
        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states)
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        return self.conv_out(hidden_states)


class Flux2VAELatentNorm(torch.nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.running_mean = torch.nn.Parameter(torch.zeros(num_features), requires_grad=False)
        self.running_var = torch.nn.Parameter(torch.ones(num_features), requires_grad=False)
        self.num_batches_tracked = torch.nn.Parameter(torch.zeros((), dtype=torch.long), requires_grad=False)

    def forward(self):
        return self.running_mean, self.running_var


class Flux2VAE(torch.nn.Module):
    """
    The VAE used by FLUX.2 and related models. Image -> latent (`encode`) and latent -> image (`decode`).
    Latents are normalized with statistics stored in the checkpoint (`bn.running_mean` / `bn.running_var`).
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        latent_channels=32,
        block_out_channels=(128, 256, 512, 512),
        decoder_block_out_channels=None,
        layers_per_block=2,
        num_groups=32,
        patch_size=(2, 2),
        use_quant_conv=True,
        use_post_quant_conv=True,
        mid_block_add_attention=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_latent_channels = patch_size[0] * patch_size[1] * latent_channels
        self.encoder = Flux2VAEEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            num_groups=num_groups,
            add_attention=mid_block_add_attention,
        )
        self.decoder = Flux2VAEDecoder(
            out_channels=out_channels,
            latent_channels=latent_channels,
            block_out_channels=decoder_block_out_channels or block_out_channels,
            layers_per_block=layers_per_block,
            num_groups=num_groups,
            add_attention=mid_block_add_attention,
        )
        self.quant_conv = torch.nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = torch.nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None
        self.bn = Flux2VAELatentNorm(self.num_latent_channels)

    @property
    def vae_dtype(self):
        for module in self.modules():
            if hasattr(module, "computation_dtype"):
                return module.computation_dtype
        return next(self.parameters()).dtype

    def latent_stats(self, device, dtype):
        # Call the module instead of reading its parameters, so that VRAM management loads the statistics on demand.
        mean, var = self.bn()
        mean = mean.to(device=device, dtype=dtype).view(1, -1, 1, 1)
        std = torch.sqrt(var.to(device=device, dtype=dtype) + 1e-4).view(1, -1, 1, 1)
        return mean, std

    def _encode(self, image):
        # Returns the predicted mean and variance, both are not normalized or packed.
        hidden_states = self.encoder(image.to(dtype=self.vae_dtype))
        if self.quant_conv is not None:
            hidden_states = self.quant_conv(hidden_states)
        return hidden_states

    def _decode(self, latents):
        latents = latents.to(dtype=self.vae_dtype)
        if self.post_quant_conv is not None:
            latents = self.post_quant_conv(latents)
        return self.decoder(latents)

    def encode(self, image):
        hidden_states = self._encode(image)
        hidden_states = rearrange(
            hidden_states, "B C (H P) (W Q) -> B (C P Q) H W", P=self.patch_size[0], Q=self.patch_size[1]
        )
        # The encoder predicts mean and variance, only the mean is used as the latent.
        hidden_states = hidden_states[:, :self.num_latent_channels]
        mean, std = self.latent_stats(hidden_states.device, hidden_states.dtype)
        return (hidden_states - mean) / std

    def decode(self, latents):
        latents = latents.to(dtype=self.vae_dtype)
        mean, std = self.latent_stats(latents.device, latents.dtype)
        latents = latents * std + mean
        latents = rearrange(
            latents, "B (C P Q) H W -> B C (H P) (W Q)", P=self.patch_size[0], Q=self.patch_size[1]
        )
        return self._decode(latents)
