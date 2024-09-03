import torch
from einops import rearrange, repeat
from .tiler import TileWorker2Dto3D



class Downsample3D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
        compress_time: bool = False,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.compress_time = compress_time

    def forward(self, x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
        if self.compress_time:
            batch_size, channels, frames, height, width = x.shape

            # (batch_size, channels, frames, height, width) -> (batch_size, height, width, channels, frames) -> (batch_size * height * width, channels, frames)
            x = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, channels, frames)

            if x.shape[-1] % 2 == 1:
                x_first, x_rest = x[..., 0], x[..., 1:]
                if x_rest.shape[-1] > 0:
                    # (batch_size * height * width, channels, frames - 1) -> (batch_size * height * width, channels, (frames - 1) // 2)
                    x_rest = torch.nn.functional.avg_pool1d(x_rest, kernel_size=2, stride=2)

                x = torch.cat([x_first[..., None], x_rest], dim=-1)
                # (batch_size * height * width, channels, (frames // 2) + 1) -> (batch_size, height, width, channels, (frames // 2) + 1) -> (batch_size, channels, (frames // 2) + 1, height, width)
                x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)
            else:
                # (batch_size * height * width, channels, frames) -> (batch_size * height * width, channels, frames // 2)
                x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
                # (batch_size * height * width, channels, frames // 2) -> (batch_size, height, width, channels, frames // 2) -> (batch_size, channels, frames // 2, height, width)
                x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)

        # Pad the tensor
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        batch_size, channels, frames, height, width = x.shape
        # (batch_size, channels, frames, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size * frames, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        x = self.conv(x)
        # (batch_size * frames, channels, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size, channels, frames, height, width)
        x = x.reshape(batch_size, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        return x



class Upsample3D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        compress_time: bool = False,
    ) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.compress_time = compress_time

    def forward(self, inputs: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
        if self.compress_time:
            if inputs.shape[2] > 1 and inputs.shape[2] % 2 == 1:
                # split first frame
                x_first, x_rest = inputs[:, :, 0], inputs[:, :, 1:]

                x_first = torch.nn.functional.interpolate(x_first, scale_factor=2.0)
                x_rest = torch.nn.functional.interpolate(x_rest, scale_factor=2.0)
                x_first = x_first[:, :, None, :, :]
                inputs = torch.cat([x_first, x_rest], dim=2)
            elif inputs.shape[2] > 1:
                inputs = torch.nn.functional.interpolate(inputs, scale_factor=2.0)
            else:
                inputs = inputs.squeeze(2)
                inputs = torch.nn.functional.interpolate(inputs, scale_factor=2.0)
                inputs = inputs[:, :, None, :, :]
        else:
            # only interpolate 2D
            b, c, t, h, w = inputs.shape
            inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            inputs = torch.nn.functional.interpolate(inputs, scale_factor=2.0)
            inputs = inputs.reshape(b, t, c, *inputs.shape[2:]).permute(0, 2, 1, 3, 4)

        b, c, t, h, w = inputs.shape
        inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        inputs = self.conv(inputs)
        inputs = inputs.reshape(b, t, *inputs.shape[1:]).permute(0, 2, 1, 3, 4)

        return inputs



class CogVideoXSpatialNorm3D(torch.nn.Module):
    def __init__(self, f_channels, zq_channels, groups):
        super().__init__()
        self.norm_layer = torch.nn.GroupNorm(num_channels=f_channels, num_groups=groups, eps=1e-6, affine=True)
        self.conv_y = torch.nn.Conv3d(zq_channels, f_channels, kernel_size=1, stride=1)
        self.conv_b = torch.nn.Conv3d(zq_channels, f_channels, kernel_size=1, stride=1)


    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        if f.shape[2] > 1 and f.shape[2] % 2 == 1:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            z_first, z_rest = zq[:, :, :1], zq[:, :, 1:]
            z_first = torch.nn.functional.interpolate(z_first, size=f_first_size)
            z_rest = torch.nn.functional.interpolate(z_rest, size=f_rest_size)
            zq = torch.cat([z_first, z_rest], dim=2)
        else:
            zq = torch.nn.functional.interpolate(zq, size=f.shape[-3:])

        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f



class Resnet3DBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, spatial_norm_dim, groups, eps=1e-6, use_conv_shortcut=False):
        super().__init__()
        self.nonlinearity = torch.nn.SiLU()
        if spatial_norm_dim is None:
            self.norm1 = torch.nn.GroupNorm(num_channels=in_channels, num_groups=groups, eps=eps)
            self.norm2 = torch.nn.GroupNorm(num_channels=out_channels, num_groups=groups, eps=eps)
        else:
            self.norm1 = CogVideoXSpatialNorm3D(in_channels, spatial_norm_dim, groups)
            self.norm2 = CogVideoXSpatialNorm3D(out_channels, spatial_norm_dim, groups)

        self.conv1 = CachedConv3d(in_channels, out_channels, kernel_size=3, padding=(0, 1, 1))

        self.conv2 = CachedConv3d(out_channels, out_channels, kernel_size=3, padding=(0, 1, 1))

        if in_channels != out_channels:
            if use_conv_shortcut:
                self.conv_shortcut = CachedConv3d(in_channels, out_channels, kernel_size=3, padding=(0, 1, 1))
            else:
                self.conv_shortcut = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_shortcut = lambda x: x


    def forward(self, hidden_states, zq):
        residual = hidden_states

        hidden_states = self.norm1(hidden_states, zq) if isinstance(self.norm1, CogVideoXSpatialNorm3D) else self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states, zq) if isinstance(self.norm2, CogVideoXSpatialNorm3D) else self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        hidden_states = hidden_states + self.conv_shortcut(residual)

        return hidden_states
    


class CachedConv3d(torch.nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.cached_tensor = None


    def clear_cache(self):
        self.cached_tensor = None
    

    def forward(self, input: torch.Tensor, use_cache = True) -> torch.Tensor:
        if use_cache:
            if self.cached_tensor is None:
                self.cached_tensor = torch.concat([input[:, :, :1]] * 2, dim=2)
            input = torch.concat([self.cached_tensor, input], dim=2)
            self.cached_tensor = input[:, :, -2:]
        return super().forward(input)



class CogVAEDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scaling_factor = 0.7
        self.conv_in = CachedConv3d(16, 512, kernel_size=3, stride=1, padding=(0, 1, 1))

        self.blocks = torch.nn.ModuleList([
            Resnet3DBlock(512, 512, 16, 32),
            Resnet3DBlock(512, 512, 16, 32),
            Resnet3DBlock(512, 512, 16, 32),
            Resnet3DBlock(512, 512, 16, 32),
            Resnet3DBlock(512, 512, 16, 32),
            Resnet3DBlock(512, 512, 16, 32),
            Upsample3D(512, 512, compress_time=True),
            Resnet3DBlock(512, 256, 16, 32),
            Resnet3DBlock(256, 256, 16, 32),
            Resnet3DBlock(256, 256, 16, 32),
            Resnet3DBlock(256, 256, 16, 32),
            Upsample3D(256, 256, compress_time=True),
            Resnet3DBlock(256, 256, 16, 32),
            Resnet3DBlock(256, 256, 16, 32),
            Resnet3DBlock(256, 256, 16, 32),
            Resnet3DBlock(256, 256, 16, 32),
            Upsample3D(256, 256, compress_time=False),
            Resnet3DBlock(256, 128, 16, 32),
            Resnet3DBlock(128, 128, 16, 32),
            Resnet3DBlock(128, 128, 16, 32),
            Resnet3DBlock(128, 128, 16, 32),
        ])

        self.norm_out = CogVideoXSpatialNorm3D(128, 16, 32)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = CachedConv3d(128, 3, kernel_size=3, stride=1, padding=(0, 1, 1))


    def forward(self, sample):
        sample = sample / self.scaling_factor
        hidden_states = self.conv_in(sample)

        for block in self.blocks:
            hidden_states = block(hidden_states, sample)
        
        hidden_states = self.norm_out(hidden_states, sample)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states
    

    def decode_video(self, sample, tiled=True, tile_size=(60, 90), tile_stride=(30, 45), progress_bar=lambda x:x):
        if tiled:
            B, C, T, H, W = sample.shape
            return TileWorker2Dto3D().tiled_forward(
                forward_fn=lambda x: self.decode_small_video(x),
                model_input=sample,
                tile_size=tile_size, tile_stride=tile_stride,
                tile_device=sample.device, tile_dtype=sample.dtype,
                computation_device=sample.device, computation_dtype=sample.dtype,
                scales=(3/16, (T//2*8+T%2)/T, 8, 8),
                progress_bar=progress_bar
            )
        else:
            return self.decode_small_video(sample)
    

    def decode_small_video(self, sample):
        B, C, T, H, W = sample.shape
        computation_device = self.conv_in.weight.device
        computation_dtype = self.conv_in.weight.dtype
        value = []
        for i in range(T//2):
            tl = i*2 + T%2 - (T%2 and i==0)
            tr = i*2 + 2 + T%2
            model_input = sample[:, :, tl: tr, :, :].to(dtype=computation_dtype, device=computation_device)
            model_output = self.forward(model_input).to(dtype=sample.dtype, device=sample.device)
            value.append(model_output)
        value = torch.concat(value, dim=2)
        for name, module in self.named_modules():
            if isinstance(module, CachedConv3d):
                module.clear_cache()
        return value
    

    @staticmethod
    def state_dict_converter():
        return CogVAEDecoderStateDictConverter()
    


class CogVAEEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scaling_factor = 0.7
        self.conv_in = CachedConv3d(3, 128, kernel_size=3, stride=1, padding=(0, 1, 1))

        self.blocks = torch.nn.ModuleList([
            Resnet3DBlock(128, 128, None, 32),
            Resnet3DBlock(128, 128, None, 32),
            Resnet3DBlock(128, 128, None, 32),
            Downsample3D(128, 128, compress_time=True),
            Resnet3DBlock(128, 256, None, 32),
            Resnet3DBlock(256, 256, None, 32),
            Resnet3DBlock(256, 256, None, 32),
            Downsample3D(256, 256, compress_time=True),
            Resnet3DBlock(256, 256, None, 32),
            Resnet3DBlock(256, 256, None, 32),
            Resnet3DBlock(256, 256, None, 32),
            Downsample3D(256, 256, compress_time=False),
            Resnet3DBlock(256, 512, None, 32),
            Resnet3DBlock(512, 512, None, 32),
            Resnet3DBlock(512, 512, None, 32),
            Resnet3DBlock(512, 512, None, 32),
            Resnet3DBlock(512, 512, None, 32),
        ])

        self.norm_out = torch.nn.GroupNorm(32, 512, eps=1e-06, affine=True)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = CachedConv3d(512, 32, kernel_size=3, stride=1, padding=(0, 1, 1))


    def forward(self, sample):
        hidden_states = self.conv_in(sample)

        for block in self.blocks:
            hidden_states = block(hidden_states, sample)
        
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)[:, :16]
        hidden_states = hidden_states * self.scaling_factor

        return hidden_states
    

    def encode_video(self, sample, tiled=True, tile_size=(60, 90), tile_stride=(30, 45), progress_bar=lambda x:x):
        if tiled:
            B, C, T, H, W = sample.shape
            return TileWorker2Dto3D().tiled_forward(
                forward_fn=lambda x: self.encode_small_video(x),
                model_input=sample,
                tile_size=(i * 8 for i in tile_size), tile_stride=(i * 8 for i in tile_stride),
                tile_device=sample.device, tile_dtype=sample.dtype,
                computation_device=sample.device, computation_dtype=sample.dtype,
                scales=(16/3, (T//4+T%2)/T, 1/8, 1/8),
                progress_bar=progress_bar
            )
        else:
            return self.encode_small_video(sample)
    

    def encode_small_video(self, sample):
        B, C, T, H, W = sample.shape
        computation_device = self.conv_in.weight.device
        computation_dtype = self.conv_in.weight.dtype
        value = []
        for i in range(T//8):
            t = i*8 + T%2 - (T%2 and i==0)
            t_ = i*8 + 8 + T%2
            model_input = sample[:, :, t: t_, :, :].to(dtype=computation_dtype, device=computation_device)
            model_output = self.forward(model_input).to(dtype=sample.dtype, device=sample.device)
            value.append(model_output)
        value = torch.concat(value, dim=2)
        for name, module in self.named_modules():
            if isinstance(module, CachedConv3d):
                module.clear_cache()
        return value
    

    @staticmethod
    def state_dict_converter():
        return CogVAEEncoderStateDictConverter()



class CogVAEEncoderStateDictConverter:
    def __init__(self):
        pass


    def from_diffusers(self, state_dict):
        rename_dict = {
            "encoder.conv_in.conv.weight": "conv_in.weight",
            "encoder.conv_in.conv.bias": "conv_in.bias",
            "encoder.down_blocks.0.downsamplers.0.conv.weight": "blocks.3.conv.weight",
            "encoder.down_blocks.0.downsamplers.0.conv.bias": "blocks.3.conv.bias",
            "encoder.down_blocks.1.downsamplers.0.conv.weight": "blocks.7.conv.weight",
            "encoder.down_blocks.1.downsamplers.0.conv.bias": "blocks.7.conv.bias",
            "encoder.down_blocks.2.downsamplers.0.conv.weight": "blocks.11.conv.weight",
            "encoder.down_blocks.2.downsamplers.0.conv.bias": "blocks.11.conv.bias",
            "encoder.norm_out.weight": "norm_out.weight",
            "encoder.norm_out.bias": "norm_out.bias",
            "encoder.conv_out.conv.weight": "conv_out.weight",
            "encoder.conv_out.conv.bias": "conv_out.bias",
        }
        prefix_dict = {
            "encoder.down_blocks.0.resnets.0.": "blocks.0.",
            "encoder.down_blocks.0.resnets.1.": "blocks.1.",
            "encoder.down_blocks.0.resnets.2.": "blocks.2.",
            "encoder.down_blocks.1.resnets.0.": "blocks.4.",
            "encoder.down_blocks.1.resnets.1.": "blocks.5.",
            "encoder.down_blocks.1.resnets.2.": "blocks.6.",
            "encoder.down_blocks.2.resnets.0.": "blocks.8.",
            "encoder.down_blocks.2.resnets.1.": "blocks.9.",
            "encoder.down_blocks.2.resnets.2.": "blocks.10.",
            "encoder.down_blocks.3.resnets.0.": "blocks.12.",
            "encoder.down_blocks.3.resnets.1.": "blocks.13.",
            "encoder.down_blocks.3.resnets.2.": "blocks.14.",
            "encoder.mid_block.resnets.0.": "blocks.15.",
            "encoder.mid_block.resnets.1.": "blocks.16.",
        }
        suffix_dict = {
            "norm1.norm_layer.weight": "norm1.norm_layer.weight",
            "norm1.norm_layer.bias": "norm1.norm_layer.bias",
            "norm1.conv_y.conv.weight": "norm1.conv_y.weight",
            "norm1.conv_y.conv.bias": "norm1.conv_y.bias",
            "norm1.conv_b.conv.weight": "norm1.conv_b.weight",
            "norm1.conv_b.conv.bias": "norm1.conv_b.bias",
            "norm2.norm_layer.weight": "norm2.norm_layer.weight",
            "norm2.norm_layer.bias": "norm2.norm_layer.bias",
            "norm2.conv_y.conv.weight": "norm2.conv_y.weight",
            "norm2.conv_y.conv.bias": "norm2.conv_y.bias",
            "norm2.conv_b.conv.weight": "norm2.conv_b.weight",
            "norm2.conv_b.conv.bias": "norm2.conv_b.bias",
            "conv1.conv.weight": "conv1.weight",
            "conv1.conv.bias": "conv1.bias",
            "conv2.conv.weight": "conv2.weight",
            "conv2.conv.bias": "conv2.bias",
            "conv_shortcut.weight": "conv_shortcut.weight",
            "conv_shortcut.bias": "conv_shortcut.bias",
            "norm1.weight": "norm1.weight",
            "norm1.bias": "norm1.bias",
            "norm2.weight": "norm2.weight",
            "norm2.bias": "norm2.bias",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                for prefix in prefix_dict:
                    if name.startswith(prefix):
                        suffix = name[len(prefix):]
                        state_dict_[prefix_dict[prefix] + suffix_dict[suffix]] = param
        return state_dict_
    

    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)



class CogVAEDecoderStateDictConverter:
    def __init__(self):
        pass


    def from_diffusers(self, state_dict):
        rename_dict = {
            "decoder.conv_in.conv.weight": "conv_in.weight",
            "decoder.conv_in.conv.bias": "conv_in.bias",
            "decoder.up_blocks.0.upsamplers.0.conv.weight": "blocks.6.conv.weight",
            "decoder.up_blocks.0.upsamplers.0.conv.bias": "blocks.6.conv.bias",
            "decoder.up_blocks.1.upsamplers.0.conv.weight": "blocks.11.conv.weight",
            "decoder.up_blocks.1.upsamplers.0.conv.bias": "blocks.11.conv.bias",
            "decoder.up_blocks.2.upsamplers.0.conv.weight": "blocks.16.conv.weight",
            "decoder.up_blocks.2.upsamplers.0.conv.bias": "blocks.16.conv.bias",
            "decoder.norm_out.norm_layer.weight": "norm_out.norm_layer.weight",
            "decoder.norm_out.norm_layer.bias": "norm_out.norm_layer.bias",
            "decoder.norm_out.conv_y.conv.weight": "norm_out.conv_y.weight",
            "decoder.norm_out.conv_y.conv.bias": "norm_out.conv_y.bias",
            "decoder.norm_out.conv_b.conv.weight": "norm_out.conv_b.weight",
            "decoder.norm_out.conv_b.conv.bias": "norm_out.conv_b.bias",
            "decoder.conv_out.conv.weight": "conv_out.weight",
            "decoder.conv_out.conv.bias": "conv_out.bias"
        }
        prefix_dict = {
            "decoder.mid_block.resnets.0.": "blocks.0.",
            "decoder.mid_block.resnets.1.": "blocks.1.",
            "decoder.up_blocks.0.resnets.0.": "blocks.2.",
            "decoder.up_blocks.0.resnets.1.": "blocks.3.",
            "decoder.up_blocks.0.resnets.2.": "blocks.4.",
            "decoder.up_blocks.0.resnets.3.": "blocks.5.",
            "decoder.up_blocks.1.resnets.0.": "blocks.7.",
            "decoder.up_blocks.1.resnets.1.": "blocks.8.",
            "decoder.up_blocks.1.resnets.2.": "blocks.9.",
            "decoder.up_blocks.1.resnets.3.": "blocks.10.",
            "decoder.up_blocks.2.resnets.0.": "blocks.12.",
            "decoder.up_blocks.2.resnets.1.": "blocks.13.",
            "decoder.up_blocks.2.resnets.2.": "blocks.14.",
            "decoder.up_blocks.2.resnets.3.": "blocks.15.",
            "decoder.up_blocks.3.resnets.0.": "blocks.17.",
            "decoder.up_blocks.3.resnets.1.": "blocks.18.",
            "decoder.up_blocks.3.resnets.2.": "blocks.19.",
            "decoder.up_blocks.3.resnets.3.": "blocks.20.",
        }
        suffix_dict = {
            "norm1.norm_layer.weight": "norm1.norm_layer.weight",
            "norm1.norm_layer.bias": "norm1.norm_layer.bias",
            "norm1.conv_y.conv.weight": "norm1.conv_y.weight",
            "norm1.conv_y.conv.bias": "norm1.conv_y.bias",
            "norm1.conv_b.conv.weight": "norm1.conv_b.weight",
            "norm1.conv_b.conv.bias": "norm1.conv_b.bias",
            "norm2.norm_layer.weight": "norm2.norm_layer.weight",
            "norm2.norm_layer.bias": "norm2.norm_layer.bias",
            "norm2.conv_y.conv.weight": "norm2.conv_y.weight",
            "norm2.conv_y.conv.bias": "norm2.conv_y.bias",
            "norm2.conv_b.conv.weight": "norm2.conv_b.weight",
            "norm2.conv_b.conv.bias": "norm2.conv_b.bias",
            "conv1.conv.weight": "conv1.weight",
            "conv1.conv.bias": "conv1.bias",
            "conv2.conv.weight": "conv2.weight",
            "conv2.conv.bias": "conv2.bias",
            "conv_shortcut.weight": "conv_shortcut.weight",
            "conv_shortcut.bias": "conv_shortcut.bias",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                for prefix in prefix_dict:
                    if name.startswith(prefix):
                        suffix = name[len(prefix):]
                        state_dict_[prefix_dict[prefix] + suffix_dict[suffix]] = param
        return state_dict_
    

    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)

