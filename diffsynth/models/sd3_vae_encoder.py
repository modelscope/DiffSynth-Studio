import torch
from .sd_unet import ResnetBlock, DownSampler
from .sd_vae_encoder import VAEAttentionBlock, SDVAEEncoderStateDictConverter
from .tiler import TileWorker
from einops import rearrange


class SD3VAEEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scaling_factor = 1.5305 # Different from SD 1.x
        self.shift_factor = 0.0609 # Different from SD 1.x
        self.conv_in = torch.nn.Conv2d(3, 128, kernel_size=3, padding=1)

        self.blocks = torch.nn.ModuleList([
            # DownEncoderBlock2D
            ResnetBlock(128, 128, eps=1e-6),
            ResnetBlock(128, 128, eps=1e-6),
            DownSampler(128, padding=0, extra_padding=True),
            # DownEncoderBlock2D
            ResnetBlock(128, 256, eps=1e-6),
            ResnetBlock(256, 256, eps=1e-6),
            DownSampler(256, padding=0, extra_padding=True),
            # DownEncoderBlock2D
            ResnetBlock(256, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            DownSampler(512, padding=0, extra_padding=True),
            # DownEncoderBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
            # UNetMidBlock2D
            ResnetBlock(512, 512, eps=1e-6),
            VAEAttentionBlock(1, 512, 512, 1, eps=1e-6),
            ResnetBlock(512, 512, eps=1e-6),
        ])

        self.conv_norm_out = torch.nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6)
        self.conv_act = torch.nn.SiLU()
        self.conv_out = torch.nn.Conv2d(512, 32, kernel_size=3, padding=1)

    def tiled_forward(self, sample, tile_size=64, tile_stride=32):
        hidden_states = TileWorker().tiled_forward(
            lambda x: self.forward(x),
            sample,
            tile_size,
            tile_stride,
            tile_device=sample.device,
            tile_dtype=sample.dtype
        )
        return hidden_states

    def forward(self, sample, tiled=False, tile_size=64, tile_stride=32, **kwargs):
        # For VAE Decoder, we do not need to apply the tiler on each layer.
        if tiled:
            return self.tiled_forward(sample, tile_size=tile_size, tile_stride=tile_stride)
        
        # 1. pre-process
        hidden_states = self.conv_in(sample)
        time_emb = None
        text_emb = None
        res_stack = None

        # 2. blocks
        for i, block in enumerate(self.blocks):
            hidden_states, time_emb, text_emb, res_stack = block(hidden_states, time_emb, text_emb, res_stack)
        
        # 3. output
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = hidden_states[:, :16]
        hidden_states = (hidden_states - self.shift_factor) * self.scaling_factor

        return hidden_states
    
    def encode_video(self, sample, batch_size=8):
        B = sample.shape[0]
        hidden_states = []

        for i in range(0, sample.shape[2], batch_size):

            j = min(i + batch_size, sample.shape[2])
            sample_batch = rearrange(sample[:,:,i:j], "B C T H W -> (B T) C H W")

            hidden_states_batch = self(sample_batch)
            hidden_states_batch = rearrange(hidden_states_batch, "(B T) C H W -> B C T H W", B=B)

            hidden_states.append(hidden_states_batch)
        
        hidden_states = torch.concat(hidden_states, dim=2)
        return hidden_states
    
    @staticmethod
    def state_dict_converter():
        return SDVAEEncoderStateDictConverter()
