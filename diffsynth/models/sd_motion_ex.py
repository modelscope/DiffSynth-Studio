from .attention import Attention
from .svd_unet import get_timestep_embedding
import torch
from einops import rearrange, repeat



class ExVideoMotionBlock(torch.nn.Module):

    def __init__(self, num_attention_heads, attention_head_dim, in_channels, max_position_embeddings=16, num_layers=1, add_positional_conv=None):
        super().__init__()

        emb = get_timestep_embedding(torch.arange(max_position_embeddings), in_channels, True, 0).reshape(max_position_embeddings, in_channels, 1, 1)
        self.positional_embedding = torch.nn.Parameter(emb)
        self.positional_conv = torch.nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1) if add_positional_conv is not None else None
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(in_channels) for _ in range(num_layers)])
        self.attns = torch.nn.ModuleList([Attention(q_dim=in_channels, num_heads=num_attention_heads, head_dim=attention_head_dim, bias_out=True) for _ in range(num_layers)])

    def frame_id_to_position_id(self, frame_id, max_id, repeat_length):
        if frame_id < max_id:
            position_id = frame_id
        else:
            position_id = (frame_id - max_id) % (repeat_length * 2)
            if position_id < repeat_length:
                position_id = max_id - 2 - position_id
            else:
                position_id = max_id - 2 * repeat_length + position_id
        return position_id
    
    def positional_ids(self, num_frames):
        max_id = self.positional_embedding.shape[0]
        positional_ids = torch.IntTensor([self.frame_id_to_position_id(i, max_id, max_id - 1) for i in range(num_frames)])
        return positional_ids

    def forward(self, hidden_states, time_emb, text_emb, res_stack, batch_size=1, **kwargs):
        batch, inner_dim, height, width = hidden_states.shape
        residual = hidden_states

        pos_emb = self.positional_ids(batch // batch_size)
        pos_emb = self.positional_embedding[pos_emb]
        pos_emb = pos_emb.repeat(batch_size)
        hidden_states = hidden_states + pos_emb
        if self.positional_conv is not None:
            hidden_states = rearrange(hidden_states, "(B T) C H W -> B C T H W", B=batch_size)
            hidden_states = self.positional_conv(hidden_states)
            hidden_states = rearrange(hidden_states, "B C T H W -> (B H W) T C")
        else:
            hidden_states = rearrange(hidden_states, "(B T) C H W -> (B H W) T C", B=batch_size)

        for norm, attn in zip(self.norms, self.attns):
            norm_hidden_states = norm(hidden_states)
            attn_output = attn(norm_hidden_states)
            hidden_states = hidden_states + attn_output

        hidden_states = rearrange(hidden_states, "(B H W) T C -> (B T) C H W", B=batch_size, H=height, W=width)
        hidden_states = hidden_states + residual
        return hidden_states, time_emb, text_emb, res_stack



class ExVideoMotionModel(torch.nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        self.motion_modules = torch.nn.ModuleList([
            ExVideoMotionBlock(8, 40, 320, num_layers=num_layers),
            ExVideoMotionBlock(8, 40, 320, num_layers=num_layers),
            ExVideoMotionBlock(8, 80, 640, num_layers=num_layers),
            ExVideoMotionBlock(8, 80, 640, num_layers=num_layers),
            ExVideoMotionBlock(8, 160, 1280, num_layers=num_layers),
            ExVideoMotionBlock(8, 160, 1280, num_layers=num_layers),
            ExVideoMotionBlock(8, 160, 1280, num_layers=num_layers),
            ExVideoMotionBlock(8, 160, 1280, num_layers=num_layers),
            ExVideoMotionBlock(8, 160, 1280, num_layers=num_layers),
            ExVideoMotionBlock(8, 160, 1280, num_layers=num_layers),
            ExVideoMotionBlock(8, 160, 1280, num_layers=num_layers),
            ExVideoMotionBlock(8, 160, 1280, num_layers=num_layers),
            ExVideoMotionBlock(8, 160, 1280, num_layers=num_layers),
            ExVideoMotionBlock(8, 160, 1280, num_layers=num_layers),
            ExVideoMotionBlock(8, 160, 1280, num_layers=num_layers),
            ExVideoMotionBlock(8, 80, 640, num_layers=num_layers),
            ExVideoMotionBlock(8, 80, 640, num_layers=num_layers),
            ExVideoMotionBlock(8, 80, 640, num_layers=num_layers),
            ExVideoMotionBlock(8, 40, 320, num_layers=num_layers),
            ExVideoMotionBlock(8, 40, 320, num_layers=num_layers),
            ExVideoMotionBlock(8, 40, 320, num_layers=num_layers),
        ])
        self.call_block_id = {
            1: 0,
            4: 1,
            9: 2,
            12: 3,
            17: 4,
            20: 5,
            24: 6,
            26: 7,
            29: 8,
            32: 9,
            34: 10,
            36: 11,
            40: 12,
            43: 13,
            46: 14,
            50: 15,
            53: 16,
            56: 17,
            60: 18,
            63: 19,
            66: 20
        }
        
    def forward(self):
        pass

    def state_dict_converter(self):
        pass
