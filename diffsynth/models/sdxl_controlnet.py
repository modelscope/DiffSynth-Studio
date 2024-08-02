import torch
from .sd_unet import Timesteps, ResnetBlock, AttentionBlock, PushBlock, DownSampler
from .sdxl_unet import SDXLUNet
from .tiler import TileWorker
from .sd_controlnet import ControlNetConditioningLayer
from collections import OrderedDict



class QuickGELU(torch.nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class ResidualAttentionBlock(torch.nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = torch.nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = torch.nn.LayerNorm(d_model)
        self.mlp = torch.nn.Sequential(OrderedDict([
            ("c_fc", torch.nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", torch.nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = torch.nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class SDXLControlNetUnion(torch.nn.Module):
    def __init__(self, global_pool=False):
        super().__init__()
        self.time_proj = Timesteps(320)
        self.time_embedding = torch.nn.Sequential(
            torch.nn.Linear(320, 1280),
            torch.nn.SiLU(),
            torch.nn.Linear(1280, 1280)
        )
        self.add_time_proj = Timesteps(256)
        self.add_time_embedding = torch.nn.Sequential(
            torch.nn.Linear(2816, 1280),
            torch.nn.SiLU(),
            torch.nn.Linear(1280, 1280)
        )
        self.control_type_proj = Timesteps(256)
        self.control_type_embedding = torch.nn.Sequential(
            torch.nn.Linear(256 * 8, 1280),
            torch.nn.SiLU(),
            torch.nn.Linear(1280, 1280)
        )
        self.conv_in = torch.nn.Conv2d(4, 320, kernel_size=3, padding=1)

        self.controlnet_conv_in = ControlNetConditioningLayer(channels=(3, 16, 32, 96, 256, 320))
        self.controlnet_transformer = ResidualAttentionBlock(320, 8)
        self.task_embedding = torch.nn.Parameter(torch.randn(8, 320))
        self.spatial_ch_projs = torch.nn.Linear(320, 320)

        self.blocks = torch.nn.ModuleList([
            # DownBlock2D
            ResnetBlock(320, 320, 1280),
            PushBlock(),
            ResnetBlock(320, 320, 1280),
            PushBlock(),
            DownSampler(320),
            PushBlock(),
            # CrossAttnDownBlock2D
            ResnetBlock(320, 640, 1280),
            AttentionBlock(10, 64, 640, 2, 2048),
            PushBlock(),
            ResnetBlock(640, 640, 1280),
            AttentionBlock(10, 64, 640, 2, 2048),
            PushBlock(),
            DownSampler(640),
            PushBlock(),
            # CrossAttnDownBlock2D
            ResnetBlock(640, 1280, 1280),
            AttentionBlock(20, 64, 1280, 10, 2048),
            PushBlock(),
            ResnetBlock(1280, 1280, 1280),
            AttentionBlock(20, 64, 1280, 10, 2048),
            PushBlock(),
            # UNetMidBlock2DCrossAttn
            ResnetBlock(1280, 1280, 1280),
            AttentionBlock(20, 64, 1280, 10, 2048),
            ResnetBlock(1280, 1280, 1280),
            PushBlock()
        ])

        self.controlnet_blocks = torch.nn.ModuleList([
            torch.nn.Conv2d(320, 320, kernel_size=(1, 1)),
            torch.nn.Conv2d(320, 320, kernel_size=(1, 1)),
            torch.nn.Conv2d(320, 320, kernel_size=(1, 1)),
            torch.nn.Conv2d(320, 320, kernel_size=(1, 1)),
            torch.nn.Conv2d(640, 640, kernel_size=(1, 1)),
            torch.nn.Conv2d(640, 640, kernel_size=(1, 1)),
            torch.nn.Conv2d(640, 640, kernel_size=(1, 1)),
            torch.nn.Conv2d(1280, 1280, kernel_size=(1, 1)),
            torch.nn.Conv2d(1280, 1280, kernel_size=(1, 1)),
            torch.nn.Conv2d(1280, 1280, kernel_size=(1, 1)),
        ])

        self.global_pool = global_pool

        # 0 -- openpose
        # 1 -- depth
        # 2 -- hed/pidi/scribble/ted
        # 3 -- canny/lineart/anime_lineart/mlsd
        # 4 -- normal
        # 5 -- segment
        # 6 -- tile
        # 7 -- repaint
        self.task_id = {
            "openpose": 0,
            "depth": 1,
            "softedge": 2,
            "canny": 3,
            "lineart": 3,
            "lineart_anime": 3,
            "tile": 6,
            "inpaint": 7
        }


    def fuse_condition_to_input(self, hidden_states, task_id, conditioning):
        controlnet_cond = self.controlnet_conv_in(conditioning)
        feat_seq = torch.mean(controlnet_cond, dim=(2, 3))
        feat_seq = feat_seq + self.task_embedding[task_id]
        x = torch.stack([feat_seq, torch.mean(hidden_states, dim=(2, 3))], dim=1)
        x = self.controlnet_transformer(x)

        alpha = self.spatial_ch_projs(x[:,0]).unsqueeze(-1).unsqueeze(-1)
        controlnet_cond_fuser = controlnet_cond + alpha

        hidden_states = hidden_states + controlnet_cond_fuser
        return hidden_states
    

    def forward(
        self,
        sample, timestep, encoder_hidden_states,
        conditioning, processor_id, add_time_id, add_text_embeds,
        tiled=False, tile_size=64, tile_stride=32,
        unet:SDXLUNet=None,
        **kwargs
    ):
        task_id = self.task_id[processor_id]

        # 1. time
        t_emb = self.time_proj(timestep).to(sample.dtype)
        t_emb = self.time_embedding(t_emb)
        
        time_embeds = self.add_time_proj(add_time_id)
        time_embeds = time_embeds.reshape((add_text_embeds.shape[0], -1))
        add_embeds = torch.concat([add_text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(sample.dtype)
        if unet is not None and unet.is_kolors:
            add_embeds = unet.add_time_embedding(add_embeds)
        else:
            add_embeds = self.add_time_embedding(add_embeds)

        control_type = torch.zeros((sample.shape[0], 8), dtype=sample.dtype, device=sample.device)
        control_type[:, task_id] = 1
        control_embeds = self.control_type_proj(control_type.flatten())
        control_embeds = control_embeds.reshape((sample.shape[0], -1))
        control_embeds = control_embeds.to(sample.dtype)
        control_embeds = self.control_type_embedding(control_embeds)
        time_emb = t_emb + add_embeds + control_embeds

        # 2. pre-process
        height, width = sample.shape[2], sample.shape[3]
        hidden_states = self.conv_in(sample)
        hidden_states = self.fuse_condition_to_input(hidden_states, task_id, conditioning)
        text_emb = encoder_hidden_states
        if unet is not None and unet.is_kolors:
            text_emb = unet.text_intermediate_proj(text_emb)
        res_stack = [hidden_states]

        # 3. blocks
        for i, block in enumerate(self.blocks):
            if tiled and not isinstance(block, PushBlock):
                _, _, inter_height, _ = hidden_states.shape
                resize_scale = inter_height / height
                hidden_states = TileWorker().tiled_forward(
                    lambda x: block(x, time_emb, text_emb, res_stack)[0],
                    hidden_states,
                    int(tile_size * resize_scale),
                    int(tile_stride * resize_scale),
                    tile_device=hidden_states.device,
                    tile_dtype=hidden_states.dtype
                )
            else:
                hidden_states, _, _, _ = block(hidden_states, time_emb, text_emb, res_stack)

        # 4. ControlNet blocks
        controlnet_res_stack = [block(res) for block, res in zip(self.controlnet_blocks, res_stack)]

        # pool
        if self.global_pool:
            controlnet_res_stack = [res.mean(dim=(2, 3), keepdim=True) for res in controlnet_res_stack]

        return controlnet_res_stack

    @staticmethod
    def state_dict_converter():
        return SDXLControlNetUnionStateDictConverter()



class SDXLControlNetUnionStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        # architecture
        block_types = [
            "ResnetBlock", "PushBlock", "ResnetBlock", "PushBlock", "DownSampler", "PushBlock",
            "ResnetBlock", "AttentionBlock", "PushBlock", "ResnetBlock", "AttentionBlock", "PushBlock", "DownSampler", "PushBlock",
            "ResnetBlock", "AttentionBlock", "PushBlock", "ResnetBlock", "AttentionBlock", "PushBlock",
            "ResnetBlock", "AttentionBlock", "ResnetBlock", "PushBlock"
        ]

        # controlnet_rename_dict
        controlnet_rename_dict = {
            "controlnet_cond_embedding.conv_in.weight": "controlnet_conv_in.blocks.0.weight",
            "controlnet_cond_embedding.conv_in.bias": "controlnet_conv_in.blocks.0.bias",
            "controlnet_cond_embedding.blocks.0.weight": "controlnet_conv_in.blocks.2.weight",
            "controlnet_cond_embedding.blocks.0.bias": "controlnet_conv_in.blocks.2.bias",
            "controlnet_cond_embedding.blocks.1.weight": "controlnet_conv_in.blocks.4.weight",
            "controlnet_cond_embedding.blocks.1.bias": "controlnet_conv_in.blocks.4.bias",
            "controlnet_cond_embedding.blocks.2.weight": "controlnet_conv_in.blocks.6.weight",
            "controlnet_cond_embedding.blocks.2.bias": "controlnet_conv_in.blocks.6.bias",
            "controlnet_cond_embedding.blocks.3.weight": "controlnet_conv_in.blocks.8.weight",
            "controlnet_cond_embedding.blocks.3.bias": "controlnet_conv_in.blocks.8.bias",
            "controlnet_cond_embedding.blocks.4.weight": "controlnet_conv_in.blocks.10.weight",
            "controlnet_cond_embedding.blocks.4.bias": "controlnet_conv_in.blocks.10.bias",
            "controlnet_cond_embedding.blocks.5.weight": "controlnet_conv_in.blocks.12.weight",
            "controlnet_cond_embedding.blocks.5.bias": "controlnet_conv_in.blocks.12.bias",
            "controlnet_cond_embedding.conv_out.weight": "controlnet_conv_in.blocks.14.weight",
            "controlnet_cond_embedding.conv_out.bias": "controlnet_conv_in.blocks.14.bias",
            "control_add_embedding.linear_1.weight": "control_type_embedding.0.weight",
            "control_add_embedding.linear_1.bias": "control_type_embedding.0.bias",
            "control_add_embedding.linear_2.weight": "control_type_embedding.2.weight",
            "control_add_embedding.linear_2.bias": "control_type_embedding.2.bias",
        }

        # Rename each parameter
        name_list = sorted([name for name in state_dict])
        rename_dict = {}
        block_id = {"ResnetBlock": -1, "AttentionBlock": -1, "DownSampler": -1, "UpSampler": -1}
        last_block_type_with_id = {"ResnetBlock": "", "AttentionBlock": "", "DownSampler": "", "UpSampler": ""}
        for name in name_list:
            names = name.split(".")
            if names[0] in ["conv_in", "conv_norm_out", "conv_out", "task_embedding", "spatial_ch_projs"]:
                pass
            elif name in controlnet_rename_dict:
                names = controlnet_rename_dict[name].split(".")
            elif names[0] == "controlnet_down_blocks":
                names[0] = "controlnet_blocks"
            elif names[0] == "controlnet_mid_block":
                names = ["controlnet_blocks", "9", names[-1]]
            elif names[0] in ["time_embedding", "add_embedding"]:
                if names[0] == "add_embedding":
                    names[0] = "add_time_embedding"
                names[1] = {"linear_1": "0", "linear_2": "2"}[names[1]]
            elif names[0] == "control_add_embedding":
                names[0] = "control_type_embedding"
            elif names[0] == "transformer_layes":
                names[0] = "controlnet_transformer"
                names.pop(1)
            elif names[0] in ["down_blocks", "mid_block", "up_blocks"]:
                if names[0] == "mid_block":
                    names.insert(1, "0")
                block_type = {"resnets": "ResnetBlock", "attentions": "AttentionBlock", "downsamplers": "DownSampler", "upsamplers": "UpSampler"}[names[2]]
                block_type_with_id = ".".join(names[:4])
                if block_type_with_id != last_block_type_with_id[block_type]:
                    block_id[block_type] += 1
                last_block_type_with_id[block_type] = block_type_with_id
                while block_id[block_type] < len(block_types) and block_types[block_id[block_type]] != block_type:
                    block_id[block_type] += 1
                block_type_with_id = ".".join(names[:4])
                names = ["blocks", str(block_id[block_type])] + names[4:]
                if "ff" in names:
                    ff_index = names.index("ff")
                    component = ".".join(names[ff_index:ff_index+3])
                    component = {"ff.net.0": "act_fn", "ff.net.2": "ff"}[component]
                    names = names[:ff_index] + [component] + names[ff_index+3:]
                if "to_out" in names:
                    names.pop(names.index("to_out") + 1)
            else:
                print(name, state_dict[name].shape)
                # raise ValueError(f"Unknown parameters: {name}")
            rename_dict[name] = ".".join(names)

        # Convert state_dict
        state_dict_ = {}
        for name, param in state_dict.items():
            if name not in rename_dict:
                continue
            if ".proj_in." in name or ".proj_out." in name:
                param = param.squeeze()
            state_dict_[rename_dict[name]] = param
        return state_dict_
    
    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)