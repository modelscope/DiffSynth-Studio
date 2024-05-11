from .sd_motion import TemporalBlock
import torch



class SDXLMotionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_modules = torch.nn.ModuleList([
            TemporalBlock(8, 320//8, 320, eps=1e-6),
            TemporalBlock(8, 320//8, 320, eps=1e-6),

            TemporalBlock(8, 640//8, 640, eps=1e-6),
            TemporalBlock(8, 640//8, 640, eps=1e-6),

            TemporalBlock(8, 1280//8, 1280, eps=1e-6),
            TemporalBlock(8, 1280//8, 1280, eps=1e-6),

            TemporalBlock(8, 1280//8, 1280, eps=1e-6),
            TemporalBlock(8, 1280//8, 1280, eps=1e-6),
            TemporalBlock(8, 1280//8, 1280, eps=1e-6),

            TemporalBlock(8, 640//8, 640, eps=1e-6),
            TemporalBlock(8, 640//8, 640, eps=1e-6),
            TemporalBlock(8, 640//8, 640, eps=1e-6),

            TemporalBlock(8, 320//8, 320, eps=1e-6),
            TemporalBlock(8, 320//8, 320, eps=1e-6),
            TemporalBlock(8, 320//8, 320, eps=1e-6),
        ])
        self.call_block_id = {
            0: 0,
            2: 1,
            7: 2,
            10: 3,
            15: 4,
            18: 5,
            25: 6,
            28: 7,
            31: 8,
            35: 9,
            38: 10,
            41: 11,
            44: 12,
            46: 13,
            48: 14,
        }
        
    def forward(self):
        pass

    def state_dict_converter(self):
        return SDMotionModelStateDictConverter()


class SDMotionModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "norm": "norm",
            "proj_in": "proj_in",
            "transformer_blocks.0.attention_blocks.0.to_q": "transformer_blocks.0.attn1.to_q",
            "transformer_blocks.0.attention_blocks.0.to_k": "transformer_blocks.0.attn1.to_k",
            "transformer_blocks.0.attention_blocks.0.to_v": "transformer_blocks.0.attn1.to_v",
            "transformer_blocks.0.attention_blocks.0.to_out.0": "transformer_blocks.0.attn1.to_out",
            "transformer_blocks.0.attention_blocks.0.pos_encoder": "transformer_blocks.0.pe1",
            "transformer_blocks.0.attention_blocks.1.to_q": "transformer_blocks.0.attn2.to_q",
            "transformer_blocks.0.attention_blocks.1.to_k": "transformer_blocks.0.attn2.to_k",
            "transformer_blocks.0.attention_blocks.1.to_v": "transformer_blocks.0.attn2.to_v",
            "transformer_blocks.0.attention_blocks.1.to_out.0": "transformer_blocks.0.attn2.to_out",
            "transformer_blocks.0.attention_blocks.1.pos_encoder": "transformer_blocks.0.pe2",
            "transformer_blocks.0.norms.0": "transformer_blocks.0.norm1",
            "transformer_blocks.0.norms.1": "transformer_blocks.0.norm2",
            "transformer_blocks.0.ff.net.0.proj": "transformer_blocks.0.act_fn.proj",
            "transformer_blocks.0.ff.net.2": "transformer_blocks.0.ff",
            "transformer_blocks.0.ff_norm": "transformer_blocks.0.norm3",
            "proj_out": "proj_out",
        }
        name_list = sorted([i for i in state_dict if i.startswith("down_blocks.")])
        name_list += sorted([i for i in state_dict if i.startswith("mid_block.")])
        name_list += sorted([i for i in state_dict if i.startswith("up_blocks.")])
        state_dict_ = {}
        last_prefix, module_id = "", -1
        for name in name_list:
            names = name.split(".")
            prefix_index = names.index("temporal_transformer") + 1
            prefix = ".".join(names[:prefix_index])
            if prefix != last_prefix:
                last_prefix = prefix
                module_id += 1
            middle_name = ".".join(names[prefix_index:-1])
            suffix = names[-1]
            if "pos_encoder" in names:
                rename = ".".join(["motion_modules", str(module_id), rename_dict[middle_name]])
            else:
                rename = ".".join(["motion_modules", str(module_id), rename_dict[middle_name], suffix])
            state_dict_[rename] = state_dict[name]
        return state_dict_
    
    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)
