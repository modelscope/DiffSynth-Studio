from diffsynth import load_state_dict
from safetensors.torch import save_file
import torch


def Flux2DiTStateDictConverter(state_dict):
    rename_dict = {
        "time_guidance_embed.timestep_embedder.linear_1.weight": "time_guidance_embed.timestep_embedder.0.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight": "time_guidance_embed.timestep_embedder.2.weight",
        "x_embedder.weight": "img_embedder.weight",
        "context_embedder.weight": "txt_embedder.weight",
    }
    state_dict_ = {}
    for name in state_dict:
        if name in rename_dict:
            state_dict_[rename_dict[name]] = state_dict[name]
        elif name.startswith("transformer_blocks"):
            if name.endswith("attn.to_q.weight"):
                state_dict_[name.replace("to_q", "img_to_qkv").replace(".attn.", ".")] = torch.concat([
                    state_dict[name.replace("to_q", "to_q")],
                    state_dict[name.replace("to_q", "to_k")],
                    state_dict[name.replace("to_q", "to_v")],
                ], dim=0)
            elif name.endswith("attn.to_k.weight") or name.endswith("attn.to_v.weight"):
                continue
            elif name.endswith("attn.to_out.0.weight"):
                state_dict_[name.replace("attn.to_out.0.weight", "img_to_out.weight")] = state_dict[name]
            elif name.endswith("attn.norm_q.weight"):
                state_dict_[name.replace("attn.norm_q.weight", "img_norm_q.weight")] = state_dict[name]
            elif name.endswith("attn.norm_k.weight"):
                state_dict_[name.replace("attn.norm_k.weight", "img_norm_k.weight")] = state_dict[name]
            elif name.endswith("attn.norm_added_q.weight"):
                state_dict_[name.replace("attn.norm_added_q.weight", "txt_norm_q.weight")] = state_dict[name]
            elif name.endswith("attn.norm_added_k.weight"):
                state_dict_[name.replace("attn.norm_added_k.weight", "txt_norm_k.weight")] = state_dict[name]
            elif name.endswith("attn.to_add_out.weight"):
                state_dict_[name.replace("attn.to_add_out.weight", "txt_to_out.weight")] = state_dict[name]
            elif name.endswith("attn.add_q_proj.weight"):
                state_dict_[name.replace("add_q_proj", "txt_to_qkv").replace(".attn.", ".")] = torch.concat([
                    state_dict[name.replace("add_q_proj", "add_q_proj")],
                    state_dict[name.replace("add_q_proj", "add_k_proj")],
                    state_dict[name.replace("add_q_proj", "add_v_proj")],
                ], dim=0)
            elif ".ff." in name:
                state_dict_[name.replace(".ff.", ".img_ff.")] = state_dict[name]
            elif ".ff_context." in name:
                state_dict_[name.replace(".ff_context.", ".txt_ff.")] = state_dict[name]
            elif name.endswith("attn.add_k_proj.weight") or name.endswith("attn.add_v_proj.weight"):
                continue
            else:
                state_dict_[name] = state_dict[name]
        elif name.startswith("single_transformer_blocks"):
            state_dict_[name.replace(".attn.", ".")] = state_dict[name]
        else:
            state_dict_[name] = state_dict[name]
    return state_dict_


state_dict = load_state_dict("xxx.safetensors")
save_file(state_dict, "yyy.safetensors")
