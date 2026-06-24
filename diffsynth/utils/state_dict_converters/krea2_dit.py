def Krea2DiTStateDictConverter(state_dict):
    state_dict_ = {i: state_dict[i] for i in state_dict if i not in ["last.down.weight", "last.up.weight"]}
    return state_dict_

def Krea2DiTStateDictConverter_Diffusers(state_dict):
    rename_dict = {
        "first.bias": "img_in.bias",
        "first.weight": "img_in.weight",
        "last.linear.bias": "final_layer.linear.bias",
        "last.linear.weight": "final_layer.linear.weight",
        "last.modulation.lin": "final_layer.scale_shift_table",
        "last.norm.scale": "final_layer.norm.weight",
        "tmlp.0.bias": "time_embed.linear_1.bias",
        "tmlp.0.weight": "time_embed.linear_1.weight",
        "tmlp.2.bias": "time_embed.linear_2.bias",
        "tmlp.2.weight": "time_embed.linear_2.weight",
        "tproj.1.bias": "time_mod_proj.bias",
        "tproj.1.weight": "time_mod_proj.weight",
        "txtfusion.layerwise_blocks.0.attn.gate.weight": "text_fusion.layerwise_blocks.0.attn.to_gate.weight",
        "txtfusion.layerwise_blocks.0.attn.qknorm.knorm.scale": "text_fusion.layerwise_blocks.0.attn.norm_k.weight",
        "txtfusion.layerwise_blocks.0.attn.qknorm.qnorm.scale": "text_fusion.layerwise_blocks.0.attn.norm_k.weight",
        "txtfusion.layerwise_blocks.0.attn.wk.weight": "text_fusion.layerwise_blocks.0.attn.to_k.weight",
        "txtfusion.layerwise_blocks.0.attn.wo.weight": "text_fusion.layerwise_blocks.0.attn.to_out.0.weight",
        "txtfusion.layerwise_blocks.0.attn.wq.weight": "text_fusion.layerwise_blocks.0.attn.to_q.weight",
        "txtfusion.layerwise_blocks.0.attn.wv.weight": "text_fusion.layerwise_blocks.0.attn.to_v.weight",
        "txtfusion.layerwise_blocks.0.mlp.down.weight": "text_fusion.layerwise_blocks.0.ff.down.weight",
        "txtfusion.layerwise_blocks.0.mlp.gate.weight": "text_fusion.layerwise_blocks.0.ff.gate.weight",
        "txtfusion.layerwise_blocks.0.mlp.up.weight": "text_fusion.layerwise_blocks.0.ff.up.weight",
        "txtfusion.layerwise_blocks.0.postnorm.scale": "text_fusion.layerwise_blocks.0.norm2.weight",
        "txtfusion.layerwise_blocks.0.prenorm.scale": "text_fusion.layerwise_blocks.0.norm1.weight",
        "txtfusion.layerwise_blocks.1.attn.gate.weight": "text_fusion.layerwise_blocks.1.attn.to_gate.weight",
        "txtfusion.layerwise_blocks.1.attn.qknorm.knorm.scale": "text_fusion.layerwise_blocks.1.attn.norm_k.weight",
        "txtfusion.layerwise_blocks.1.attn.qknorm.qnorm.scale": "text_fusion.layerwise_blocks.1.attn.norm_k.weight",
        "txtfusion.layerwise_blocks.1.attn.wk.weight": "text_fusion.layerwise_blocks.1.attn.to_k.weight",
        "txtfusion.layerwise_blocks.1.attn.wo.weight": "text_fusion.layerwise_blocks.1.attn.to_out.0.weight",
        "txtfusion.layerwise_blocks.1.attn.wq.weight": "text_fusion.layerwise_blocks.1.attn.to_q.weight",
        "txtfusion.layerwise_blocks.1.attn.wv.weight": "text_fusion.layerwise_blocks.1.attn.to_v.weight",
        "txtfusion.layerwise_blocks.1.mlp.down.weight": "text_fusion.layerwise_blocks.1.ff.down.weight",
        "txtfusion.layerwise_blocks.1.mlp.gate.weight": "text_fusion.layerwise_blocks.1.ff.gate.weight",
        "txtfusion.layerwise_blocks.1.mlp.up.weight": "text_fusion.layerwise_blocks.1.ff.up.weight",
        "txtfusion.layerwise_blocks.1.postnorm.scale": "text_fusion.layerwise_blocks.1.norm2.weight",
        "txtfusion.layerwise_blocks.1.prenorm.scale": "text_fusion.layerwise_blocks.1.norm1.weight",
        "txtfusion.projector.weight": "text_fusion.projector.weight",
        "txtfusion.refiner_blocks.0.attn.gate.weight": "text_fusion.refiner_blocks.0.attn.to_gate.weight",
        "txtfusion.refiner_blocks.0.attn.qknorm.knorm.scale": "text_fusion.refiner_blocks.0.attn.norm_k.weight",
        "txtfusion.refiner_blocks.0.attn.qknorm.qnorm.scale": "text_fusion.refiner_blocks.0.attn.norm_k.weight",
        "txtfusion.refiner_blocks.0.attn.wk.weight": "text_fusion.refiner_blocks.0.attn.to_k.weight",
        "txtfusion.refiner_blocks.0.attn.wo.weight": "text_fusion.refiner_blocks.0.attn.to_out.0.weight",
        "txtfusion.refiner_blocks.0.attn.wq.weight": "text_fusion.refiner_blocks.0.attn.to_q.weight",
        "txtfusion.refiner_blocks.0.attn.wv.weight": "text_fusion.refiner_blocks.0.attn.to_v.weight",
        "txtfusion.refiner_blocks.0.mlp.down.weight": "text_fusion.refiner_blocks.0.ff.down.weight",
        "txtfusion.refiner_blocks.0.mlp.gate.weight": "text_fusion.refiner_blocks.0.ff.gate.weight",
        "txtfusion.refiner_blocks.0.mlp.up.weight": "text_fusion.refiner_blocks.0.ff.up.weight",
        "txtfusion.refiner_blocks.0.postnorm.scale": "text_fusion.refiner_blocks.0.norm2.weight",
        "txtfusion.refiner_blocks.0.prenorm.scale": "text_fusion.refiner_blocks.0.norm1.weight",
        "txtfusion.refiner_blocks.1.attn.gate.weight": "text_fusion.refiner_blocks.1.attn.to_gate.weight",
        "txtfusion.refiner_blocks.1.attn.qknorm.knorm.scale": "text_fusion.refiner_blocks.1.attn.norm_k.weight",
        "txtfusion.refiner_blocks.1.attn.qknorm.qnorm.scale": "text_fusion.refiner_blocks.1.attn.norm_k.weight",
        "txtfusion.refiner_blocks.1.attn.wk.weight": "text_fusion.refiner_blocks.1.attn.to_k.weight",
        "txtfusion.refiner_blocks.1.attn.wo.weight": "text_fusion.refiner_blocks.1.attn.to_out.0.weight",
        "txtfusion.refiner_blocks.1.attn.wq.weight": "text_fusion.refiner_blocks.1.attn.to_q.weight",
        "txtfusion.refiner_blocks.1.attn.wv.weight": "text_fusion.refiner_blocks.1.attn.to_v.weight",
        "txtfusion.refiner_blocks.1.mlp.down.weight": "text_fusion.refiner_blocks.1.ff.down.weight",
        "txtfusion.refiner_blocks.1.mlp.gate.weight": "text_fusion.refiner_blocks.1.ff.gate.weight",
        "txtfusion.refiner_blocks.1.mlp.up.weight": "text_fusion.refiner_blocks.1.ff.up.weight",
        "txtfusion.refiner_blocks.1.postnorm.scale": "text_fusion.refiner_blocks.1.norm2.weight",
        "txtfusion.refiner_blocks.1.prenorm.scale": "text_fusion.refiner_blocks.1.norm1.weight",
        "txtmlp.0.scale": "txt_in.norm.weight",
        "txtmlp.1.bias": "txt_in.linear_1.bias",
        "txtmlp.1.weight": "txt_in.linear_1.weight",
        "txtmlp.3.bias": "txt_in.linear_2.bias",
        "txtmlp.3.weight": "txt_in.linear_2.weight",
    }
    for i in range(28):
        rename_dict[f"blocks.{i}.attn.gate.weight"] = f"transformer_blocks.{i}.attn.to_gate.weight"
        rename_dict[f"blocks.{i}.attn.qknorm.knorm.scale"] = f"transformer_blocks.{i}.attn.norm_k.weight"
        rename_dict[f"blocks.{i}.attn.qknorm.qnorm.scale"] = f"transformer_blocks.{i}.attn.norm_q.weight"
        rename_dict[f"blocks.{i}.attn.wk.weight"] = f"transformer_blocks.{i}.attn.to_k.weight"
        rename_dict[f"blocks.{i}.attn.wo.weight"] = f"transformer_blocks.{i}.attn.to_out.0.weight"
        rename_dict[f"blocks.{i}.attn.wq.weight"] = f"transformer_blocks.{i}.attn.to_q.weight"
        rename_dict[f"blocks.{i}.attn.wv.weight"] = f"transformer_blocks.{i}.attn.to_v.weight"
        rename_dict[f"blocks.{i}.mlp.down.weight"] = f"transformer_blocks.{i}.ff.down.weight"
        rename_dict[f"blocks.{i}.mlp.gate.weight"] = f"transformer_blocks.{i}.ff.gate.weight"
        rename_dict[f"blocks.{i}.mlp.up.weight"] = f"transformer_blocks.{i}.ff.up.weight"
        rename_dict[f"blocks.{i}.postnorm.scale"] = f"transformer_blocks.{i}.norm2.weight"
        rename_dict[f"blocks.{i}.prenorm.scale"] = f"transformer_blocks.{i}.norm1.weight"
        rename_dict[f"blocks.{i}.mod.lin"] = f"transformer_blocks.{i}.scale_shift_table"
    state_dict_ = {i: state_dict[rename_dict[i]] for i in rename_dict}
    for i in state_dict_:
        if i.endswith(".mod.lin"):
            state_dict_[i] = state_dict_[i].flatten()
    return state_dict_
