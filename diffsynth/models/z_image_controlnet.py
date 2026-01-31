from .z_image_dit import ZImageTransformerBlock
from ..core.gradient import gradient_checkpoint_forward
from torch.nn.utils.rnn import pad_sequence
import torch
from torch import nn


class ZImageControlTransformerBlock(ZImageTransformerBlock):
    def __init__(
        self, 
        layer_id: int = 1000,
        dim: int = 3840,
        n_heads: int = 30,
        n_kv_heads: int = 30,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        modulation = True,
        block_id = 0
    ):
        super().__init__(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
        self.after_proj = nn.Linear(self.dim, self.dim)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class ZImageControlNet(torch.nn.Module):
    def __init__(
        self,
        control_layers_places=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        control_in_dim=33,
        dim=3840,
        n_refiner_layers=2,
    ):
        super().__init__()
        self.control_layers = nn.ModuleList([ZImageControlTransformerBlock(layer_id=i, block_id=i) for i in control_layers_places])
        self.control_all_x_embedder = nn.ModuleDict({"2-1": nn.Linear(1 * 2 * 2 * control_in_dim, dim, bias=True)})
        self.control_noise_refiner = nn.ModuleList([ZImageControlTransformerBlock(block_id=layer_id) for layer_id in range(n_refiner_layers)])
        self.control_layers_mapping = {0: 0, 2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 12: 6, 14: 7, 16: 8, 18: 9, 20: 10, 22: 11, 24: 12, 26: 13, 28: 14}

    def forward_layers(
        self,
        x,
        cap_feats,
        control_context,
        control_context_item_seqlens,
        kwargs,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        bsz = len(control_context)
        # unified
        cap_item_seqlens = [len(_) for _ in cap_feats]
        control_context_unified = []
        for i in range(bsz):
            control_context_len = control_context_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            control_context_unified.append(torch.cat([control_context[i][:control_context_len], cap_feats[i][:cap_len]]))
        c = pad_sequence(control_context_unified, batch_first=True, padding_value=0.0)

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)
        
        for layer in self.control_layers:
            c = gradient_checkpoint_forward(
                layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                c=c, **new_kwargs
            )
 
        hints = torch.unbind(c)[:-1]
        return hints
    
    def forward_refiner(
        self,
        dit,
        x,
        cap_feats,
        control_context,
        kwargs,
        t=None,
        patch_size=2,
        f_patch_size=1,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        # embeddings
        bsz = len(control_context)
        device = control_context[0].device
        (
            control_context,
            control_context_size,
            control_context_pos_ids,
            control_context_inner_pad_mask,
        ) = dit.patchify_controlnet(control_context, patch_size, f_patch_size, cap_feats[0].size(0))

        # control_context embed & refine
        control_context_item_seqlens = [len(_) for _ in control_context]
        assert all(_ % 2 == 0 for _ in control_context_item_seqlens)
        control_context_max_item_seqlen = max(control_context_item_seqlens)

        control_context = torch.cat(control_context, dim=0)
        control_context = self.control_all_x_embedder[f"{patch_size}-{f_patch_size}"](control_context)

        # Match t_embedder output dtype to control_context for layerwise casting compatibility
        adaln_input = t.type_as(control_context)
        control_context[torch.cat(control_context_inner_pad_mask)] = dit.x_pad_token.to(dtype=control_context.dtype, device=control_context.device)
        control_context = list(control_context.split(control_context_item_seqlens, dim=0))
        control_context_freqs_cis = list(dit.rope_embedder(torch.cat(control_context_pos_ids, dim=0)).split(control_context_item_seqlens, dim=0))

        control_context = pad_sequence(control_context, batch_first=True, padding_value=0.0)
        control_context_freqs_cis = pad_sequence(control_context_freqs_cis, batch_first=True, padding_value=0.0)
        control_context_attn_mask = torch.zeros((bsz, control_context_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(control_context_item_seqlens):
            control_context_attn_mask[i, :seq_len] = 1
        c = control_context

        # arguments
        new_kwargs = dict(
            x=x, 
            attn_mask=control_context_attn_mask,
            freqs_cis=control_context_freqs_cis, 
            adaln_input=adaln_input,
        )
        new_kwargs.update(kwargs)
        
        for layer in self.control_noise_refiner:
            c = gradient_checkpoint_forward(
                layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                c=c, **new_kwargs
            )
 
        hints = torch.unbind(c)[:-1]
        control_context = torch.unbind(c)[-1]

        return hints, control_context, control_context_item_seqlens