from .qwen_image_21_dit import QwenImage21TransformerBlock
from ..core.gradient import gradient_checkpoint_forward
import torch
import torch.nn as nn


class QwenImage21ControlNetBlock(QwenImage21TransformerBlock):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: int = 3,
        eps: float = 1e-6,
        block_id: int = 0,
    ):
        super().__init__(dim, num_attention_heads, attention_head_dim, mlp_ratio, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(dim, dim)
        self.after_proj = nn.Linear(dim, dim)

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


class QwenImage21ControlNet(nn.Module):
    _repeated_blocks = ["QwenImage21ControlNetBlock"]

    def __init__(
        self,
        control_layers: tuple = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30),
        control_in_dim: int = 129,
        dim: int = 4096,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        mlp_ratio: int = 3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.control_layers = tuple(sorted({int(i) for i in control_layers}))
        self.control_in_dim = control_in_dim
        self.control_layers_mapping = {i: n for n, i in enumerate(self.control_layers)}
        self.control_img_in = nn.Linear(control_in_dim, dim)
        self.control_blocks = nn.ModuleList(
            [
                QwenImage21ControlNetBlock(dim, num_attention_heads, attention_head_dim, mlp_ratio, eps, block_id=i)
                for i in self.control_layers
            ]
        )

    def forward(
        self,
        joint_hidden_states: torch.Tensor,
        control_context: torch.Tensor,
        image_positions: torch.Tensor,
        modulation: torch.Tensor,
        rotary_emb: torch.Tensor,
        attention_mask,
        target_token_mask: torch.Tensor | None,
        segments,
        key_valid: torch.Tensor | None,
        control_scale: float = 1.0,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> dict[int, torch.Tensor]:
        control_features = self.control_img_in(control_context)
        c = torch.zeros_like(joint_hidden_states)
        c[:, image_positions] = control_features.to(joint_hidden_states.dtype)
        kwargs = dict(
            modulation=modulation,
            rotary_emb=rotary_emb,
            attention_mask=attention_mask,
            target_token_mask=target_token_mask,
            kv_cache=None,
            cache_write_slice=None,
            segments=segments,
            key_valid=key_valid,
        )
        for block in self.control_blocks:
            c = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                c,
                joint_hidden_states,
                **kwargs,
            )
        hints = torch.unbind(c)[:-1]
        return {block_id: hints[index] * control_scale for block_id, index in self.control_layers_mapping.items()}
