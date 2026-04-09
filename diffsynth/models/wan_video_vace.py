import torch
from .wan_video_dit import DiTBlock
from ..core.gradient import gradient_checkpoint_forward


class VaceWanAttentionBlock(DiTBlock):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
        self.after_proj = torch.nn.Linear(self.dim, self.dim)

    def forward(self, c, x, context, t_mod, freqs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, context, t_mod, freqs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class VaceWanModel(torch.nn.Module):
    def __init__(
        self,
        vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        vace_in_dim=96,
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        eps=1e-6,
        depth_fusion_scale=0.4,
        depth_mode="adapter",
    ):
        super().__init__()
        self.vace_layers = vace_layers
        self.vace_in_dim = vace_in_dim
        self.depth_fusion_scale = depth_fusion_scale
        if depth_mode not in ("adapter", "concat"):
            raise ValueError(f"Unsupported depth_mode={depth_mode!r}. Expected 'adapter' or 'concat'.")
        self.depth_mode = depth_mode
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # vace blocks
        self.vace_blocks = torch.nn.ModuleList([
            VaceWanAttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
            for i in self.vace_layers
        ])

        # vace patch embeddings
        self.vace_patch_embedding = torch.nn.Conv3d(vace_in_dim, dim, kernel_size=patch_size, stride=patch_size)
        if self.depth_mode == "adapter":
            self.depth_adapter = torch.nn.Sequential(
                torch.nn.Conv3d(16, 64, kernel_size=3, padding=1),
                torch.nn.SiLU(),
                torch.nn.Conv3d(64, 128, kernel_size=3, padding=1),
                torch.nn.SiLU(),
                torch.nn.Conv3d(128, dim, kernel_size=patch_size, stride=patch_size),
            )

    def forward(
        self, x, vace_context, context, t_mod, freqs, depth_latents=None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        if self.depth_mode == "adapter" and hasattr(self, "depth_adapter") and depth_latents is not None:
            c_depth = [self.depth_adapter(u.unsqueeze(0)) for u in depth_latents]
            c = [u + self.depth_fusion_scale * v for u, v in zip(c, c_depth)]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])
        
        for block in self.vace_blocks:
            c = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                c, x, context, t_mod, freqs
            )
            
        hints = torch.unbind(c)[:-1]
        return hints