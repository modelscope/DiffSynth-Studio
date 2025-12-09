import torch


class CompressedMLP(torch.nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, bias=False):
        super().__init__()
        self.proj_in = torch.nn.Linear(in_dim, mid_dim, bias=bias)
        self.proj_out = torch.nn.Linear(mid_dim, out_dim, bias=bias)
        
    def forward(self, x, residual=None):
        x = self.proj_in(x)
        if residual is not None: x = x + residual
        x = self.proj_out(x)
        return x


class ImageEmbeddingToLoraMatrix(torch.nn.Module):
    def __init__(self, in_dim, compress_dim, lora_a_dim, lora_b_dim, rank):
        super().__init__()
        self.proj_a = CompressedMLP(in_dim, compress_dim, lora_a_dim * rank)
        self.proj_b = CompressedMLP(in_dim, compress_dim, lora_b_dim * rank)
        self.lora_a_dim = lora_a_dim
        self.lora_b_dim = lora_b_dim
        self.rank = rank
        
    def forward(self, x, residual=None):
        lora_a = self.proj_a(x, residual).view(self.rank, self.lora_a_dim)
        lora_b = self.proj_b(x, residual).view(self.lora_b_dim, self.rank)
        return lora_a, lora_b


class SequencialMLP(torch.nn.Module):
    def __init__(self, length, in_dim, mid_dim, out_dim, bias=False):
        super().__init__()
        self.proj_in = torch.nn.Linear(in_dim, mid_dim, bias=bias)
        self.proj_out = torch.nn.Linear(length * mid_dim, out_dim, bias=bias)
        self.length = length
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        
    def forward(self, x):
        x = x.view(self.length, self.in_dim)
        x = self.proj_in(x)
        x = x.view(1, self.length * self.mid_dim)
        x = self.proj_out(x)
        return x


class LoRATrainerBlock(torch.nn.Module):
    def __init__(self, lora_patterns, in_dim=1536+4096, compress_dim=128, rank=4, block_id=0, use_residual=True, residual_length=64+7, residual_dim=3584, residual_mid_dim=1024):
        super().__init__()
        self.lora_patterns = lora_patterns
        self.block_id = block_id
        self.layers = []
        for name, lora_a_dim, lora_b_dim in self.lora_patterns:
            self.layers.append(ImageEmbeddingToLoraMatrix(in_dim, compress_dim, lora_a_dim, lora_b_dim, rank))
        self.layers = torch.nn.ModuleList(self.layers)
        if use_residual:
            self.proj_residual = SequencialMLP(residual_length, residual_dim, residual_mid_dim, compress_dim)
        else:
            self.proj_residual = None
    
    def forward(self, x, residual=None):
        lora = {}
        if self.proj_residual is not None: residual = self.proj_residual(residual)
        for lora_pattern, layer in zip(self.lora_patterns, self.layers):
            name = lora_pattern[0]
            lora_a, lora_b = layer(x, residual=residual)
            lora[f"transformer_blocks.{self.block_id}.{name}.lora_A.default.weight"] = lora_a
            lora[f"transformer_blocks.{self.block_id}.{name}.lora_B.default.weight"] = lora_b
        return lora
    

class QwenImageImage2LoRAModel(torch.nn.Module):
    def __init__(self, num_blocks=60, use_residual=True, compress_dim=128, rank=4, residual_length=64+7, residual_mid_dim=1024):
        super().__init__()
        self.lora_patterns = [
            [
                ("attn.to_q", 3072, 3072),
                ("attn.to_k", 3072, 3072),
                ("attn.to_v", 3072, 3072),
                ("attn.to_out.0", 3072, 3072),
            ],
            [
                ("img_mlp.net.2", 3072*4, 3072),
                ("img_mod.1", 3072, 3072*6),
            ],
            [
                ("attn.add_q_proj", 3072, 3072),
                ("attn.add_k_proj", 3072, 3072),
                ("attn.add_v_proj", 3072, 3072),
                ("attn.to_add_out", 3072, 3072),
            ],
            [
                ("txt_mlp.net.2", 3072*4, 3072),
                ("txt_mod.1", 3072, 3072*6),
            ],
        ]
        self.num_blocks = num_blocks
        self.blocks = []
        for lora_patterns in self.lora_patterns:
            for block_id in range(self.num_blocks):
                self.blocks.append(LoRATrainerBlock(lora_patterns, block_id=block_id, use_residual=use_residual, compress_dim=compress_dim, rank=rank, residual_length=residual_length, residual_mid_dim=residual_mid_dim))
        self.blocks = torch.nn.ModuleList(self.blocks)
        self.residual_scale = 0.05
        self.use_residual = use_residual
        
    def forward(self, x, residual=None):
        if residual is not None:
            if self.use_residual:
                residual = residual * self.residual_scale
            else:
                residual = None
        lora = {}
        for block in self.blocks:
            lora.update(block(x, residual))
        return lora
    
    def initialize_weights(self):
        state_dict = self.state_dict()
        for name in state_dict:
            if ".proj_a." in name:
                state_dict[name] = state_dict[name] * 0.3
            elif ".proj_b.proj_out." in name:
                state_dict[name] = state_dict[name] * 0
            elif ".proj_residual.proj_out." in name:
                state_dict[name] = state_dict[name] * 0.3
        self.load_state_dict(state_dict)
