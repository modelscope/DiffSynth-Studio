import torch
from .qwen_image_image2lora import ImageEmbeddingToLoraMatrix, SequencialMLP


class LoRATrainerBlock(torch.nn.Module):
    def __init__(self, lora_patterns, in_dim=1536+4096, compress_dim=128, rank=4, block_id=0, use_residual=True, residual_length=64+7, residual_dim=3584, residual_mid_dim=1024, prefix="transformer_blocks"):
        super().__init__()
        self.prefix = prefix
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
            lora[f"{self.prefix}.{self.block_id}.{name}.lora_A.default.weight"] = lora_a
            lora[f"{self.prefix}.{self.block_id}.{name}.lora_B.default.weight"] = lora_b
        return lora


class ZImageImage2LoRAComponent(torch.nn.Module):
    def __init__(self, lora_patterns, prefix, num_blocks=60, use_residual=True, compress_dim=128, rank=4, residual_length=64+7, residual_mid_dim=1024):
        super().__init__()
        self.lora_patterns = lora_patterns
        self.num_blocks = num_blocks
        self.blocks = []
        for lora_patterns in self.lora_patterns:
            for block_id in range(self.num_blocks):
                self.blocks.append(LoRATrainerBlock(lora_patterns, block_id=block_id, use_residual=use_residual, compress_dim=compress_dim, rank=rank, residual_length=residual_length, residual_mid_dim=residual_mid_dim, prefix=prefix))
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


class ZImageImage2LoRAModel(torch.nn.Module):
    def __init__(self, use_residual=False, compress_dim=64, rank=4, residual_length=64+7, residual_mid_dim=1024):
        super().__init__()
        lora_patterns = [
            [
                ("attention.to_q", 3840, 3840),
                ("attention.to_k", 3840, 3840),
                ("attention.to_v", 3840, 3840),
                ("attention.to_out.0", 3840, 3840),
            ],
            [
                ("feed_forward.w1", 3840, 10240),
                ("feed_forward.w2", 10240, 3840),
                ("feed_forward.w3", 3840, 10240),
            ],
        ]
        config = {
            "lora_patterns": lora_patterns,
            "use_residual": use_residual,
            "compress_dim": compress_dim,
            "rank": rank,
            "residual_length": residual_length,
            "residual_mid_dim": residual_mid_dim,
        }
        self.layers_lora = ZImageImage2LoRAComponent(
            prefix="layers",
            num_blocks=30,
            **config,
        )
        self.context_refiner_lora = ZImageImage2LoRAComponent(
            prefix="context_refiner",
            num_blocks=2,
            **config,
        )
        self.noise_refiner_lora = ZImageImage2LoRAComponent(
            prefix="noise_refiner",
            num_blocks=2,
            **config,
        )
        
    def forward(self, x, residual=None):
        lora = {}
        lora.update(self.layers_lora(x, residual=residual))
        lora.update(self.context_refiner_lora(x, residual=residual))
        lora.update(self.noise_refiner_lora(x, residual=residual))
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


class ImageEmb2LoRAWeightCompressed(torch.nn.Module):
    def __init__(self, in_dim, out_dim, emb_dim, rank):
        super().__init__()
        self.lora_a = torch.nn.Parameter(torch.randn((rank, in_dim)))
        self.lora_b = torch.nn.Parameter(torch.randn((out_dim, rank)))
        self.proj = torch.nn.Linear(emb_dim, rank * rank, bias=True)
        self.rank = rank
    
    def forward(self, x):
        x = self.proj(x).view(self.rank, self.rank)
        lora_a = x @ self.lora_a
        lora_b = self.lora_b
        return lora_a, lora_b


class ZImageImage2LoRAModelCompressed(torch.nn.Module):
    def __init__(self, emb_dim=1536+4096, rank=32):
        super().__init__()
        target_layers = [
            ("attention.to_q", 3840, 3840),
            ("attention.to_k", 3840, 3840),
            ("attention.to_v", 3840, 3840),
            ("attention.to_out.0", 3840, 3840),
            ("feed_forward.w1", 3840, 10240),
            ("feed_forward.w2", 10240, 3840),
            ("feed_forward.w3", 3840, 10240),
        ]
        self.lora_patterns = [
            {
                "prefix": "layers",
                "num_layers": 30,
                "target_layers": target_layers,
            },
            {
                "prefix": "context_refiner",
                "num_layers": 2,
                "target_layers": target_layers,
            },
            {
                "prefix": "noise_refiner",
                "num_layers": 2,
                "target_layers": target_layers,
            },
        ]
        module_dict = {}
        for lora_pattern in self.lora_patterns:
            prefix, num_layers, target_layers = lora_pattern["prefix"], lora_pattern["num_layers"], lora_pattern["target_layers"]
            for layer_id in range(num_layers):
                for layer_name, in_dim, out_dim in target_layers:
                    name = f"{prefix}.{layer_id}.{layer_name}".replace(".", "___")
                    model = ImageEmb2LoRAWeightCompressed(in_dim, out_dim, emb_dim, rank)
                    module_dict[name] = model
        self.module_dict = torch.nn.ModuleDict(module_dict)

    def forward(self, x, residual=None):
        lora = {}
        for name, module in self.module_dict.items():
            name = name.replace("___", ".")
            name_a, name_b = f"{name}.lora_A.default.weight", f"{name}.lora_B.default.weight"
            lora_a, lora_b = module(x)
            lora[name_a] = lora_a
            lora[name_b] = lora_b
        return lora

    def initialize_weights(self):
        state_dict = self.state_dict()
        for name in state_dict:
            if "lora_b" in name:
                state_dict[name] = state_dict[name] * 0
            elif "lora_a" in name:
                state_dict[name] = state_dict[name] * 0.2
            elif "proj.weight" in name:
                print(name)
                state_dict[name] = state_dict[name] * 0.2
        self.load_state_dict(state_dict)
