import torch

def decomposite(tensor_A, tensor_B, rank):
    dtype, device = tensor_A.dtype, tensor_A.device
    weight = tensor_B @ tensor_A
    U, S, V = torch.pca_lowrank(weight.float(), q=rank)
    tensor_A = (V.T).to(dtype=dtype, device=device).contiguous()
    tensor_B = (U @ torch.diag(S)).to(dtype=dtype, device=device).contiguous()
    return tensor_A, tensor_B

def reset_lora_rank(lora, rank):
    lora_merged = {}
    keys = [i for i in lora.keys() if ".lora_A." in i]
    for key in keys:
        tensor_A = lora[key]
        tensor_B = lora[key.replace(".lora_A.", ".lora_B.")]
        tensor_A, tensor_B = decomposite(tensor_A, tensor_B, rank)
        lora_merged[key] = tensor_A
        lora_merged[key.replace(".lora_A.", ".lora_B.")] = tensor_B
    return lora_merged