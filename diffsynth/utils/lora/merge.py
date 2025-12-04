import torch
from typing import Dict, List


def merge_lora_weight(tensors_A, tensors_B):
    lora_A = torch.concat(tensors_A, dim=0)
    lora_B = torch.concat(tensors_B, dim=1)
    return lora_A, lora_B


def merge_lora(loras: List[Dict[str, torch.Tensor]], alpha=1):
    lora_merged = {}
    keys = [i for i in loras[0].keys() if ".lora_A." in i]
    for key in keys:
        tensors_A = [lora[key] for lora in loras]
        tensors_B = [lora[key.replace(".lora_A.", ".lora_B.")] for lora in loras]
        lora_A, lora_B = merge_lora_weight(tensors_A, tensors_B)
        lora_merged[key] = lora_A * alpha
        lora_merged[key.replace(".lora_A.", ".lora_B.")] = lora_B
    return lora_merged
