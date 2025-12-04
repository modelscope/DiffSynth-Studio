import torch


class GeneralLoRALoader:
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype
    
    
    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}
        for key in lora_state_dict:
            if ".lora_up." in key:
                lora_A_key = "lora_down"
                lora_B_key = "lora_up"
            else:
                lora_A_key = "lora_A"
                lora_B_key = "lora_B"
            if lora_B_key not in key:
                continue
            keys = key.split(".")
            if len(keys) > keys.index(lora_B_key) + 2:
                keys.pop(keys.index(lora_B_key) + 1)
            keys.pop(keys.index(lora_B_key))
            if keys[0] == "diffusion_model":
                keys.pop(0)
            keys.pop(-1)
            target_name = ".".join(keys)
            lora_name_dict[target_name] = (key, key.replace(lora_B_key, lora_A_key))
        return lora_name_dict
    
    
    def convert_state_dict(self, state_dict, suffix=".weight"):
        name_dict = self.get_name_dict(state_dict)
        state_dict_ = {}
        for name in name_dict:
            weight_up = state_dict[name_dict[name][0]]
            weight_down = state_dict[name_dict[name][1]]
            state_dict_[name + f".lora_B{suffix}"] = weight_up
            state_dict_[name + f".lora_A{suffix}"] = weight_down
        return state_dict_


    def fuse_lora_to_base_model(self, model: torch.nn.Module, state_dict, alpha=1.0):
        updated_num = 0
        state_dict = self.convert_state_dict(state_dict)
        lora_layer_names = set([i.replace(".lora_B.weight", "") for i in state_dict if i.endswith(".lora_B.weight")])
        for name, module in model.named_modules():
            if name in lora_layer_names:
                weight_up = state_dict[name + ".lora_B.weight"].to(device=self.device, dtype=self.torch_dtype)
                weight_down = state_dict[name + ".lora_A.weight"].to(device=self.device, dtype=self.torch_dtype)
                if len(weight_up.shape) == 4:
                    weight_up = weight_up.squeeze(3).squeeze(2)
                    weight_down = weight_down.squeeze(3).squeeze(2)
                    weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                else:
                    weight_lora = alpha * torch.mm(weight_up, weight_down)
                state_dict_base = module.state_dict()
                state_dict_base["weight"] = state_dict_base["weight"].to(device=self.device, dtype=self.torch_dtype) + weight_lora
                module.load_state_dict(state_dict_base)
                updated_num += 1
        print(f"{updated_num} tensors are fused by LoRA. Fused LoRA layers cannot be cleared by `pipe.clear_lora()`.")
