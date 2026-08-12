import torch
from .general import GeneralLoRALoader


class MiniMaxH3TurboLoRALoader(GeneralLoRALoader):

    @staticmethod
    def _source_prefix(target_prefix):
        if target_prefix.startswith("transformer_blocks."):
            return target_prefix.replace("transformer_blocks.", "blocks.", 1)
        if target_prefix.startswith("token_refiner.refiner_blocks."):
            return target_prefix.replace(
                "token_refiner.refiner_blocks.", "token_refiner.blocks.", 1
            )
        return target_prefix

    @staticmethod
    def _pair(state_dict, prefix):
        names = (
            f"{prefix}.lora_A.default.weight",
            f"{prefix}.lora_B.default.weight",
        )
        return state_dict[names[0]], state_dict[names[1]]

    def convert_state_dict(self, state_dict, suffix=".weight"):
        if state_dict and all(
            key.endswith((".lora_A.weight", ".lora_B.weight"))
            for key in state_dict
        ):
            return state_dict
        converted = {}
        prefixes = {
            key.removesuffix(".lora_A.default.weight")
            for key in state_dict
            if key.endswith(".lora_A.default.weight")
        }

        attention_prefixes = sorted(
            prefix.removesuffix(".to_q")
            for prefix in prefixes
            if prefix.endswith(".attn.to_q")
        )
        consumed = set()
        for prefix in attention_prefixes:
            pairs = [self._pair(state_dict, f"{prefix}.to_{name}") for name in "qkv"]
            ranks = {a.shape[0] for a, _ in pairs}
            rank = ranks.pop()
            out_features_set = {b.shape[0] for _, b in pairs}
            out_features = out_features_set.pop()

            a_fused = torch.cat([a for a, _ in pairs], dim=0)
            b_fused = torch.zeros(
                out_features * 3,
                rank * 3,
                dtype=pairs[0][1].dtype,
                device=pairs[0][1].device,
            )
            heads = out_features // 128
            for modality, (_, b) in enumerate(pairs):
                rows = (
                    torch.arange(out_features, device=b.device).reshape(heads, 128)
                    + modality * 128
                    + torch.arange(heads, device=b.device)[:, None] * 256
                ).reshape(-1)
                b_fused[rows, modality * rank : (modality + 1) * rank] = b

            target = self._source_prefix(prefix) + ".qkv_proj"
            converted[target + f".lora_A{suffix}"] = a_fused
            converted[target + f".lora_B{suffix}"] = b_fused
            consumed.update(f"{prefix}.to_{name}" for name in "qkv")

        for prefix in sorted(prefixes - consumed):
            a, b = self._pair(state_dict, prefix)
            target = self._source_prefix(prefix)
            if target.endswith(".attn.to_out.0"):
                target = target.removesuffix(".to_out.0") + ".out_proj"
            elif target.endswith(".ff.net.0.proj"):
                target = target.removesuffix(".ff.net.0.proj") + ".mlp.fc1"
                up, gate = b.chunk(2, dim=0)
                b = torch.cat([gate, up], dim=0)
            elif target.endswith(".ff.net.2"):
                target = target.removesuffix(".ff.net.2") + ".mlp.fc2"
            else:
                raise ValueError(f"Unsupported LoRA target: {prefix}")
            converted[target + f".lora_A{suffix}"] = a
            converted[target + f".lora_B{suffix}"] = b
        return converted
