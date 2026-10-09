from .general import GeneralLoRALoader
import torch


class MiniMaxH3LoRALoader(GeneralLoRALoader):
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        super().__init__(device=device, torch_dtype=torch_dtype)

    @staticmethod
    def is_lightx2v_format(state_dict):
        for key in state_dict:
            if key.startswith("transformer_blocks.") and (".attn.to_q.lora_A." in key or ".attn.orig.to_q.lora_A." in key):
                return True
        return False

    def convert_state_dict(self, state_dict, suffix=".weight"):
        if self.is_lightx2v_format(state_dict):
            state_dict = MiniMaxH3LoRAConverter.align_to_diffsynth_format(state_dict)
        return super().convert_state_dict(state_dict, suffix=suffix)


class MiniMaxH3LoRAConverter:

    @staticmethod
    def _source_prefix(target_prefix):
        if target_prefix.startswith("transformer_blocks."):
            return target_prefix.replace("transformer_blocks.", "blocks.", 1)
        if target_prefix.startswith("token_refiner.refiner_blocks."):
            return target_prefix.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.", 1)
        return target_prefix

    @staticmethod
    def _detect_adapter_name(state_dict):
        for key in state_dict:
            if ".lora_A." in key:
                return key.split(".lora_A.")[1].split(".")[0]
        return "default"

    @classmethod
    def _pair(cls, state_dict, prefix, adapter_name):
        return (
            state_dict[f"{prefix}.lora_A.{adapter_name}.weight"],
            state_dict[f"{prefix}.lora_B.{adapter_name}.weight"],
        )

    @classmethod
    def align_to_diffsynth_format(cls, state_dict, adapter_name=None):
        if adapter_name is None:
            adapter_name = cls._detect_adapter_name(state_dict)
        # VDN adapters spell the wrapped backbone projections as `.attn.orig.*`; the
        # token refiner is never wrapped so its keys carry no `.orig.`.
        vdn = any(".attn.orig." in key for key in state_dict)

        def base_prefix(prefix):
            return prefix.replace(".attn.orig.", ".attn.")

        a_suffix = f".lora_A.{adapter_name}.weight"
        orig_prefixes = {key.removesuffix(a_suffix) for key in state_dict if key.endswith(a_suffix)}
        base_to_orig = {base_prefix(p): p for p in orig_prefixes}

        def vdn_target(target):
            if vdn and target.startswith("blocks.") and ".attn." in target:
                return target.replace(".attn.", ".attn.orig.", 1)
            return target

        attention_prefixes = sorted(
            bp.removesuffix(".to_q")
            for bp in base_to_orig
            if bp.endswith(".attn.to_q")
        )
        consumed = set()
        converted = {}
        for prefix in attention_prefixes:
            src_root = base_to_orig[prefix + ".to_q"].removesuffix(".to_q")
            pairs = [cls._pair(state_dict, f"{src_root}.to_{name}", adapter_name) for name in "qkv"]
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

            target = vdn_target(cls._source_prefix(prefix) + ".qkv_proj")
            converted[target + ".lora_A.default.weight"] = a_fused
            converted[target + ".lora_B.default.weight"] = b_fused
            consumed.update(f"{src_root}.to_{name}" for name in "qkv")

        for src in sorted(orig_prefixes - consumed):
            a, b = cls._pair(state_dict, src, adapter_name)
            target = cls._source_prefix(base_prefix(src))
            if target.endswith(".attn.to_out.0"):
                target = vdn_target(target.removesuffix(".to_out.0") + ".out_proj")
            elif target.endswith(".ff.net.0.proj"):
                target = target.removesuffix(".ff.net.0.proj") + ".mlp.fc1"
                up, gate = b.chunk(2, dim=0)
                b = torch.cat([gate, up], dim=0)
            elif target.endswith(".ff.net.2"):
                target = target.removesuffix(".ff.net.2") + ".mlp.fc2"
            elif target == "norm_out.linear":
                target = "final_layer.adaln_proj.linear"
            elif target.endswith(".adaln_proj.linear"):
                pass
            else:
                raise ValueError(f"Unsupported LoRA target: {prefix}")
            converted[target + ".lora_A.default.weight"] = a
            converted[target + ".lora_B.default.weight"] = b
        return converted
