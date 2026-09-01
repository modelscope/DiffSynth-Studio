import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


class LTX25GemmaTokenizer:
    def __init__(self, model_path: str | Path, max_length: int = 1024):
        model_path = Path(model_path)
        with safe_open(model_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if "tokenizer_json" not in handle.keys():
                raise ValueError(f"{model_path} does not contain packed tokenizer_json assets.")
            tokenizer_bytes = handle.get_tensor("tokenizer_json").detach().cpu().numpy().astype(np.uint8).tobytes()
            raw_config = metadata.get("tokenizer_config.json")
            if raw_config is None and "hf_asset__tokenizer_config.json" in handle.keys():
                raw_config = handle.get_tensor("hf_asset__tokenizer_config.json").detach().cpu().numpy().astype(np.uint8).tobytes().decode()
        config = json.loads(raw_config) if raw_config else {}
        ignored = {"tokenizer_class", "auto_map", "model_max_length", "backend", "is_local", "local_files_only", "processor_class", "added_tokens_decoder"}
        config = {key: value for key, value in config.items() if key not in ignored}
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_buffer(tokenizer_bytes),
            model_max_length=max_length,
            **config,
        )
        self.tokenizer.model_max_length = max_length
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def tokenize_with_weights(self, text: str) -> dict[str, list[tuple[int, int]]]:
        text = text.strip()
        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            raise ValueError("Packed Gemma tokenizer has no BOS token id.")
        encoded = self.tokenizer(text, padding=False, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = encoded.input_ids[0].tolist()
        if not input_ids or input_ids[0] != bos_id:
            input_ids = [bos_id, *input_ids][: self.max_length]
        padded = self.tokenizer.pad(
            {"input_ids": [input_ids]},
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {"gemma": list(zip(padded.input_ids[0].tolist(), padded.attention_mask[0].tolist(), strict=True))}
