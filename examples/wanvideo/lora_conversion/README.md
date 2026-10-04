# Wan DiT LoRA conversion

Convert a single Wan DiT adapter between DiffSynth and Diffusers/PEFT naming:

```bash
python examples/wanvideo/lora_conversion/convert.py input.safetensors exported.safetensors --to diffusers
python examples/wanvideo/lora_conversion/convert.py exported.safetensors restored.safetensors --to diffsynth
```

The same stateless API can be used without the CLI:

```python
from diffsynth.utils.lora.wan import WanLoRAConverter

exported = WanLoRAConverter.align_to_opensource_format(state_dict, scale=1.0)
restored = WanLoRAConverter.align_to_diffsynth_format(exported)
```

The converter supports self-attention, cross-attention and FFN linear layers in
Wan DiT blocks. Both `.lora_A.weight` and legacy `.lora_A.default.weight` names
are accepted, with the corresponding B factors. Outputs use canonical names;
Diffusers outputs carry the `transformer.` prefix.

DiffSynth's default training configuration sets `lora_alpha = rank`, giving unit
scaling. Per-layer tensor alpha/rank is folded into B exactly once. Use `--scale`
for any additional input runtime scaling. Safetensors metadata scaling is not
inferred: inspect it and pass the appropriate scale explicitly if needed.

Missing factors, rank mismatches, non-finite scaling, non-floating-point factors,
unsupported modules and duplicate factors
raise errors instead of silently dropping weights. Existing output files are
not overwritten, including files created by another writer during conversion.
Text-encoder adapters, VACE-specific layers, fused-QKV adapters
and other architectures are outside this converter's scope.
