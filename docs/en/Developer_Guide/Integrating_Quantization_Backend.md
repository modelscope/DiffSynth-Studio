# Integrating a Quantization Backend

The quantization framework of `DiffSynth-Studio` lives in `diffsynth.core.quant` and ships with bitsandbytes, torchao, and comfy-kitchen backends (see [Model Quantization](../Pipeline_Usage/Quantization.md)). If you have your own quantization algorithm, or want to plug in another quantization library, you only need to implement a `QuantBackend` — online quantization, saving/loading pre-quantized checkpoints, mixed quantization, VRAM management, and quantization + LoRA training are all reused as-is.

This guide walks through the whole process with a toy backend: **INT9** — 9-bit symmetric weight-only quantization, genuinely stored at 9 bits per weight, with one fp32 scale per output channel. INT9 does not exist on any hardware; it is used here because it keeps the example short while still covering every interface you have to implement. For the full interface signatures and contracts, see the [`diffsynth.core.quant` API documentation](../API_Reference/core/quant.md#extension-interface-custom-backends).

## Framework Structure

The framework has three layers:

- **`QuantizeConfig`**: the user-facing config and entry point, responsible for traversing the model, matching layers, and replacing `nn.Linear`. You never need to touch it.
- **`QuantBackend`**: the adapter layer, which only ever deals with a **single** `nn.Linear`: how to quantize it, how to build an empty shell, how to dequantize it, how to read and write its state dict. This is the part you implement.
- **The quantized Linear**: the module that actually holds the quantized weight and performs dequantization + matmul in `forward`.

The quantized Linear must satisfy four contract clauses:

- **(a)** It is a drop-in replacement for `nn.Linear`, with `forward(x)` doing dequantization + matmul internally. It must subclass `torch.nn.Linear`, otherwise LoRA injection and VRAM management cannot see it.
- **(b)** `.to(...)` only moves devices, never re-types the packed weight or quant state. VRAM management performs dtype conversions on the model; if a packed weight were cast to bf16, the quant state would be corrupted.
- **(c)** `state_dict()` and `load_state_dict(assign=True)` round-trip, via `flatten_state_dict` / `unflatten_state_dict` when necessary.
- **(d)** (Training only) `forward` is differentiable w.r.t. its input, so gradients can pass through frozen quantized layers to reach LoRA branches.

## Step 1: Write the Quantized Linear

INT9's storage layout needs a little thought: there is no native 9-bit dtype, and simply putting the codes into an int16 tensor would still spend 16 bits per weight — exactly as much as bf16, so the quantization would save nothing. Each weight is therefore split in two: the low 8 bits go into a uint8 `weight` buffer, and the 9th (most significant) bit forms a separate bit plane where 8 weights are packed into one byte in `weight_msb`, plus one fp32 `weight_scale` per output channel. That is 9 bits per weight, 56% of bf16.

Two more details matter:

- Delete `nn.Linear`'s original `weight` parameter and register a buffer under the same name, so checkpoint keys stay `layer_name.weight`. Disk offload and mixed-quantization key ownership rely on this.
- Guard the dtype of the packed tensors by overriding `_apply`, i.e. contract clause (b). Every conversion (`.to()`, `.half()`, `.float()`, ...) funnels through `_apply`, so a conversion that would change the dtype is downgraded to a device-only move.

```python
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from diffsynth.core.quant import BackendConfig, QuantBackend, register_quant_backend, register_quant_method


def pack_msb(bits):
    """Pack a 0/1 bit plane into one bit per weight, 8 weights per byte."""
    flat = bits.reshape(-1)
    padding = (-flat.numel()) % 8
    if padding:
        flat = torch.cat([flat, flat.new_zeros(padding)])
    groups = flat.view(-1, 8)
    packed = torch.zeros(groups.shape[0], dtype=torch.uint8, device=flat.device)
    for index in range(8):
        packed |= groups[:, index] << index
    return packed


def unpack_msb(packed, numel):
    bits = torch.stack([(packed >> index) & 1 for index in range(8)], dim=1)
    return bits.reshape(-1)[:numel]


class Int9Linear(torch.nn.Linear):
    """int9 weight: low 8 bits in the uint8 `weight`, the 9th bit packed into `weight_msb`,
    plus one fp32 scale per output channel. 9 bits per weight, 56% of bf16."""

    dtype_guarded_tensor_names = ("weight", "weight_msb", "weight_scale")

    def __init__(self, in_features, out_features, bias, compute_dtype):
        with torch.device("meta"):
            super().__init__(in_features, out_features, bias=bias, dtype=compute_dtype)
        del self.weight
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.uint8, device="meta"))
        self.register_buffer("weight_msb", torch.empty((in_features * out_features + 7) // 8, dtype=torch.uint8, device="meta"))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32, device="meta"))
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def _apply(self, fn, recurse=True):
        protected = {id(tensor) for name in self.dtype_guarded_tensor_names
                     if (tensor := getattr(self, name, None)) is not None}

        def guard(tensor):
            converted = fn(tensor)
            if id(tensor) in protected and converted.dtype != tensor.dtype:
                return tensor.to(device=converted.device)
            return converted

        return super()._apply(guard, recurse)

    def dequantize_weight(self, dtype):
        msb = unpack_msb(self.weight_msb, self.weight.numel()).view_as(self.weight)
        codes = self.weight.to(torch.int16) | (msb.to(torch.int16) << 8)
        return ((codes - 256).float() * self.weight_scale.unsqueeze(1)).to(dtype)

    def forward(self, x):
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.dequantize_weight(x.dtype), bias)
```

Dequantization in `forward` uses ordinary tensor ops, so gradients flow back to the input `x` through `F.linear` and contract clause (d) holds automatically. Unpacking here is written bit by bit with PyTorch ops purely for clarity; a real backend fuses unpacking into the matmul kernel instead of materializing an fp weight on every forward.

There is an easy trap in `dequantize_weight`: the integer codes must be reconstructed in fp32. bf16 only carries 8 bits of significand, so integers above 256 are not representable exactly; casting the codes to bf16 before applying the scale rounds the 9th bit away and throws the accuracy gain out (measured: the advantage over int8 collapses from 2.25x to 1.15x). This applies to any format whose code width exceeds the significand of the compute dtype.

## Step 2: Write the Backend

Every backend method operates on a single `nn.Linear`:

- `capabilities()`: declares what the backend supports; all four flags default to `False`. Saving quantized weights requires `is_serializable=True`, and quantization + LoRA training requires `is_differentiable=True`.
- `quantized_linear_classes()`: declares the Linear classes this backend produces; `is_quantized_linear` defaults to an `isinstance` check against them.
- `create_quantized_linear()`: online quantization, turning an fp `nn.Linear` into a quantized one. `compute_device` is where quantization runs and `model_device` is where the result is stored, so the two together stream the work layer by layer with only one layer on the accelerator at a time.
- `create_quantized_linear_shell()`: builds an empty shell, used for loading pre-quantized checkpoints and for disk offload. It is rebuilt on every offload cycle, so build it on the `meta` device and keep it cheap.
- `dequantize_to_linear()`: restores a plain `nn.Linear`, used by `mode="dequant_once"`.
- `flatten_state_dict` / `unflatten_state_dict`: conversion between the state dict and flat tensors. INT9's state dict already holds plain tensors, so the base class implementation is enough; only backends with composite tensors (tensor subclasses, nested quant state) such as bitsandbytes and torchao need to override them.

Unimplemented methods raise a descriptive exception from the base class, so a backend that only supports some capabilities can implement just what it needs. `self.config` is the backend config instance injected by the framework — the `Int9WeightOnlyConfig` written in the next step.

```python
@register_quant_backend("toy_int9")
class Int9QuantBackend(QuantBackend):
    project_url = "https://example.com/toy-int9"

    def capabilities(self):
        return {**super().capabilities(), "is_serializable": True, "is_differentiable": True}

    def quantized_linear_classes(self):
        return (Int9Linear,)

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        weight = linear.weight.data
        if compute_device is not None:
            weight = weight.to(device=compute_device)
        amax = weight.abs().amax(dim=1) if self.config.per_channel else weight.abs().amax().expand(weight.shape[0])
        scale = (amax.float() / 255).clamp(min=1e-8)
        codes = (weight.float() / scale.unsqueeze(1)).round().clamp(-256, 255).to(torch.int16) + 256

        quant_linear = Int9Linear(linear.in_features, linear.out_features, bias=linear.bias is not None, compute_dtype=weight.dtype)
        quant_linear.weight = (codes & 0xFF).to(torch.uint8)
        quant_linear.weight_msb = pack_msb((codes >> 8).to(torch.uint8))
        quant_linear.weight_scale = scale
        if linear.bias is not None:
            quant_linear.bias = torch.nn.Parameter(linear.bias.data.to(device=scale.device), requires_grad=False)
        return quant_linear if model_device is None else quant_linear.to(device=model_device)

    def create_quantized_linear_shell(self, linear, compute_dtype):
        return Int9Linear(linear.in_features, linear.out_features, bias=linear.bias is not None, compute_dtype=compute_dtype)

    def dequantize_to_linear(self, module, compute_dtype, compute_device=None, model_device=None):
        if compute_device is not None:
            module = module.to(device=compute_device)
        fp_weight = module.dequantize_weight(compute_dtype)
        linear = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None, device="meta")
        linear.weight = torch.nn.Parameter(fp_weight, requires_grad=False)
        if module.bias is not None:
            linear.bias = torch.nn.Parameter(module.bias.data.to(dtype=compute_dtype, device=fp_weight.device), requires_grad=False)
        return linear if model_device is None else linear.to(device=model_device)
```

## Step 3: Write the Backend Config

The backend config subclasses `BackendConfig`: user-tunable parameters are ordinary dataclass fields, while values pinned by the method are declared with `field(init=False, default=...)`. `describe_quant_method` reports the two groups separately, and `from_kwargs` raises when a user passes unknown `backend_config_kwargs`.

```python
@dataclass
class Int9WeightOnlyConfig(BackendConfig):
    per_channel: bool = True                     # user-tunable: per-channel or per-tensor
    bits: int = field(init=False, default=9)      # pinned by the method, not overridable
```

## Step 4: Register the Quantization Method

One backend can register several methods, distinguished by the fields pinned in its config (the bitsandbytes backend, for example, distinguishes nf4 from fp4 via `quant_type`). Method names should follow the `<backend>_<format>_w<weight bits>a<activation bits>` convention:

```python
register_quant_method("toy_int9_w9a16", "toy_int9", Int9WeightOnlyConfig.from_kwargs, label="9bit, int9, weight-only (toy)")
```

There are two ways to register a backend and its methods:

**Option 1: keep it in your own code (recommended, plug-and-play).** Put the code above in any module; as long as that module is imported before you construct `QuantizeConfig`, the method is already in `QUANT_METHODS` and can be used just like a built-in one, with no framework changes:

```python
import my_project.toy_int9   # triggers register_quant_backend / register_quant_method

from diffsynth.core.quant import QuantizeConfig

quantize = QuantizeConfig(method="toy_int9_w9a16", backend_config_kwargs={"per_channel": True})
```

**Option 2: ship it as a built-in backend (permanent).** Put the backend file under `diffsynth/core/quant/backends/` and register it in `_LAZY_BACKENDS` in `diffsynth/core/quant/backends/__init__.py`; the framework then imports it on demand and users do not need to import anything:

```python
_LAZY_BACKENDS = {
    "bitsandbytes": ".bitsandbytes",
    "torchao": ".torchao",
    "comfy_kitchen": ".comfy_kitchen",
    "toy_int9": ".toy_int9",
}
```

If your quantization algorithm or library is generally useful, you are welcome to submit it as a PR following Option 2, so that more users can benefit from it. A backend that depends on a third-party library should check its dependencies in `validate_environment()` with an installation hint, and point `project_url` at the upstream project.

## Step 5: Self-Check

The framework provides two verification tools; run them right after integrating. `check_backend_contract` verifies that the backend declares its Linear classes, that both factory methods return instances of those classes, that all declared classes subclass `nn.Linear`, and that every checkpoint key the backend actually writes lives under the layer name (a missing scale would make disk offload silently load corrupted layers). Unsupported factory methods are skipped rather than counted as failures.

```python
from diffsynth.core.quant import QUANT_BACKENDS, QUANT_METHODS, check_backend_contract, check_differentiable, describe_quant_method

describe_quant_method("toy_int9_w9a16")

spec = QUANT_METHODS["toy_int9_w9a16"]
check_backend_contract(QUANT_BACKENDS[spec.backend](spec.config_factory({})), compute_device="cpu")
```

The output is as follows; `describe_quant_method` also confirms that the split between user-tunable and pinned parameters is what you intended:

```
method: toy_int9_w9a16
backend: toy_int9
detail: 9bit, int9, weight-only (toy)
backend config: my_project.toy_int9.Int9WeightOnlyConfig
backend_config_kwargs (user-tunable):
  per_channel = True
pinned by method (not overridable):
  bits = 9
check_backend_contract (toy_int9):
  [PASS] quantized_linear_classes() is non-empty: ['Int9Linear']
  [PASS] Int9Linear subclasses torch.nn.Linear
  [PASS] a plain nn.Linear is not reported as quantized
  [PASS] create_quantized_linear_shell() returns a declared class, got Int9Linear
  [PASS] the shell is recognized before load_state_dict (disk offload routing)
  [PASS] create_quantized_linear() returns a declared class, got Int9Linear
  [PASS] every stored key lives under the layer name; uncovered: []
  => OK
```

Next, check the numerical error, the real memory saving, the dtype guard of clause (b), and the differentiability of clause (d) on a small model:

```python
import torch
from diffsynth.core.quant import QuantizeConfig, check_differentiable


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(256, 512)
        self.fc2 = torch.nn.Linear(512, 256, bias=False)

    def forward(self, x):
        return self.fc2(torch.nn.functional.silu(self.fc1(x)))


def footprint(model):
    return sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))


torch.manual_seed(0)
model = ToyModel().to(torch.bfloat16)
x = torch.randn(4, 256, dtype=torch.bfloat16)
reference = model(x)
fp_bytes = footprint(model)

QuantizeConfig(method="toy_int9_w9a16").quantize_model(model, compute_device="cpu")
print("relative error:", ((model(x) - reference).norm() / reference.norm()).item())
print(f"footprint: {fp_bytes} -> {footprint(model)} bytes ({footprint(model) / fp_bytes:.3f} of bf16)")

model.to(torch.float32)                                   # clause (b): packed dtypes must not change
print(model.fc1.weight.dtype, model.fc1.weight_msb.dtype, model.fc1.weight_scale.dtype, model.fc1.bias.dtype)

check_differentiable(model.fc1)                           # clause (d)
```

```
2 nn.Linear layers quantized (method: toy_int9_w9a16).
relative error: 0.004150390625
footprint: 525312 -> 299008 bytes (0.569 of bf16)
torch.uint8 torch.uint8 torch.float32 torch.float32
check_differentiable (Int9Linear): OK -- gradients pass through the module to its input
```

The measured footprint is 0.569 of bf16, slightly above 9/16 = 0.5625 because of the fp32 scales and the unquantized bias. If this ratio comes out close to 1, the packing format is not actually compressing the weights and you should revisit the storage layout in Step 1.

### Inference on a Real Model: Z-Image

Once the small-model checks pass, the backend is ready for real models — a custom backend is used exactly like a built-in method. Import the module that registers it, then pass the method to `ModelConfig(quantize=...)`:

```python
import torch

import my_project.toy_int9   # registers the toy_int9 backend and the toy_int9_w9a16 method
from diffsynth.core.quant import QuantizeConfig
from diffsynth.pipelines.z_image import ModelConfig, ZImagePipeline

pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="Tongyi-MAI/Z-Image-Turbo",
            origin_file_pattern="transformer/*.safetensors",
            quantize=QuantizeConfig(method="toy_int9_w9a16"),
        ),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)

dit_bytes = sum(t.numel() * t.element_size() for t in list(pipe.dit.parameters()) + list(pipe.dit.buffers()))
print(f"dit weights: {dit_bytes / 1024 ** 3:.3f} GiB")

prompt = "A delicate portrait of an underwater girl, blue dress flowing, hair gently drifting, light and shadow clear, surrounded by bubbles, serene expression, exquisite details, dreamlike and beautiful."
image = pipe(prompt=prompt, seed=42, rand_device="cuda")
image.save("z_image_toy_int9.jpg")
```

Measured DiT weight footprint on Z-Image Turbo (the 8-step Turbo generation works normally, with no visible quality difference from bf16):

| | DiT weights |
| --- | --- |
| bf16 | 11.464 GiB |
| `toy_int9_w9a16` | 6.456 GiB (0.563x) |

Note that peak memory and weight footprint are not the same thing: this toy materializes a temporary fp weight on every forward, so the peak saving is smaller than the storage saving. Measured on a synthetic model with 48 Linears, all weights resident on the GPU:

| | Weights | Forward peak |
| --- | --- | --- |
| bf16 | 1.500 GiB | 1.527 GiB |
| `toy_int9_w9a16` | 0.845 GiB (0.563x) | 1.036 GiB (0.678x) |

That temporary weight depends only on the **largest single layer** and does not grow with depth, so the deeper the model, the closer the peak saving gets to the weight ratio; a real backend fusing unpacking into the matmul kernel does not need it at all. To push the peak down further, stack [VRAM management](../Pipeline_Usage/VRAM_management.md) on top and move weights layer by layer (pass a `vram_config` to each `ModelConfig` above — measured peak drops to 2.1 GiB).

### Accuracy: int9 vs int8

Does the extra bit actually buy accuracy? Quantize the same weight with the **identical** per-channel symmetric scheme at 8 and 9 bits, then compare the dequantized weight error and the layer output error. This is the general recipe for an accuracy regression on a new backend: hold everything else fixed and change only the bit width.

```python
import torch
from my_project.toy_int9 import Int9QuantBackend, Int9WeightOnlyConfig


def quantize_int8(linear):
    """The same per-channel symmetric scheme with one bit less: codes in [-128, 127]."""
    weight = linear.weight.data
    scale = (weight.abs().amax(dim=1).float() / 127).clamp(min=1e-8)
    codes = (weight.float() / scale.unsqueeze(1)).round().clamp(-128, 127)
    return (codes * scale.unsqueeze(1)).to(weight.dtype)


def relative_error(reference, value):
    return ((value.float() - reference.float()).norm() / reference.float().norm()).item()


torch.manual_seed(0)
backend = Int9QuantBackend(Int9WeightOnlyConfig())
linear = torch.nn.Linear(2048, 2048, bias=False).to(torch.bfloat16)
fp_weight = linear.weight.data.clone()

int9_weight = backend.create_quantized_linear(linear).dequantize_weight(torch.bfloat16)
int8_weight = quantize_int8(linear)
error8, error9 = relative_error(fp_weight, int8_weight), relative_error(fp_weight, int9_weight)
print(f"weight error: int8 {error8:.6f} | int9 {error9:.6f} ({error8 / error9:.2f}x lower)")

x = torch.randn(64, 2048, dtype=torch.bfloat16)
reference = torch.nn.functional.linear(x, fp_weight)
out8 = relative_error(reference, torch.nn.functional.linear(x, int8_weight))
out9 = relative_error(reference, torch.nn.functional.linear(x, int9_weight))
print(f"output error: int8 {out8:.6f} | int9 {out9:.6f} ({out8 / out9:.2f}x lower)")
```

```
weight error: int8 0.004353 | int9 0.001937 (2.25x lower)
output error: int8 0.004947 | int9 0.002816 (1.76x lower)
```

This matches the theory: going from 255 to 511 levels halves the quantization step, and for uniform quantization the error is proportional to the step, so the weight error drops to roughly half (2.25x measured). The end-to-end layer output gain is smaller (1.76x) because the activations themselves are bf16 and the matmul's own rounding noise eats part of the benefit — a reminder to evaluate bit-width gains at the actual compute precision, not only on the weights.

Finally, verify clause (c): save the quantized weights, load them back into shells, and confirm both produce identical outputs.

```python
from safetensors.torch import load_file, save_file

save_config = QuantizeConfig(method="toy_int9_w9a16")
tensors, metadata = save_config.flatten_state_dict(model.state_dict())
save_file(tensors, "toy_int9.safetensors", metadata=metadata)

loaded = ToyModel().to(torch.bfloat16)
load_config = QuantizeConfig(method="toy_int9_w9a16", load_prequantized=True)
load_config.prepare_for_prequantized_load(loaded, compute_dtype=torch.bfloat16)
loaded.load_state_dict(load_config.unflatten_state_dict(load_file("toy_int9.safetensors"), metadata), assign=True)
print("reload match:", torch.equal(loaded(x.float()), model(x.float())))
```

```
reload match: True
```

### Combining with Disk Offload

Disk offload, part of [VRAM management](../Pipeline_Usage/VRAM_management.md), places the strictest demands on a quantization backend: the resident model keeps only `meta` shells, and each layer's tensors are streamed back from disk at forward time and dropped right after. It relies on two things:

- It only supports **pre-quantized checkpoints**, so `load_prequantized=True` is required and `prepare_for_prequantized_load` must first swap the target layers for shells.
- Which tensors a layer needs is resolved by a prefix scan over the checkpoint keys using the layer's dotted name, and the result is loaded with a strict `load_state_dict(assign=True)`. So the only requirement on a backend is that every tensor lives under `{layer_name}.` — flat siblings like `layer.weight_scale` and nested quant state like bnb's both work. A missing or extra key raises instead of silently loading a corrupted layer.

```python
import torch
from safetensors.torch import save_file

from diffsynth.core.loader.model import load_metadata_from_safetensors
from diffsynth.core.quant import QuantizeConfig
from diffsynth.core.vram.disk_map import DiskMap
from diffsynth.core.vram.layers import AutoWrappedLinear, enable_vram_management_recursively

resident = ToyModel().to(torch.bfloat16)
x = torch.randn(2, 256, dtype=torch.bfloat16, device="cuda")

save_config = QuantizeConfig(method="toy_int9_w9a16")
save_config.quantize_model(resident, compute_device="cuda")
resident = resident.to("cuda")
reference = resident(x)

tensors, metadata = save_config.flatten_state_dict(resident.state_dict())
save_file({key: value.cpu() for key, value in tensors.items()}, "toy_int9.safetensors", metadata=metadata)

fresh = ToyModel().to(torch.bfloat16)
load_config = QuantizeConfig(method="toy_int9_w9a16", load_prequantized=True)
load_config.prepare_for_prequantized_load(fresh, compute_dtype=torch.bfloat16)
enable_vram_management_recursively(
    fresh,
    module_map={torch.nn.Linear: AutoWrappedLinear},
    vram_config={
        "offload_dtype": "disk", "offload_device": "disk",
        "onload_dtype": "disk", "onload_device": "disk",
        "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16, "computation_device": "cuda",
    },
    disk_map=DiskMap(["toy_int9.safetensors"], "cuda", torch_dtype=None),
    quantize=load_config,
    metadata=load_metadata_from_safetensors("toy_int9.safetensors"),
)

for name, module in fresh.named_modules():
    if getattr(module, "disk_offload", False):
        print(f"{name}: {module._disk_required_keys()}")

resident_bytes = sum(t.numel() * t.element_size() for t in list(resident.parameters()) + list(resident.buffers()))
offloaded_bytes = sum(t.numel() * t.element_size() for t in list(fresh.parameters()) + list(fresh.buffers()) if not t.is_meta)
print(f"resident {resident_bytes} bytes -> in memory after disk offload {offloaded_bytes} bytes")
print("output matches:", torch.equal(fresh(x), reference), "| repeatable:", torch.equal(fresh(x), reference))
```

Measured on the same `ToyModel` (`torch_dtype=None` on `DiskMap` is essential — it guarantees the packed tensors are not re-typed while being read):

```
2 nn.Linear layers replaced for loading the pre-quantized checkpoint (method: toy_int9_w9a16).
fc1: ['fc1.bias', 'fc1.weight', 'fc1.weight_msb', 'fc1.weight_scale']
fc2: ['fc2.weight', 'fc2.weight_msb', 'fc2.weight_scale']
resident 299008 bytes -> in memory after disk offload 0 bytes
output matches: True | repeatable: True
```

Each layer's `weight` / `weight_msb` / `weight_scale` / `bias` is correctly attributed to that layer, the resident footprint drops to 0 bytes (everything is a `meta` shell), the output is bit-identical to the resident quantized model, and repeated forwards stay stable — so rebuilding shells and streaming from disk has no side effects.

On a real model, use the standard workflows from [Model Quantization](../Pipeline_Usage/Quantization.md) for end-to-end validation: pass `QuantizeConfig(method="toy_int9_w9a16")` to `ModelConfig(quantize=...)` for online-quantized inference, save the quantized weights with `save_quantized_model` and load them back after registering the hash, and inject LoRA into the quantized model for training.

## Integration Checklist

- The packing format really shrinks the weights: the measured footprint ratio should be close to the theoretical bit-width ratio, not close to 1.
- The accuracy gain is verified: compared against the same scheme with one bit less, the error really goes down; otherwise precision is being lost somewhere in the dequantization path.
- The quantized Linear subclasses `torch.nn.Linear`, and all `state_dict` keys live under the layer name.
- `_apply` guards the dtype of every packed tensor and quant state.
- `capabilities()` matches reality: declaring `is_serializable` requires a round-tripping state dict, and declaring `is_differentiable` requires passing `check_differentiable`.
- `create_quantized_linear` honors `compute_device` / `model_device`, so layer-by-layer streaming quantization works.
- Disk offload works: the shell is built on `meta` and cheap to rebuild, every stored tensor lives under the layer's dotted name, and `unflatten_state_dict` tolerates being called with a single layer's subdict plus whole-file metadata.
- When depending on a third-party library, `validate_environment()` gives a clear installation hint and `project_url` points at the upstream project.
- `check_backend_contract` passes completely.
