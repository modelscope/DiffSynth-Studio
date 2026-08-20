# `diffsynth.core.quant`: Model Quantization

This document introduces the low-level quantization interfaces in `diffsynth.core.quant`. Refer to it if you want to use these features in another codebase. If you only want to enable quantization in a `Pipeline`, see [Model Quantization](../../Pipeline_Usage/Quantization.md).

The module exports the following interfaces through `diffsynth.core.quant`, organized in three categories:

| Category | Interfaces |
| --- | --- |
| User interfaces | `QuantizeConfig`, `MixedQuantizeConfig`, `describe_quant_method`, `QUANT_METHODS` |
| Extension interfaces | `QuantBackend`, `BackendConfig`, `register_quant_backend`, `register_quant_method`, `QuantMethodSpec`, `QUANT_BACKENDS` |
| Verification tools | `check_differentiable`, `check_backend_contract` |

Quantization operates on the `nn.Linear` layers in a model: the framework traverses the model and replaces the matched `nn.Linear` layers with the backend's quantized Linears (all subclasses of `nn.Linear`, so LoRA injection, VRAM management, and other mechanisms recognize them without modification). A backend is only responsible for quantizing a single layer; model-level traversal and replacement is done by `QuantizeConfig`.

## User Interfaces

### QuantizeConfig

`QuantizeConfig` is both the quantization config and the operation entry point for any `nn.Module`.

Fields:

| Field | Type | Description |
| --- | --- | --- |
| `method` | `str` | Quantization method name, from `QUANT_METHODS`; determines the backend, scheme, and backend config. Required |
| `mode` | `str` | `"dynamic"` (default) keeps the backend-native quantized Linears, dequantizing at every forward; `"dequant_once"` restores plain fp `nn.Linear` right after the weights are quantized or loaded |
| `target_modules` | `list` | Only quantize the matched layers; `None` means no restriction |
| `exclude_modules` | `list` | Exclude the matched layers |
| `backend_config_kwargs` | `dict` | Parameters passed to the method's backend config factory, determining the quantization behavior, e.g. nf4's `blocksize` |
| `load_prequantized` | `bool` | The checkpoint already holds quantized weights; load them directly instead of quantizing online |

Matching rule for `target_modules` / `exclude_modules`: a layer matches if its full dotted name equals an entry, or ends with `"." + entry`. For example, `"img_mod.1"` matches `transformer_blocks.0.img_mod.1`.

Constructing a `QuantizeConfig` validates the backend dependencies and parameters, and raises immediately (with installation instructions) when they are not satisfied, rather than failing later at inference time.

Main methods:

#### `quantize_model(model, compute_device=None, model_device=None)`

Quantizes the matched `nn.Linear` layers in `model` in place, keeping each layer's existing dtype. Must be called **after** `load_state_dict`. Does nothing when `load_prequantized=True` (such a checkpoint is already quantized).

- `compute_device`: the device where quantization computation happens; `None` means quantize in place.
- `model_device`: the device where each layer is stored after quantization; `None` means leaving it on `compute_device`.

With an fp model on the CPU and `compute_device="cuda", model_device="cpu"`, quantization streams layer by layer, so the accelerator only ever holds one layer at a time:

```python
import torch
from diffsynth.core.quant import QuantizeConfig

cfg = QuantizeConfig(method="bitsandbytes_nf4")
model.load_state_dict(fp_state_dict)
cfg.quantize_model(model, compute_device="cuda", model_device="cpu")
```

#### `prepare_for_prequantized_load(model, compute_dtype=torch.bfloat16)`

Replaces the matched `nn.Linear` layers with empty quantized layers ("shells") matching the structure of a pre-quantized checkpoint. Must be called **before** `load_state_dict(assign=True)`. `compute_dtype` is the dtype the quantized layers dequantize to at forward time.

#### `unflatten_state_dict(state_dict, metadata)` / `flatten_state_dict(state_dict)`

Quantized weights are often composite structures of "packed tensors + quant state", while `.safetensors` can only store plain tensors. These two methods convert between the two forms.

- `unflatten_state_dict(state_dict, metadata)`: rebuilds composite quantized tensors from the flat tensors read out of a checkpoint; the result can be given to `load_state_dict(assign=True)`.
- `flatten_state_dict(state_dict)`: flattens a quantized model's state dict into plain tensors and string-only metadata, returning `(tensors, metadata)`, which can be passed directly to `safetensors.torch.save_file(tensors, path, metadata=metadata)`. Raises `NotImplementedError` if the backend does not declare `is_serializable`.

The complete flow for loading a pre-quantized checkpoint:

```python
import torch
from diffsynth.core.quant import QuantizeConfig

cfg = QuantizeConfig(method="bitsandbytes_nf4", load_prequantized=True)
cfg.prepare_for_prequantized_load(model, compute_dtype=torch.bfloat16)
state_dict = cfg.unflatten_state_dict(state_dict, metadata)
model.load_state_dict(state_dict, assign=True)
```

#### `dequantize_model(model, compute_dtype=torch.bfloat16, compute_device=None, model_device=None)`

Replaces all quantized Linears in the model with plain fp `nn.Linear`; the restored weights carry the quantization error. **Only takes effect when `mode="dequant_once"`**; otherwise returns directly. Can be called after either of the two flows above:

```python
cfg.dequantize_model(model, compute_dtype=torch.bfloat16)
```

#### `is_quantized_linear(module)`

Whether `module` is one of the quantized Linears produced by this config's backend.

#### `build_quantized_shell(module, compute_dtype)`

Builds an empty quantized Linear matching `module`'s shape and bias presence. Used to release a layer's weights while keeping it routable, and to stage a transient copy on the computation device — a companion interface for VRAM management.

### MixedQuantizeConfig

Combines multiple `QuantizeConfig`s into one mixed quantization; each sub-config is responsible for a mutually disjoint set of layers. It exposes the same interface as a single `QuantizeConfig` (`quantize_model`, `prepare_for_prequantized_load`, `dequantize_model`, `flatten_state_dict`, `unflatten_state_dict`, `is_quantized_linear`, `build_quantized_shell`, plus the two read-only properties `method` / `mode`).

```python
from diffsynth.core.quant import QuantizeConfig, MixedQuantizeConfig

mod_layers = ["img_mod.1", "txt_mod.1", "norm_out.linear", "img_in", "txt_in", "proj_out"]
cfg = MixedQuantizeConfig(configs=[
    QuantizeConfig(method="bitsandbytes_nf4", exclude_modules=mod_layers),
    QuantizeConfig(method="torchao_int8_w8a16", target_modules=mod_layers),
])
cfg.quantize_model(model, compute_device="cuda")
```

Fields and constraints:

- `configs`: a list of `QuantizeConfig`, executed in order. All sub-configs must share the same `mode`, and their `load_prequantized` must be `False`.
- `load_prequantized`: set on this wrapper when loading a mixed quantized checkpoint, not on the sub-configs.
- The layer sets matched by the sub-configs must be pairwise disjoint. `quantize_model` and `prepare_for_prequantized_load` verify this before touching the model, and raise on conflict, naming the overlapping layers.

`build_quantized_shell(module, compute_dtype, layer_name=None)` gains an extra `layer_name` parameter here: when multiple sub-configs share the same backend, the quantized Linears they produce are the same class, and ownership can only be determined by layer name.

### describe_quant_method and QUANT_METHODS

`QUANT_METHODS` is a registry of `{method name: QuantMethodSpec}`. `QuantMethodSpec` has three fields: `backend` (backend name), `config_factory` (a callable turning `backend_config_kwargs` into the backend config), and `label` (a human-readable description).

Call `backends.load_all_backends()` before enumerating all methods:

```python
from diffsynth.core.quant import QUANT_METHODS, backends

backends.load_all_backends()
print(sorted(QUANT_METHODS))
```

`describe_quant_method(name)` prints a method's backend, description, and the accepted `backend_config_kwargs` with defaults (it loads the backend internally):

```python
from diffsynth.core.quant import describe_quant_method

describe_quant_method("comfy_kitchen_int8_w8a8")
```

```
method: comfy_kitchen_int8_w8a8
backend: comfy_kitchen
detail: W8A8, int8 weight + int8 dynamic activation (ComfyUI int8_tensorwise)
backend config: diffsynth.core.quant.backends.comfy_kitchen.ComfyKitchenInt8Config
backend_config_kwargs (user-tunable):
  per_channel       = True
  convrot           = True
  convrot_groupsize = 256
  orig_dtype        = torch.bfloat16
pinned by method (not overridable):
  format = 'int8_tensorwise'
```

`user-tunable` are the parameters that can be modified via `backend_config_kwargs`; `pinned by method` are fixed for the method and cannot be modified (e.g. `comfy_kitchen_int8_w8a8` and `comfy_kitchen_fp8_w8a8` share one backend and are distinguished by `format`). Passing an unaccepted key raises an error listing the available keys.

## Extension Interface: Custom Backends

### The QuantBackend Contract

`QuantBackend` is the adapter layer between the framework and a concrete quantization library (bitsandbytes / torchao / custom). Subclasses are registered into `QUANT_BACKENDS` via `register_quant_backend`, instantiated by `QuantizeConfig`, and injected with the method's backend config.

The quantized Linear produced by a backend must satisfy the following four contract clauses:

- **(a)** It is a drop-in replacement for `nn.Linear`: `forward(x)` performs dequantization + matmul internally.
- **(b)** `.to(...)` only moves devices, never re-types the packed weight / quant state: dtype casts (`.to(dtype)`, `.half()`, `.float()`, etc.) must leave their storage format and values intact.
- **(c)** `state_dict()` and `load_state_dict(assign=True)` round-trip (via `flatten_state_dict` / `unflatten_state_dict` when necessary).
- **(d)** (Training only) `forward` is differentiable with respect to its input, so gradients can pass through frozen quantized layers to reach LoRA branches. Declared statically by `capabilities()["is_differentiable"]` and verifiable at runtime with `check_differentiable`.

Clause (b) is necessary because VRAM management performs dtype/device conversions on the model; if a packed weight were accidentally cast to bf16, the quant state would be corrupted. See `Fp8Linear._apply` in `diffsynth/models/ideogram4_dit.py` for a reference: register the tensor names that need protection, and downgrade conversions that would change their dtype to device-only moves inside `_apply`.

Members to implement or override:

| Member | Description |
| --- | --- |
| `name` | Set automatically by `register_quant_backend` |
| `project_url` | The project page of the library this backend belongs to; `announce_environment()` prints it, pointing hardware compatibility issues upstream |
| `capabilities()` | Returns four boolean flags `is_serializable` / `is_differentiable` / `is_compileable` / `requires_calibration`, all defaulting to `False` |
| `validate_environment()` | Checks dependencies and hardware, raising an exception with installation instructions when missing. Called when constructing `QuantizeConfig` |
| `quantized_linear_classes()` | Declares the Linear classes this backend produces; they must all be subclasses of `torch.nn.Linear`. `is_quantized_linear` defaults to an `isinstance` check against them |
| `create_quantized_linear(linear, compute_device, model_device)` | Online quantization: turns an fp `nn.Linear` into a quantized Linear. If unimplemented, the backend does not support online quantization |
| `create_quantized_linear_shell(linear, compute_dtype)` | Builds an empty shell for loading pre-quantized checkpoints. If unimplemented, the backend does not support pre-quantized loading |
| `dequantize_to_linear(module, compute_dtype, compute_device, model_device)` | Restores a plain `nn.Linear`. If unimplemented, `mode="dequant_once"` is unavailable |
| `flatten_state_dict` / `unflatten_state_dict` | Conversion between quantized state dicts and flat tensors; must be implemented when `is_serializable=True` |

The base class provides clear error messages for unimplemented methods, so a backend supporting only some capabilities can implement just the ones it needs.

### BackendConfig

`BackendConfig` is the base class for a backend's typed config. User-tunable parameters are written as ordinary dataclass fields; values pinned by the method are declared with `field(init=False, default=...)`, so they are both shown separately by `describe_quant_method` and impossible to modify via `backend_config_kwargs`.

The classmethod `from_kwargs(kwargs)` validates the keys passed in: unknown keys raise a `ValueError` listing all accepted keys. It is typically used directly as the `config_factory` of `register_quant_method`.

The bitsandbytes backend is a canonical example of this pattern — the shared 4bit parameters live in the base class, while `quant_type` is pinned by each method's subclass:

```python
from dataclasses import dataclass, field
import torch
from diffsynth.core.quant import BackendConfig, register_quant_method


@dataclass
class BitsAndBytes4bitConfig(BackendConfig):
    compress_statistics: bool = True
    blocksize: int = None
    quant_storage: torch.dtype = torch.uint8


@dataclass
class BitsAndBytesNF4Config(BitsAndBytes4bitConfig):
    quant_type: str = field(init=False, default="nf4")


register_quant_method("bitsandbytes_nf4", "bitsandbytes", BitsAndBytesNF4Config.from_kwargs, label="4bit, nf4, weight-only")
```

`config_factory` is not required to return a `BackendConfig`: if the backend directly consumes a third-party library's config object, you can pass any function that turns a `dict` into that object (the torchao backend does this, building `Int8WeightOnlyConfig` and the like directly).

### register_quant_backend and register_quant_method

- `register_quant_backend(name)`: a class decorator that registers a backend class into `QUANT_BACKENDS` and sets its `name`.
- `register_quant_method(name, backend, config_factory, label="")`: registers a method name into `QUANT_METHODS`, specifying which backend it uses and how its backend config is built. One backend can register multiple methods, distinguished by pinned fields.

A complete skeleton of a minimal backend:

```python
import torch
from diffsynth.core.quant import QuantBackend, register_quant_backend, register_quant_method


class MyQuantLinear(torch.nn.Linear):
    """Custom quantized Linear; must satisfy contract clauses (a)-(d)."""


@register_quant_backend("my_backend")
class MyQuantBackend(QuantBackend):
    project_url = "https://example.com/my-quant-lib"

    def capabilities(self):
        return {**super().capabilities(), "is_serializable": True, "is_differentiable": True}

    def validate_environment(self):
        ...   # raise ImportError when dependencies are missing

    def quantized_linear_classes(self):
        return (MyQuantLinear,)

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        ...

    def create_quantized_linear_shell(self, linear, compute_dtype):
        ...

    def dequantize_to_linear(self, module, compute_dtype, compute_device=None, model_device=None):
        ...


register_quant_method("my_method", "my_backend", lambda kwargs: dict(kwargs), label="my custom method")
```

Once registered, it can be used just like a built-in method: `QuantizeConfig(method="my_method")`. If the backend is defined outside `diffsynth/core/quant/backends/` (e.g. alongside a model), it only needs to be imported before constructing `QuantizeConfig`.

## Verification Tools

### check_differentiable

```python
check_differentiable(module, example_input=None, verbose=True) -> bool
```

Checks whether gradients can pass through `module` to its input: runs a real backward pass from the output (`torch.autograd.grad`) and confirms a finite gradient arrives at the input. This is exactly what LoRA training requires from frozen (quantized) layers. The module is cast to bfloat16 in place and probed with a bfloat16 input; when `example_input` is `None`, a random input is constructed automatically for modules exposing `in_features`.

```python
import torch
from diffsynth.core.quant import check_differentiable
from torchao.quantization import quantize_, Int8WeightOnlyConfig

linear = torch.nn.Linear(1024, 1024, dtype=torch.bfloat16, device="cuda")
quantize_(linear, Int8WeightOnlyConfig(version=2))
check_differentiable(linear)
```

### check_backend_contract

```python
check_backend_contract(backend, in_features=512, out_features=512,
                       compute_dtype=torch.bfloat16, compute_device="cuda", verbose=True) -> bool
```

An admission self-check for new backends: verifies that it declares its Linear classes, that both factory methods return instances of those classes, and that every declared class is a subclass of `torch.nn.Linear` (otherwise LoRA target detection and VRAM management cannot see it). It also checks that the checkpoint keys the backend actually writes all live under the layer name — a key pattern missing a scale would make Disk Offload silently load corrupted layers. Unsupported factory methods are skipped rather than counted as failures.

```python
from diffsynth.core.quant import QUANT_BACKENDS, QUANT_METHODS, check_backend_contract

spec = QUANT_METHODS["bitsandbytes_nf4"]
check_backend_contract(QUANT_BACKENDS[spec.backend](spec.config_factory({})))
```
