# `diffsynth.core.distributed`: Distributed Parallelism

`diffsynth.core.distributed` provides distributed parallelism: process-group and parallel-axis management, collective primitives, and the configuration and installation entry points for each axis.

Parallelism is organized by **axis**: one axis per capability, each communicating on its own process group, so axes compose. There are currently two:

* **Sequence parallelism** (SP) splits the tokens of one long sequence across GPUs and restores the full sequence at attention through an all-to-all exchange. Splitting and gathering are done by forward hooks on the blocks, so the model only declares token dimensions on its block class; attention reads a thread-local state through `attention_forward` to decide whether to exchange, with its signature unchanged.
* **CFG parallelism** puts the positive and negative classifier-free-guidance forwards on two groups of GPUs, each computing one branch, then exchanges the results. It acts on the pipeline's denoising loop; the model needs no changes.

See [Parallel Inference](../../Pipeline_Usage/Parallel_Inference.md) for how to enable it and which models are covered.

## Module Layout

| Module | Contents | Scope |
| --- | --- | --- |
| `config` | `ParallelConfig`, `SequenceParallelConfig`, `CFGParallelConfig` | Common entry / per-axis config |
| `context` | `ParallelContext`, `parallel_context`, `init_parallel`, `destroy_parallel` | Common |
| `collectives` | `all_to_all_4d`, `all_gather_sequence` | Common |
| `sp_hooks` | `sequence_parallel_recipe`, `install_sequence_parallel`, `SequenceParallelStack` | Sequence parallelism |
| `sp_state` | `SequenceParallelState`, `current_sequence_parallel_state`, `sequence_parallel_disabled` | Sequence parallelism |
| `sp_backend` | `UlyssesBackend`, `SEQUENCE_PARALLEL_BACKENDS` | Sequence parallelism |

All of these names can be imported directly from `diffsynth.core.distributed`, except `SEQUENCE_PARALLEL_BACKENDS`, which lives in `sp_backend`.

## Parallel Configuration: `ParallelConfig`

```python
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig, CFGParallelConfig

parallel_config = ParallelConfig()                                    # No parallelism
parallel_config = ParallelConfig(sequence=SequenceParallelConfig())     # Sequence parallelism only
parallel_config = ParallelConfig(cfg=CFGParallelConfig())               # CFG parallelism only
parallel_config = ParallelConfig(sequence=SequenceParallelConfig(), cfg=CFGParallelConfig())   # Both axes
```

`ParallelConfig` is the single configuration object accepted by a pipeline's `from_pretrained(..., parallel_config=...)`. The axes are peers, and **an axis is enabled by passing its configuration**:

| Field | Default | Description |
| --- | --- | --- |
| `sequence` | `None` | Sequence-parallel axis; enabled by passing `SequenceParallelConfig(...)` |
| `cfg` | `None` | CFG-parallel axis; enabled by passing `CFGParallelConfig(...)` |
| `timeout_seconds` | `1800` | Process-group rendezvous timeout; below 1 raises `ValueError` |

`ParallelConfig.sequence_enabled` and `ParallelConfig.cfg_enabled` report whether each axis is on.

`SequenceParallelConfig`:

| Field | Default | Description |
| --- | --- | --- |
| `backend` | `"ulysses"` | Exchange backend name from `SEQUENCE_PARALLEL_BACKENDS`. An unknown name raises `ValueError` at construction |
| `gather_between_blocks` | `False` | `False`: the whole block stack splits on entry and gathers on exit, passing local shards between blocks. `True`: each block forms its own stack, so sequences are full between blocks |

`CFGParallelConfig` has no fields yet; passing it enables the axis, which takes 2 shares for the positive and negative guidance branches.

Both axis configs are `frozen=True` dataclasses; use `dataclasses.replace` to modify them.

## Process Groups and Parallel Axes: `ParallelContext`, `init_parallel`, `destroy_parallel`

```python
from diffsynth.core.distributed import init_parallel, destroy_parallel, parallel_context

init_parallel(parallel_config=parallel_config)   # Initialize or reuse the process group, then lay out the axes
sp = parallel_context.get_sequence_group()       # Sequence axis; None when that axis is off
cfg = parallel_context.get_cfg_group()           # CFG axis; None when that axis is off
world = parallel_context.get_world_group()       # Every rank
destroy_parallel()                               # Release the communicator created by this module
```

`parallel_context` is the module's single `ParallelContext` instance, created at import. It holds each axis's process group, and reaching an axis is a method on it:

| Method | Returns |
| --- | --- |
| `get_world_group()` | The group spanning every rank; `None` before `init_parallel` |
| `get_sequence_group()` | The sequence axis group; `None` when that axis is off |
| `get_cfg_group()` | The CFG axis group; `None` when that axis is off |
| `get_sequence_rank()` / `get_sequence_world_size()` | This process's index within the sequence axis / that axis's rank count; `0` / `1` when the axis is off |
| `get_cfg_rank()` / `get_cfg_world_size()` | The same for the CFG axis |

**A disabled axis has group `None`**, not a group containing only this process. To test whether an axis is usable, check the group for `None`.

* `init_parallel(timeout_seconds=1800, parallel_config=None)` establishes the process group through an `env://` rendezvous, so it requires the `RANK`, `WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT` environment variables, usually provided by `torchrun`.
* It calls `set_device` with `LOCAL_RANK` (falling back to `RANK`), binding the GPU used by this process.
* If `torch.distributed` is already initialized, the existing group is reused instead of creating a new one.
* The communication backend follows the device type: `cuda → nccl`, `npu → hccl`.
* `destroy_parallel()` destroys the communicator only if this module created it; calling it at process exit is optional.

`init_parallel` is called internally by a pipeline's `from_pretrained(..., parallel_config=...)`.

### How the Axes Are Laid Out

With both axes on, a named-dimension `DeviceMesh` divides the ranks in the order `("cfg", "sequence")`:

* **The sequence axis is innermost** and gets adjacent ranks. It performs an all-to-all at every block, so it communicates most often.
* **The CFG axis is outer** and its ranks are strided. It exchanges one prediction per step.

`world_size = cfg_size × sp_size`. With 8 GPUs and `cfg_size=2`, the sequence axis is `[0,1,2,3]` and `[4,5,6,7]`, and the CFG axis is `[0,4]`, `[1,5]`, `[2,6]`, `[3,7]`.

When only one axis has size above 1, that axis spans every rank and uses the world group directly, without creating sub-groups.

The layout is built once in `init_parallel`; reaching an axis afterwards is a lookup. Each rank receives the sub-group **it belongs to** — `DeviceMesh` selects it by this process's global rank, so callers need no test of their own.

## Collective Primitives: `all_to_all_4d`, `all_gather_sequence`

```python
from diffsynth.core.distributed import all_to_all_4d, all_gather_sequence

y = all_to_all_4d(x, scatter_dim=2, gather_dim=1, group=group)   # [B,S/P,H,D] -> [B,S,H/P,D]
y = all_to_all_4d(x, scatter_dim=1, gather_dim=2, group=group)   # Reverse
full = all_gather_sequence(local, dim=1, group=group)            # Concatenate in rank order
```

| Function | Constraints | Backward |
| --- | --- | --- |
| `all_to_all_4d` | Must be 4-D; only `(scatter_dim, gather_dim)` of `(2, 1)` or `(1, 2)`; the `scatter_dim` length must be divisible by the number of ranks, else `ValueError` | The same exchange with the dimensions swapped |
| `all_gather_sequence` | Any dimension; negative dimensions normalize as `dim + tensor.dim()` | `reduce_scatter` sums the copies and takes this rank's shard |

Both are wrapped as `autograd.Function` and can be used in training; with a single rank they return the input tensor unchanged and issue no communication.

`group` must be passed explicitly; `None` raises `ValueError`.

## Sequence Parallelism

### Declaring How a Block Stack Is Split: `sequence_parallel_recipe`

`sequence_parallel_recipe` is the only parallel code a model needs to write. It is a class decorator:

```python
from diffsynth.core.distributed import sequence_parallel_recipe

@sequence_parallel_recipe(
    split_dimensions_for_inputs={"x": 1, "freqs": 0},
    gather_dimensions_for_outputs={"x": 1},
)
class Block(torch.nn.Module):
    def forward(self, x, freqs):
        ...
        return x
```

Four parameters:

| Parameter | Type | Description |
| --- | --- | --- |
| `split_dimensions_for_inputs` | `dict[str, int]` | Input name → token dimension. Every tensor-bearing input is padded and split **independently** according to its own length |
| `gather_dimensions_for_outputs` | `dict[str, int] \| None` | Input name → output gather dimension. **Insertion order is the return-value order**, not the input order |
| `no_exchange_modules` | `tuple[str, ...]` | Submodule names whose internal attention performs no exchange (their K/V is replicated context) |
| `kv_prefix_inputs` | `dict[str, int] \| None` | Replicated-prefix input name → sequence dimension. These inputs are **not split**; their lengths are summed and used as the K/V prefix |

#### Input Declaration Semantics: `split_dimensions_for_inputs`

* Undeclared arguments pass through untouched.
* `None` and other non-tensors produce no length entry and must remain non-tensor throughout the stack.
* Tuples and lists (including namedtuples) are traversed recursively; all tensor leaves under one argument must share the declared token dimension and length. Container types and non-tensor elements are preserved.
* When a length is not divisible by the number of ranks, **zeros are appended at the tail**. The padding amount is recorded per input and trimmed at the exit through the associated input name.
* The first block receives full inputs. Later blocks may receive the previous block's local shard, or a full companion tensor re-supplied by the caller on every iteration (such as a RoPE table); the hooks distinguish the two by length.

#### Output Declaration Semantics: `gather_dimensions_for_outputs`

Each key identifies which input's padding amount is used to trim that output; the value gives the gather dimension, which may differ from that input's split dimension. Every key must appear in the input declaration and correspond to a tensor-bearing argument at stack entry.

* A single returned tensor: declare one entry.
* A flat returned tuple/list (including a namedtuple): declare one entry per returned tensor, in return order; the type is preserved.
* Dict returns, nested containers, and several outputs associated with the same input name are not supported.
* When a stream is absent entirely (its input is `None`), that output is returned as is, without gathering.

Two streams whose return order is the reverse of their input order:

```python
@sequence_parallel_recipe(
    split_dimensions_for_inputs={"image": 1, "text": 1},
    gather_dimensions_for_outputs={"text": 1, "image": 1},
)
class Block(torch.nn.Module):
    def forward(self, image, text):
        ...
        return text, image
```

With two ranks and image/text lengths of 7/10, the local lengths are 4/5 and the padding amounts are 1/0: the first output gathers text without trimming, the second gathers image and removes one trailing token.

#### Replicated Cross-Attention: `no_exchange_modules`

When text serves only as cross-attention context and every rank holds a full copy, list that submodule in `no_exchange_modules`:

```python
@sequence_parallel_recipe(
    split_dimensions_for_inputs={"x": 1, "freqs": 0},
    gather_dimensions_for_outputs={"x": 1},
    no_exchange_modules=("cross_attn",),
)
```

The `attention_forward` inside it attends the local query against the full K/V directly, without exchange.

#### Replicated KV Prefix: `kv_prefix_inputs`

When K/V has the form "replicated prefix | sharded suffix" while Q contains only the suffix (KV cache, template prefix), declare the prefix input:

```python
@sequence_parallel_recipe(
    split_dimensions_for_inputs={"hidden_states": 1, "rotary_emb": 0},
    gather_dimensions_for_outputs={"hidden_states": 1},
    kv_prefix_inputs={"kv_prefix": 1},
)
```

`kv_prefix` is not split; every rank holds the full prefix. Its length is summed into the state, attention exchanges only the suffix, and the prefix is sliced locally by **KV head count** and prepended, with a single softmax throughout. The prefix length comes from the explicit declaration.

The pass that writes the cache must produce full outputs; wrap it in `sequence_parallel_disabled()` (see below).

### Installation and Execution: `install_sequence_parallel`

```python
from diffsynth.core.distributed import install_sequence_parallel

install_sequence_parallel(model, parallel_config)
```

`model_loader` calls it for every loaded model. Behavior:

* It is a **no-op** when `parallel_config` is `None`, `sequence_enabled` is false, the process group is missing, or there is only one rank.
* Stacks are discovered from structure rather than declared by the model: a module whose class (or the class inside a residency wrapper) carries a recipe declaration is a block; the longest run of blocks **under one container, with consecutive indices and one class** forms a stack.
* Only a class's own `__dict__` counts, so declarations are **not inherited**. A subclass of a declared block (some ControlNet blocks, for example) does not automatically become an SP stack.
* Wrappers such as `AutoWrappedNonRecurseModule` are unwrapped one level: matching uses the inner class, hooks attach to the wrapper, and argument order comes from the declared class's `forward` signature.
* With `gather_between_blocks=True`, each block forms its own stack.

`SequenceParallelStack` is the object carrying the hooks. `install` hooks **every** block in the stack, not just the first and last, because the loop driving a stack often re-supplies the same full companion tensor on each iteration.

### Runtime State: `SequenceParallelState`

```python
from diffsynth.core.distributed import current_sequence_parallel_state, sequence_parallel_disabled

state = current_sequence_parallel_state()   # None outside a stack
if state is not None:
    ...                                     # Currently inside a sharded stack

with sequence_parallel_disabled():
    cache = model.encode_prefix(...)        # This pass is not split; outputs stay full
```

`SequenceParallelState` is thread-local, with at most one active instance per thread: the stack activates it at the first block and deactivates it after gathering the last block's outputs.

| Member | Description |
| --- | --- |
| `backend` | Exchange backend used by this stack |
| `local_lengths` / `paddings` | Local token counts and tail padding amounts, keyed by declared input name |
| `kv_prefix_len` | Sum of replicated prefix lengths; 0 when none is declared |
| `attention_parallel` | Whether attention should exchange; `False` inside a `no_exchange_modules` submodule |
| `no_exchange_depth` | Nesting depth of replicated submodules |

`set_state` does not support nesting; activating twice raises `RuntimeError`.

`sequence_parallel_disabled()` works both as a context manager and as a decorator (`@sequence_parallel_disabled()`). It makes an installed stack **skip splitting for that pass**, which is needed for forwards that must produce full tensors, such as writing a KV cache that later serves as a replicated prefix.

### Integration with the Attention Entry: `attention_forward`

`diffsynth.core.attention.attention_forward` is the unified attention entry and the single point where sequence parallelism intervenes:

```python
from diffsynth.core.attention import attention_forward

out = attention_forward(q, k, v, q_pattern="b n s d", k_pattern="b n s d",
                        v_pattern="b n s d", out_pattern="b n s d")
```

It reads `current_sequence_parallel_state()`:

* State is `None`, or `attention_parallel` is false → `local_attention_forward`, identical to single-GPU execution.
* Otherwise → `state.backend.forward(...)`, which calls the same local kernel before and after the exchange.

Calls in a layout other than `"b s n d"` are converted on the way in and restored on the way out. `cu_seqlens` (packed varlen) still applies to global segment boundaries after the exchange.

### Exchange Backend: `UlyssesBackend`

```python
from diffsynth.core.distributed import UlyssesBackend
from diffsynth.core.distributed.sp_backend import SEQUENCE_PARALLEL_BACKENDS   # {"ulysses": UlyssesBackend}
```

`UlyssesBackend` performs one all-to-all before and after the local attention, turning sequence shards into head shards and back.

* When Q/K/V have identical shapes they ride in **one** collective (concatenated along `head_dim` so whole heads are still scattered), bit-identical to three separate exchanges; differing shapes are exchanged separately.
* A replicated KV prefix takes another path: only the suffix is exchanged, and the prefix is sliced locally by KV heads and prepended. This path **does not support `attn_mask`**; passing one raises `ValueError`.
* The group, rank and world size are looked up on every call, never cached.

To register a new backend, add a name-to-class mapping to `SEQUENCE_PARALLEL_BACKENDS`. The class must provide `forward(q, k, v, *, local_attention, q_pattern, k_pattern, v_pattern, out_pattern, dims, **attn_kwargs)` with the same signature as `UlyssesBackend.forward`, and can then be selected through `SequenceParallelConfig(backend=...)`.

### Constraints and Errors

| Item | Description |
| --- | --- |
| Head divisibility | Both the Q head count and the KV head count taking part in the exchange must be divisible by the number of ranks, otherwise `all_to_all_4d` raises. For GQA models the KV head count sets the limit |
| Sequence length | When not divisible, zeros are appended at the tail and trimmed at the exit according to the declaration; padded tokens take part in attention |
| Replicated KV prefix | `attn_mask` is not supported; passing one raises `ValueError` |
| Declaration consistency | Keys of gather outputs must appear in the input declaration; keys of `kv_prefix_inputs` must not. Violations raise `ValueError` in the decorator |
## CFG Parallelism

Classifier-free guidance (CFG) runs two forwards per step: one conditioned on the positive prompt, one on the negative prompt, then combines them as `nega + cfg_scale * (posi - nega)`. CFG parallelism puts those two forwards on the two groups of the CFG axis, each GPU computing one branch.

Configuration and axis access are covered above; the scheduling lives in `diffsynth.diffusion.BasePipeline.cfg_guided_model_fn`, the single CFG entry point in a pipeline's denoising loop.

### Entry Point: `cfg_guided_model_fn`

```python
noise_pred = self.cfg_guided_model_fn(
    model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega,
    **models, timestep=timestep,
)
```

| Argument | Description |
| --- | --- |
| `model_fn` | Callable performing one forward |
| `cfg_scale` | Guidance strength |
| `inputs_shared` | Inputs common to both branches |
| `inputs_posi` / `inputs_nega` | Inputs specific to each branch |
| `**inputs_others` | Remaining keyword arguments forwarded to `model_fn`, common to both branches |

Each branch is invoked as `model_fn(**inputs_branch, **inputs_shared, **inputs_others)`. **Every difference between the branches is expressed through the branch dicts.**

With the CFG axis off, both branches run sequentially in this process. With it on:

1. GPUs at `parallel_context.get_cfg_rank() == 0` compute the positive branch, the others the negative branch
2. `all_gather` along the batch dimension gives every GPU both predictions
3. Every GPU combines them locally with the original expression

The combined result is identical on every GPU, so `scheduler.step` also runs locally and no broadcast is needed.

### Constraints and Errors

| Item | Description |
| --- | --- |
| `cfg_scale` | Must not equal 1 while the CFG axis is on, else `ValueError` — there is no negative branch to hand to the other group |
| Head divisibility | Constrains only the sequence axis size, independent of the CFG axis and the total GPU count |
| Output shape | Both branches must return the same shape (input lengths may differ), else `all_gather` fails |
| Mutable state shared across branches | The branches run in different processes, so a shared cache counter accumulates independently on each. Such state must be kept per branch |
