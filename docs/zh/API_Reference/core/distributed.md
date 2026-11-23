# `diffsynth.core.distributed`: 分布式并行

`diffsynth.core.distributed` 提供分布式并行能力：进程组管理、集合通信原语，以及序列并行的配置与安装入口。

序列并行（Sequence Parallel，SP）把一条长序列的 token 切分到多张卡上计算，在注意力处通过 all-to-all 交换还原出完整序列。切分与还原由挂在 block 上的 forward hook 完成，模型侧只需在 block 类上声明 token 维；注意力通过 `attention_forward` 读取线程局部状态判断是否交换，签名不变。

启用方式与可并行的模型清单见[并行推理](../../Pipeline_Usage/Parallel_Inference.md)。

## 模块构成

| 模块 | 内容 | 归属 |
| --- | --- | --- |
| `config` | `ParallelConfig`、`SequenceParallelConfig` | 通用入口 / 序列并行 |
| `context` | `init_parallel`、`get_parallel_group`、`destroy_parallel` | 通用 |
| `collectives` | `all_to_all_4d`、`all_gather_sequence` | 通用 |
| `sp_hooks` | `sequence_parallel_recipe`、`install_sequence_parallel`、`SequenceParallelStack` | 序列并行 |
| `sp_state` | `SequenceParallelState`、`current_sequence_parallel_state`、`sequence_parallel_disabled` | 序列并行 |
| `sp_backend` | `UlyssesBackend`、`SEQUENCE_PARALLEL_BACKENDS` | 序列并行 |

以上名称均可从 `diffsynth.core.distributed` 直接导入（`SEQUENCE_PARALLEL_BACKENDS` 除外，它在 `sp_backend` 中）。

## 并行配置：`ParallelConfig`、`SequenceParallelConfig`

```python
from diffsynth.core.distributed import ParallelConfig, SequenceParallelConfig

parallel_config = ParallelConfig()                                    # 默认启用 Ulysses 序列并行
parallel_config = ParallelConfig(sequence=SequenceParallelConfig(backend="ulysses"))
parallel_config = ParallelConfig(sequence=SequenceParallelConfig(backend=None))   # 关闭 SP
```

`ParallelConfig` 是并行的统一配置对象，pipeline 的 `from_pretrained(..., parallel_config=...)` 接收它：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `sequence` | `SequenceParallelConfig()` | 序列并行配置 |
| `timeout_seconds` | `1800` | 进程组 rendezvous 超时，小于 1 时抛 `ValueError` |

`ParallelConfig.sp_enabled` 是「序列并行是否启用」的便捷属性。

`SequenceParallelConfig`：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `backend` | `"ulysses"` | 交换后端名，取自 `SEQUENCE_PARALLEL_BACKENDS`；`None` 表示不启用序列并行。非法名在构造时抛 `ValueError` |
| `gather_between_blocks` | `False` | `False`：整个 block 栈只在入口切分、出口还原，块间传递局部分片。`True`：每个 block 各自成一个栈，块间序列是完整的 |

两个配置都是 `frozen=True` 的 dataclass，修改请用 `dataclasses.replace`。

## 进程组管理：`init_parallel`、`get_parallel_group`、`destroy_parallel`

```python
from diffsynth.core.distributed import init_parallel, get_parallel_group, destroy_parallel

group = init_parallel()          # 初始化或复用进程组
group = get_parallel_group()     # 未初始化时返回 None
destroy_parallel()               # 释放本模块创建的通信组
```

* `init_parallel(timeout_seconds=1800)` 通过 `env://` rendezvous 建立进程组，因此需要 `RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT` 环境变量，通常由 `torchrun` 提供。
* 它会按 `LOCAL_RANK`（缺省用 `RANK`）调用 `set_device`，绑定本进程使用的卡。
* 若 `torch.distributed` 已初始化，则复用现有组，不重复创建。
* 通信后端按设备类型选择：`cuda → nccl`、`npu → hccl`。
* `destroy_parallel()` 只在本模块创建了组时才真正销毁通信子；进程退出时不调用也可以。

`init_parallel` 由 pipeline 的 `from_pretrained(..., parallel_config=...)` 内部调用。

## 集合通信原语：`all_to_all_4d`、`all_gather_sequence`

```python
from diffsynth.core.distributed import all_to_all_4d, all_gather_sequence

y = all_to_all_4d(x, scatter_dim=2, gather_dim=1, group=group)   # [B,S/P,H,D] -> [B,S,H/P,D]
y = all_to_all_4d(x, scatter_dim=1, gather_dim=2, group=group)   # 反向
full = all_gather_sequence(local, dim=1, group=group)            # 按 rank 顺序拼接
```

| 函数 | 约束 | 反向 |
| --- | --- | --- |
| `all_to_all_4d` | 必须是 4 维；只支持 `(scatter_dim, gather_dim)` 为 `(2, 1)` 或 `(1, 2)`；`scatter_dim` 长度必须被卡数整除，否则 `ValueError` | 交换两个维度后做同一次交换 |
| `all_gather_sequence` | 任意维；负数维按 `dim + tensor.dim()` 归一 | `reduce_scatter` 求和后取本 rank 分片 |

两者都封装成 `autograd.Function`，可用于训练；卡数为 1 时直接返回原张量，不产生通信。

## 序列并行

### 声明块栈的切分：`sequence_parallel_recipe`

`sequence_parallel_recipe` 是模型侧唯一需要写的并行代码，它是一个类装饰器：

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

四个参数：

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `split_dimensions_for_inputs` | `dict[str, int]` | 输入名 → token 维。每个带张量的输入**独立**按自己的长度补齐并切分 |
| `gather_dimensions_for_outputs` | `dict[str, int] \| None` | 输入名 → 输出 gather 维。**插入顺序即返回值顺序**，不是输入顺序 |
| `no_exchange_modules` | `tuple[str, ...]` | 子模块名，其内部注意力不做交换（K/V 是复制的上下文） |
| `kv_prefix_inputs` | `dict[str, int] \| None` | 复制前缀输入名 → 序列维。这些输入**不切分**，长度求和后作为 K/V 前缀 |

#### 输入声明的语义：`split_dimensions_for_inputs`

* 未声明的参数原样透传，hook 不碰。
* `None` 与其它非张量不产生长度条目，且必须在整个栈内保持非张量。
* tuple/list（含 namedtuple）会递归遍历，同一参数下所有张量叶子必须共享声明的 token 维与长度，容器类型与非张量元素保持不变。
* 长度不能被卡数整除时，在**尾部补零**；补齐量按输入各自记录，出口按关联输入名裁掉。
* 首块收到完整输入；后续块可能收到上一块的局部分片，也可能收到调用方每次重新传入的完整伴随张量（如 RoPE 表），hook 按长度区分两者。

#### 输出声明的语义：`gather_dimensions_for_outputs`

`gather_dimensions_for_outputs` 的每个键标识「用哪个输入的补齐量来裁剪这个输出」，值给出 gather 维（可以与该输入的切分维不同）。每个键必须出现在输入声明里，且在栈入口对应一个带张量的参数。

* 返回单张量：声明一项。
* 返回扁平 tuple/list（含 namedtuple）：按返回顺序逐项声明，类型保持不变。
* 不支持 dict 返回、嵌套容器、以及多个输出关联同一输入名。
* 某条流整体缺席（对应输入为 `None`）时，该输出不 gather、原样返回。

两条流、返回顺序与输入顺序相反的例子：

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

两卡、image/text 长度 7/10 时，局部长度为 4/5，补齐量为 1/0：第一个输出 gather text 且不裁剪，第二个 gather image 并去掉 1 个尾部 token。

#### 复制的 cross-attention：`no_exchange_modules`

文本只作为 cross-attention 的上下文、且各卡持有完整副本时，把该子模块列入 `no_exchange_modules`：

```python
@sequence_parallel_recipe(
    split_dimensions_for_inputs={"x": 1, "freqs": 0},
    gather_dimensions_for_outputs={"x": 1},
    no_exchange_modules=("cross_attn",),
)
```

其内部的 `attention_forward` 会用本地 query 直接对完整 K/V 做注意力，不交换。

#### 复制的 KV 前缀：`kv_prefix_inputs`

K/V 形如「复制前缀 | 分片后缀」而 Q 只有后缀时（KV cache、模板前缀），声明前缀输入：

```python
@sequence_parallel_recipe(
    split_dimensions_for_inputs={"hidden_states": 1, "rotary_emb": 0},
    gather_dimensions_for_outputs={"hidden_states": 1},
    kv_prefix_inputs={"kv_prefix": 1},
)
```

`kv_prefix` 不切分，各卡持有完整前缀；其长度被求和记入状态，注意力只交换后缀，前缀按 **KV 头数**本地切片后拼回，全程一次 softmax。前缀长度来自显式声明，不由 `k` 与 `q` 的长度差推断。

写缓存的那一遍前向需要完整输出，用 `sequence_parallel_disabled()` 包住（见下文）。

### 安装与运行：`install_sequence_parallel`

```python
from diffsynth.core.distributed import install_sequence_parallel

install_sequence_parallel(model, parallel_config)
```

`model_loader` 对每个加载的模型调用它。行为：

* `parallel_config` 为 `None`、`sp_enabled` 为假、进程组缺失或卡数为 1 时，**为空操作**。
* 栈是从结构里发现的，不由模型声明：一个模块的类（或 residency 包装内层的类）带有 recipe 声明，它就是一个 block；**同一容器下、索引连续、类相同**的最长一段构成一个栈。
* 声明只认类自己的 `__dict__`，**不被继承**。因此继承已声明 block 的子类（例如某些 ControlNet block）不会自动成为 SP 栈。
* `AutoWrappedNonRecurseModule` 这类包装会被拆一层：匹配取内层类，hook 挂在包装上，参数顺序取被声明类的 `forward` 签名。
* `gather_between_blocks=True` 时每个 block 各自成栈。

`SequenceParallelStack` 是承载 hook 的对象，`install` 为栈内**每个** block 挂 hook，而不只是首尾——因为驱动栈的循环常常每次重新传入同一个完整伴随张量。

### 运行时状态：`SequenceParallelState`

```python
from diffsynth.core.distributed import current_sequence_parallel_state, sequence_parallel_disabled

state = current_sequence_parallel_state()   # 栈外为 None
if state is not None:
    ...                                     # 当前处于某个分片栈内部

with sequence_parallel_disabled():
    cache = model.encode_prefix(...)        # 这一遍不切分，输出完整
```

`SequenceParallelState` 是线程局部的，每线程至多一个活跃实例：栈在首块激活、末块 gather 后注销。

| 成员 | 说明 |
| --- | --- |
| `backend` | 本次栈使用的交换后端 |
| `local_lengths` / `paddings` | 按声明输入名记录的局部 token 数与尾部补齐量 |
| `kv_prefix_len` | 复制前缀长度之和，无声明时为 0 |
| `attention_parallel` | 是否应该交换；处于 `no_exchange_modules` 子模块内时为 `False` |
| `no_exchange_depth` | 复制子模块的嵌套深度 |

`set_state` 不支持嵌套，重复激活抛 `RuntimeError`。

`sequence_parallel_disabled()` 既是上下文管理器也是装饰器（`@sequence_parallel_disabled()`）。它让已安装的栈**这一遍不切分**，用于必须产出完整张量的前向，例如写入之后要充当复制前缀的 KV cache。

### 与注意力入口的对接：`attention_forward`

`diffsynth.core.attention.attention_forward` 是统一的注意力入口，也是序列并行的唯一介入点：

```python
from diffsynth.core.attention import attention_forward

out = attention_forward(q, k, v, q_pattern="b n s d", k_pattern="b n s d",
                        v_pattern="b n s d", out_pattern="b n s d")
```

它读取 `current_sequence_parallel_state()`：

* 状态为 `None`，或 `attention_parallel` 为假 → 走 `local_attention_forward`，与单卡完全一致；
* 否则 → 交给 `state.backend.forward(...)`，在交换前后调用同一个本地内核。

布局不是 `"b s n d"` 的调用会在入口转换、出口还原；`cu_seqlens`（packed varlen）在交换后仍按全局段边界生效。

### 交换后端：`UlyssesBackend`

```python
from diffsynth.core.distributed import UlyssesBackend
from diffsynth.core.distributed.sp_backend import SEQUENCE_PARALLEL_BACKENDS   # {"ulysses": UlyssesBackend}
```

`UlyssesBackend` 在本地注意力前后各做一次 all-to-all，把序列分片换成头分片，算完再换回来。

* Q/K/V 形状相同时合并成**一次**集合通信（沿 `head_dim` 拼接，保证按整头切分），与三次独立交换逐位一致；形状不同则分别交换。
* 复制 KV 前缀走另一条路径：只交换后缀，前缀按 KV 头本地切片后拼回；该路径**不支持 `attn_mask`**，传入会抛 `ValueError`。
* 组与 rank/卡数每次调用现查，不缓存。

注册新后端：向 `SEQUENCE_PARALLEL_BACKENDS` 加入名称到类的映射，类需提供与 `UlyssesBackend.forward` 相同签名的 `forward(q, k, v, *, local_attention, q_pattern, k_pattern, v_pattern, out_pattern, dims, **attn_kwargs)`，然后在 `SequenceParallelConfig(backend=...)` 里选用。

### 约束与报错

| 项 | 说明 |
| --- | --- |
| 头数整除 | 参与交换的 Q 头数与 KV 头数都必须被卡数整除，否则 `all_to_all_4d` 抛错。GQA 模型的上限由 KV 头数决定 |
| 序列长度 | 不整除时在尾部补零，出口按声明裁掉；补齐 token 会参与注意力计算 |
| 复制 KV 前缀 | 不支持 `attn_mask`，传入抛 `ValueError` |
| 声明一致性 | gather 输出的键必须在输入声明中；`kv_prefix_inputs` 的键不得出现在输入声明中；违反时装饰器抛 `ValueError` |
