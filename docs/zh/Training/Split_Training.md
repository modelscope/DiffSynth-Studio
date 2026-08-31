# 两阶段拆分训练

本文档介绍拆分训练，能够自动将训练过程拆分为两阶段进行，减少显存占用，同时加快训练速度。

（拆分训练是实验性特性，尚未进行大规模验证，如果在使用中出现问题，请在 GitHub 上给我们提 issue。）

## 拆分训练

在大部分模型的训练过程中，大量计算发生在“前处理”中，即“与去噪模型无关的计算”，包括 VAE 编码、文本编码等。当对应的模型参数固定时，这部分计算的结果是重复的，在多个 epoch 中每个数据样本的计算结果完全相同，因此我们提供了“拆分训练”功能，该功能可以自动分析并拆分训练过程。

对于普通文生图模型的标准监督训练，拆分过程是非常简单的，只需要把所有 [`Pipeline Units`](../Developer_Guide/Building_a_Pipeline.md#units) 的计算拆分到第一阶段，将计算结果存储到硬盘中，然后在第二阶段从硬盘中读取这些结果并进行后续计算即可。但如果前处理过程中需要梯度回传，情况就变得极其复杂，为此，我们引入了一个计算图拆分算法用于分析如何拆分计算。

## 启用拆分训练

拆分训练已支持[标准监督训练](../Training/Supervised_Fine_Tuning.md)和[直接蒸馏训练](../Training/Direct_Distill.md)，在训练命令中通过 `--task` 参数控制，以 Qwen-Image 模型的 LoRA 训练为例，拆分前的训练命令为：

```shell
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "qwen_image/Qwen-Image/*" --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/qwen_image/Qwen-Image \
  --dataset_metadata_path data/diffsynth_example_dataset/qwen_image/Qwen-Image/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters
```

拆分后，在第一阶段中，做如下修改：

* 将 `--dataset_repeat` 改为 1，避免重复计算
* 将 `--output_path` 改为第一阶段计算结果保存的路径
* 添加额外参数 `--task "sft:data_process"`
* 在 `offload_models` 中填入不需要进行 forward 计算的模型，格式与 `model_id_with_origin_paths` 相同
  * 直接删除 `--model_id_with_origin_paths` 中不需要进行 forward 计算的模型也可，但你必须确保对应的模型在 pipeline 中不会被间接调用，这意味着你必须了解 Pipeline 的运行细节

```shell
accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/qwen_image/Qwen-Image \
  --dataset_metadata_path data/diffsynth_example_dataset/qwen_image/Qwen-Image/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors,Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors" \
  --offload_models "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-LoRA-splited-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --task "sft:data_process"
```

在第二阶段，做如下修改：

* 将 `--dataset_base_path` 改为第一阶段的 `--output_path`
* 删除 `--dataset_metadata_path`
* 添加额外参数 `--task "sft:train"`
* 在 `offload_models` 中填入不需要进行 forward 计算的模型，格式与 `model_id_with_origin_paths` 相同
  * 直接删除 `--model_id_with_origin_paths` 中不需要进行 forward 计算的模型也可，但你必须确保对应的模型在 pipeline 中不会被间接调用，这意味着你必须了解 Pipeline 的运行细节

```shell
accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path "./models/train/Qwen-Image-LoRA-splited-cache" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors,Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors" \
  --offload_models "Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-LoRA-splited" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --task "sft:train"
```

我们提供了样例训练脚本和验证脚本，位于 `examples/qwen_image/model_training/special/split_training`。

## 计算图拆分算法原理

训练框架通过 `DiffusionTrainingModule` 的 `split_pipeline_units` 方法拆分 `Pipeline` 中的计算单元，以下是计算图拆分算法的详细原理。

### 问题定义

为精确刻画拆分过程，本节对计算流水线进行形式化描述。设流水线由 $n$ 个计算单元（[`Pipeline Unit`](../Developer_Guide/Building_a_Pipeline.md#units)）构成，记单元集合为 $V=\{u_1,u_2,\dots,u_n\}$。每个单元 $u\in V$ 具有如下属性：

* 输入参数集合 $\operatorname{in}(u)$：由 `input_params`、`input_params_posi` 与 `input_params_nega` 声明，表示 $u$ 计算前必须读取的数据项；
* 输出参数集合 $\operatorname{out}(u)$：由 `output_params` 声明，表示 $u$ 计算完成后产生并写入数据缓存的数据项；
* 关联模型集合 $\mathcal{M}(u)$：由 `onload_model_names` 声明，表示 $u$ 的计算所依赖的模型。

全体参数构成参数空间 $\mathcal{P}=\bigcup_{u\in V}\left(\operatorname{in}(u)\cup\operatorname{out}(u)\right)$。

**定义 1（数据依赖边）** 设参数 $p\in\mathcal{P}$。若存在单元 $u_i,u_j\in V$，使得 $p\in\operatorname{out}(u_i)\cap\operatorname{in}(u_j)$，且 $u_i$ 为 $p$ 的最近生产者（即全部产生 $p$ 的单元中执行顺序最靠后者），则称 $u_i$ 与 $u_j$ 之间存在数据依赖边 $(u_i,u_j)$，其语义为 $u_j$ 的计算必须发生在 $u_i$ 完成之后。

由此，计算流水线被抽象为有向无环图 $G=(V,E)$，其中 $E$ 为全部数据依赖边的集合。

**定义 2（直接相关单元）** 给定需梯度回传的模型集合 $\mathcal{W}$（由 `trainable_models` 与 `lora_base_model` 指定，分别是正在训练的模型组件和正在以 LoRA 训练的模型组件）。若单元 $u\in V$ 满足 $\mathcal{M}(u)\cap\mathcal{W}\neq\varnothing$，则称 $u$ 为直接相关单元，其计算过程涉及可训练模型的调用。

**定义 3（计算图拆分问题）** 给定图 $G=(V,E)$ 与模型集合 $\mathcal{W}$，求 $V$ 的一个二分 $(V_1,V_2)$，使得 $V_1$ 为包含全部直接相关单元且满足下述闭包条件的最小集合，$V_2=V\setminus V_1$：

（C1）前向闭包：若 $u\in V_1$ 且 $(u,v)\in E$，则 $v\in V_1$，即 $V_2$ 中不存在任何依赖 $V_1$ 输出的单元；

（C2）更新链闭包：对任意参数 $p\in\mathcal{P}$，设其更新链 $\mathbf{c}(p)=(u^{(1)},u^{(2)},\dots,u^{(k)})$ 为按执行顺序产生 $p$ 的全部单元。若 $p$ 在 $V_1$ 内首次被消费于 $u^{(i)}$ 且 $i<k$，则 $u^{(i+1)},\dots,u^{(k)}\in V_1$，即 $V_1$ 内部对参数的消费与更新顺序与全图执行顺序一致。

拆分目的：$V_2$ 中单元的计算结果与模型参数无关，可在多个 epoch 间复用，故将其执行于数据预处理阶段（第一阶段）并缓存至磁盘；$V_1$ 中单元的计算需梯度回传，执行于训练阶段（第二阶段）。

### 算法设计

考虑集合算子 $T:2^V\to 2^V$：

$$
T(X)=X\cup F(X)\cup U(X),
$$

其中 $F$ 为前向可达算子，$F(X)$ 为 $X$ 沿数据依赖边（方向 producer $\to$ consumer）可达的全部后继单元的传递闭包；$U$ 为更新链回溯算子，$U(X)$ 定义为：对任意参数 $p$，若 $p$ 在 $X$ 内首次被消费于 $u^{(i)}$ 且 $i<k$（$k$ 为更新链 $\mathbf{c}(p)$ 的长度），则 $U(X)$ 包含 $\mathbf{c}(p)$ 中后续更新单元 $u^{(i+1)},\dots,u^{(k)}$ 中不属于 $X$ 者。

**命题 1（单调性与终止性）** $T$ 为单调算子，即 $X\subseteq Y\Rightarrow T(X)\subseteq T(Y)$。以直接相关单元集合 $X_0=\{u\in V\mid\mathcal{M}(u)\cap\mathcal{W}\neq\varnothing\}$ 为初值，迭代 $X_{k+1}=T(X_k)$。由于 $X_k$ 单调递增且 $X_k\subseteq V$（$|V|=n$），迭代至多 $n$ 步终止于最小不动点 $X^*$，满足 $T(X^*)=X^*$。

**命题 2（一致性）** 令 $V_1=X^*$，$V_2=V\setminus V_1$，则 $(V_1,V_2)$ 满足条件（C1）与（C2）。由 $X^*$ 的不动点性质可知，$F(X^*)\subseteq X^*$ 蕴含（C1）成立，$U(X^*)\subseteq X^*$ 蕴含（C2）成立。

### 算法伪代码

上述算法的实现如下：

```python
def split_pipeline_units(units, model_names):
    # 步骤 1：初始化 X_0，即直接相关单元集合
    related = {id for id, unit in enumerate(units)
               if unit.onload_model_names is not None
               and any(m in unit.onload_model_names for m in model_names)}

    # 步骤 2：构建数据依赖边 E 与各参数的更新链 c(p)
    edges = build_edges(units)
    chains = build_chains(units)

    # 步骤 3：不动点迭代 X_{k+1} = X_k ∪ F(X_k) ∪ U(X_k)
    while True:
        before = len(related)
        related = forward_reachable(edges, related)       # F(X_k)
        related = updating_units(units, chains, related)  # U(X_k)
        if len(related) == before:
            break

    # 步骤 4：输出二分 (V1, V2)
    related_units   = [units[i] for i in sorted(related)]
    unrelated_units = [units[i] for i in range(len(units)) if i not in related]
    return related_units, unrelated_units
```

各子函数的构造细节如下。

`build_edges` 扫描单元序列，维护"参数 $\to$ 最近生产者"的映射：当单元 $u_j$ 的输入参数 $p$ 已存在生产者 $u_i$ 时记录边 $(u_i,u_j)$；随后以 $u_j$ 的每个输出参数 $p$ 更新该映射。

```python
def build_edges(units):
    last_producer = {}
    edges = []
    for id, unit in enumerate(units):
        for param in unit.fetch_input_params():
            if param in last_producer:
                edges.append((last_producer[param], id))
        for param in unit.fetch_output_params():
            last_producer[param] = id
    return edges
```

`build_chains` 按执行顺序记录每个参数的更新链 $\mathbf{c}(p)$。

```python
def build_chains(units):
    params = sorted(set(sum([unit.fetch_input_params() + unit.fetch_output_params()
                             for unit in units], [])))
    chains = {param: [] for param in params}
    for id, unit in enumerate(units):
        for param in unit.fetch_output_params():
            chains[param].append(id)
    return chains
```

`forward_reachable` 实现前向可达算子 $F$，沿 producer$\to$consumer 方向迭代求闭包，直至不再有新单元加入。

```python
def forward_reachable(edges, related):
    while True:
        neighbors = {target for source, target in edges
                     if source in related and target not in related}
        if not neighbors:
            break
        related |= neighbors
    return related
```

`updating_units` 实现更新链回溯算子 $U$。对相关集合 $X$ 内每个单元的输入参数 $p$，确定 $p$ 在 $X$ 内首次被消费的单元 $u^{(i)}$；若 $i<k$，则将更新链 $\mathbf{c}(p)$ 中位于 $u^{(i)}$ 之后的全部单元并入 $X$，从而保证条件（C2）。

```python
def updating_units(units, chains, related):
    # 按执行顺序（单元 id 升序）确定 p 在 X 内首次被消费的单元
    first_consumer = {}
    for unit_id in sorted(related):
        for param in units[unit_id].fetch_input_params():
            if param not in first_consumer:
                first_consumer[param] = unit_id
    for param, first in first_consumer.items():
        chain = chains[param]
        if first in chain and chain.index(first) != len(chain) - 1:
            for later in chain[chain.index(first) + 1:]:
                related.add(later)
    return related
```

### 两阶段拆分流程

获得划分 $(V_1,V_2)$ 后，训练框架依据任务类型配置 `Pipeline`，并分别交由 `launch_data_process_task` 与 `launch_training_task` 执行：

```python
def split_pipeline_units(task, pipe, trainable_models, lora_base_model, ...):
    models_require_backward = []
    if trainable_models is not None:
        models_require_backward += trainable_models.split(",")
    if lora_base_model is not None:
        models_require_backward.append(lora_base_model)

    if task.endswith(":data_process"):           # 第一阶段（数据预处理）
        other_units, pipe.units = pipe.split_pipeline_units(models_require_backward)
        # pipe.units = V2，仅执行与模型无关的计算（如 VAE 编码、文本编码）
        # 可选：追加 GeneralUnit_RemoveCache，剔除冗余缓存项以减小缓存体积
    elif task.endswith(":train"):                # 第二阶段（训练）
        pipe.units, _ = pipe.split_pipeline_units(models_require_backward)
        # pipe.units = V1，仅执行与模型相关的计算
```

**阶段一（数据预处理）** 在 `torch.no_grad()` 模式下遍历数据集，对每个样本执行 $V_2$ 中的单元，并将中间结果按进程分片序列化至磁盘：

```python
for data in dataloader:
    with torch.no_grad():
        cache = model(data)                      # 仅执行 V2 中的单元
        torch.save(cache, cache_path)            # 逐样本保存为 .pth
```

**阶段二（训练）** 数据加载器以第一阶段产生的缓存为输入，执行 $V_1$ 中的单元，完成前向传播、反向传播与参数更新：

```python
for data in dataloader:                          # 读取第一阶段缓存
    with accelerator.accumulate(model):
        loss = model({}, inputs=data)            # 仅执行 V1 中的单元
        accelerator.backward(loss)               # 梯度仅流经可训练模型
        optimizer.step()
```

**命题 3（缓存可复用性）** 由定义 3 可知，$V_2$ 中的单元不依赖任何可训练模型，其计算结果与模型参数 $\theta$ 无关。因此，阶段一产生的缓存可在后续任意 epoch 中直接复用，无需重复计算。基于该性质，阶段一的 `--dataset_repeat` 应设为 1，阶段二可保留原始取值。