# Two-Stage Split Training

This document introduces split training, which can automatically divide the training process into two stages, reducing VRAM usage while accelerating training speed.

(Split training is an experimental feature that has not yet undergone large-scale validation. If you encounter any issues while using it, please submit an issue on GitHub.)

## Split Training

In the training process of most models, a large amount of computation occurs in "preprocessing," i.e., "computations unrelated to the denoising model," including VAE encoding, text encoding, etc. When the corresponding model parameters are fixed, the results of these computations are repetitive. For each data sample, the computational results are identical across multiple epochs. Therefore, we provide a "split training" feature that can automatically analyze and split the training process.

For standard supervised training of ordinary text-to-image models, the splitting process is straightforward. It only requires splitting the computation of all [`Pipeline Units`](../Developer_Guide/Building_a_Pipeline.md#units) into the first stage, storing the computational results to disk, and then reading these results from disk in the second stage for subsequent computations. However, if gradient backpropagation is required during preprocessing, the situation becomes extremely complex. To address this, we introduced a computational graph splitting algorithm to analyze how to split the computation.

## Enabling Split Training

Split training already supports [Standard Supervised Training](../Training/Supervised_Fine_Tuning.md) and [Direct Distillation Training](../Training/Direct_Distill.md). The `--task` parameter in the training command controls this. Taking LoRA training of the Qwen-Image model as an example, the pre-split training command is:

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

After splitting, in the first stage, make the following modifications:

* Change `--dataset_repeat` to 1 to avoid redundant computation
* Change `--output_path` to the path where the first-stage computation results are saved
* Add the additional parameter `--task "sft:data_process"`
* Fill in `offload_models` with the models that do not require forward computation, in the same format as `model_id_with_origin_paths`
  * Alternatively, you can directly remove from `--model_id_with_origin_paths` the models that do not require forward computation. However, you must ensure that the corresponding models are not indirectly invoked in the pipeline, which means you need to understand the internal details of the Pipeline.

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

In the second stage, make the following modifications:

* Change `--dataset_base_path` to the `--output_path` of the first stage
* Remove `--dataset_metadata_path`
* Add the additional parameter `--task "sft:train"`
* Fill in `offload_models` with the models that do not require forward computation, in the same format as `model_id_with_origin_paths`
  * Alternatively, you can directly remove from `--model_id_with_origin_paths` the models that do not require forward computation. However, you must ensure that the corresponding models are not indirectly invoked in the pipeline, which means you need to understand the internal details of the Pipeline.

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

We provide sample training scripts and validation scripts located at `examples/qwen_image/model_training/special/split_training`.

## Principles of the Computational Graph Splitting Algorithm

The training framework splits the computational units in the `Pipeline` through the `split_pipeline_units` method of `DiffusionTrainingModule`. The following describes the detailed principles of the computational graph splitting algorithm.

### Problem Definition

To precisely characterize the splitting process, we first formalize the computation pipeline. Suppose the pipeline consists of $n$ computational units ([`Pipeline Unit`](../Developer_Guide/Building_a_Pipeline.md#units)), and let the set of units be $V=\{u_1,u_2,\dots,u_n\}$. Each unit $u\in V$ has the following properties:

* Input parameter set $\operatorname{in}(u)$: declared by `input_params`, `input_params_posi` and `input_params_nega`, representing the data items that must be read before the computation of $u$;
* Output parameter set $\operatorname{out}(u)$: declared by `output_params`, representing the data items produced and written into the data cache after the computation of $u$;
* Associated model set $\mathcal{M}(u)$: declared by `onload_model_names`, representing the models that the computation of $u$ depends on.

All parameters constitute the parameter space $\mathcal{P}=\bigcup_{u\in V}\left(\operatorname{in}(u)\cup\operatorname{out}(u)\right)$.

**Definition 1 (Data Dependency Edge)** Let $p\in\mathcal{P}$ be a parameter. If there exist units $u_i,u_j\in V$ such that $p\in\operatorname{out}(u_i)\cap\operatorname{in}(u_j)$, and $u_i$ is the most recent producer of $p$ (i.e., the unit with the latest execution order among all units that produce $p$), then there exists a data dependency edge $(u_i,u_j)$ between $u_i$ and $u_j$, whose semantics is that the computation of $u_j$ must occur after the computation of $u_i$ completes.

Accordingly, the computation pipeline is abstracted as a directed acyclic graph $G=(V,E)$, where $E$ is the set of all data dependency edges.

**Definition 2 (Directly Related Unit)** Given a set of models $\mathcal{W}$ that require gradient backpropagation (specified by `trainable_models` and `lora_base_model`, which are respectively the model components being trained and the model components being trained with LoRA). If a unit $u\in V$ satisfies $\mathcal{M}(u)\cap\mathcal{W}\neq\varnothing$, then $u$ is called a directly related unit, whose computation involves the invocation of trainable models.

**Definition 3 (Computational Graph Splitting Problem)** Given a graph $G=(V,E)$ and a model set $\mathcal{W}$, find a bipartition $(V_1,V_2)$ of $V$ such that $V_1$ is the minimal set that contains all directly related units and satisfies the following closure conditions, with $V_2=V\setminus V_1$:

(C1) Forward closure: if $u\in V_1$ and $(u,v)\in E$, then $v\in V_1$; that is, $V_2$ contains no unit that depends on the outputs of $V_1$;

(C2) Updating-chain closure: for any parameter $p\in\mathcal{P}$, let its updating chain $\mathbf{c}(p)=(u^{(1)},u^{(2)},\dots,u^{(k)})$ be the sequence of all units that produce $p$ in execution order. If $p$ is first consumed at $u^{(i)}$ within $V_1$ and $i<k$, then $u^{(i+1)},\dots,u^{(k)}\in V_1$; that is, the order in which parameters are consumed and updated inside $V_1$ is consistent with the execution order of the whole graph.

Purpose of splitting: the computational results of the units in $V_2$ are independent of the model parameters and can be reused across multiple epochs. Therefore, they are executed in the data preprocessing stage (the first stage) and cached to disk; the computation of the units in $V_1$ requires gradient backpropagation and is executed in the training stage (the second stage).

### Algorithm Design

Consider the set operator $T:2^V\to 2^V$:

$$
T(X)=X\cup F(X)\cup U(X),
$$

where $F$ is the forward reachability operator, and $F(X)$ is the transitive closure of all successor units reachable from $X$ along data dependency edges (in the direction producer $\to$ consumer); $U$ is the updating-chain backtracking operator, defined as follows: for any parameter $p$, if $p$ is first consumed at $u^{(i)}$ within $X$ and $i<k$ ($k$ being the length of the updating chain $\mathbf{c}(p)$), then $U(X)$ contains those units among the subsequent updating units $u^{(i+1)},\dots,u^{(k)}$ of $\mathbf{c}(p)$ that are not in $X$.

**Proposition 1 (Monotonicity and Termination)** $T$ is a monotone operator, i.e., $X\subseteq Y\Rightarrow T(X)\subseteq T(Y)$. Starting from the set of directly related units $X_0=\{u\in V\mid\mathcal{M}(u)\cap\mathcal{W}\neq\varnothing\}$, iterate $X_{k+1}=T(X_k)$. Since $X_k$ is monotonically non-decreasing and $X_k\subseteq V$ ($|V|=n$), the iteration terminates within at most $n$ steps at the least fixed point $X^*$, satisfying $T(X^*)=X^*$.

**Proposition 2 (Consistency)** Let $V_1=X^*$ and $V_2=V\setminus V_1$. Then $(V_1,V_2)$ satisfies conditions (C1) and (C2). From the fixed-point property of $X^*$, $F(X^*)\subseteq X^*$ implies (C1), and $U(X^*)\subseteq X^*$ implies (C2).

### Algorithm Pseudocode

The implementation of the above algorithm is as follows:

```python
def split_pipeline_units(units, model_names):
    # Step 1: initialize X_0, the set of directly related units
    related = {id for id, unit in enumerate(units)
               if unit.onload_model_names is not None
               and any(m in unit.onload_model_names for m in model_names)}

    # Step 2: build the data dependency edges E and the updating chain c(p) of each parameter
    edges = build_edges(units)
    chains = build_chains(units)

    # Step 3: fixed-point iteration X_{k+1} = X_k ∪ F(X_k) ∪ U(X_k)
    while True:
        before = len(related)
        related = forward_reachable(edges, related)       # F(X_k)
        related = updating_units(units, chains, related)  # U(X_k)
        if len(related) == before:
            break

    # Step 4: output the bipartition (V1, V2)
    related_units   = [units[i] for i in sorted(related)]
    unrelated_units = [units[i] for i in range(len(units)) if i not in related]
    return related_units, unrelated_units
```

The construction details of each sub-function are as follows.

`build_edges` scans the unit sequence and maintains a mapping from "parameter $\to$ most recent producer": when the input parameter $p$ of unit $u_j$ already has a producer $u_i$, an edge $(u_i,u_j)$ is recorded; subsequently, the mapping is updated with each output parameter $p$ of $u_j$.

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

`build_chains` records the updating chain $\mathbf{c}(p)$ of each parameter in execution order.

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

`forward_reachable` implements the forward reachability operator $F$, iteratively computing the closure along the producer$\to$consumer direction until no new units are added.

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

`updating_units` implements the updating-chain backtracking operator $U$. For each input parameter $p$ of every unit in the related set $X$, it determines the unit $u^{(i)}$ where $p$ is first consumed within $X$; if $i<k$, all units after $u^{(i)}$ in the updating chain $\mathbf{c}(p)$ are merged into $X$, thereby guaranteeing condition (C2).

```python
def updating_units(units, chains, related):
    # Determine, in execution order (ascending unit id), the unit where p is first consumed in X
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

### Two-Stage Splitting Workflow

After obtaining the bipartition $(V_1,V_2)$, the training framework configures the `Pipeline` according to the task type and delegates execution to `launch_data_process_task` and `launch_training_task`, respectively:

```python
def split_pipeline_units(task, pipe, trainable_models, lora_base_model, ...):
    models_require_backward = []
    if trainable_models is not None:
        models_require_backward += trainable_models.split(",")
    if lora_base_model is not None:
        models_require_backward.append(lora_base_model)

    if task.endswith(":data_process"):           # Stage 1 (data preprocessing)
        other_units, pipe.units = pipe.split_pipeline_units(models_require_backward)
        # pipe.units = V2, executes only computations unrelated to the models (e.g., VAE encoding, text encoding)
        # Optional: append GeneralUnit_RemoveCache to drop redundant cached items and reduce cache size
    elif task.endswith(":train"):                # Stage 2 (training)
        pipe.units, _ = pipe.split_pipeline_units(models_require_backward)
        # pipe.units = V1, executes only computations related to the models
```

**Stage 1 (data preprocessing)** In `torch.no_grad()` mode, iterate over the dataset, execute the units in $V_2$ for each sample, and serialize the intermediate results to disk sharded by process:

```python
for data in dataloader:
    with torch.no_grad():
        cache = model(data)                      # executes only the units in V2
        torch.save(cache, cache_path)            # saves each sample as a .pth file
```

**Stage 2 (training)** The data loader takes the cache produced in the first stage as input, executes the units in $V_1$, and performs forward propagation, backpropagation, and parameter updates:

```python
for data in dataloader:                          # reads the first-stage cache
    with accelerator.accumulate(model):
        loss = model({}, inputs=data)            # executes only the units in V1
        accelerator.backward(loss)               # gradients flow only through trainable models
        optimizer.step()
```

**Proposition 3 (Cache Reusability)** By Definition 3, the units in $V_2$ do not depend on any trainable model, and their computational results are independent of the model parameters $\theta$. Therefore, the cache produced in the first stage can be directly reused in any subsequent epoch without redundant computation. Based on this property, `--dataset_repeat` should be set to 1 in the first stage, while the second stage can keep its original value.
