# 推理改进优化技术

DiffSynth-Studio 旨在以基础框架驱动技术创新。本文以 Inference-time scaling 为例，展示如何基于 DiffSynth-Studio 构建免训练（Training-free）的图像生成增强方案。

Notebook: https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/zh/Research_Tutorial/inference_time_scaling.ipynb

## 1. 图像质量量化

首先，我们需要找到一个指标来量化图像生成模型生成的图像质量。最简单直接的方案是人工打分，但这样做的成本太高，无法大规模使用。不过，收集人工打分后，训练一个图像分类模型来预测人类的打分结果，是完全可行的。PickScore [[1]](https://arxiv.org/abs/2305.01569) 就是这样一个模型，运行下面的代码，将会自动下载并加载 [PickScore 模型](https://modelscope.cn/models/AI-ModelScope/PickScore_v1)。

```python
from modelscope import AutoProcessor, AutoModel
import torch

class PickScore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained("AI-ModelScope/PickScore_v1").eval().to("cuda")

    def forward(self, image, prompt):
        image_inputs = self.processor(images=image, padding=True, truncation=True, max_length=77, return_tensors="pt").to("cuda")
        text_inputs = self.processor(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            image_embs = self.model.get_image_features(**image_inputs).pooler_output
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            text_embs = self.model.get_text_features(**text_inputs).pooler_output
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            score = (text_embs @ image_embs.T).flatten().item()
        return score

reward_model = PickScore()
```

## 2. Inference-time Scaling 技术

Inference-time Scaling [[2]](https://arxiv.org/abs/2504.00294) 是一类有趣的技术，旨在通过增加推理时的计算量来提升生成结果的质量。例如，在语言模型中，[Qwen/Qwen3.5-27B](https://modelscope.cn/models/Qwen/Qwen3.5-27B)、[deepseek-ai/DeepSeek-R1](deepseek-ai/DeepSeek-R1) 等模型通过“思考模式”引导模型花更多时间仔细思考，让回答结果更准确。接下来我们以模型 [black-forest-labs/FLUX.2-klein-4B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-4B) 为例，探讨如何为图像生成模型设计 Inference-time Scaling 方案。

> 在开始前，我们稍微改造了 `Flux2ImagePipeline` 的代码，使其能够根据输入的特定高斯噪声矩阵进行初始化，便于复现结果，详见 [diffsynth/pipelines/flux2_image.py](https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/pipelines/flux2_image.py) 中的 `Flux2Unit_NoiseInitializer`。

运行以下代码，加载模型 [black-forest-labs/FLUX.2-klein-4B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-4B)。

```python
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

pipe = Flux2ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-4B", origin_file_pattern="tokenizer/"),
)
```

用提示词 `"sketch, a cat"` 生成一只素描猫猫，并用 PickScore 模型打分。

```python
def evaluate_noise(noise, pipe, reward_model, prompt):
    # Generate an image and compute the score.
    image = pipe(
        prompt=prompt,
        num_inference_steps=4,
        initial_noise=noise,
        progress_bar_cmd=lambda x: x,
    )
    score = reward_model(image, prompt)
    return score

torch.manual_seed(1)
prompt = "sketch, a cat"
noise = pipe.generate_noise((1, 128, 64, 64), rand_device="cuda", rand_torch_dtype=pipe.torch_dtype)

image_1 = pipe(prompt, num_inference_steps=4, initial_noise=noise)
print("Score:", reward_model(image_1, prompt))
image_1
```

![Image](https://github.com/user-attachments/assets/b6546c6d-b368-4463-b703-d561a9134ba0)

### 2.1 Best-of-N 随机搜索

模型的生成结果具有一定的随机性，如果用不同的随机种子，生成的图像结果也是不同的，有时图像质量高，有时图像质量低。那么，我们有一个简单的 Inference-time scaling 方案：使用多个不同的随机种子分别生成图像，然后利用 PickScore 进行打分，只保留分数最高的那一张。

```python
from tqdm import tqdm

def random_search(base_latents, objective_reward_fn, total_eval_budget):
    # Search for the noise randomly.
    best_noise = base_latents
    best_score = objective_reward_fn(base_latents)
    for it in tqdm(range(total_eval_budget - 1)):
        noise = pipe.generate_noise((1, 128, 64, 64), seed=None)
        score = objective_reward_fn(noise)
        if score > best_score:
            best_score, best_noise = score, noise
    return best_noise

best_noise = random_search(
    base_latents=noise,
    objective_reward_fn=lambda noise: evaluate_noise(noise, pipe, reward_model, prompt),
    total_eval_budget=50,
)
image_2 = pipe(prompt, num_inference_steps=4, initial_noise=best_noise)
print("Score:", reward_model(image_2, prompt))
image_2
```

![Image](https://github.com/user-attachments/assets/b8dba70a-daa8-4368-8f32-a6c150daecb5)

我们可以清晰地看到，经过多次随机搜索后，最终选出的猫猫毛发细节更加丰富，PickScore 分数也有明显提升。但这种暴力的随机搜索效率极低，生成时间成倍增长，且很容易触及质量上限。因此，我们希望能够找到一种更高效的搜索方法，在同等计算预算下达到更高的分数。

### 2.2 SES 搜索

为了突破随机搜索的瓶颈，我们引入了 SES (Spectral Evolution Search) 算法 [[3]](https://arxiv.org/abs/2602.03208)，详细的代码位于 [diffsynth/utils/ses](https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/utils/ses)。

扩散模型生成的图像，很大程度上由初始噪声的低频分量决定。SES 算法通过小波变换将高斯噪声分解，固定高频细节，专门针对低频部分使用交叉熵方法进行演化搜索，能以更高的效率找到优质的初始噪声。

运行下面的代码，即可使用 SES 更高效地搜索最佳的高斯噪声矩阵。

```python
from diffsynth.utils.ses import ses_search

best_noise = ses_search(
    base_latents=noise,
    objective_reward_fn=lambda noise: evaluate_noise(noise, pipe, reward_model, prompt),
    total_eval_budget=50,
)
image_3 = pipe(prompt, num_inference_steps=4, initial_noise=best_noise)
print("Score:", reward_model(image_3, prompt))
image_3
```

![Image](https://github.com/user-attachments/assets/9a3f7598-3812-46d2-b333-cd65e49886ab)

可以观察到，在同样的计算预算下，相比于随机搜索，SES 的结果在 PickScore 得分上取得了显著的提升。“素描猫猫”展现出了更精致的整体构图以及更具层次感的明暗对比。

Inference-time scaling 能够以更长推理时间为代价获得更高的图像质量，那么它生成的图像数据也可以用 DPO [[4]](https://arxiv.org/abs/2311.12908)、差分训练 [[5]](https://arxiv.org/abs/2412.12888) 等方式赋予模型自身，那就是另外一个有趣的探索方向了。
