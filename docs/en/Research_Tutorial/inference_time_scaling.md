# Inference Optimization Techniques

DiffSynth-Studio aims to drive technological innovation through its foundational framework. This article demonstrates how to build a training-free image generation enhancement solution using DiffSynth-Studio, taking Inference-time scaling as an example.

Notebook: https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Research_Tutorial/inference_time_scaling.ipynb

## 1. Image Quality Quantification

First, we need to find an indicator to quantify image quality from generation models. Manual scoring is the most straightforward solution but too costly for large-scale applications. However, after collecting manual scores, training an image classification model to predict human scoring is completely feasible. PickScore [[1]](https://arxiv.org/abs/2305.01569) is such a model. Running the following code will automatically download and load the [PickScore model](https://modelscope.cn/models/AI-ModelScope/PickScore_v1).

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

## 2. Inference-time Scaling Techniques

Inference-time Scaling [[2]](https://arxiv.org/abs/2504.00294) is an interesting technique aiming to improve generation quality by increasing computational costs during inference. For example, in language models, models like [Qwen/Qwen3.5-27B](https://modelscope.cn/models/Qwen/Qwen3.5-27B) and [deepseek-ai/DeepSeek-R1](deepseek-ai/DeepSeek-R1) use "thinking mode" to guide the model to spend more time considering results more carefully, producing more accurate answers. Next, we'll use the [black-forest-labs/FLUX.2-klein-4B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-4B) model as an example to explore how to design Inference-time Scaling solutions for image generation models.

> Before starting, we slightly modified the `Flux2ImagePipeline` code to allow initialization with specific Gaussian noise matrices for result reproducibility. See `Flux2Unit_NoiseInitializer` in [diffsynth/pipelines/flux2_image.py](https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/pipelines/flux2_image.py).

Run the following code to load the [black-forest-labs/FLUX.2-klein-4B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-4B) model.

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

Generate a sketch cat image using the prompt `"sketch, a cat"` and score it with the PickScore model.

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

### 2.1 Best-of-N Random Search

Model generation results have inherent randomness. Different random seeds produce different images - sometimes high quality, sometimes low. This leads to a simple Inference-time scaling solution: generate images using multiple random seeds, score them with PickScore, and retain only the highest-scoring image.

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

We can clearly see that after multiple random searches, the final selected cat image shows richer fur details and significantly improved PickScore. However, this brute-force random search is extremely inefficient - generation time multiplies while easily hitting quality limits. Therefore, we need a more efficient search method that achieves higher scores within the same computational budget.

### 2.2 SES Search

To overcome random search limitations, we introduce the Spectral Evolution Search (SES) algorithm [[3]](https://arxiv.org/abs/2602.03208). Detailed code is available at [diffsynth/utils/ses](https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/utils/ses).

Image generation in diffusion models is largely determined by low-frequency components in the initial noise. The SES algorithm decomposes Gaussian noise through wavelet transforms, fixes high-frequency details, and applies an evolution search using the cross-entropy method specifically on low-frequency components to find optimal initial noise with higher efficiency.

Run the following code to perform efficient best Gaussian noise matrix search using SES.

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

Observing the results, under the same computational budget, SES achieves significantly higher PickScore compared to random search. The "sketch cat" demonstrates more refined overall composition and more layered contrast between light and shadow.

Inference-time scaling can achieve higher image quality at the cost of longer inference time. The generated image data can then be used to train the model itself through methods like DPO [[4]](https://arxiv.org/abs/2311.12908) or differential training [[5]](https://arxiv.org/abs/2412.12888), opening another interesting research direction.
