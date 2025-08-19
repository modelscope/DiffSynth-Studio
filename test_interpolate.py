from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, QwenImageUnit_PromptEmbedder, load_state_dict
import torch, os
from tqdm import tqdm
from diffsynth.models.svd_unet import TemporalTimesteps
from einops import rearrange, repeat



class ValueEncoder(torch.nn.Module):
    def __init__(self, dim_in=256, dim_out=3584, value_emb_length=32):
        super().__init__()
        self.value_emb = TemporalTimesteps(num_channels=dim_in, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.positional_emb = torch.nn.Parameter(torch.randn(1, value_emb_length, dim_out))
        self.proj_value = torch.nn.Linear(dim_in, dim_out)
        self.proj_out = torch.nn.Linear(dim_out, dim_out)
        self.value_emb_length = value_emb_length
        
    def forward(self, value):
        value = value * 1
        emb = self.value_emb(value).to(value.dtype)
        emb = self.proj_value(emb)
        emb = repeat(emb, "b d -> b s d", s=self.value_emb_length)
        emb = emb + self.positional_emb.to(dtype=emb.dtype, device=emb.device)
        emb = torch.nn.functional.silu(emb)
        emb = self.proj_out(emb)
        return emb


class TextInterpolationModel(torch.nn.Module):
    def __init__(self, dim_in=256, dim_out=3584, value_emb_length=32, num_heads=32):
        super().__init__()
        self.to_q = ValueEncoder(dim_in=dim_in, dim_out=dim_out, value_emb_length=value_emb_length)
        self.xk_emb = torch.nn.Parameter(torch.randn(1, 1, dim_out))
        self.yk_emb = torch.nn.Parameter(torch.randn(1, 1, dim_out))
        self.xv_emb = torch.nn.Parameter(torch.randn(1, 1, dim_out))
        self.yv_emb = torch.nn.Parameter(torch.randn(1, 1, dim_out))
        self.to_k = torch.nn.Linear(dim_out, dim_out, bias=False)
        self.to_v = torch.nn.Linear(dim_out, dim_out, bias=False)
        self.to_out = torch.nn.Linear(dim_out, dim_out)
        self.num_heads = num_heads

    def forward(self, value, x, y):
        q = self.to_q(value)
        k = self.to_k(torch.concat([x + self.xk_emb, y + self.yk_emb], dim=1))
        v = self.to_v(torch.concat([x + self.xv_emb, y + self.yv_emb], dim=1))
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h s d -> b s (h d)')
        out = self.to_out(out)
        return out





pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
unit = QwenImageUnit_PromptEmbedder()

dataset_prompt = [
    (
        "超级黑暗的画面，整体在黑暗中，暗无天日，暗淡无光，阴森黑暗，几乎全黑",
        "超级明亮的画面，爆闪，相机过曝，整个画面都是白色的眩光，几乎全是白色",
    ),
]
dataset_tensors = []
for prompt_x, prompt_y in tqdm(dataset_prompt):
    with torch.no_grad():
        x = unit.process(pipe, prompt_x)["prompt_emb"]
        y = unit.process(pipe, prompt_y)["prompt_emb"]
    dataset_tensors.append((x, y))

model = TextInterpolationModel().to(dtype=torch.bfloat16, device="cuda")
model.load_state_dict(load_state_dict("models/interpolate.pth"))

def sample_tokens(emb, p):
    perm = torch.randperm(emb.shape[1])[:max(0, int(emb.shape[1]*p))]
    return emb[:, perm]


def loss_fn(x, y):
    s, l = x.shape[1], y.shape[1]
    x = repeat(x, "b s d -> b s l d", l=l)
    y = repeat(y, "b l d -> b s l d", s=s)
    d = torch.square(x - y).mean(dim=-1)
    loss_x = d.min(dim=1).values.mean()
    loss_y = d.min(dim=2).values.mean()
    return loss_x + loss_y


def get_target(x, y, p):
    x = sample_tokens(x, 1-p)
    y = sample_tokens(y, p)
    return torch.concat([x, y], dim=1)

name = "brightness"
for i in range(6):
    v = i/5
    with torch.no_grad():
        data_id = 0
        x, y = dataset_tensors[data_id]
        x, y = x.to("cuda"), y.to("cuda")
        value = torch.tensor([v], dtype=torch.bfloat16, device="cuda")
        value_emb = model(value, x, y)

    prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
    image = pipe(prompt, seed=0, num_inference_steps=40, extra_prompt_emb=value_emb)
    os.makedirs(f"data/qwen_image_value/{name}", exist_ok=True)
    image.save(f"data/qwen_image_value/{name}/image_{v}.jpg")
