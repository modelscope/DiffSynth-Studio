import torch
import lightning as pl
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, SequenceParallel, PrepareModuleInput, PrepareModuleOutput
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module
from lightning.pytorch.strategies import ModelParallelStrategy
from diffsynth import ModelManager, WanVideoPipeline, save_video
from tqdm import tqdm
from modelscope import snapshot_download



class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, tasks=[]):
        self.tasks = tasks

    def __getitem__(self, data_id):
        return self.tasks[data_id]

    def __len__(self):
        return len(self.tasks)


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model_manager = ModelManager(device="cpu")
        model_manager.load_models(
            [
                [
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
                ],
                "models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
                "models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
            ],
            torch_dtype=torch.bfloat16,
        )
        self.pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")

    def configure_model(self):
        tp_mesh = self.device_mesh["tensor_parallel"]
        plan = {
            "text_embedding.0": ColwiseParallel(),
            "text_embedding.2": RowwiseParallel(),
            "time_projection.1": ColwiseParallel(output_layouts=Replicate()),
            "text_embedding.0": ColwiseParallel(),
            "text_embedding.2": RowwiseParallel(),
            "blocks.0": PrepareModuleInput(
                input_layouts=(Replicate(), None, None, None),
                desired_input_layouts=(Replicate(), None, None, None),
            ),
            "head": PrepareModuleInput(
                input_layouts=(Replicate(), None),
                desired_input_layouts=(Replicate(), None),
                use_local_output=True,
            )
        }
        self.pipe.dit = parallelize_module(self.pipe.dit, tp_mesh, plan)
        for block_id, block in enumerate(self.pipe.dit.blocks):
            layer_tp_plan = {
                "self_attn": PrepareModuleInput(
                    input_layouts=(Shard(1), Replicate()),
                    desired_input_layouts=(Shard(1), Shard(0)),
                ),
                "self_attn.q": SequenceParallel(),
                "self_attn.k": SequenceParallel(),
                "self_attn.v": SequenceParallel(),
                "self_attn.norm_q": SequenceParallel(),
                "self_attn.norm_k": SequenceParallel(),
                "self_attn.attn": PrepareModuleInput(
                    input_layouts=(Shard(1), Shard(1), Shard(1)),
                    desired_input_layouts=(Shard(2), Shard(2), Shard(2)),
                ),
                "self_attn.o": RowwiseParallel(input_layouts=Shard(2), output_layouts=Replicate()),

                "cross_attn": PrepareModuleInput(
                    input_layouts=(Shard(1), Replicate()),
                    desired_input_layouts=(Shard(1), Replicate()),
                ),
                "cross_attn.q": SequenceParallel(),
                "cross_attn.k": SequenceParallel(),
                "cross_attn.v": SequenceParallel(),
                "cross_attn.norm_q": SequenceParallel(),
                "cross_attn.norm_k": SequenceParallel(),
                "cross_attn.attn": PrepareModuleInput(
                    input_layouts=(Shard(1), Shard(1), Shard(1)),
                    desired_input_layouts=(Shard(2), Shard(2), Shard(2)),
                ),
                "cross_attn.o": RowwiseParallel(input_layouts=Shard(2), output_layouts=Replicate(), use_local_output=False),

                "ffn.0": ColwiseParallel(input_layouts=Shard(1)),
                "ffn.2": RowwiseParallel(output_layouts=Replicate()),

                "norm1": SequenceParallel(use_local_output=True),
                "norm2": SequenceParallel(use_local_output=True),
                "norm3": SequenceParallel(use_local_output=True),
                "gate": PrepareModuleInput(
                    input_layouts=(Shard(1), Replicate(), Replicate()),
                    desired_input_layouts=(Replicate(), Replicate(), Replicate()),
                )
            }
            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_tp_plan,
            )


    def test_step(self, batch):
        data = batch[0]
        data["progress_bar_cmd"] = tqdm if self.local_rank == 0 else lambda x: x
        output_path = data.pop("output_path")
        with torch.no_grad(), torch.inference_mode(False):
            video = self.pipe(**data)
        if self.local_rank == 0:
            save_video(video, output_path, fps=15, quality=5)


if __name__ == "__main__":
    snapshot_download("Wan-AI/Wan2.1-T2V-14B", local_dir="models/Wan-AI/Wan2.1-T2V-14B")
    dataloader = torch.utils.data.DataLoader(
        ToyDataset([
            {
                "prompt": "一名宇航员身穿太空服，面朝镜头骑着一匹机械马在火星表面驰骋。红色的荒凉地表延伸至远方，点缀着巨大的陨石坑和奇特的岩石结构。机械马的步伐稳健，扬起微弱的尘埃，展现出未来科技与原始探索的完美结合。宇航员手持操控装置，目光坚定，仿佛正在开辟人类的新疆域。背景是深邃的宇宙和蔚蓝的地球，画面既科幻又充满希望，让人不禁畅想未来的星际生活。",
                "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                "num_inference_steps": 50,
                "seed": 0,
                "tiled": False,
                "output_path": "video1.mp4",
            },
            {
                "prompt": "一名宇航员身穿太空服，面朝镜头骑着一匹机械马在火星表面驰骋。红色的荒凉地表延伸至远方，点缀着巨大的陨石坑和奇特的岩石结构。机械马的步伐稳健，扬起微弱的尘埃，展现出未来科技与原始探索的完美结合。宇航员手持操控装置，目光坚定，仿佛正在开辟人类的新疆域。背景是深邃的宇宙和蔚蓝的地球，画面既科幻又充满希望，让人不禁畅想未来的星际生活。",
                "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                "num_inference_steps": 50,
                "seed": 1,
                "tiled": False,
                "output_path": "video2.mp4",
            },
        ]),
        collate_fn=lambda x: x
    )
    model = LitModel()
    trainer = pl.Trainer(accelerator="gpu", devices=torch.cuda.device_count(), strategy=ModelParallelStrategy())
    trainer.test(model, dataloader)