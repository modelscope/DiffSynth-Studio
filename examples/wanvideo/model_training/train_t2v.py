import torch, os, json
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, VideoDataset, launch_training_task, wan_parser
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(self, model_paths, task="train_lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32):
        super().__init__()
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=[ModelConfig(path=path) for path in model_paths],
        )
        self.pipe.scheduler.set_timesteps(1000, training=True)
        if task == "train_lora":
            self.pipe.freeze_except([])
            self.pipe.dit = self.add_lora_to_model(self.pipe.dit, target_modules=lora_target_modules.split(","), lora_rank=lora_rank)
        else:
            self.pipe.freeze_except(["dit"])
        
    def forward_preprocess(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": True,
            "cfg_merge": False,
        }
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    def forward(self, data):
        inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    dataset = VideoDataset(args=args)
    model = WanTrainingModule(json.loads(args.model_paths), task=args.task, lora_target_modules=args.lora_target_modules, lora_rank=args.lora_rank)
    launch_training_task(model, dataset, args=args)
