import torch, os
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, VideoDataset, launch_training_task
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(self, model_paths):
        super().__init__()
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=[ModelConfig(path=path) for path in model_paths],
        )
        self.pipe.freeze_except([])
        self.pipe.dit = self.add_lora_to_model(self.pipe.dit, target_modules="q,k,v,o,ffn.0,ffn.2".split(","), lora_alpha=16)
        
        
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


    
def add_general_parsers(parser):
    parser.add_argument("--dataset_base_path", type=str, default="", help="Base path of the Dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default="", required=True, help="Metadata path of the Dataset.")
    parser.add_argument("--height", type=int, default=None, help="Image or video height. Leave `height` and `width` None to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Image or video width. Leave `height` and `width` None to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in metadata. Separated by commas.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times the dataset is repeated in each epoch.")
    parser.add_argument("--model_paths", type=str, default="", help="Model paths to be loaded. Separated by commas.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    return parser


if __name__ == "__main__":
    dataset = VideoDataset(
        base_path="data/pixabay100/train",
        metadata_path="data/pixabay100/metadata_example.csv",
        height=480, width=832,
        data_file_keys=["video"],
        repeat=400,
    )
    model = WanTrainingModule([
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    launch_training_task(model, dataset)
    
