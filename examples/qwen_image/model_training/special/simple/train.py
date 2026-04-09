import torch, accelerate
from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.diffusion import *

class QwenImageTrainingModule(DiffusionTrainingModule):
    def __init__(self, device):
        super().__init__()
        # Load the pipeline
        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
            ],
            tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
        )
        # Switch to training mode
        self.switch_pipe_to_training_mode(
            self.pipe,
            lora_base_model="dit",
            lora_target_modules="to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj",
            lora_rank=32,
        )

    def forward(self, data):
        # Preprocess
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": True,
            "use_gradient_checkpointing_offload": False,
        }
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        # Loss
        loss = FlowMatchSFTLoss(self.pipe, **inputs_shared, **inputs_posi)
        return loss

if __name__ == "__main__":
    accelerator = accelerate.Accelerator(
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    dataset = UnifiedDataset(
        base_path="data/example_image_dataset",
        metadata_path="data/example_image_dataset/metadata.csv",
        repeat=50,
        data_file_keys="image",
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path="data/example_image_dataset",
            height=512,
            width=512,
            height_division_factor=16,
            width_division_factor=16,
        )
    )
    model = QwenImageTrainingModule(accelerator.device)
    model_logger = ModelLogger(
        output_path="models/toy_model",
        remove_prefix_in_ckpt="pipe.dit.",
    )
    launch_training_task(
        accelerator, dataset, model, model_logger,
        learning_rate=1e-5, num_epochs=1,
    )
