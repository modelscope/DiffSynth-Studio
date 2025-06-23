from diffsynth import FluxImagePipeline, ModelManager, load_state_dict
from diffsynth.models.lora import FluxLoRAConverter
from diffsynth.pipelines.flux_image import lets_dance_flux
from lora.dataset import LoraDataset
from lora.merger import LoraPatcher
from lora.utils import load_lora
import torch, os
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm


model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda", model_id_list=["FLUX.1-dev"])
pipe = FluxImagePipeline.from_model_manager(model_manager)
pipe.enable_auto_lora()

lora_patcher = LoraPatcher().to(dtype=torch.bfloat16, device="cuda")
lora_patcher.load_state_dict(load_state_dict("models/lora_merger/epoch-3.safetensors"))

dataset = LoraDataset("data/lora/models", "data/lora/lora_dataset_1000.csv", steps_per_epoch=800, loras_per_item=4)

for seed in range(100):
    batch = dataset[0]
    num_lora = torch.randint(1, len(batch), (1,))[0]
    lora_state_dicts = [
        FluxLoRAConverter.align_to_diffsynth_format(load_lora(batch[i]["model_file"], device="cuda")) for i in range(num_lora)
    ]
    image = pipe(
        prompt=batch[0]["text"],
        seed=seed,
    )
    image.save(f"data/lora/lora_outputs/image_{seed}_nolora.jpg")
    for i in range(num_lora):
        image = pipe(
            prompt=batch[0]["text"],
            lora_state_dicts=[lora_state_dicts[i]], 
            lora_patcher=lora_patcher,
            seed=seed,
        )
        image.save(f"data/lora/lora_outputs/image_{seed}_{i}.jpg")
    image = pipe(
        prompt=batch[0]["text"],
        lora_state_dicts=lora_state_dicts, 
        lora_patcher=lora_patcher,
        seed=seed,
    )
    image.save(f"data/lora/lora_outputs/image_{seed}_merger.jpg")
