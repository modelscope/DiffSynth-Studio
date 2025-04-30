import torch
from diffsynth import ModelManager
from .flux_image_pipeline import FluxImagePipelineAll2All

class FluxDecoder:

    def __init__(self, flux_all2all_modelpath, flux_path, device='cuda', torch_dtype=torch.bfloat16):
        self.device = device
        self.torch_dtype = torch_dtype
        self.pipe, self.adapter = self.get_pipe(flux_all2all_modelpath, flux_path, device, torch_dtype)

    def get_pipe(self, flux_all2all_modelpath, flux_path, device="cuda", torch_dtype=torch.bfloat16):
        model_manager = ModelManager(torch_dtype=torch_dtype, device=device)
        model_manager.load_models([
            f"{flux_path}/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
            f"{flux_path}/FLUX/FLUX.1-dev/text_encoder_2",
            f"{flux_path}/FLUX/FLUX.1-dev/ae.safetensors",
            f"{flux_path}/FLUX/FLUX.1-dev/flux1-dev.safetensors"
        ])

        state_dict = torch.load(flux_all2all_modelpath, weights_only=True, map_location='cpu')
        adapter_states = ['0.weight', '0.bias', '1.weight', '1.bias', '3.weight', '3.bias', '4.weight', '4.bias']
        adapter_state_dict = {}
        for key in adapter_states:
            adapter_state_dict[key] = state_dict.pop(key)

        in_channel = 3584
        out_channel = 4096
        expand_ratio = 1
        adapter = torch.nn.Sequential(torch.nn.Linear(in_channel, out_channel * expand_ratio),
                                      torch.nn.LayerNorm(out_channel * expand_ratio), torch.nn.ReLU(),
                                      torch.nn.Linear(out_channel * expand_ratio, out_channel),
                                      torch.nn.LayerNorm(out_channel))
        adapter.load_state_dict(adapter_state_dict)
        adapter.to(device, dtype=torch_dtype)

        pipe = FluxImagePipelineAll2All.from_model_manager(model_manager)
        pipe.dit.load_state_dict(state_dict)

        return pipe, adapter

    @torch.no_grad()
    def decode_image_embeds(self,
                            output_image_embeddings,
                            height=512,
                            width=512,
                            num_inference_steps=50,
                            seed=42,
                            negative_prompt="",
                            cfg_scale=1.0,
                            **pipe_kwargs):
        output_image_embeddings = output_image_embeddings.to(device=self.device, dtype=self.torch_dtype)
        image_embed = self.adapter(output_image_embeddings)
        image = self.pipe(prompt="",
                          image_embed=image_embed,
                          num_inference_steps=num_inference_steps,
                          embedded_guidance=3.5,
                          negative_prompt=negative_prompt,
                          cfg_scale=cfg_scale,
                          height=height,
                          width=width,
                          seed=seed,
                          **pipe_kwargs)
        return image
