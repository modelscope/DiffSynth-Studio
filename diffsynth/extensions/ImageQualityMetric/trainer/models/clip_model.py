from dataclasses import dataclass
from transformers import CLIPModel as HFCLIPModel
from transformers import AutoTokenizer

from torch import nn, einsum

from .base_model import BaseModelConfig

from transformers import CLIPConfig
from typing import Any, Optional, Tuple, Union
import torch

from .cross_modeling import Cross_model

import json, os

class XCLIPModel(HFCLIPModel):
    def __init__(self, config: CLIPConfig):
        super().__init__(config)
    
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # pooled_output = text_outputs[1]
        # text_features = self.text_projection(pooled_output)
        last_hidden_state = text_outputs[0]
        text_features = self.text_projection(last_hidden_state)

        pooled_output = text_outputs[1]
        text_features_EOS = self.text_projection(pooled_output)


        # del last_hidden_state, text_outputs
        # gc.collect()

        return text_features, text_features_EOS

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # pooled_output = vision_outputs[1]  # pooled_output
        # image_features = self.visual_projection(pooled_output)
        last_hidden_state = vision_outputs[0]
        image_features = self.visual_projection(last_hidden_state)  

        return image_features



@dataclass
class ClipModelConfig(BaseModelConfig):
    _target_: str = "diffsynth.extensions.QualityMetric.trainer.models.clip_model.CLIPModel"
    pretrained_model_name_or_path: str ="checkpoints/clip-vit-base-patch32"


class CLIPModel(nn.Module):
    def __init__(self, ckpt, config_file=False):
        super().__init__()
        if config_file is None:
            self.model = XCLIPModel.from_pretrained(ckpt)
        else:
            with open(os.path.join(ckpt, "config.json"), "r", encoding="utf-8") as f:
                config = json.load(f)
            config = CLIPConfig(**config)
            self.model = XCLIPModel._from_config(config)
        self.cross_model = Cross_model(dim=1024, layer_num=4, heads=16)
    
    def get_text_features(self, *args, **kwargs):
        return self.model.get_text_features(*args, **kwargs)

    def get_image_features(self, *args, **kwargs):
        return self.model.get_image_features(*args, **kwargs)

    def forward(self, text_inputs=None, image_inputs=None, condition_inputs=None):
        outputs = ()

        text_f, text_EOS = self.model.get_text_features(text_inputs) # B*77*1024
        outputs += text_EOS,

        image_f = self.model.get_image_features(image_inputs.half()) # 2B*257*1024
        condition_f, _ = self.model.get_text_features(condition_inputs) # B*5*1024

        sim_text_condition = einsum('b i d, b j d -> b j i', text_f, condition_f)
        sim_text_condition = torch.max(sim_text_condition, dim=1, keepdim=True)[0]
        sim_text_condition = sim_text_condition / sim_text_condition.max()
        mask = torch.where(sim_text_condition > 0.01, 0, float('-inf')) # B*1*77

        mask = mask.repeat(1,image_f.shape[1],1) # B*257*77
        bc = int(image_f.shape[0]/2)

        sim0 = self.cross_model(image_f[:bc,:,:], text_f,mask.half())
        sim1 = self.cross_model(image_f[bc:,:,:], text_f,mask.half())
        outputs += sim0[:,0,:],
        outputs += sim1[:,0,:],

        return outputs

    @property
    def logit_scale(self):
        return self.model.logit_scale

    def save(self, path):
        self.model.save_pretrained(path)

