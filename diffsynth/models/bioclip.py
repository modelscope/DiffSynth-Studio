import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel as HFCLIPModel, CLIPConfig


class BioCLIPv2Model(HFCLIPModel):
    def __init__(self):
        super().__init__(self._build_config())

    @staticmethod
    def _build_config():
        return CLIPConfig(
            projection_dim=768,
            logit_scale_init_value=2.6592,
            text_config={
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "max_position_embeddings": 77,
                "vocab_size": 49408,
                "hidden_act": "gelu",
                "layer_norm_eps": 1e-5,
                "projection_dim": 768,
                "bos_token_id": 0,
                "eos_token_id": 2,
                "pad_token_id": 1,
            },
            vision_config={
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_attention_heads": 16,
                "num_hidden_layers": 24,
                "image_size": 224,
                "patch_size": 14,
                "hidden_act": "gelu",
                "layer_norm_eps": 1e-5,
                "projection_dim": 768,
            },
        )


class BioCLIPv2Compute(torch.nn.Module):
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, model: BioCLIPv2Model, tokenizer, max_length: int = 77):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def _preprocess_images(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        images = [img.convert("RGB") for img in images]
        pixel_values = torch.stack([self.image_transform(img) for img in images])
        return pixel_values.to(device=self.device, dtype=self.dtype)

    def _tokenize(self, text):
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(
            text, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in tokens.items()}

    @torch.no_grad()
    def get_image_features(self, images):
        pixel_values = self._preprocess_images(images)
        features = self.model.get_image_features(pixel_values=pixel_values)
        return torch.nn.functional.normalize(features, dim=-1)

    @torch.no_grad()
    def get_text_features(self, text):
        tokens = self._tokenize(text)
        features = self.model.get_text_features(**tokens)
        return torch.nn.functional.normalize(features, dim=-1)

    @torch.no_grad()
    def forward(self, text: str | list[str], images):
        if isinstance(text, str):
            text = [text]
        if isinstance(images, Image.Image):
            images = [images]
        if len(text) == 1 and len(images) > 1:
            text = text * len(images)
        if len(images) == 1 and len(text) > 1:
            images = images * len(text)
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(text)
        scores = (text_features * image_features).sum(dim=-1)
        scores = self.model.logit_scale.exp() * scores
        return scores

    @torch.no_grad()
    def similarity_matrix(self, text: str | list[str], images):
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(text)
        scores = text_features @ image_features.T
        scores = self.model.logit_scale.exp() * scores
        return scores
