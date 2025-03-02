import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from functools import partial
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[Union[float, Tuple[float, float, float]]] = None
    interpolation: Optional[str] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None
    use_timm: bool = False


class ResizeMaxSize(nn.Module):

    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[1:]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
        return img


def _convert_to_rgb_or_rgba(image):
    if image.mode == 'RGBA':
        return image
    else:
        return image.convert('RGB')

# def transform_and_split(merged, transform_fn, normalize_fn):
#     transformed = transform_fn(merged)
#     crop_img, crop_label = torch.split(transformed, [3,1], dim=0)

#     # crop_img = _convert_to_rgb(crop_img)
#     crop_img = normalize_fn(ToTensor()(crop_img))
#     return crop_img, crop_label

class MaskAwareNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.normalize = Normalize(mean=mean, std=std)

    def forward(self, tensor):
        if tensor.shape[0] == 4:
            return torch.cat([self.normalize(tensor[:3]), tensor[3:]], dim=0)
        else:
            return self.normalize(tensor)

def image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()
    normalize = MaskAwareNormalize(mean=mean, std=std)
    if is_train:
        aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}
        use_timm = aug_cfg_dict.pop('use_timm', False)
        if use_timm:
            assert False, "not tested for augmentation with mask"
            from timm.data import create_transform  # timm can still be optional
            if isinstance(image_size, (tuple, list)):
                assert len(image_size) >= 2
                input_size = (3,) + image_size[-2:]
            else:
                input_size = (3, image_size, image_size)
            # by default, timm aug randomly alternates bicubic & bilinear for better robustness at inference time
            aug_cfg_dict.setdefault('interpolation', 'random')
            aug_cfg_dict.setdefault('color_jitter', None)  # disable by default
            train_transform = create_transform(
                input_size=input_size,
                is_training=True,
                hflip=0.,
                mean=mean,
                std=std,
                re_mode='pixel',
                **aug_cfg_dict,
            )
        else:
            train_transform = Compose([
                _convert_to_rgb_or_rgba,
                ToTensor(),
                RandomResizedCrop(
                    image_size,
                    scale=aug_cfg_dict.pop('scale'),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                normalize,
            ])
            if aug_cfg_dict:
                warnings.warn(f'Unused augmentation cfg items, specify `use_timm` to use ({list(aug_cfg_dict.keys())}).')
        return train_transform
    else:
        transforms = [
            _convert_to_rgb_or_rgba,
            ToTensor(),
        ]
        if resize_longest_max:
            transforms.extend([
                ResizeMaxSize(image_size, fill=fill_color)
            ])
        else:
            transforms.extend([
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ])
        transforms.extend([
            normalize,
        ])
        return Compose(transforms)


# def image_transform_region(
#         image_size: int,
#         is_train: bool,
#         mean: Optional[Tuple[float, ...]] = None,
#         std: Optional[Tuple[float, ...]] = None,
#         resize_longest_max: bool = False,
#         fill_color: int = 0,
#         aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
# ):
#     mean = mean or OPENAI_DATASET_MEAN
#     if not isinstance(mean, (list, tuple)):
#         mean = (mean,) * 3

#     std = std or OPENAI_DATASET_STD
#     if not isinstance(std, (list, tuple)):
#         std = (std,) * 3

#     if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
#         # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
#         image_size = image_size[0]

#     if isinstance(aug_cfg, dict):
#         aug_cfg = AugmentationCfg(**aug_cfg)
#     else:
#         aug_cfg = aug_cfg or AugmentationCfg()
#     normalize = Normalize(mean=mean, std=std)
#     if is_train:
#         aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}
        
#         transform = Compose([
#                 RandomResizedCrop(
#                     image_size,
#                     scale=aug_cfg_dict.pop('scale'),
#                     interpolation=InterpolationMode.BICUBIC,
#                 ),
#                 ])
#         train_transform = Compose([
#             partial(transform_and_split, transform_fn=transform,normalize_fn=normalize)
#         ])
#         return train_transform
#     else:
#         if resize_longest_max:
#             transform = [
#                 ResizeMaxSize(image_size, fill=fill_color)
#             ]
#             val_transform = Compose([
#                 partial(transform_and_split, transform_fn=transform,normalize_fn=normalize),
#             ])
#         else:
#             transform = [
#                 Resize(image_size, interpolation=InterpolationMode.BICUBIC),
#                 CenterCrop(image_size),
#             ]
#             val_transform = Compose([
#                 partial(transform_and_split, transform_fn=transform,normalize_fn=normalize),
#             ])
#         return val_transform