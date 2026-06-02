import os
from collections import defaultdict
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

ImageInput = Union[str, os.PathLike, Image.Image]

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".pgm", ".png", ".ppm", ".tif", ".tiff", ".webp"}

LPIPS_NET_CHOICES = ("alex", "vgg", "squeeze")


def _list_image_files(path: Union[str, os.PathLike]):
    path = os.fspath(path)
    if not os.path.isdir(path):
        raise ValueError(f"Expected a directory for LPIPS, got: {path}")
    files = []
    for entry in sorted(os.listdir(path)):
        full = os.path.join(path, entry)
        if os.path.isfile(full) and os.path.splitext(entry)[1].lower() in IMAGE_EXTENSIONS:
            files.append(full)
    if not files:
        raise ValueError(f"No images found under {path}.")
    return files


def _pair_directories_by_stem(dir_a, dir_b):
    files_a = _list_image_files(dir_a)
    files_b = _list_image_files(dir_b)
    by_stem_a = defaultdict(list)
    for f in files_a:
        by_stem_a[Path(f).stem].append(f)
    by_stem_b = defaultdict(list)
    for f in files_b:
        by_stem_b[Path(f).stem].append(f)
    common = sorted(set(by_stem_a.keys()) & set(by_stem_b.keys()))
    if not common:
        raise ValueError(f"No matching filename stems between {dir_a} and {dir_b}.")
    pairs = []
    for stem in common:
        pairs.append((sorted(by_stem_a[stem])[0], sorted(by_stem_b[stem])[0]))
    return pairs


def _open_rgb(image: ImageInput) -> Image.Image:
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    if not isinstance(image, Image.Image):
        raise TypeError(f"LPIPS expects PIL images or image paths, got {type(image)}.")
    return image.convert("RGB")


class _AlexFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.slice1.add_module("0", nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2))
        self.slice1.add_module("1", nn.ReLU(inplace=True))
        self.slice2.add_module("2", nn.MaxPool2d(kernel_size=3, stride=2))
        self.slice2.add_module("3", nn.Conv2d(64, 192, kernel_size=5, padding=2))
        self.slice2.add_module("4", nn.ReLU(inplace=True))
        self.slice3.add_module("5", nn.MaxPool2d(kernel_size=3, stride=2))
        self.slice3.add_module("6", nn.Conv2d(192, 384, kernel_size=3, padding=1))
        self.slice3.add_module("7", nn.ReLU(inplace=True))
        self.slice4.add_module("8", nn.Conv2d(384, 256, kernel_size=3, padding=1))
        self.slice4.add_module("9", nn.ReLU(inplace=True))
        self.slice5.add_module("10", nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.slice5.add_module("11", nn.ReLU(inplace=True))

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return [h1, h2, h3, h4, h5]


class _VGG16Features(nn.Module):
    def __init__(self):
        super().__init__()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        cfg = [
            (1, 0, nn.Conv2d(3, 64, 3, padding=1)),
            (1, 1, nn.ReLU(inplace=True)),
            (1, 2, nn.Conv2d(64, 64, 3, padding=1)),
            (1, 3, nn.ReLU(inplace=True)),
            (2, 4, nn.MaxPool2d(2, 2)),
            (2, 5, nn.Conv2d(64, 128, 3, padding=1)),
            (2, 6, nn.ReLU(inplace=True)),
            (2, 7, nn.Conv2d(128, 128, 3, padding=1)),
            (2, 8, nn.ReLU(inplace=True)),
            (3, 9, nn.MaxPool2d(2, 2)),
            (3, 10, nn.Conv2d(128, 256, 3, padding=1)),
            (3, 11, nn.ReLU(inplace=True)),
            (3, 12, nn.Conv2d(256, 256, 3, padding=1)),
            (3, 13, nn.ReLU(inplace=True)),
            (3, 14, nn.Conv2d(256, 256, 3, padding=1)),
            (3, 15, nn.ReLU(inplace=True)),
            (4, 16, nn.MaxPool2d(2, 2)),
            (4, 17, nn.Conv2d(256, 512, 3, padding=1)),
            (4, 18, nn.ReLU(inplace=True)),
            (4, 19, nn.Conv2d(512, 512, 3, padding=1)),
            (4, 20, nn.ReLU(inplace=True)),
            (4, 21, nn.Conv2d(512, 512, 3, padding=1)),
            (4, 22, nn.ReLU(inplace=True)),
            (5, 23, nn.MaxPool2d(2, 2)),
            (5, 24, nn.Conv2d(512, 512, 3, padding=1)),
            (5, 25, nn.ReLU(inplace=True)),
            (5, 26, nn.Conv2d(512, 512, 3, padding=1)),
            (5, 27, nn.ReLU(inplace=True)),
            (5, 28, nn.Conv2d(512, 512, 3, padding=1)),
            (5, 29, nn.ReLU(inplace=True)),
        ]
        for slice_idx, orig_idx, module in cfg:
            getattr(self, f"slice{slice_idx}").add_module(str(orig_idx), module)

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return [h1, h2, h3, h4, h5]


class _Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x)),
            ],
            dim=1,
        )


class _SqueezeNet11Features(nn.Module):
    def __init__(self):
        super().__init__()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.slice6 = nn.Sequential()
        self.slice7 = nn.Sequential()
        self.slice1.add_module("0", nn.Conv2d(3, 64, kernel_size=3, stride=2))
        self.slice1.add_module("1", nn.ReLU(inplace=True))
        self.slice2.add_module("2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        self.slice2.add_module("3", _Fire(64, 16, 64, 64))
        self.slice2.add_module("4", _Fire(128, 16, 64, 64))
        self.slice3.add_module("5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        self.slice3.add_module("6", _Fire(128, 32, 128, 128))
        self.slice3.add_module("7", _Fire(256, 32, 128, 128))
        self.slice4.add_module("8", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        self.slice4.add_module("9", _Fire(256, 48, 192, 192))
        self.slice5.add_module("10", _Fire(384, 48, 192, 192))
        self.slice6.add_module("11", _Fire(384, 64, 256, 256))
        self.slice7.add_module("12", _Fire(512, 64, 256, 256))

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        h6 = self.slice6(h5)
        h7 = self.slice7(h6)
        return [h1, h2, h3, h4, h5, h6, h7]


_NET_CONFIG = {
    "alex": {"factory": _AlexFeatures, "channels": (64, 192, 384, 256, 256)},
    "vgg": {"factory": _VGG16Features, "channels": (64, 128, 256, 512, 512)},
    "squeeze": {"factory": _SqueezeNet11Features, "channels": (64, 128, 256, 384, 384, 512, 512)},
}


class _ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.tensor([-0.030, -0.088, -0.188]).view(1, 3, 1, 1))
        self.register_buffer("scale", torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.shift) / self.scale


class _NetLinLayer(nn.Module):
    def __init__(self, chn_in, use_dropout=True):
        super().__init__()
        layers = []
        if use_dropout:
            layers.append(nn.Dropout())
        layers.append(nn.Conv2d(chn_in, 1, kernel_size=1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def _normalize_tensor(x, eps=1e-10):
    norm = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm + eps)


def _spatial_average(x):
    return x.mean(dim=(2, 3), keepdim=True)


class LPIPSModel(nn.Module):
    def __init__(self, net: str = "alex", use_dropout: bool = True):
        super().__init__()
        if net not in _NET_CONFIG:
            raise ValueError(f"net must be one of {LPIPS_NET_CHOICES}, got {net!r}")
        self.net_name = net
        self.scaling_layer = _ScalingLayer()
        self.net = _NET_CONFIG[net]["factory"]()
        chns = _NET_CONFIG[net]["channels"]
        for i, chn in enumerate(chns):
            setattr(self, f"lin{i}", _NetLinLayer(chn, use_dropout=use_dropout))
        self.num_layers = len(chns)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, in0, in1):
        in0 = self.scaling_layer(in0)
        in1 = self.scaling_layer(in1)
        feats0 = self.net(in0)
        feats1 = self.net(in1)
        val = 0
        for i in range(self.num_layers):
            diff = (_normalize_tensor(feats0[i]) - _normalize_tensor(feats1[i])) ** 2
            lin = getattr(self, f"lin{i}")
            val = val + _spatial_average(lin(diff))
        return val.view(-1)


class LPIPSCompute(nn.Module):
    def __init__(
        self,
        model: LPIPSModel,
        device: Union[str, torch.device] = "cpu",
        batch_size: int = 16,
        target_size: int = 512,
    ):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.target_size = target_size
        self._resize_transform = transforms.Compose(
            [
                transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
            ]
        )
        self._raw_transform = transforms.ToTensor()
        self.to(device)

    @property
    def device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _to_tensor(self, image: Image.Image, do_resize: bool) -> torch.Tensor:
        transform = self._resize_transform if do_resize else self._raw_transform
        x = transform(image).clamp(0.0, 1.0) * 2.0 - 1.0
        return x

    @torch.no_grad()
    def _compute_pair(self, img_a: Image.Image, img_b: Image.Image, do_resize: bool) -> float:
        x0 = self._to_tensor(img_a, do_resize).unsqueeze(0).to(self.device)
        x1 = self._to_tensor(img_b, do_resize).unsqueeze(0).to(self.device)
        return float(self.model(x0, x1).item())

    @torch.no_grad()
    def _compute_pairs(self, pairs, do_resize: bool) -> float:
        scores = []
        batch_size = max(1, self.batch_size)
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            xs0 = torch.stack([self._to_tensor(_open_rgb(a), do_resize) for a, _ in chunk]).to(self.device)
            xs1 = torch.stack([self._to_tensor(_open_rgb(b), do_resize) for _, b in chunk]).to(self.device)
            scores.append(self.model(xs0, xs1).detach().cpu())
        merged = torch.cat(scores, dim=0)
        return float(merged.mean().item())

    @staticmethod
    def _is_dir(value) -> bool:
        return isinstance(value, (str, os.PathLike)) and os.path.isdir(os.fspath(value))

    @staticmethod
    def _is_image_input(value) -> bool:
        if isinstance(value, Image.Image):
            return True
        if isinstance(value, (str, os.PathLike)):
            return os.path.isfile(os.fspath(value))
        return False

    def compute(self, image_a, image_b) -> float:
        a_is_dir = self._is_dir(image_a)
        b_is_dir = self._is_dir(image_b)
        if a_is_dir != b_is_dir:
            raise ValueError("LPIPS.compute requires both inputs to be directories or both to be single images.")

        if a_is_dir:
            pairs = _pair_directories_by_stem(image_a, image_b)
            sizes = set()
            for path_a, path_b in pairs:
                with Image.open(path_a) as ia, Image.open(path_b) as ib:
                    sizes.add(ia.size)
                    sizes.add(ib.size)
            do_resize = len(sizes) > 1
            return self._compute_pairs(pairs, do_resize=do_resize)

        if not (self._is_image_input(image_a) and self._is_image_input(image_b)):
            raise ValueError("LPIPS.compute inputs must be image paths, PIL images, or directories.")
        img_a = _open_rgb(image_a)
        img_b = _open_rgb(image_b)
        do_resize = img_a.size != img_b.size
        return self._compute_pair(img_a, img_b, do_resize=do_resize)

    def forward(self, image_a, image_b):
        return self.compute(image_a, image_b)
