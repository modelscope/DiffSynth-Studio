from pathlib import Path
from typing import Iterable, Union
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.models.inception import InceptionA, InceptionC, InceptionE

ImageInput = Union[str, Path, Image.Image]

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".pgm", ".png", ".ppm", ".tif", ".tiff", ".webp"}

def _resolve_device(device: Union[str, torch.device, None]):
    if device is None:
        device = "cuda" if _is_cuda_usable("cuda", warn=False) else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not _is_cuda_usable(device, warn=True):
        return torch.device("cpu")
    return device


def _is_cuda_usable(device: Union[str, torch.device], warn: bool = True):
    try:
        if not torch.cuda.is_available():
            if warn:
                warnings.warn("CUDA was requested but torch.cuda.is_available() is False. FID will run on CPU instead.", RuntimeWarning)
            return False
        torch.empty(1, device=device)
        return True
    except Exception as error:
        if warn:
            warnings.warn(f"CUDA was requested but cannot be initialized ({error}). FID will run on CPU instead.", RuntimeWarning)
        return False


def _image_files(path: Union[str, Path]):
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension for FID: {path}")
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"FID path does not exist: {path}")
    files = [item for item in sorted(path.rglob("*")) if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise ValueError(f"No images found under {path}.")
    return files


class _ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images: Iterable[ImageInput], transform):
        self.images = list(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if not isinstance(image, Image.Image):
            raise TypeError(f"FID expects PIL images or image paths, but received {type(image)}.")
        return self.transform(image.convert("RGB"))


class _InceptionFeatures(nn.Module):
    def __init__(self, weights_path: str = None, pretrained: bool = True, use_fid_inception: bool = True):
        super().__init__()
        if use_fid_inception and weights_path is not None:
            model = _fid_inception_v3(weights_path)
            self.normalize_input = "fid"
        elif use_fid_inception and weights_path is None:
            warnings.warn(
                "FID-specific Inception weights were not provided. Falling back to torchvision Inception weights; "
                "scores are useful for relative comparisons but are not directly comparable to standard pytorch-fid values.",
                RuntimeWarning,
            )
            weights = Inception_V3_Weights.DEFAULT if pretrained else None
            model = inception_v3(weights=weights, aux_logits=True, init_weights=False)
            model.fc = nn.Identity()
            self.normalize_input = "imagenet" if pretrained else None
        else:
            weights = Inception_V3_Weights.DEFAULT if pretrained else None
            model = inception_v3(weights=weights, aux_logits=True, init_weights=False)
            model.fc = nn.Identity()
            self.normalize_input = "imagenet" if pretrained else None
        model.eval()
        self.model = model

    def forward(self, images):
        if self.normalize_input == "fid":
            images = 2 * images - 1
        elif self.normalize_input == "imagenet":
            mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
            images = (images - mean) / std
        return self.model(images)


def _fid_inception_v3(weights_path: str):
    model = inception_v3(weights=None, aux_logits=False, num_classes=1008, init_weights=False)
    model.Mixed_5b = _FIDInceptionA(192, pool_features=32)
    model.Mixed_5c = _FIDInceptionA(256, pool_features=64)
    model.Mixed_5d = _FIDInceptionA(288, pool_features=64)
    model.Mixed_6b = _FIDInceptionC(768, channels_7x7=128)
    model.Mixed_6c = _FIDInceptionC(768, channels_7x7=160)
    model.Mixed_6d = _FIDInceptionC(768, channels_7x7=160)
    model.Mixed_6e = _FIDInceptionC(768, channels_7x7=192)
    model.Mixed_7b = _FIDInceptionE1(1280)
    model.Mixed_7c = _FIDInceptionE2(2048)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.fc = nn.Identity()
    return model


class _FIDInceptionA(InceptionA):
    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 1)


class _FIDInceptionC(InceptionC):
    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 1)


class _FIDInceptionE1(InceptionE):
    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = torch.cat([self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)], 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = torch.cat([self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)], 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 1)


class _FIDInceptionE2(InceptionE):
    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = torch.cat([self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)], 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = torch.cat([self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)], 1)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 1)


class FIDModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, device: Union[str, torch.device] = "cpu", batch_size: int = 50, num_workers: int = 0):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [
                transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        )
        self.to(device)

    @classmethod
    def from_pretrained(
        cls,
        weights_path: str = None,
        pretrained: bool = True,
        device: Union[str, torch.device] = "cpu",
        batch_size: int = 50,
        num_workers: int = 0,
        use_fid_inception: bool = True,
    ):
        if isinstance(weights_path, (list, tuple)):
            if len(weights_path) == 1:
                weights_path = weights_path[0]
            elif len(weights_path) == 0:
                raise FileNotFoundError(
                    "FID weights were not found. Please check the ModelScope model id and file pattern."
                )
            else:
                raise ValueError(
                    f"FID expects a single weights file, but got {len(weights_path)} paths: {weights_path}"
                )
        device = _resolve_device(device)
        model = _InceptionFeatures(weights_path=weights_path, pretrained=pretrained, use_fid_inception=use_fid_inception).to(device).eval()
        return cls(model=model, device=device, batch_size=batch_size, num_workers=num_workers)

    @property
    def device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _as_images(self, images):
        if isinstance(images, (str, Path)):
            return _image_files(images)
        if isinstance(images, Image.Image):
            return [images]
        return list(images)

    @torch.no_grad()
    def get_activations(self, images, batch_size: int = None, num_workers: int = None):
        images = self._as_images(images)
        batch_size = self.batch_size if batch_size is None else batch_size
        num_workers = self.num_workers if num_workers is None else num_workers
        dataset = _ImageDataset(images, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False, num_workers=num_workers)
        activations = []
        self.model.eval()
        for batch in dataloader:
            batch = batch.to(self.device)
            features = self.model(batch)
            if isinstance(features, tuple):
                features = features[0]
            if features.ndim == 4:
                features = F.adaptive_avg_pool2d(features, output_size=(1, 1)).flatten(1)
            activations.append(features.detach().cpu().to(torch.float64))
        return torch.cat(activations, dim=0)

    def statistics(self, images, batch_size: int = None, num_workers: int = None):
        activations = self.get_activations(images, batch_size=batch_size, num_workers=num_workers)
        return self.activation_statistics(activations)

    @staticmethod
    def activation_statistics(activations):
        activations = activations.to(torch.float64)
        mean = activations.mean(dim=0)
        centered = activations - mean
        if activations.shape[0] <= 1:
            covariance = torch.zeros((activations.shape[1], activations.shape[1]), dtype=torch.float64)
        else:
            covariance = centered.T @ centered / (activations.shape[0] - 1)
        return mean, covariance

    @staticmethod
    def _sqrtm_psd(matrix, eps: float = 1e-10):
        matrix = (matrix + matrix.T) * 0.5
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        eigenvalues = eigenvalues.clamp_min(eps).sqrt()
        return (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.T

    @classmethod
    def frechet_distance(cls, mean1, covariance1, mean2, covariance2, eps: float = 1e-6):
        mean1 = mean1.to(torch.float64)
        covariance1 = covariance1.to(torch.float64)
        mean2 = mean2.to(torch.float64)
        covariance2 = covariance2.to(torch.float64)
        diff = mean1 - mean2
        offset = torch.eye(covariance1.shape[0], dtype=torch.float64) * eps
        sqrt_cov1 = cls._sqrtm_psd(covariance1 + offset)
        covmean = cls._sqrtm_psd(sqrt_cov1 @ (covariance2 + offset) @ sqrt_cov1)
        distance = diff.dot(diff) + torch.trace(covariance1) + torch.trace(covariance2) - 2 * torch.trace(covmean)
        return distance.clamp_min(0)

    def compute(self, reference_images, generated_images, batch_size: int = None, num_workers: int = None):
        mean1, covariance1 = self.statistics(reference_images, batch_size=batch_size, num_workers=num_workers)
        mean2, covariance2 = self.statistics(generated_images, batch_size=batch_size, num_workers=num_workers)
        return self.frechet_distance(mean1, covariance1, mean2, covariance2)

    def forward(self, reference_images, generated_images):
        return self.compute(reference_images, generated_images)
