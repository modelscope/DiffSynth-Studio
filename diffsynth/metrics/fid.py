import csv
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
from zipfile import ZipFile
from typing import Union

import torch

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..models.fid import FIDModel, IMAGE_EXTENSIONS
from .base import Metric


class FIDMetric(Metric):
    DEFAULT_REFERENCE_NAME = "coco_2014_caption_validation"
    DEFAULT_REFERENCE_DATASET_ID = "modelscope/coco_2014_caption"
    DEFAULT_REFERENCE_METADATA_URL = "https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/coco_2014_caption/val2014.csv.zip"
    DEFAULT_REFERENCE_SPLIT = "validation"

    def __init__(self, model: FIDModel):
        super().__init__()
        self.model = model

    @staticmethod
    def default_weights_config():
        return ModelConfig(
            model_id="diffusionTry/weights-inception-2015-12-05-6726825d",
            origin_file_pattern="weights-inception-2015-12-05-6726825d.pth",
        )

    @staticmethod
    def default_reference_root():
        base_path = os.environ.get("DIFFSYNTH_DATA_BASE_PATH", "./data")
        return Path(base_path) / "fid_reference" / FIDMetric.DEFAULT_REFERENCE_NAME

    @staticmethod
    def _image_files(path: Union[str, Path]):
        path = Path(path)
        if not path.exists():
            return []
        return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)

    @staticmethod
    def _download_file(url: str, path: Path, timeout: int = 60, retries: int = 3):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            return path
        temp_path = path.with_suffix(path.suffix + ".tmp")
        last_error = None
        for _ in range(retries):
            try:
                with urlopen(url, timeout=timeout) as response, open(temp_path, "wb") as file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        file.write(chunk)
                temp_path.replace(path)
                return path
            except Exception as error:
                last_error = error
                if temp_path.exists():
                    temp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {last_error}")

    @staticmethod
    def _metadata_rows(metadata_zip_path: Path):
        with ZipFile(metadata_zip_path) as archive:
            csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV file found in {metadata_zip_path}.")
            with archive.open(csv_names[0]) as file:
                reader = csv.DictReader(line.decode("utf-8") for line in file)
                rows = []
                seen = set()
                for row in reader:
                    url = row.get("image", "")
                    if not url or url in seen:
                        continue
                    rows.append(row)
                    seen.add(url)
                return rows

    @staticmethod
    def _image_filename(row: dict, index: int):
        url_path = urlparse(row["image"]).path
        name = Path(url_path).name
        if name and Path(name).suffix.lower() in IMAGE_EXTENSIONS:
            return name
        image_id = row.get("image_id") or f"{index:08d}"
        return f"{image_id}.jpg"

    @classmethod
    def download_reference_dir(
        cls,
        local_dir: Union[str, Path] = None,
        max_images: int = None,
        force: bool = False,
        metadata_url: str = None,
        timeout: int = 60,
        retries: int = 3,
        verbose: bool = True,
    ):
        """
        Download the default COCO 2014 caption validation reference images.

        The ModelScope dataset stores a small CSV archive whose image column
        points to ModelScope OSS image URLs. This helper downloads that metadata
        and materializes the referenced real images as a normal image directory.
        """

        root = Path(local_dir) if local_dir is not None else cls.default_reference_root()
        images_dir = root / "images"
        metadata_dir = root / "metadata"
        metadata_url = cls.DEFAULT_REFERENCE_METADATA_URL if metadata_url is None else metadata_url
        existing = cls._image_files(images_dir)
        manifest_path = root / "reference_manifest.json"
        if not force and existing:
            if max_images is not None and len(existing) >= max_images:
                return str(images_dir)
            if max_images is None and manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as file:
                    manifest = json.load(file)
                if manifest.get("max_images") is None and len(existing) >= manifest.get("image_count", 0):
                    return str(images_dir)

        metadata_zip_path = metadata_dir / "val2014.csv.zip"
        cls._download_file(metadata_url, metadata_zip_path, timeout=timeout, retries=retries)
        rows = cls._metadata_rows(metadata_zip_path)
        if max_images is not None:
            rows = rows[:max_images]
        if not rows:
            raise ValueError("No reference images were found in the COCO 2014 caption metadata.")

        images_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        for index, row in enumerate(rows):
            image_path = images_dir / cls._image_filename(row, index)
            if not force and image_path.exists() and image_path.stat().st_size > 0:
                downloaded += 1
                continue
            cls._download_file(row["image"], image_path, timeout=timeout, retries=retries)
            downloaded += 1
            if verbose and downloaded % 100 == 0:
                print(f"Downloaded {downloaded}/{len(rows)} FID reference images to {images_dir}")

        manifest = {
            "name": cls.DEFAULT_REFERENCE_NAME,
            "dataset_id": cls.DEFAULT_REFERENCE_DATASET_ID,
            "split": cls.DEFAULT_REFERENCE_SPLIT,
            "metadata_url": metadata_url,
            "max_images": max_images,
            "image_count": len(rows),
            "images_dir": str(images_dir),
        }
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)
        return str(images_dir)

    @classmethod
    def default_reference_dir(
        cls,
        local_dir: Union[str, Path] = None,
        max_images: int = None,
        download: bool = True,
        **download_kwargs,
    ):
        root = Path(local_dir) if local_dir is not None else cls.default_reference_root()
        images_dir = root / "images"
        existing = cls._image_files(images_dir)
        if existing:
            if max_images is not None and len(existing) >= max_images:
                return str(images_dir)
            manifest_path = root / "reference_manifest.json"
            if max_images is None and manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as file:
                    manifest = json.load(file)
                if manifest.get("max_images") is None and len(existing) >= manifest.get("image_count", 0):
                    return str(images_dir)
        if not download:
            raise FileNotFoundError(
                f"FID reference directory does not exist: {images_dir}. "
                "Call FIDMetric.download_reference_dir(...) first or pass your own reference directory."
            )
        return cls.download_reference_dir(local_dir=root, max_images=max_images, **download_kwargs)

    @classmethod
    def from_pretrained(
        cls,
        weights_config: Union[ModelConfig, str] = None,
        pretrained: bool = True,
        device: Union[str, torch.device] = get_device_type(),
        batch_size: int = 50,
        num_workers: int = 0,
        use_fid_inception: bool = True,
    ):
        if weights_config is None and use_fid_inception:
            weights_config = cls.default_weights_config()
        weights_config = cls.resolve_model_config(weights_config) if weights_config is not None else None
        model = FIDModel.from_pretrained(
            weights_path=None if weights_config is None else weights_config.path,
            pretrained=pretrained,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            use_fid_inception=use_fid_inception,
        )
        return cls(model)

    @torch.no_grad()
    def compute(self, reference_images, generated_images, batch_size: int = None, num_workers: int = None):
        score = self.model.compute(reference_images, generated_images, batch_size=batch_size, num_workers=num_workers)
        return self.tensor_to_float(score)

    @torch.no_grad()
    def compute_with_default_reference(
        self,
        generated_images,
        reference_dir: Union[str, Path] = None,
        max_reference_images: int = None,
        batch_size: int = None,
        num_workers: int = None,
        download_kwargs: dict = None,
    ):
        reference_dir = self.default_reference_dir(
            local_dir=reference_dir,
            max_images=max_reference_images,
            **({} if download_kwargs is None else download_kwargs),
        )
        return self.compute(reference_dir, generated_images, batch_size=batch_size, num_workers=num_workers)

    def statistics(self, images, batch_size: int = None, num_workers: int = None):
        return self.model.statistics(images, batch_size=batch_size, num_workers=num_workers)

    def forward(self, reference_images, generated_images):
        return self.compute(reference_images, generated_images)
