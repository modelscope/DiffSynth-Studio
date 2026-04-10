from PIL import Image
from typing import Tuple
import math
import torchvision.transforms.functional as TF

class BucketGroup:
    """Manages dynamic batch grouping buckets for image inference."""

    def __init__(
        self,
        bucket_configs: list[tuple[int, int, int, int, int]],
        prioritize_frame_matching: bool = True,
    ):
        """
        Initialize bucket group with predefined configurations.

        Args:
            bucket_configs: List of (batch_size, num_items, num_frames, height, width) tuples
            prioritize_frame_matching: Unused, kept for API compatibility.
        """
        self.bucket_configs = [tuple(b) for b in bucket_configs]

    def find_best_bucket(self, media_shape: tuple[int, int, int, int]) -> tuple[int, int, int, int, int]:
        """
        Find the best matching bucket for given media dimensions.

        Args:
            media_shape: (num_items, num_frames, height, width) of input media

        Returns:
            Best matching bucket as (batch_size, num_items, num_frames, height, width)
        """
        num_items, num_frames, height, width = media_shape
        target_aspect_ratio = height / width

        if num_frames != 1:
            raise ValueError(
                f"Only image inference (num_frames=1) is supported, got num_frames={num_frames}")

        valid_buckets = [
            b for b in self.bucket_configs
            if b[1] == num_items and b[2] == 1
        ]
        if not valid_buckets:
            raise ValueError(
                f"No image buckets found for shape {media_shape}")

        return min(
            valid_buckets,
            key=lambda bucket: abs(
                (bucket[3] / bucket[4]) - target_aspect_ratio)
        )

    def __repr__(self) -> str:
        return (
            f"BucketGroup("
            f"total_buckets={len(self.bucket_configs)}, "
            f"configs={self.bucket_configs})"
        )


def _generate_hw_buckets(base_height=256, base_width=256, step_width=16, step_height=16, max_ratio=4.0) -> list[tuple[int, int, int, int, int]]:
    """Generate dimension buckets based on aspect ratios."""
    buckets = []
    target_pixels = base_height * base_width

    height = target_pixels // step_width
    width = step_width

    while height >= step_height:
        if max(height, width) / min(height, width) <= max_ratio:
            buckets.append((1, 1, 1, height, width))
        if height * (width + step_width) <= target_pixels:
            width += step_width
        else:
            height -= step_height

    return buckets


def generate_video_image_bucket(basesize=256, min_temporal=65, max_temporal=129, bs_img=8, bs_vid=1, bs_mimg=4, min_items=1, max_items=1):
    """Generate bucket configs for image inference.

    Returns:
        List of (batch_size, num_items, num_frames, height, width) tuples.
    """
    assert basesize in [
        256, 512, 768, 1024], f"[generate_video_image_bucket] wrong basesize {basesize}"
    bucket_list = []

    base_bucket_list = _generate_hw_buckets()
    # image
    for _bucket in base_bucket_list:
        bucket = list(_bucket)
        bucket[0] = bs_img
        bucket_list.append(bucket)
    # multiple images
    for num_items in range(min_items, max_items + 1):
        for _bucket in base_bucket_list:
            bucket = list(_bucket)
            bucket[0] = bs_mimg
            bucket[1] = num_items
            bucket_list.append(bucket)
    # spatial resize
    if basesize > 256:
        ratio = basesize // 256

        def resize(bucket, r):
            bucket[-2] *= r
            bucket[-1] *= r
            return bucket
        bucket_list = [resize(bucket, ratio) for bucket in bucket_list]
    return bucket_list


def _dynamic_resize_from_bucket(image: Image, basesize: int = 512):
    def resize_center_crop(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        w, h = img.size  # PIL: (width, height)
        bh, bw = target_size
        scale = max(bh / h, bw / w)
        resize_h, resize_w = math.ceil(h * scale), math.ceil(w * scale)
        img = TF.resize(img, (resize_h, resize_w),
                        interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        img = TF.center_crop(img, target_size)
        return img

    bucket_config = generate_video_image_bucket(
        basesize=basesize, min_temporal=56, max_temporal=56, bs_img=4, bs_vid=4, bs_mimg=8, min_items=2, max_items=2
    )
    bucket_group = BucketGroup(bucket_config)
    img_w, img_h = image.size
    bucket = bucket_group.find_best_bucket((1, 1, img_h, img_w))
    target_height, target_width = bucket[-2], bucket[-1]  # (height, width)
    img_proc = resize_center_crop(image, (target_height, target_width))
    return img_proc
