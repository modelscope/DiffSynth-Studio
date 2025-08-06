from typing import Optional, Tuple, List, Dict, Any, Union, Set, Callable
import imageio, os, torch, warnings, torchvision, argparse, json
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from .dataset_key_configs import get_default_config, get_loader

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path: str = None,
        metadata_path: str = None,
        max_pixels: int = 1920 * 1080,
        height: int = None,
        width: int = None,
        height_division_factor: int = 16,
        width_division_factor: int = 16,
        default_key_model: str = "flux",
        input_configs: str = None,
        file_extensions: Tuple[str, ...] = ("jpg", "mp4"),
        generated_target_key: str = "image",
        repeat: int = 1,
    ):
        self.base_path = base_path if base_path is not None else ""
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.file_extensions = file_extensions
        self.generated_target_key = generated_target_key
        self.repeat = repeat
        self.keyconfigs = self.parse_keyconfigs(input_configs, default_key_model)
        data = self.load_meta(metadata_path)
        self.data = self.preprocess(data)

        if height is not None and width is not None:
            print("Fixed resolution. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        else:
            print("Dynamic resolution enabled.")
            self.dynamic_resolution = True


    def parse_keyconfigs(self, input_configs, default_key_model):
        if input_configs is None:
            print("No input configs provided, invalid dataset.")
            return {}
        default_configs = get_default_config(default_key_model)
        keyconfigs = {}
        for item in input_configs.split(","):
            if ":" in item:
                key, value = item.split(":", 1)
                keyconfigs[key] = value
            else:
                keyconfigs[item] = default_configs[item]
        print(f"Using dataset key configurations: {keyconfigs}")
        return keyconfigs


    def load_meta(self, metadata_path, base_path=None):
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            metadata = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
        else:
            metadata = pd.read_csv(metadata_path)
            metadata = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        print(f"successfully loaded {len(metadata)} metadata from {metadata_path}.")
        return metadata


    def generate_metadata(self, folder):
        file_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.file_extensions:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            file_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata[self.generated_target_key] = file_list
        metadata["prompt"] = prompt_list
        return metadata


    def convert_to_absolute_path(self, path):
        if isinstance(path, list) or isinstance(path, tuple):
            return [os.path.join(self.base_path, p) for p in path]
        else:
            return os.path.join(self.base_path, path)


    def check_file_existence(self, path):
        if isinstance(path, list) or isinstance(path, tuple):
            for p in path:
                assert os.path.exists(p), f"file {p} does not exist."
        else:
            assert os.path.exists(path), f"file {path} does not exist."


    def preprocess(self, data):
        required_keys = list(self.keyconfigs.keys())
        file_keys = [k for k in required_keys if self.keyconfigs[k] in ("image", "video", "tensor")]
        new_data = []
        for cur_data in tqdm(data):
            try:
                # fetch all required keys
                cur_data = {k: cur_data[k] for k in required_keys}
                # convert file paths to absolute paths and check existence
                for file_key in file_keys:
                    cur_data[file_key] = self.convert_to_absolute_path(cur_data[file_key])
                    # self.check_file_existence(cur_data[file_key])
                # add to filtered data
                new_data.append(cur_data)
            except:
                continue
        print(f"get {len(new_data)} valid data from total {len(data)} metadata.")
        return new_data


    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width


    def parse_image(self, image):
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image


    def parse_video(self, video):
        for i in range(len(video)):
            video[i] = self.crop_and_resize(video[i], *self.get_height_width(video[i]))
        return video


    def load_data(self, item, data_type):
        loader = get_loader(data_type)
        if isinstance(item, list) or isinstance(item, tuple):
            return [loader(p) for p in item]
        else:
            return loader(item)


    def type_parser(self, item, data_type):
        if data_type in ("raw", "int", "float"):
            return item
        elif data_type == "image":
            if isinstance(item, list) or isinstance(item, tuple):
                return [self.parse_image(img) for img in item]
            else:
                return self.parse_image(item)
        elif data_type == "video":
            return self.parse_video(item)
        elif data_type == "tensor":
            # TODO: implement tensor parsing
            return item
        else:
            return item


    def __getitem__(self, data_id):
        max_retries = 10
        while True:
            data = self.data[data_id % len(self.data)].copy()
            try:
                for key in data.keys():
                    data_type = self.keyconfigs.get(key, "raw")
                    item = self.load_data(data[key], data_type)
                    item = self.type_parser(item, data_type)
                    data[key] = item
                return data
            except:
                warnings.warn(f"Error loading data with id {data_id}. Replacing with another data.")
                data_id = torch.randint(0, len(self), (1,)).item()
                max_retries -= 1
                if max_retries <= 0:
                    warnings.warn("Max retries reached. Returning None.")
                    return None


    def __len__(self):
        return len(self.data) * self.repeat


class ImageDataset(BaseDataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        default_key_model="flux", input_configs="prompt:raw,image:image",
        file_extensions=("jpg", "jpeg", "png", "webp"),
        generated_target_key="image", repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            input_configs = args.dataset_input_configs if args.dataset_input_configs else "prompt:raw,image:image"
            repeat = args.dataset_repeat

        super().__init__(
            base_path=base_path,
            metadata_path=metadata_path,
            max_pixels=max_pixels,
            height=height,
            width=width,
            height_division_factor=height_division_factor,
            width_division_factor=width_division_factor,
            default_key_model=default_key_model,
            input_configs=input_configs,
            file_extensions=file_extensions,
            generated_target_key=generated_target_key,
            repeat=repeat,
        )

    def parse_image(self, image):
        return self.crop_and_resize(image, *self.get_height_width(image))


class VideoDataset(BaseDataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        default_key_model="wan", input_configs="prompt:raw,video:video",
        file_extensions=("jpg", "jpeg", "png", "webp", "mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        generated_target_key="video", repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            input_configs = args.dataset_input_configs if args.dataset_input_configs else "prompt:raw,video:video"
            repeat = args.dataset_repeat
            num_frames = args.num_frames
        
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        super().__init__(
            base_path=base_path,
            metadata_path=metadata_path,
            max_pixels=max_pixels,
            height=height,
            width=width,
            height_division_factor=height_division_factor,
            width_division_factor=width_division_factor,
            default_key_model=default_key_model,
            input_configs=input_configs,
            file_extensions=file_extensions,
            generated_target_key=generated_target_key,
            repeat=repeat,
        )


    def parse_video(self, video):
        num_frames = self.get_num_frames(video)
        video = video[:num_frames]
        for i in range(len(video)):
            video[i] = self.crop_and_resize(video[i], *self.get_height_width(video[i]))
        return video


    def parse_image(self, image):
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return [image]


    def get_num_frames(self, video):
        num_frames = self.num_frames
        if len(video) < num_frames:
            num_frames = len(video)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames


def general_dataset_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--dataset_input_configs", type=str, default=None, help="Data file keys and data types in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")

    return parser


def video_dataset_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to sample from each video.")
    return parser
