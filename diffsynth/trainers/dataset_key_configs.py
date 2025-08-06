from easydict import EasyDict
from PIL import Image
import imageio
import torch


"""
types:
- raw: raw info which does not need any processing, example: "string"
- int: integer number to represent a value, example: 123
- float: floating point number to represent a value, example: 123.45
- image: image file, postfix: "jpg", "jpeg", "png", "webp"
- video: video file, postfix: "mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"
- tensor: pytorch tensor, postfix: "pt"
"""

# shared dataset key configuration for all models
shared_config = EasyDict(__name__='Datatype Config: Base')
shared_config.prompt = 'raw'


# dataset key configuration for Flux.1-dev model
flux_config = EasyDict(__name__='Datatype Config: Flux.1-dev')
flux_config.update(shared_config)
flux_config.image = 'image'

flux_config.kontext_images = 'image'

flux_config.ipadapter_images = 'image'

flux_config.eligen_entity_prompts = 'raw'
flux_config.eligen_entity_masks = 'image'

flux_config.infinityou_id_image = 'image'
flux_config.infinityou_guidance = 'float'

flux_config.step1x_reference_image = 'image'

flux_config.nexus_gen_reference_image = 'image'

flux_config.value_controller_inputs = 'float'

flux_config.input_latents = 'tensor'

flux_config.controlnet_image = 'image'
flux_config.controlnet_inpaint_mask = 'image'
flux_config.controlnet_processor_id = 'raw'


# dataset key configuration for qwen-image
qwen_image_config = EasyDict(__name__='Datatype Config: Qwen-Image')
qwen_image_config.update(shared_config)
qwen_image_config.image = 'image'


# dataset key configuration for Wan model
wan_config = EasyDict(__name__='Datatype Config: Wan')
wan_config.update(shared_config)
wan_config.video = 'video'

wan_config.motion_bucket_id = 'int'

wan_config.input_image = 'image'
wan_config.end_image = 'image'

wan_config.control_video = 'video'

wan_config.camera_control_direction = 'raw'
wan_config.camera_control_speed = 'float'

wan_config.reference_image = 'image'

wan_config.vace_video = 'video'
wan_config.vace_reference_image = 'image'


def get_default_config(model_name):
    """
    Get the default dataset key configuration for the given model name.
    :param model_name: Name of the model
    :return: EasyDict containing the default dataset key configuration
    """
    if model_name.lower() == 'flux':
        return flux_config
    elif model_name.lower() == 'qwen-image':
        return qwen_image_config
    elif model_name.lower() == 'wan':
        return wan_config
    else:
        return shared_config

def raw_loader(value):
    """
    Load a raw value.
    :param value: The raw value to load
    :return: The loaded raw value
    """
    return value


def int_loader(value):
    """
    Load an integer value.
    :param value: The integer value to load
    :return: The loaded integer value
    """
    return int(value)


def float_loader(value):
    """
    Load a floating point value.
    :param value: The floating point value to load
    :return: The loaded floating point value
    """
    return float(value)


def image_loader(file_path):
    """
    Load an image file.
    :param value: The image file path to load
    :return: The loaded image
    """
    return Image.open(file_path).convert('RGB')


def video_loader(file_path):
    """
    Load a video file.
    :param value: The video file path to load
    :return: The loaded video
    """
    reader = imageio.get_reader(file_path)
    num_frames = int(reader.count_frames())
    frames = []
    for frame_id in range(num_frames):
        frame = reader.get_data(frame_id)
        frame = Image.fromarray(frame)
        frames.append(frame)
    reader.close()
    return frames

def tensor_loader(file_path):
    """
    Load a PyTorch tensor file.
    :param file_path: The tensor file path to load
    :return: The loaded tensor
    """
    return torch.load(file_path, map_location='cpu')

def get_loader(data_type):
    """
    Get the loader function for the given data type.
    :param data_type: The data type to get the loader for
    :return: The loader function
    """
    if data_type == 'raw':
        return raw_loader
    elif data_type == 'int':
        return int_loader
    elif data_type == 'float':
        return float_loader
    elif data_type == 'image':
        return image_loader
    elif data_type == 'video':
        return video_loader
    elif data_type == 'tensor':
        return tensor_loader
    else:
        raise ValueError(f"Unsupported data type: {data_type}")