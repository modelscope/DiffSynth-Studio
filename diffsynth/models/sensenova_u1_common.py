import torch, math
import torchvision.transforms as T
from PIL import Image


# patch_size (16) * merge_size (2). Output height/width must be divisible by this.
PATCH_SIZE = 32

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SYSTEM_MESSAGE_FOR_GEN = (
    "You are an image generation and editing assistant that accurately understands and executes "
    "user intent.\n\nYou support two modes:\n\n1. Think Mode:\nIf the task requires reasoning, you "
    "MUST start with a <think></think> block. Put all reasoning inside the block using plain text. "
    "DO NOT include any image tags. Keep it reasonable and directly useful for producing the final "
    "image.\n\n2. Non-Think Mode:\nIf no reasoning is needed, directly produce the final image.\n\n"
    "Task Types:\n\nA. Text-to-Image Generation:\n"
    "- Generate a high-quality image based on the user's description.\n"
    "- Ensure visual clarity, semantic consistency, and completeness.\n"
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification or transformation.\n"
    "- The result can be an edited image or a new image based on the reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n"
    "- For any visible text in the image, follow the language specified for the rendered text in "
    "the user's description, not the language of the prompt. If no language is specified, use the "
    "user's input language."
)


IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

# Plain literal (not a special token) marking where an input image belongs in a prompt.
IMAGE_PLACEHOLDER = "<image>"

# Suffix that closes an empty reasoning block, used when think mode is off.
NON_THINK_PREFIX = "<think>\n\n</think>\n\n"

# Opens a reasoning block that the model closes itself, used when think mode is on.
THINK_PREFIX = "<think>\n"


def build_conversation_prompt(prompt, system_message="", append_text=""):
    """Assemble a neo1_0 template prompt.

    The system block is dropped entirely when `system_message` is empty, and the assistant
    block is left open so `append_text` continues the assistant turn.
    """
    ret = "" if not system_message else f"<|im_start|>system\n{system_message}<|im_end|>\n"
    ret += f"<|im_start|>user\n{prompt}<|im_end|>\n"
    ret += "<|im_start|>assistant\n"
    return ret + append_text


def patchify(images, patch_size, channel_first=False):
    h, w = images.shape[2] // patch_size, images.shape[3] // patch_size
    x = images.reshape(shape=(images.shape[0], 3, h, patch_size, w, patch_size))
    if channel_first:
        x = torch.einsum('nchpwq->nhwcpq', x)
    else:
        x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(images.shape[0], h * w, patch_size ** 2 * 3))
    return x


def unpatchify(x, patch_size, h=None, w=None):
    if h is None or w is None:
        h = w = int(x.shape[1] ** .5)
    else:
        h = h // patch_size
        w = w // patch_size
    x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    images = x.reshape(shape=(x.shape[0], 3, h * patch_size, w * patch_size))
    return images


def create_block_causal_mask(index: torch.Tensor):
    """Block-wise causal mask: tokens sharing the same time index attend to each other."""
    L = index.size(0)
    idx_i = index.unsqueeze(1).expand(L, L)
    idx_j = index.unsqueeze(0).expand(L, L)

    arange = torch.arange(L, device=index.device)
    mask = (idx_j == idx_i) | (arange.unsqueeze(0) <= arange.unsqueeze(1))

    return torch.where(mask[None, None, :, :] > 0, torch.tensor(0.0), torch.tensor(float('-inf')))


def build_abs_positions_from_grid_hw(grid_hw: torch.Tensor, device=None):
    """Compute per-patch (x, y) coordinates from a (B, 2) tensor of per-image (H, W) grids."""
    device = grid_hw.device
    B = grid_hw.shape[0]

    H = grid_hw[:, 0]
    W = grid_hw[:, 1]
    N = H * W
    N_total = N.sum()

    patch_to_sample = torch.repeat_interleave(torch.arange(B, device=device), N)

    patch_id_within_image = torch.arange(N_total, device=device)
    patch_id_within_image = patch_id_within_image - torch.cumsum(
        torch.cat([torch.tensor([0], device=device), N[:-1]]), dim=0
    )[patch_to_sample]

    W_per_patch = W[patch_to_sample]
    abs_x = patch_id_within_image % W_per_patch
    abs_y = patch_id_within_image // W_per_patch

    return abs_x, abs_y


def build_image_token_block(num_patch_token):
    """Assemble the token run that stands in for one input image."""
    return IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_patch_token + IMG_END_TOKEN


def build_thw_indexes(input_ids, img_start_token_id, img_context_token_id, grid_hw=None, merge_size=2):
    """Compute per-token (t, h, w) position indexes for a possibly interleaved sequence.

    Image tokens share a single time index so they form one block, and carry their patch
    grid coordinates in h/w. A pure text sequence reduces to `t = arange`, `h = w = 0`.
    """
    img_start_shift = torch.cat([
        torch.zeros(1, dtype=torch.long, device=input_ids.device),
        (input_ids == img_start_token_id).long(),
    ], dim=0)[:-1]
    not_img_token = (input_ids != img_context_token_id).long()
    t_indexes = (img_start_shift + not_img_token).cumsum(0) - 1
    h_indexes = torch.zeros_like(t_indexes)
    w_indexes = torch.zeros_like(t_indexes)

    if grid_hw is not None:
        selected = input_ids == img_context_token_id
        if selected.long().sum() > 0:
            abs_pos_w, abs_pos_h = build_abs_positions_from_grid_hw(grid_hw // merge_size)
            h_indexes[selected] = abs_pos_h.to(t_indexes.device, t_indexes.dtype)
            w_indexes[selected] = abs_pos_w.to(t_indexes.device, t_indexes.dtype)
    return torch.stack([t_indexes, h_indexes, w_indexes], dim=0)


def round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int = 32, min_pixels: int = 65536, max_pixels: int = 4194304):
    """Rescale so H/W are divisible by `factor` and total pixels fall inside [min_pixels, max_pixels]."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def preprocess_pixel_values(pixel_values: torch.Tensor, patch_size: int = 16):
    c, h, w = pixel_values.shape
    grid_h = h // patch_size
    grid_w = w // patch_size

    flatten_pixel_values = (
        pixel_values.view(c, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_h * grid_w, c * patch_size ** 2)
    )

    grid_hw = torch.tensor([[grid_h, grid_w]], device=pixel_values.device)
    return flatten_pixel_values, grid_hw


def load_image_native(
    image,
    patch_size: int = 16,
    downsample_ratio: float = 0.5,
    min_pixels: int = 65536,
    max_pixels: int = 4194304,
    upscale: bool = False,
):
    """Load and preprocess an image: RGB convert, smart-resize, ImageNet normalize, patchify."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background.convert("RGB")
    else:
        image = image.convert("RGB")

    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)

    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    resized_height, resized_width = smart_resize(
        image.height, image.width,
        factor=int(patch_size // downsample_ratio),
        min_pixels=min_pixels, max_pixels=max_pixels,
    )
    new_image = image.resize((resized_width, resized_height))
    pixel_values, grid_hw = preprocess_pixel_values(
        transform(new_image).to(torch.float32), patch_size=patch_size
    )
    return pixel_values, grid_hw
