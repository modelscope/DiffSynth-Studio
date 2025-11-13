import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Tuple, Optional, List
from einops import rearrange



MEMORY_LAYOUT = {
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}


def attention(
    q,
    k,
    v,
    mode="torch",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    max_seqlen_q=None,
    batch_size=1,
):
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]

    if mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)

    x = post_attn_layout(x)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


class CausalConv1d(nn.Module):

    def __init__(self, chan_in, chan_out, kernel_size=3, stride=1, dilation=1, pad_mode="replicate", **kwargs):
        super().__init__()

        self.pad_mode = pad_mode
        padding = (kernel_size - 1, 0)  # T
        self.time_causal_padding = padding

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)



class FaceEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads=int, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.num_heads = num_heads
        self.conv1_local = CausalConv1d(in_dim, 1024 * num_heads, 3, stride=1)
        self.norm1 = nn.LayerNorm(hidden_dim // 8, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(1024, 1024, 3, stride=2)
        self.conv3 = CausalConv1d(1024, 1024, 3, stride=2)

        self.out_proj = nn.Linear(1024, hidden_dim)
        self.norm1 = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.norm2 = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.norm3 = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        
        x = rearrange(x, "b t c -> b c t")
        b, c, t = x.shape

        x = self.conv1_local(x)
        x = rearrange(x, "b (n c) t -> (b n) t c", n=self.num_heads)
        
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.conv2(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.conv3(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm3(x)
        x = self.act(x)
        x = self.out_proj(x)
        x = rearrange(x, "(b n) t c -> b t n c", b=b)
        padding = self.padding_tokens.repeat(b, x.shape[1], 1, 1)
        x = torch.cat([x, padding], dim=-2)
        x_local = x.clone()

        return x_local



class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


def get_norm_layer(norm_layer):
    """
    Get the normalization layer.

    Args:
        norm_layer (str): The type of normalization layer.

    Returns:
        norm_layer (nn.Module): The normalization layer.
    """
    if norm_layer == "layer":
        return nn.LayerNorm
    elif norm_layer == "rms":
        return RMSNorm
    else:
        raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")


class FaceAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads_num: int,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        num_adapter_layers: int = 1,
        dtype=None,
        device=None,
    ):

        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.hidden_size = hidden_dim
        self.heads_num = heads_num
        self.fuser_blocks = nn.ModuleList(
            [
                FaceBlock(
                    self.hidden_size,
                    self.heads_num,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(num_adapter_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        motion_embed: torch.Tensor,
        idx: int,
        freqs_cis_q: Tuple[torch.Tensor, torch.Tensor] = None,
        freqs_cis_k: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:

        return self.fuser_blocks[idx](x, motion_embed, freqs_cis_q, freqs_cis_k)



class FaceBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.scale = qk_scale or head_dim**-0.5
       
        self.linear1_kv = nn.Linear(hidden_size, hidden_size * 2, **factory_kwargs)
        self.linear1_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)

        self.linear2 = nn.Linear(hidden_size, hidden_size, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.pre_norm_feat = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.pre_norm_motion = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        motion_vec: torch.Tensor,
        motion_mask: Optional[torch.Tensor] = None,
        use_context_parallel=False,
    ) -> torch.Tensor:
        
        B, T, N, C = motion_vec.shape
        T_comp = T

        x_motion = self.pre_norm_motion(motion_vec)
        x_feat = self.pre_norm_feat(x)

        kv = self.linear1_kv(x_motion)
        q = self.linear1_q(x_feat)

        k, v = rearrange(kv, "B L N (K H D) -> K B L N H D", K=2, H=self.heads_num)
        q = rearrange(q, "B S (H D) -> B S H D", H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        k = rearrange(k, "B L N H D -> (B L) H N D")  
        v = rearrange(v, "B L N H D -> (B L) H N D") 

        q = rearrange(q, "B (L S) H D -> (B L) H S D", L=T_comp)  
        # Compute attention.
        attn = F.scaled_dot_product_attention(q, k, v)

        attn = rearrange(attn, "(B L) H S D -> B (L S) (H D)", L=T_comp)

        output = self.linear2(attn)

        if motion_mask is not None:
            output = output * rearrange(motion_mask, "B T H W -> B (T H W)").unsqueeze(-1)

        return output



def custom_qr(input_tensor):
    original_dtype = input_tensor.dtype
    if original_dtype == torch.bfloat16:
        q, r = torch.linalg.qr(input_tensor.to(torch.float32))
        return q.to(original_dtype), r.to(original_dtype)
    return torch.linalg.qr(input_tensor)

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
	return F.leaky_relu(input + bias, negative_slope) * scale


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
	_, minor, in_h, in_w = input.shape
	kernel_h, kernel_w = kernel.shape

	out = input.view(-1, minor, in_h, 1, in_w, 1)
	out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
	out = out.view(-1, minor, in_h * up_y, in_w * up_x)

	out = F.pad(out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
	out = out[:, :, max(-pad_y0, 0): out.shape[2] - max(-pad_y1, 0),
		  max(-pad_x0, 0): out.shape[3] - max(-pad_x1, 0), ]

	out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
	w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
	out = F.conv2d(out, w)
	out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
					  in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1, )
	return out[:, :, ::down_y, ::down_x]


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
	return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])


def make_kernel(k):
	k = torch.tensor(k, dtype=torch.float32)
	if k.ndim == 1:
		k = k[None, :] * k[:, None]
	k /= k.sum()
	return k


class FusedLeakyReLU(nn.Module):
	def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
		super().__init__()
		self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
		self.negative_slope = negative_slope
		self.scale = scale

	def forward(self, input):
		out = fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
		return out


class Blur(nn.Module):
	def __init__(self, kernel, pad, upsample_factor=1):
		super().__init__()

		kernel = make_kernel(kernel)

		if upsample_factor > 1:
			kernel = kernel * (upsample_factor ** 2)

		self.register_buffer('kernel', kernel)

		self.pad = pad

	def forward(self, input):
		return upfirdn2d(input, self.kernel, pad=self.pad)


class ScaledLeakyReLU(nn.Module):
	def __init__(self, negative_slope=0.2):
		super().__init__()

		self.negative_slope = negative_slope

	def forward(self, input):
		return F.leaky_relu(input, negative_slope=self.negative_slope)


class EqualConv2d(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
		super().__init__()

		self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
		self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

		self.stride = stride
		self.padding = padding

		if bias:
			self.bias = nn.Parameter(torch.zeros(out_channel))
		else:
			self.bias = None

	def forward(self, input):

		return F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)

	def __repr__(self):
		return (
			f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
			f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
		)


class EqualLinear(nn.Module):
	def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
		super().__init__()

		self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

		if bias:
			self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
		else:
			self.bias = None

		self.activation = activation

		self.scale = (1 / math.sqrt(in_dim)) * lr_mul
		self.lr_mul = lr_mul

	def forward(self, input):

		if self.activation:
			out = F.linear(input, self.weight * self.scale)
			out = fused_leaky_relu(out, self.bias * self.lr_mul)
		else:
			out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

		return out

	def __repr__(self):
		return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')


class ConvLayer(nn.Sequential):
	def __init__(
			self,
			in_channel,
			out_channel,
			kernel_size,
			downsample=False,
			blur_kernel=[1, 3, 3, 1],
			bias=True,
			activate=True,
	):
		layers = []

		if downsample:
			factor = 2
			p = (len(blur_kernel) - factor) + (kernel_size - 1)
			pad0 = (p + 1) // 2
			pad1 = p // 2

			layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

			stride = 2
			self.padding = 0

		else:
			stride = 1
			self.padding = kernel_size // 2

		layers.append(EqualConv2d(in_channel, out_channel, kernel_size, padding=self.padding, stride=stride,
								  bias=bias and not activate))

		if activate:
			if bias:
				layers.append(FusedLeakyReLU(out_channel))
			else:
				layers.append(ScaledLeakyReLU(0.2))

		super().__init__(*layers)


class ResBlock(nn.Module):
	def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
		super().__init__()

		self.conv1 = ConvLayer(in_channel, in_channel, 3)
		self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

		self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)

	def forward(self, input):
		out = self.conv1(input)
		out = self.conv2(out)

		skip = self.skip(input)
		out = (out + skip) / math.sqrt(2)

		return out


class EncoderApp(nn.Module):
	def __init__(self, size, w_dim=512):
		super(EncoderApp, self).__init__()

		channels = {
			4: 512,
			8: 512,
			16: 512,
			32: 512,
			64: 256,
			128: 128,
			256: 64,
			512: 32,
			1024: 16
		}

		self.w_dim = w_dim
		log_size = int(math.log(size, 2))

		self.convs = nn.ModuleList()
		self.convs.append(ConvLayer(3, channels[size], 1))

		in_channel = channels[size]
		for i in range(log_size, 2, -1):
			out_channel = channels[2 ** (i - 1)]
			self.convs.append(ResBlock(in_channel, out_channel))
			in_channel = out_channel

		self.convs.append(EqualConv2d(in_channel, self.w_dim, 4, padding=0, bias=False))

	def forward(self, x):

		res = []
		h = x
		for conv in self.convs:
			h = conv(h)
			res.append(h)

		return res[-1].squeeze(-1).squeeze(-1), res[::-1][2:]


class Encoder(nn.Module):
	def __init__(self, size, dim=512, dim_motion=20):
		super(Encoder, self).__init__()

		# appearance netmork
		self.net_app = EncoderApp(size, dim)

		# motion network
		fc = [EqualLinear(dim, dim)]
		for i in range(3):
			fc.append(EqualLinear(dim, dim))

		fc.append(EqualLinear(dim, dim_motion))
		self.fc = nn.Sequential(*fc)

	def enc_app(self, x):
		h_source = self.net_app(x)
		return h_source

	def enc_motion(self, x):
		h, _ = self.net_app(x)
		h_motion = self.fc(h)
		return h_motion


class Direction(nn.Module):
    def __init__(self, motion_dim):
        super(Direction, self).__init__()
        self.weight = nn.Parameter(torch.randn(512, motion_dim))

    def forward(self, input):

        weight = self.weight + 1e-8
        Q, R = custom_qr(weight)
        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)
            return out


class Synthesis(nn.Module):
    def __init__(self, motion_dim):
        super(Synthesis, self).__init__()
        self.direction = Direction(motion_dim)


class Generator(nn.Module):
    def __init__(self, size, style_dim=512, motion_dim=20):
        super().__init__()

        self.enc = Encoder(size, style_dim, motion_dim)
        self.dec = Synthesis(motion_dim)

    def get_motion(self, img):
        #motion_feat = self.enc.enc_motion(img)
        motion_feat = torch.utils.checkpoint.checkpoint((self.enc.enc_motion), img, use_reentrant=True)
        motion = self.dec.direction(motion_feat)
        return motion


class WanAnimateAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pose_patch_embedding = torch.nn.Conv3d(16, 5120, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.motion_encoder = Generator(size=512, style_dim=512, motion_dim=20)
        self.face_adapter = FaceAdapter(heads_num=40, hidden_dim=5120, num_adapter_layers=40 // 5)
        self.face_encoder = FaceEncoder(in_dim=512, hidden_dim=5120, num_heads=4)
    
    def after_patch_embedding(self, x: List[torch.Tensor], pose_latents, face_pixel_values):
        pose_latents = self.pose_patch_embedding(pose_latents)
        x[:, :, 1:] += pose_latents
        
        b,c,T,h,w = face_pixel_values.shape
        face_pixel_values = rearrange(face_pixel_values, "b c t h w -> (b t) c h w")

        encode_bs = 8
        face_pixel_values_tmp = []
        for i in range(math.ceil(face_pixel_values.shape[0]/encode_bs)):
            face_pixel_values_tmp.append(self.motion_encoder.get_motion(face_pixel_values[i*encode_bs:(i+1)*encode_bs]))

        motion_vec = torch.cat(face_pixel_values_tmp)
        
        motion_vec = rearrange(motion_vec, "(b t) c -> b t c", t=T)
        motion_vec = self.face_encoder(motion_vec)

        B, L, H, C = motion_vec.shape
        pad_face = torch.zeros(B, 1, H, C).type_as(motion_vec)
        motion_vec = torch.cat([pad_face, motion_vec], dim=1)
        return x, motion_vec
    
    def after_transformer_block(self, block_idx, x, motion_vec, motion_masks=None):
        if block_idx % 5 == 0:
            adapter_args = [x, motion_vec, motion_masks, False]
            residual_out = self.face_adapter.fuser_blocks[block_idx // 5](*adapter_args)
            x = residual_out + x
        return x
    
    @staticmethod
    def state_dict_converter():
        return WanAnimateAdapterStateDictConverter()


class WanAnimateAdapterStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        return state_dict
    
    def from_civitai(self, state_dict):
        state_dict_ = {}
        for name, param in state_dict.items():
            if name.startswith("pose_patch_embedding.") or name.startswith("face_adapter") or name.startswith("face_encoder") or name.startswith("motion_encoder"):
                state_dict_[name] = param
        return state_dict_

