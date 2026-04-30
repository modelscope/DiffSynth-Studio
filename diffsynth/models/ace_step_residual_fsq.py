"""
Code adapted from https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/residual_fsq.py
"""
from functools import wraps, partial
from contextlib import nullcontext

import random
from math import ceil

import torch
from torch import nn, tensor, Tensor, int32, tanh, atanh, clamp
from torch.nn import Module
import torch.nn.functional as F
from torch.amp import autocast
import torch.distributed as dist
from einops import rearrange, reduce, pack, unpack


# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def identity(t):
    return t

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers

def round_ste(z):
    """ round with straight through gradients. """
    zhat = z.round()
    return z + (zhat - z).detach()

def floor_ste(z):
    """ floor with straight through gradients. """
    zhat = z.floor()
    return z + (zhat - z).detach()

# main class

class FSQ(Module):
    def __init__(
        self,
        levels: list[int] | tuple[int, ...],
        dim: int | None = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: bool | None = None,
        scale: float | None = None,
        allowed_dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first = False,
        projection_has_bias = True,
        return_indices = True,
        force_quantization_f32 = True,
        preserve_symmetry = False,
        noise_dropout = 0.,
        bound_hard_clamp = False,                   # for residual fsq, if input is pre-softclamped to the right range
        orthogonal_rotation = False                 # increase codebook utilization. ensure levels are symmetric! https://arxiv.org/abs/2307.13304v2
    ):
        super().__init__()

        assert not (any([l == 2 for l in levels]) and not preserve_symmetry), 'turn on `preserve_symmetry` for using any levels == 2, or use a greater level'

        if isinstance(levels, tuple):
            levels = list(levels)

        _levels = tensor(levels, dtype = int32)
        self.register_buffer('_levels', _levels, persistent = False)

        _basis = torch.cumprod(tensor([1] + levels[:-1]), dim = 0, dtype = int32)
        self.register_buffer('_basis', _basis, persistent = False)

        self.scale = scale

        assert not (noise_dropout > 0 and not preserve_symmetry)
        self.preserve_symmetry = preserve_symmetry
        self.noise_dropout = noise_dropout

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias = projection_has_bias) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim, bias = projection_has_bias) if has_projections else nn.Identity()

        self.has_projections = has_projections

        self.return_indices = return_indices

        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer('implicit_codebook', implicit_codebook, persistent = False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

        # allow for a hard clamp

        self.bound_hard_clamp = bound_hard_clamp

        self.orthogonal_rotation = orthogonal_rotation

        if orthogonal_rotation:
            is_symmetric = len(set(levels)) == 1
            if not is_symmetric:
                print('orthogonal_rotation is not recommended for FSQ with asymmetric levels (i.e. where the number of bins differ across dimensions)')

            orthogonal_rot = torch.empty(codebook_dim, codebook_dim)
            nn.init.orthogonal_(orthogonal_rot)
            self.register_buffer('orthogonal_rot', orthogonal_rot)

    def bound(self, z, eps = 1e-3, hard_clamp = False):
        """ Bound `z`, an array of shape (..., d). """
        maybe_tanh = tanh if not hard_clamp else partial(clamp, min = -1., max = 1.)
        maybe_atanh = atanh if not hard_clamp else identity

        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = maybe_atanh(offset / half_l)
        bounded_z = maybe_tanh(z + shift) * half_l - offset
        half_width = self._levels // 2
        return round_ste(bounded_z) / half_width

    # symmetry-preserving and noise-approximated quantization, section 3.2 in https://arxiv.org/abs/2411.19842

    def symmetry_preserving_bound(self, z, hard_clamp = False):
        """ QL(x) = 2 / (L - 1) * [(L - 1) * (tanh(x) + 1) / 2 + 0.5] - 1 """
        maybe_tanh = tanh if not hard_clamp else partial(clamp, min = -1., max = 1.)

        levels_minus_1 = (self._levels - 1)
        scale = 2. / levels_minus_1
        bracket = (levels_minus_1 * (maybe_tanh(z) + 1) / 2.) + 0.5
        bracket = floor_ste(bracket)
        return scale * bracket - 1.

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """

        shape, device, preserve_symmetry = z.shape[0], z.device, self.preserve_symmetry
        bound_fn = self.symmetry_preserving_bound if preserve_symmetry else self.bound

        return bound_fn(z, hard_clamp = self.bound_hard_clamp)

    def maybe_apply_noise(self, bounded_z):
        noise_dropout = self.noise_dropout

        if not self.training or noise_dropout == 0.:
            return bounded_z

        # determine where to add a random offset elementwise
        # if using noise dropout

        offset_mask = torch.full_like(bounded_z, noise_dropout).bernoulli_().bool()
        offset = torch.rand_like(bounded_z) - 0.5

        bounded_z = torch.where(offset_mask, bounded_z + offset, bounded_z)

        return bounded_z.clamp(-1., 1.)

    def _scale_and_shift(self, zhat_normalized):
        if self.preserve_symmetry:
            return (zhat_normalized + 1.) / (2. / (self._levels - 1))

        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        if self.preserve_symmetry:
            return zhat * (2. / (self._levels - 1)) - 1.

        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim = -1).round().to(int32)

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """
        assert exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.orthogonal_rotation:
            codes = codes @ self.orthogonal_rot.t()

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)

        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        if self.orthogonal_rotation:
            z = z @ self.orthogonal_rot

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        with quantization_context():
            orig_dtype = z.dtype

            if force_f32 and orig_dtype not in self.allowed_dtypes:
                z = z.float()

            codes = self.quantize(z)

            # returning indices could be optional

            indices = None

            if self.return_indices:
                indices = self.codes_to_indices(codes)

            codes = self.maybe_apply_noise(codes)

            if self.orthogonal_rotation:
                codes = codes @ self.orthogonal_rot.t()

            codes = rearrange(codes, 'b n c d -> b n (c d)')

            codes = codes.to(orig_dtype)

        # project out

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = maybe(unpack_one)(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim and self.return_indices:
            indices = maybe(rearrange)(indices, '... 1 -> ...')

        # return quantized output and indices

        return out, indices

# helper functions

def first(l):
    return l[0]

def default_residual_fsq(val, d):
    return val if exists(val) else d

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

# distributed helpers

def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def get_maybe_sync_seed(device, max_size = 10_000):
    rand_int = torch.randint(0, max_size, (), device = device)

    if is_distributed():
        dist.all_reduce(rand_int)

    return rand_int.item()

# main class

class ResidualFSQ(Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(
        self,
        *,
        levels: list[int],
        num_quantizers,
        dim = None,
        is_channel_first = False,
        quantize_dropout = False,
        quantize_dropout_cutoff_index = 0,
        quantize_dropout_multiple_of = 1,
        soft_clamp_input_value: float | list[float] | Tensor | None = None,
        bound_hard_clamp = True,
        **kwargs
    ):
        super().__init__()
        codebook_dim = len(levels)
        dim = default_residual_fsq(dim, codebook_dim)

        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.is_channel_first = is_channel_first
        self.num_quantizers = num_quantizers

        # layers

        self.levels = levels
        self.layers = nn.ModuleList([])

        levels_tensor = tensor(levels)
        assert (levels_tensor > 1).all()

        scales = []

        for ind in range(num_quantizers):
            scales.append(levels_tensor.float() ** -ind)

            fsq = FSQ(
                levels = levels,
                dim = codebook_dim,
                preserve_symmetry = True,
                bound_hard_clamp = bound_hard_clamp,
                **kwargs
            )

            self.layers.append(fsq)

        assert all([not fsq.has_projections for fsq in self.layers])

        self.codebook_size = self.layers[0].codebook_size

        self.register_buffer('scales', torch.stack(scales), persistent = False)

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

        # soft clamping the input value

        if bound_hard_clamp:
            assert not exists(soft_clamp_input_value)
            soft_clamp_input_value = 1 + (1 / (levels_tensor - 1))

        if isinstance(soft_clamp_input_value, (list, float)):
            soft_clamp_input_value = tensor(soft_clamp_input_value)

        self.register_buffer('soft_clamp_input_value', soft_clamp_input_value, persistent = False)

    @property
    def codebooks(self):
        codebooks = [layer.implicit_codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        return codebooks

    def get_codes_from_indices(self, indices):

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # may also receive indices in the shape of 'b h w q' (accept_image_fmap)

        indices, ps = pack([indices], 'b * q')

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0., 'quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations'
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # take care of quantizer dropout

        mask = indices == -1
        indices = indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        all_codes = torch.stack([F.embedding(indices[:, :, i], self.codebooks[i]) for i in range(self.num_quantizers)], dim=0)  # (q, b, n, d)

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(rearrange(mask, 'b n q -> q b n 1'), 0.)

        # scale the codes

        scales = rearrange(self.scales, 'q d -> q 1 1 d')
        all_codes = all_codes * scales

        # if (accept_image_fmap = True) then return shape (quantize, batch, height, width, dimension)

        all_codes, = unpack(all_codes, ps, 'q b * d')

        return all_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def forward(
        self,
        x,
        return_all_codes = False,
        rand_quantize_dropout_fixed_seed = None
    ):
        num_quant, quant_dropout_multiple_of, device = self.num_quantizers, self.quantize_dropout_multiple_of, x.device

        # handle channel first

        if self.is_channel_first:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack([x], 'b * d')

        # maybe project in

        x = self.project_in(x)

        # maybe softclamp input before residual layers

        if exists(self.soft_clamp_input_value):
            clamp_value = self.soft_clamp_input_value
            x = (x / clamp_value).tanh() * clamp_value

        # ready some variables to be accumulated

        quantized_out = 0.
        residual = x

        all_indices = []

        should_quantize_dropout = self.training and self.quantize_dropout and torch.is_grad_enabled()

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices

        if should_quantize_dropout:

            # check if seed is manually passed in

            if not exists(rand_quantize_dropout_fixed_seed):
                rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device)

            rand = random.Random(rand_quantize_dropout_fixed_seed)

            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1

            null_indices = torch.full(x.shape[:2], -1., device = device, dtype = torch.long)

        # go through the layers

        with autocast('cuda', enabled = False):
            for quantizer_index, (layer, scale) in enumerate(zip(self.layers, self.scales)):

                if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                    all_indices.append(null_indices)
                    continue

                quantized, indices = layer(residual / scale)

                quantized = quantized * scale

                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized

                all_indices.append(indices)

        # project out, if needed

        quantized_out = self.project_out(quantized_out)

        # stack all indices

        all_indices = torch.stack(all_indices, dim = -1)

        # channel first out

        if self.is_channel_first:
            quantized_out, = unpack(quantized_out, ps, 'b * d')
            all_indices, = unpack(all_indices, ps, 'b * d')

            quantized_out = rearrange(quantized_out, 'b ... d -> b d ...')
            all_indices = rearrange(all_indices, 'b ... d -> b d ...')

        # return

        ret = (quantized_out, all_indices)

        if not return_all_codes:
            return ret

        # whether to return all codes from all codebooks across layers

        all_codes = self.get_codes_from_indices(all_indices)

        # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)

        return (*ret, all_codes)
