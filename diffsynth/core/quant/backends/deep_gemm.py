"""DeepGEMM W8A8 / W4A4 / W4A8 quantization with per-token dynamic activation quantization.

Weights are quantized once, when the model is loaded, and stay stored in fp8 (E4M3, one
scale per 128x128 block) or fp4 (E2M1, one scale per 1x32 block). Every `forward`
quantizes its own activation tile (per token, in a fused Triton kernel) and calls
DeepGEMM's GEMM (`D = A @ B.T`, the NT layout), so the matmul runs on the fp8/fp4 tensor
cores instead of bf16.

`deep_gemm_fp8_w8a8` runs on SM90 and SM100. The FP4 methods (`deep_gemm_fp4_w4a4`,
`deep_gemm_fp4_w4a8`) need SM100: DeepGEMM's FP4 MMA is Blackwell-only, and SM90 rejects
a packed-FP4 operand outright.

Every scaling factor is an exact power of two (UE8M0), because DeepGEMM's SM100 path packs
the exponent only; SM90 accepts the same values as plain `float32`. Weights keep their
`float32` scale grid in the state dict, so a checkpoint quantized on SM90 reloads on SM100,
and the packed `int32` SF layout is derived at runtime with DeepGEMM's own layout kernel.
"""

from dataclasses import dataclass, field

import torch

from ..base import BackendConfig, QuantBackend, register_quant_backend
from ..config import register_quant_method

try:
    import deep_gemm
    from deep_gemm.utils import ceil_to_ue8m0, per_token_cast_to_fp4, per_token_cast_to_fp8

    DEEP_GEMM_AVAILABLE = True
    _DEEP_GEMM_IMPORT_ERROR = None
except Exception as error:  # a missing package or an unsupported GPU/CPU-only build both land here
    deep_gemm = None
    ceil_to_ue8m0 = None
    per_token_cast_to_fp4 = None
    per_token_cast_to_fp8 = None
    DEEP_GEMM_AVAILABLE = False
    _DEEP_GEMM_IMPORT_ERROR = error

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


BLOCK_SIZE = 128            # fp8 granularity along K, and the block size of the fp8 weight scales
FP8_MAX = 448.0
FP4_MAX = 6.0
FP4_GRANULARITY = 32        # fp4 granularity along K (weights and activations)
FP4_BLOCK_K = 256           # DeepGEMM's FP4xFP4 MMA consumes 256 K elements per stage
QUANT_BLOCK_M = 64          # tokens per Triton program


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _align(value, alignment):
    return _ceil_div(value, alignment) * alignment




if TRITON_AVAILABLE:

    @triton.jit
    def _per_token_cast_kernel(x_ptr, out_ptr, scale_ptr, m, k,
                               BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
        """One program handles `BLOCK_M` tokens x one 128-channel block of one token."""
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        block = tl.program_id(1)
        columns = block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = rows[:, None] < m
        x = tl.load(x_ptr + rows[:, None] * k + columns[None, :], mask=mask, other=0.0).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4) / 448.0
        # UE8M0: DeepGEMM's SM100 path packs the exponent only, so the codes must be scaled by a power of two.
        bits = scale.to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = (tl.minimum(tl.maximum(exponent, 1), 254) << 23).to(tl.float32, bitcast=True)
        tl.store(out_ptr + rows[:, None] * k + columns[None, :], (x * (1.0 / scale)[:, None]).to(tl.float8e4nv), mask=mask)
        tl.store(scale_ptr + rows * (k // BLOCK_K) + block, scale, mask=rows < m)

    @triton.jit
    def _e2m1_codes(x):
        """Round to the nearest E2M1 value, matching `deep_gemm.utils._quantize_to_fp4_e2m1`."""
        ax = tl.abs(x)
        code = (ax > 0.25).to(tl.int32)
        code += (ax >= 0.75).to(tl.int32)
        code += (ax > 1.25).to(tl.int32)
        code += (ax >= 1.75).to(tl.int32)
        code += (ax > 2.5).to(tl.int32)
        code += (ax >= 3.5).to(tl.int32)
        code += (ax > 5.0).to(tl.int32)
        sign = (x < 0) & (code != 0)
        return (code | (sign.to(tl.int32) << 3)).to(tl.uint8)

    @triton.jit
    def _fp4_cast_kernel(x_ptr, out_ptr, scale_ptr, m, k, aligned_m,
                         BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
        """One program handles `BLOCK_M` tokens x 128 channels: 64 packed fp4 bytes per token,
        plus one UE8M0-packed `int32` scale (4 scales of 32 channels) per token."""
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        block = tl.program_id(1)
        columns = block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = rows[:, None] < m
        x = tl.load(x_ptr + rows[:, None] * k + columns[None, :], mask=mask, other=0.0).to(tl.float32)
        x = tl.reshape(x, (BLOCK_M, BLOCK_K // 32, 32))
        amax = tl.max(tl.abs(x), axis=2)
        scale = tl.maximum(amax, 1e-4) / 6.0
        exponent = (scale.to(tl.int32, bitcast=True) >> 23) & 0xFF
        exponent += ((scale.to(tl.int32, bitcast=True) & 0x7FFFFF) != 0).to(tl.int32)
        exponent = tl.minimum(tl.maximum(exponent, 1), 254)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        codes = _e2m1_codes(x * (1.0 / scale)[:, :, None])
        codes = tl.reshape(codes, (BLOCK_M, BLOCK_K // 2, 2))
        low, high = tl.split(codes)
        out_columns = block * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
        tl.store(out_ptr + rows[:, None] * (k // 2) + out_columns[None, :], low | (high << 4), mask=mask)
        pairs = tl.reshape(exponent, (BLOCK_M, 2, 2))
        even, odd = tl.split(pairs)
        e0, e2 = tl.split(even)
        e1, e3 = tl.split(odd)
        packed = e0 | (e1 << 8) | (e2 << 16) | (e3 << 24)
        tl.store(scale_ptr + rows + block * aligned_m, packed, mask=rows < m)


def _per_token_cast(x_2d):
    """Quantize each token to fp8 E4M3 with one fp32 scale per 128 channels.

    Returns the fp8 activations `[m, k]` and the untransformed SFA `[m, k / 128]`, which
    is what DeepGEMM's SM90/SM100 GEMMs take. The fused Triton path matters: the plain
    PyTorch equivalent (which `deep_gemm.utils.per_token_cast_to_fp8` uses) costs about
    as much as the GEMM it feeds.
    """
    m, k = x_2d.shape
    if TRITON_AVAILABLE and k % BLOCK_SIZE == 0:
        quantized = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x_2d.device)
        scale = torch.empty((m, k // BLOCK_SIZE), dtype=torch.float32, device=x_2d.device)
        if m > 0:
            _per_token_cast_kernel[(triton.cdiv(m, QUANT_BLOCK_M), k // BLOCK_SIZE)](
                x_2d, quantized, scale, m, k, BLOCK_M=QUANT_BLOCK_M, BLOCK_K=BLOCK_SIZE,
            )
        return quantized, scale
    return per_token_cast_to_fp8(x_2d, use_ue8m0=True)


def _per_token_cast_fp4(x_2d):
    """Quantize each token to packed fp4 E2M1 with one UE8M0 scale per 32 channels.

    Returns the packed `int8` activations `[m, k / 2]` (two E2M1 codes per byte, the dtype
    DeepGEMM names `kPackedFP4`) and the already-packed `int32` SFA in the MN-major
    TMA-aligned layout DeepGEMM validates, so no layout transform runs per call.
    """
    m, k = x_2d.shape
    if TRITON_AVAILABLE and k % BLOCK_SIZE == 0:
        quantized = torch.empty((m, k // 2), dtype=torch.uint8, device=x_2d.device)
        aligned_m = _align(m, 4)
        scale = torch.empty_strided(
            (m, _ceil_div(k, FP4_GRANULARITY * 4)), (1, aligned_m), dtype=torch.int32, device=x_2d.device,
        )
        if m > 0:
            _fp4_cast_kernel[(triton.cdiv(m, QUANT_BLOCK_M), k // BLOCK_SIZE)](
                x_2d, quantized, scale, m, k, aligned_m, BLOCK_M=QUANT_BLOCK_M, BLOCK_K=BLOCK_SIZE,
            )
        return quantized.view(torch.int8), scale
    quantized, scale = per_token_cast_to_fp4(x_2d, use_ue8m0=True, gran_k=FP4_GRANULARITY)
    return quantized, _pack_scale(scale)


def _pack_scale(scale):
    """Pack `float32` UE8M0 scales into the MN-major `int32` layout DeepGEMM takes.

    This is the same kernel DeepGEMM's own SM100 SF transform calls, so the result is
    accepted by `check_sf_layout` as a pre-transformed SF (no transform per call).
    """
    return deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(scale)


def _is_supported(in_features, out_features):
    """Whether DeepGEMM's FP8 GEMM accepts this layer's shape.

    DeepGEMM tiles K (and the weight's N) in 128-element blocks; a layer whose
    dimensions are not whole blocks would need padded weights to stay correct, so
    such layers are left in full precision by `create_quantized_linear`.
    """
    return in_features % BLOCK_SIZE == 0 and out_features % BLOCK_SIZE == 0


def _is_supported_fp4(in_features, out_features, activation):
    """FP4 weights are tiled 256 (W4A4) or 128 (W4A8, the MMA consumes fp8 on the A side) along K."""
    block_k = FP4_BLOCK_K if activation == "fp4" else BLOCK_SIZE
    return in_features % block_k == 0 and out_features % BLOCK_SIZE == 0


def _quantize_weight(weight):
    """Per-128x128-block symmetric fp8 E4M3 weight quantization.

    Returns the fp8 weight `[out_features, in_features]` and its `float32` scale
    grid `[ceil(out_features / 128), ceil(in_features / 128)]`, which is the
    untransformed SFB layout DeepGEMM expects.
    """
    out_features, in_features = weight.shape
    padded_out = _align(out_features, BLOCK_SIZE)
    padded_in = _align(in_features, BLOCK_SIZE)
    blocks = weight.detach().to(torch.float32)
    if (padded_out, padded_in) != (out_features, in_features):
        blocks = torch.nn.functional.pad(blocks, (0, padded_in - in_features, 0, padded_out - out_features))
    blocks = blocks.view(padded_out // BLOCK_SIZE, BLOCK_SIZE, padded_in // BLOCK_SIZE, BLOCK_SIZE)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_(min=1e-4)
    scale = ceil_to_ue8m0(amax / FP8_MAX)
    quantized = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    quantized = quantized.view(padded_out, padded_in)[:out_features, :in_features].contiguous()
    return quantized, scale


def _dequantize_weight(weight, scale, dtype):
    """Rebuild the fp weight from the fp8 codes and their per-block scales."""
    out_features, in_features = weight.shape
    padded_out = _align(out_features, BLOCK_SIZE)
    padded_in = _align(in_features, BLOCK_SIZE)
    blocks = weight.to(torch.float32).view(padded_out // BLOCK_SIZE, BLOCK_SIZE, padded_in // BLOCK_SIZE, BLOCK_SIZE)
    blocks = blocks * scale[:, None, :, None]
    return blocks.view(padded_out, padded_in)[:out_features, :in_features].to(dtype)


def _quantize_weight_fp4(weight):
    """Per-channel, per-32-channel-block fp4 E2M1 weight quantization.

    Returns the packed weight `[out_features, in_features / 2]` (`int8`, two codes per
    byte) and its untransformed UE8M0 `float32` scale grid `[out_features, in_features / 32]`.
    """
    quantized, scale = per_token_cast_to_fp4(weight.detach(), use_ue8m0=True, gran_k=FP4_GRANULARITY)
    return quantized.contiguous(), scale


def _dequantize_fp4_weight(weight, scale, dtype):
    """Rebuild the fp weight from the packed fp4 codes and their per-32-channel scales."""
    out_features, half_in = weight.shape
    in_features = half_in * 2
    codes = weight.to(torch.int32)
    values = torch.empty((out_features, in_features), dtype=torch.int32, device=codes.device)
    values[:, 0::2] = codes & 0x0F
    values[:, 1::2] = (codes >> 4) & 0x0F
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=codes.device,
    )
    magnitude = magnitudes[values & 0x07]
    signed = torch.where((values & 0x08) != 0, -magnitude, magnitude)
    return (signed * scale.repeat_interleave(FP4_GRANULARITY, dim=1)).to(dtype)


def _fp8_matmul(x_2d, weight, weight_scale):
    """`x_2d @ weight.T` in fp8: quantize the activation per token, then run DeepGEMM."""
    if not x_2d.is_cuda:
        raise RuntimeError("DeepGEMM fp8 layers can only run on CUDA devices; move the model to CUDA first.")
    if x_2d.dtype != torch.bfloat16:
        x_2d = x_2d.to(torch.bfloat16)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    x_fp8, x_scale = _per_token_cast(x_2d)
    out = torch.empty((x_fp8.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x_fp8.device)
    deep_gemm.fp8_gemm_nt((x_fp8, x_scale), (weight, weight_scale), out)
    return out


def _fp4_matmul(x_2d, weight, weight_scale, activation):
    """`x_2d @ weight.T` in fp4: quantize the activation per token, then run DeepGEMM's W4A4/W4A8 GEMM."""
    if not x_2d.is_cuda:
        raise RuntimeError("DeepGEMM fp4 layers can only run on CUDA devices; move the model to CUDA first.")
    if x_2d.dtype != torch.bfloat16:
        x_2d = x_2d.to(torch.bfloat16)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    if activation == "fp4":
        x_quantized, x_scale = _per_token_cast_fp4(x_2d)
        recipe_a = (1, FP4_GRANULARITY)
    else:
        x_fp8, x_scale_fp32 = _per_token_cast(x_2d)
        x_quantized, x_scale = x_fp8, _pack_scale(x_scale_fp32)
        recipe_a = (1, BLOCK_SIZE)
    out = torch.empty((x_quantized.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x_quantized.device)
    deep_gemm.fp8_fp4_gemm_nt(
        (x_quantized, x_scale), (weight, weight_scale), out,
        recipe_a=recipe_a, recipe_b=(1, FP4_GRANULARITY),
    )
    return out


class _DeepGEMMQuantizedLinear(torch.nn.Linear):
    """Shared plumbing for the quantized Linears: packed-buffer dtype guard and 2-D reshape."""

    dtype_guarded_tensor_names = ()

    def _apply(self, fn, recurse=True):
        protected = {
            id(tensor) for name in self.dtype_guarded_tensor_names
            if (tensor := getattr(self, name, None)) is not None
        }

        def guard(tensor):
            converted = fn(tensor)
            if id(tensor) in protected and converted.dtype != tensor.dtype:
                return tensor.to(device=converted.device)
            return converted

        return super()._apply(guard, recurse)

    def matmul_2d(self, x_2d):
        raise NotImplementedError

    def dequantize_weight(self, dtype):
        raise NotImplementedError

    def forward(self, x):
        x_2d = x.reshape(-1, x.shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        out = self.matmul_2d(x_2d)
        return out.view(*x.shape[:-1], out.shape[-1])


class _DeepGEMMFp8Function(torch.autograd.Function):
    """Compute in fp8, but pass the activation gradient through the dequantized weight.

    LoRA training keeps the quantized weight frozen and only back-propagates to the
    activation, so the straight-through derivative `grad @ W_dequant` is enough.
    """

    @staticmethod
    def forward(ctx, x_2d, weight, weight_scale, bias):
        out = _fp8_matmul(x_2d, weight, weight_scale)
        if bias is not None:
            out = out + bias.to(out.dtype)
        out = out.to(x_2d.dtype)
        ctx.save_for_backward(weight, weight_scale)
        ctx.input_shape = x_2d.shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        weight, weight_scale = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_x = torch.mm(grad_2d, _dequantize_weight(weight, weight_scale, grad_2d.dtype))
        grad_bias = grad_2d.sum(dim=0) if ctx.needs_input_grad[3] else None
        return grad_x.reshape(ctx.input_shape), None, None, grad_bias


class _DeepGEMMFp4Function(torch.autograd.Function):
    """Same straight-through trick as the fp8 path, on top of the W4A4/W4A8 GEMM."""

    @staticmethod
    def forward(ctx, x_2d, weight, weight_scale, packed_weight_scale, bias, activation):
        out = _fp4_matmul(x_2d, weight, packed_weight_scale, activation)
        if bias is not None:
            out = out + bias.to(out.dtype)
        out = out.to(x_2d.dtype)
        ctx.save_for_backward(weight, weight_scale)
        ctx.input_shape = x_2d.shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        weight, weight_scale = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_x = torch.mm(grad_2d, _dequantize_fp4_weight(weight, weight_scale, grad_2d.dtype))
        grad_bias = grad_2d.sum(dim=0) if ctx.needs_input_grad[4] else None
        return grad_x.reshape(ctx.input_shape), None, None, None, grad_bias, None


class DeepGEMMFp8Linear(_DeepGEMMQuantizedLinear):
    """`nn.Linear` that stores an fp8 weight plus block scales and calls DeepGEMM in `forward`."""

    dtype_guarded_tensor_names = ("weight", "weight_scale")

    def __init__(self, in_features, out_features, bias, compute_dtype):
        with torch.device("meta"):
            super().__init__(in_features, out_features, bias=bias, dtype=compute_dtype)
        del self.weight
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device="meta"))
        self.register_buffer(
            "weight_scale",
            torch.empty(
                _ceil_div(out_features, BLOCK_SIZE), _ceil_div(in_features, BLOCK_SIZE),
                dtype=torch.float32, device="meta",
            ),
        )
        self.compute_dtype = compute_dtype
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def matmul_2d(self, x_2d):
        return _DeepGEMMFp8Function.apply(x_2d, self.weight, self.weight_scale, self.bias)

    def dequantize_weight(self, dtype):
        return _dequantize_weight(self.weight, self.weight_scale, dtype)


class DeepGEMMFp4Linear(_DeepGEMMQuantizedLinear):
    """`nn.Linear` that stores a packed fp4 weight plus per-32-channel scales.

    `activation` selects what the activation side is quantized to: `"fp4"` for W4A4 or
    `"fp8"` for W4A8 (same weight format, higher-precision activations).
    """

    dtype_guarded_tensor_names = ("weight", "weight_scale")

    def __init__(self, in_features, out_features, bias, compute_dtype, activation="fp4"):
        with torch.device("meta"):
            super().__init__(in_features, out_features, bias=bias, dtype=compute_dtype)
        del self.weight
        self.register_buffer("weight", torch.empty(out_features, in_features // 2, dtype=torch.int8, device="meta"))
        self.register_buffer(
            "weight_scale",
            torch.empty(out_features, _ceil_div(in_features, FP4_GRANULARITY), dtype=torch.float32, device="meta"),
        )
        self.activation = activation
        self.compute_dtype = compute_dtype
        self._packed_weight_scale = None
        self._packed_weight_scale_key = None
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def packed_weight_scale(self):
        """The `int32` MN-major SFB layout, rebuilt whenever the float32 scales change."""
        scale = self.weight_scale
        key = (scale.data_ptr(), scale._version, scale.device)
        if self._packed_weight_scale_key != key:
            self._packed_weight_scale = _pack_scale(scale)
            self._packed_weight_scale_key = key
        return self._packed_weight_scale

    def matmul_2d(self, x_2d):
        return _DeepGEMMFp4Function.apply(
            x_2d, self.weight, self.weight_scale, self.packed_weight_scale(), self.bias, self.activation,
        )

    def dequantize_weight(self, dtype):
        return _dequantize_fp4_weight(self.weight, self.weight_scale, dtype)


def _disable_dynamo_in_forward():
    """The GEMM is a pybind kernel and the quantization is a Triton kernel: neither is traceable."""
    for cls in (DeepGEMMFp8Linear, DeepGEMMFp4Linear):
        if not getattr(cls.forward, "_torchdynamo_disable", False):
            cls.forward = torch._dynamo.disable(cls.forward)


class _DeepGEMMBackend(QuantBackend):
    """Shared backend logic: layer traversal hooks, quantization factories, checkpoint I/O."""

    project_url = "https://github.com/deepseek-ai/DeepGEMM"
    linear_cls = None

    def validate_environment(self):
        if not DEEP_GEMM_AVAILABLE:
            raise ImportError(
                "DeepGEMM is required for deep_gemm quantization methods. Install it from source "
                "with `git clone --recursive https://github.com/deepseek-ai/DeepGEMM && "
                "cd DeepGEMM && ./install.sh` (NVIDIA SM90 or SM100 GPU, CUDA >= 12.9, "
                "a compiler with C++20 <format> support)."
            ) from _DEEP_GEMM_IMPORT_ERROR
        if not torch.cuda.is_available():
            raise RuntimeError("DeepGEMM quantization requires a CUDA device.")
        major = torch.cuda.get_device_capability()[0]
        if major not in (9, 10):
            raise RuntimeError(
                f"DeepGEMM supports NVIDIA SM90 and SM100 GPUs, but this device is SM{major}0. "
                "Use another quantization method on this GPU."
            )
        _disable_dynamo_in_forward()

    def capabilities(self):
        return {
            "is_serializable": True,
            "is_differentiable": True,
            "is_compileable": False,
            "requires_calibration": False,
        }

    def quantized_linear_classes(self):
        return (self.linear_cls,)

    # Hooks each backend fills in.
    def _is_supported_layer(self, linear):
        raise NotImplementedError

    def _unsupported_hint(self):
        raise NotImplementedError

    def _quantize_weight(self, weight):
        raise NotImplementedError

    def _linear_kwargs(self):
        return {}

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        if not self._is_supported_layer(linear):
            print(
                f"DeepGEMM kept one `nn.Linear({linear.in_features}, {linear.out_features})` in "
                f"{linear.weight.dtype} (not quantized): {self._unsupported_hint()}."
            )
            return linear if model_device is None else linear.to(device=model_device)
        weight = linear.weight.data
        if compute_device is not None:
            weight = weight.to(device=compute_device)
        quantized_weight, weight_scale = self._quantize_weight(weight)
        quant_linear = self.linear_cls(
            linear.in_features, linear.out_features, bias=linear.bias is not None,
            compute_dtype=weight.dtype, **self._linear_kwargs(),
        )
        quant_linear.weight = quantized_weight
        quant_linear.weight_scale = weight_scale
        if linear.bias is not None:
            quant_linear.bias = torch.nn.Parameter(linear.bias.data.to(device=weight_scale.device), requires_grad=False)
        return quant_linear if model_device is None else quant_linear.to(device=model_device)

    def create_quantized_linear_shell(self, linear, compute_dtype):
        if not self._is_supported_layer(linear):
            return torch.nn.Linear(
                linear.in_features, linear.out_features,
                bias=linear.bias is not None, dtype=compute_dtype, device="meta",
            )
        return self.linear_cls(
            linear.in_features, linear.out_features, bias=linear.bias is not None,
            compute_dtype=compute_dtype, **self._linear_kwargs(),
        )

    def dequantize_to_linear(self, module, compute_dtype, compute_device=None, model_device=None):
        if compute_device is not None:
            module = module.to(device=compute_device)
        fp_weight = module.dequantize_weight(compute_dtype)
        linear = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None, device="meta")
        linear.weight = torch.nn.Parameter(fp_weight, requires_grad=False)
        if module.bias is not None:
            linear.bias = torch.nn.Parameter(
                module.bias.data.to(dtype=compute_dtype, device=fp_weight.device), requires_grad=False,
            )
        return linear if model_device is None else linear.to(device=model_device)


@register_quant_backend("deep_gemm")
class DeepGEMMQuantBackend(_DeepGEMMBackend):
    """Adapter over DeepGEMM's FP8 GEMM kernels for online W8A8 quantization (SM90 and SM100)."""

    linear_cls = DeepGEMMFp8Linear

    def _is_supported_layer(self, linear):
        return _is_supported(linear.in_features, linear.out_features)

    def _unsupported_hint(self):
        return f"both dimensions must be multiples of {BLOCK_SIZE}"

    def _quantize_weight(self, weight):
        return _quantize_weight(weight)


@register_quant_backend("deep_gemm_fp4")
class DeepGEMMFp4QuantBackend(_DeepGEMMBackend):
    """Adapter over DeepGEMM's FP4 GEMM kernels for online W4A4 / W4A8 quantization (SM100 only)."""

    linear_cls = DeepGEMMFp4Linear

    def validate_environment(self):
        super().validate_environment()
        major = torch.cuda.get_device_capability()[0]
        if major != 10:
            raise RuntimeError(
                f"`deep_gemm_fp4_*` needs an NVIDIA SM100 (Blackwell) GPU, but this device is SM{major}0: "
                "DeepGEMM's FP4 MMA is Blackwell-only. Use `deep_gemm_fp8_w8a8` on SM90 instead."
            )

    def _is_supported_layer(self, linear):
        return _is_supported_fp4(linear.in_features, linear.out_features, self.config.activation)

    def _unsupported_hint(self):
        block_k = FP4_BLOCK_K if self.config.activation == "fp4" else BLOCK_SIZE
        return f"in_features must be a multiple of {block_k} and out_features of {BLOCK_SIZE}"

    def _quantize_weight(self, weight):
        return _quantize_weight_fp4(weight)

    def _linear_kwargs(self):
        return {"activation": self.config.activation}


@dataclass
class DeepGEMMFp8Config(BackendConfig):
    block_size: int = field(init=False, default=BLOCK_SIZE)   # pinned by the method: DeepGEMM tiles K in 128


@dataclass
class DeepGEMMFp4Config(BackendConfig):
    activation: str = field(init=False, default="fp4")                  # pinned by the method
    weight_granularity: int = field(init=False, default=FP4_GRANULARITY)   # pinned: 1x32 blocks along K


@dataclass
class DeepGEMMFp4W4A8Config(DeepGEMMFp4Config):
    activation: str = field(init=False, default="fp8")


register_quant_method(
    "deep_gemm_fp8_w8a8", "deep_gemm", DeepGEMMFp8Config.from_kwargs,
    label="W8A8, fp8 E4M3 weight (128x128 block scale) + per-token fp8 dynamic activation (DeepGEMM)",
)
register_quant_method(
    "deep_gemm_fp4_w4a4", "deep_gemm_fp4", DeepGEMMFp4Config.from_kwargs,
    label="W4A4, fp4 E2M1 weight (1x32 block scale) + per-token fp4 dynamic activation (DeepGEMM, SM100)",
)
register_quant_method(
    "deep_gemm_fp4_w4a8", "deep_gemm_fp4", DeepGEMMFp4W4A8Config.from_kwargs,
    label="W4A8, fp4 E2M1 weight (1x32 block scale) + per-token fp8 dynamic activation (DeepGEMM, SM100)",
)
