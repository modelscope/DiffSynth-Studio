import ctypes
import ctypes.util
import importlib.util
import os
import sys
import warnings
from dataclasses import dataclass, fields

import torch

from ..base import BackendConfig, QuantBackend, register_quant_backend
from ..config import register_quant_method

try:
    import entropack as ep

    _REQUIRED_ENTROPACK_API = (
        ep.CompressedLinear, ep.CompressedFP8Linear, ep.CompressedINT8Linear,
        ep.DFloat11Config, ep.TileANSConfig, ep.LatticeRANSConfig, ep.RawConfig,
    )
    ENTROPACK_AVAILABLE = True
    _ENTROPACK_IMPORT_ERROR = None
except (ImportError, AttributeError) as error:
    ep = None
    ENTROPACK_AVAILABLE = False
    _ENTROPACK_IMPORT_ERROR = error


if ENTROPACK_AVAILABLE:

    @dataclass
    class EntroPackDFloat11Config(BackendConfig, ep.DFloat11Config):
        linear_cls = ep.CompressedLinear

    @dataclass
    class EntroPackTileANSConfig(BackendConfig, ep.TileANSConfig):
        linear_cls = ep.CompressedLinear

    @dataclass
    class EntroPackLossyQuantConfig(BackendConfig, ep.LatticeRANSConfig):
        linear_cls = ep.CompressedLinear

    @dataclass
    class EntroPackFP8CodedConfig(BackendConfig, ep.LatticeRANSConfig):
        linear_cls = ep.CompressedFP8Linear

    @dataclass
    class EntroPackFP8RawConfig(BackendConfig, ep.RawConfig):
        linear_cls = ep.CompressedFP8Linear

    @dataclass
    class EntroPackINT8CodedConfig(BackendConfig, ep.LatticeRANSConfig):
        linear_cls = ep.CompressedINT8Linear
        use_convrot: bool = True
        convrot_groupsize: int = 256

    @dataclass
    class EntroPackINT8RawConfig(BackendConfig, ep.RawConfig):
        linear_cls = ep.CompressedINT8Linear
        use_convrot: bool = True
        convrot_groupsize: int = 256


def _patch_compile_graph_breaks():
    for cls in (ep.CompressedLinear, ep.CompressedFP8Linear, ep.CompressedINT8Linear):
        if not getattr(cls.forward, "_torchdynamo_disable", False):
            cls.forward = torch._dynamo.disable(cls.forward)


_CUBLAS_LIB = None
_SM75_PATCHED = False
_PRE_SM80_CACHE = {}
_CUBLAS_CHUNK_M = 32768


def _is_pre_sm80_cuda(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cached = _PRE_SM80_CACHE.get(idx)
    if cached is None:
        cached = torch.cuda.get_device_capability(idx)[0] < 8
        _PRE_SM80_CACHE[idx] = cached
    return cached


def _iter_cublas_candidates():
    cuda_major = (torch.version.cuda or "").split(".")[0]
    is_win = sys.platform == "win32"
    candidates = []

    try:
        spec = importlib.util.find_spec("nvidia.cublas")
    except (ImportError, ValueError):
        spec = None
    if spec and spec.submodule_search_locations:
        subdir = "bin" if is_win else "lib"
        for root in spec.submodule_search_locations:
            lib_dir = os.path.join(root, subdir)
            if cuda_major:
                fname = f"cublas64_{cuda_major}.dll" if is_win else f"libcublas.so.{cuda_major}"
                candidates.append(os.path.join(lib_dir, fname))
            if not is_win:
                for ver in ("13", "12", "11"):
                    candidates.append(os.path.join(lib_dir, f"libcublas.so.{ver}"))

    if is_win:
        if cuda_major:
            candidates.append(f"cublas64_{cuda_major}.dll")
        candidates.extend(("cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll"))
    else:
        if cuda_major:
            candidates.append(f"libcublas.so.{cuda_major}")
        candidates.extend(("libcublas.so.13", "libcublas.so.12", "libcublas.so.11", "libcublas.so"))

    found = ctypes.util.find_library("cublas")
    if found:
        candidates.append(found)

    return list(dict.fromkeys(c for c in candidates if c))


def _get_cublas_lib():
    global _CUBLAS_LIB
    if _CUBLAS_LIB is None:
        for name in _iter_cublas_candidates():
            try:
                lib = ctypes.CDLL(name)
                lib.cublasGemmEx.argtypes = [
                    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                    ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int,
                ]
                lib.cublasGemmEx.restype = ctypes.c_int
                _CUBLAS_LIB = lib
                break
            except (OSError, AttributeError):
                continue
        if _CUBLAS_LIB is None:
            raise RuntimeError("Could not load libcublas for sm_75 INT8 Tensor Core GEMM.")
    return _CUBLAS_LIB


def _cublas_int8_gemm_sm75(activation, codes, activation_scale, weight_scale, bias, out_dtype):
    orig_shape = activation.shape
    a = activation.reshape(-1, orig_shape[-1]).contiguous()
    b = codes.contiguous()
    m, k = a.shape
    n = b.shape[0]
    if m == 0:
        return torch.empty((*orig_shape[:-1], n), dtype=out_dtype, device=a.device)
    if k == 0:
        out = torch.zeros((m, n), dtype=torch.float32, device=a.device)
        if bias is not None:
            out.add_(bias.reshape(1, n).float())
        return out.to(out_dtype).view(*orig_shape[:-1], n)

    # cuBLAS INT8 Tensor Core GEMM (TN layout) requires N, K and leading dimensions to be multiples of 4.
    pad_k = (-k) % 4
    pad_n = (-n) % 4
    b_pad = torch.nn.functional.pad(b, (0, pad_k, 0, pad_n)) if (pad_k or pad_n) else b
    n_pad, k_pad = b_pad.shape

    a_scale = activation_scale.reshape(m, 1) if activation_scale.numel() == m else activation_scale.reshape(1, 1)
    w_scale_f32 = (weight_scale.reshape(1, n) if weight_scale.numel() == n else weight_scale.reshape(1, 1)).float()
    bias_f32 = bias.reshape(1, n).float() if bias is not None else None

    lib = _get_cublas_lib()
    alpha = ctypes.c_int32(1)
    beta = ctypes.c_int32(0)
    b_f32_fallback = None

    def _compute_chunk_f32(start: int, end: int) -> torch.Tensor:
        nonlocal b_f32_fallback
        ac = a[start:end]
        ac_pad = torch.nn.functional.pad(ac, (0, pad_k)) if pad_k else ac
        cm = end - start
        out_i32 = torch.empty((cm, n_pad), dtype=torch.int32, device=a.device)
        with torch.cuda.device(a.device):
            handle = torch.cuda.current_blas_handle()
            # CUBLAS_OP_T=1, CUBLAS_OP_N=0, CUDA_R_8I=3, CUDA_R_32I=10, CUBLAS_COMPUTE_32I=72, CUBLAS_GEMM_DEFAULT_TENSOR_OP=99
            st = lib.cublasGemmEx(
                ctypes.c_void_p(handle), 1, 0,
                n_pad, cm, k_pad,
                ctypes.byref(alpha),
                ctypes.c_void_p(b_pad.data_ptr()), 3, k_pad,
                ctypes.c_void_p(ac_pad.data_ptr()), 3, k_pad,
                ctypes.byref(beta),
                ctypes.c_void_p(out_i32.data_ptr()), 10, n_pad,
                72, 99,
            )
        if st == 0:
            chunk_f32 = (out_i32[:, :n] if pad_n else out_i32).float()
            del out_i32
        else:
            del out_i32
            if b_f32_fallback is None:
                b_f32_fallback = b.float().t()
            chunk_f32 = torch.matmul(ac.float(), b_f32_fallback)

        a_s = a_scale[start:end].float() if a_scale.shape[0] == m else a_scale.float()
        chunk_f32.mul_(a_s).mul_(w_scale_f32)
        if bias_f32 is not None:
            chunk_f32.add_(bias_f32)
        return chunk_f32

    # Turing cublasGemmEx(INT8) returns status=15 for m >= 65536; chunk along m <= 32768 without concatenating fp32 copies.
    if m <= _CUBLAS_CHUNK_M:
        return _compute_chunk_f32(0, m).to(out_dtype).view(*orig_shape[:-1], n)

    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    for start in range(0, m, _CUBLAS_CHUNK_M):
        end = min(start + _CUBLAS_CHUNK_M, m)
        chunk_f32 = _compute_chunk_f32(start, end)
        out[start:end].copy_(chunk_f32)
        del chunk_f32
    return out.view(*orig_shape[:-1], n)


def _patch_sm75_int8_support():
    """Enable CompressedINT8Linear on Turing (sm_75, e.g. Tesla T4) via cuBLAS INT8 Tensor Cores."""
    global _SM75_PATCHED
    if _SM75_PATCHED or not torch.cuda.is_available():
        return
    capabilities = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
    if all(major >= 8 for major, _ in capabilities):
        _SM75_PATCHED = True
        return
    if getattr(ep.CompressedINT8Linear, "min_capability", (8, 0)) <= (7, 5):
        _SM75_PATCHED = True
        return
    try:
        from entropack.linear import utils as ep_utils
        qk = ep_utils.load_quant_kernels()
        if qk is None:
            warnings.warn(
                "entropack quant kernels returned None; sm_75 INT8 Tensor Core patch was not applied.",
                RuntimeWarning,
            )
            return
        orig_int8_gemm = qk.int8_gemm
        orig_quantize_rows = qk.quantize_rows

        def _patched_int8_gemm(activation, codes, activation_scale, weight_scale, bias, out_dtype):
            if _is_pre_sm80_cuda(activation.device):
                return _cublas_int8_gemm_sm75(activation, codes, activation_scale, weight_scale, bias, out_dtype)
            return orig_int8_gemm(activation, codes, activation_scale, weight_scale, bias, out_dtype)

        def _patched_quantize_rows(tensor, code_dtype, code_max, rounds_to_integer):
            if tensor.dtype == torch.bfloat16 and _is_pre_sm80_cuda(tensor.device):
                tensor = tensor.float()
            return orig_quantize_rows(tensor, code_dtype, code_max, rounds_to_integer)

        qk.int8_gemm = _patched_int8_gemm
        qk.quantize_rows = _patched_quantize_rows
        if any(cap >= (7, 5) for cap in capabilities) and ep.CompressedINT8Linear.min_capability > (7, 5):
            ep.CompressedINT8Linear.min_capability = (7, 5)
        _SM75_PATCHED = True
    except Exception as exc:
        warnings.warn(
            f"Failed to apply sm_75 INT8 support patch for entropack ({type(exc).__name__}: {exc}).",
            RuntimeWarning,
        )


@register_quant_backend("entropack")
class EntroPackQuantBackend(QuantBackend):
    """Adapter over entropack's `CompressedLinear` family; a method's config is the config its layers get."""

    project_url = "https://github.com/modelscope/entropack"

    def validate_environment(self):
        if not ENTROPACK_AVAILABLE:
            raise ImportError(
                "entropack with the CompressedLinear / CompressedFP8Linear / CompressedINT8Linear "
                "classes is required for this quantization method. Install it via "
                '`pip install "entropack[cuda13]"` (use `cuda12` for a CUDA 12 environment) '
                'or `pip install "diffsynth[quant]"`.'
            ) from _ENTROPACK_IMPORT_ERROR
        _patch_compile_graph_breaks()
        _patch_sm75_int8_support()

    def capabilities(self):
        return {
            "is_serializable": True,
            "is_differentiable": True,
            "is_compileable": True,
            "requires_calibration": False,
        }

    def quantized_linear_classes(self):
        return (ep.CompressedLinear, ep.CompressedFP8Linear, ep.CompressedINT8Linear)

    def unflatten_state_dict(self, state_dict, metadata):
        return state_dict

    def _compression_config_cls(self):
        for cls in type(self.config).__mro__:
            if cls.__module__.startswith("entropack") and issubclass(cls, ep.CompressionConfig):
                return cls
        return None

    def _compression_config(self):
        cls = self._compression_config_cls()
        if cls is None:
            return self.config
        names = {field.name for field in fields(cls) if field.init}
        return cls(**{name: getattr(self.config, name) for name in names})

    def linear_spec_configs(self):
        cls = self._compression_config_cls()
        if cls is None:
            return {}
        compression = {field.name for field in fields(cls)}
        return {
            field.name: getattr(self.config, field.name)
            for field in fields(self.config)
            if field.init and field.name not in compression
        }

    def create_quantized_linear(self, linear, compute_device=None, model_device=None):
        linear.requires_grad_(False)
        if compute_device is not None:
            linear = linear.to(device=compute_device)
        quantized = self.config.linear_cls.from_linear(
            linear, config=self._compression_config(), **self.linear_spec_configs(),
        )
        return quantized if model_device is None else quantized.to(device=model_device)

    def create_quantized_linear_shell(self, linear, compute_dtype):
        return self.config.linear_cls(
            linear.in_features, linear.out_features, bias=linear.bias is not None,
            dtype=compute_dtype, config=self._compression_config(), **self.linear_spec_configs(),
        )

    def dequantize_to_linear(self, module, compute_dtype, compute_device=None, model_device=None):
        if compute_device is not None:
            module = module.to(device=compute_device)
        weight = module.dequantize(compute_device).to(compute_dtype)
        linear = torch.nn.Linear(module.in_features, module.out_features, bias=module.bias is not None, device="meta")
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        if module.bias is not None:
            linear.bias = torch.nn.Parameter(
                module.bias.data.to(dtype=compute_dtype, device=weight.device), requires_grad=False,
            )
        return linear if model_device is None else linear.to(device=model_device)


def _df11_config(kwargs):
    return EntroPackDFloat11Config.from_kwargs(kwargs)


def _tile_ans_config(kwargs):
    return EntroPackTileANSConfig.from_kwargs(kwargs)


def _lossy_config(kwargs):
    return EntroPackLossyQuantConfig.from_kwargs(kwargs)


def _fp8_coded_config(kwargs):
    return EntroPackFP8CodedConfig.from_kwargs(kwargs)


def _fp8_raw_config(kwargs):
    return EntroPackFP8RawConfig.from_kwargs(kwargs)


def _int8_coded_config(kwargs):
    return EntroPackINT8CodedConfig.from_kwargs(kwargs)


def _int8_raw_config(kwargs):
    return EntroPackINT8RawConfig.from_kwargs(kwargs)


register_quant_method("entropack_lossless_df11", "entropack", _df11_config, label="lossless, dfloat11")
register_quant_method("entropack_lossless_tile_ans", "entropack", _tile_ans_config, label="lossless, tile_ans")
register_quant_method("entropack_lossy_quant", "entropack", _lossy_config, label="lossy, lattice rate")
register_quant_method("entropack_lossy_quant_fp8", "entropack", _fp8_coded_config, label="W8A8, fp8, lattice-coded")
register_quant_method("entropack_lossy_quant_fp8_raw", "entropack", _fp8_raw_config, label="W8A8, fp8, codes verbatim")
register_quant_method("entropack_lossy_quant_int8", "entropack", _int8_coded_config, label="W8A8, int8, lattice-coded")
register_quant_method("entropack_lossy_quant_int8_raw", "entropack", _int8_raw_config, label="W8A8, int8, codes verbatim")
