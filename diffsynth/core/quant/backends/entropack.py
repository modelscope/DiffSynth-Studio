import ctypes
import ctypes.util
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


def _get_cublas_lib():
    global _CUBLAS_LIB
    if _CUBLAS_LIB is None:
        for name in ("libcublas.so.12", "libcublas.so.11", "libcublas.so", ctypes.util.find_library("cublas")):
            if not name:
                continue
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
            except OSError:
                continue
        if _CUBLAS_LIB is None:
            raise RuntimeError("Could not load libcublas for sm_75 INT8 Tensor Core GEMM.")
    return _CUBLAS_LIB


def _cublas_int8_gemm_sm75(activation, codes, activation_scale, weight_scale, bias, out_dtype):
    a = activation.contiguous()
    b = codes.contiguous()
    m, k = a.shape
    n = b.shape[0]
    handle = torch.cuda.current_blas_handle()
    lib = _get_cublas_lib()
    alpha = ctypes.c_int32(1)
    beta = ctypes.c_int32(0)

    def _run_cublas_chunk(a_chunk):
        cm = a_chunk.shape[0]
        out_i32 = torch.empty((cm, n), dtype=torch.int32, device=a.device)
        # CUBLAS_OP_T=1, CUBLAS_OP_N=0, CUDA_R_8I=3, CUDA_R_32I=10, CUBLAS_COMPUTE_32I=72, CUBLAS_GEMM_DEFAULT_TENSOR_OP=99
        st = lib.cublasGemmEx(
            ctypes.c_void_p(handle), 1, 0,
            n, cm, k,
            ctypes.byref(alpha),
            ctypes.c_void_p(b.data_ptr()), 3, k,
            ctypes.c_void_p(a_chunk.data_ptr()), 3, k,
            ctypes.byref(beta),
            ctypes.c_void_p(out_i32.data_ptr()), 10, n,
            72, 99,
        )
        return st, out_i32

    # Turing cublasGemmEx(INT8) can return status=15 for m >= 65536; chunk along m <= 32768
    if m <= 32768:
        status, out_i32 = _run_cublas_chunk(a)
        if status == 0:
            out = out_i32.float()
        else:
            out = torch.matmul(a.float(), b.float().t())
    else:
        chunks = []
        for start in range(0, m, 32768):
            ac = a[start : start + 32768]
            st, ci32 = _run_cublas_chunk(ac)
            chunks.append(ci32.float() if st == 0 else torch.matmul(ac.float(), b.float().t()))
        out = torch.cat(chunks, dim=0)

    out.mul_(activation_scale.unsqueeze(1)).mul_(weight_scale.unsqueeze(0))
    if bias is not None:
        out.add_(bias.float().unsqueeze(0))
    return out.to(out_dtype)


def _patch_sm75_int8_support():
    """Enable CompressedINT8Linear on Turing (sm_75, e.g. Tesla T4) via cuBLAS INT8 Tensor Cores."""
    global _SM75_PATCHED
    if _SM75_PATCHED or not torch.cuda.is_available():
        return
    if torch.cuda.get_device_capability(0)[0] >= 8:
        _SM75_PATCHED = True
        return
    if ep.CompressedINT8Linear.min_capability > (7, 5):
        ep.CompressedINT8Linear.min_capability = (7, 5)
    try:
        from entropack.linear import utils as ep_utils
        qk = ep_utils.load_quant_kernels()
        if qk is not None:
            orig_int8_gemm = qk.int8_gemm
            orig_quantize_rows = qk.quantize_rows

            def _patched_int8_gemm(activation, codes, activation_scale, weight_scale, bias, out_dtype):
                if activation.is_cuda and torch.cuda.get_device_capability(activation.device)[0] < 8:
                    return _cublas_int8_gemm_sm75(activation, codes, activation_scale, weight_scale, bias, out_dtype)
                return orig_int8_gemm(activation, codes, activation_scale, weight_scale, bias, out_dtype)

            def _patched_quantize_rows(tensor, code_dtype, code_max, rounds_to_integer):
                if tensor.is_cuda and tensor.dtype == torch.bfloat16 and torch.cuda.get_device_capability(tensor.device)[0] < 8:
                    tensor = tensor.float()
                return orig_quantize_rows(tensor, code_dtype, code_max, rounds_to_integer)

            qk.int8_gemm = _patched_int8_gemm
            qk.quantize_rows = _patched_quantize_rows
    except Exception:
        pass
    _SM75_PATCHED = True


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
