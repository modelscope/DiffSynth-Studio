import optimum
import torch
from optimum.quanto.tensor.packed import PackedTensor
from optimum.quanto.tensor.weights.qbits import WeightQBitsTensor
from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

_TORCH_TENSOR_DATA_DESCRIPTOR = torch.Tensor.data

if torch.cuda.is_available():
    # the marlin fp8 kernel needs some help with dtype casting for some reason
    # see: https://github.com/huggingface/optimum-quanto/pull/296#issuecomment-2380719201
    if torch.device("cuda").type == "cuda" and torch.version.cuda:
        from optimum.quanto.library.extensions.cuda import ext as quanto_ext

        # Save the original operator
        original_gemm_f16f8_marlin = torch.ops.quanto.gemm_f16f8_marlin

        def fp8_marlin_gemm_wrapper(
            a: torch.Tensor,
            b_q_weight: torch.Tensor,
            b_scales: torch.Tensor,
            workspace: torch.Tensor,
            num_bits: int,
            size_m: int,
            size_n: int,
            size_k: int,
        ) -> torch.Tensor:
            # Ensure 'a' has the correct dtype
            a = a.to(b_scales.dtype)
            # Call the original operator
            return original_gemm_f16f8_marlin(
                a,
                b_q_weight,
                b_scales,
                workspace,
                num_bits,
                size_m,
                size_n,
                size_k,
            )

        # Monkey-patch the operator
        torch.ops.quanto.gemm_f16f8_marlin = fp8_marlin_gemm_wrapper

    class TinyGemmQBitsLinearFunction(optimum.quanto.tensor.function.QuantizedLinearFunction):
        @staticmethod
        def forward(ctx, input, other, bias):
            ctx.save_for_backward(input, other)
            if type(input) is not torch.Tensor:
                input = input.dequantize()
            in_features = input.shape[-1]
            out_features = other.shape[0]
            output_shape = input.shape[:-1] + (out_features,)
            output = torch._weight_int4pack_mm(
                input.view(-1, in_features).to(dtype=other.dtype),
                other._data._data,
                other._group_size,
                other._scale_shift,
            )
            output = output.view(output_shape)
            if bias is not None:
                output = output + bias
            return output

    from optimum.quanto.tensor.weights import tinygemm

    tinygemm.qbits.TinyGemmQBitsLinearFunction = TinyGemmQBitsLinearFunction


class WeightQBytesLinearFunction(optimum.quanto.tensor.function.QuantizedLinearFunction):
    @staticmethod
    def forward(ctx, input, other, bias=None):
        ctx.save_for_backward(input, other)
        input_device = getattr(input, "device", None)
        if input_device is None and hasattr(input, "_data"):
            input_device = input._data.device

        if input_device is not None and hasattr(other, "_data"):
            backing_data = other._data
            backing_scale = getattr(other, "_scale", None)
            if backing_data.device != input_device:
                other._data = backing_data.to(input_device, non_blocking=True)
            if backing_scale is not None and hasattr(backing_scale, "device") and backing_scale.device != input_device:
                other._scale = backing_scale.to(input_device, non_blocking=True)

        if isinstance(input, optimum.quanto.tensor.QBytesTensor):
            output = torch.ops.quanto.qbytes_mm(input._data, other._data, input._scale * other._scale)
        else:
            in_features = input.shape[-1]
            out_features = other.shape[0]
            output_shape = input.shape[:-1] + (out_features,)
            output = torch.ops.quanto.qbytes_mm(input.reshape(-1, in_features), other._data, other._scale)
            output = output.view(output_shape)
        if bias is not None:
            output = output + bias
        return output


optimum.quanto.tensor.weights.qbytes.WeightQBytesLinearFunction = WeightQBytesLinearFunction


def reshape_qlf_backward(ctx, gO):
    # another one where we need .reshape instead of .view
    input_gO = other_gO = bias_gO = None
    input, other = ctx.saved_tensors
    out_features, in_features = other.shape
    if ctx.needs_input_grad[0]:
        # grad(A@(B.t()) = gO => grad(A) = gO@(B.t().t()) = gO@B
        input_gO = torch.matmul(gO, other)
    if ctx.needs_input_grad[1]:
        # grad(B@A.t()) = gO.t() => grad(B) = gO.t()@(A.t().t()) = gO.t()@A
        other_gO = torch.matmul(
            gO.reshape(-1, out_features).t(),
            input.to(gO.dtype).reshape(-1, in_features),
        )
    if ctx.needs_input_grad[2]:
        # Bias gradient is the sum on all dimensions but the last one
        dim = tuple(range(gO.ndim - 1))
        bias_gO = gO.sum(dim)
    return input_gO, other_gO, bias_gO


optimum.quanto.tensor.function.QuantizedLinearFunction.backward = reshape_qlf_backward


def _bridge_storage_accessors(tensor_cls, data_attr: str) -> None:
    if getattr(tensor_cls, "_simpletuner_storage_bridge_applied", False):
        return

    def _backing_tensor(self):
        backing = getattr(self, data_attr, None)
        if backing is None:
            raise AttributeError(f"{tensor_cls.__name__} is missing expected backing tensor '{data_attr}'")
        return backing

    def _data_ptr(self):
        return _backing_tensor(self).data_ptr()

    def _untyped_storage(self):
        return _backing_tensor(self).untyped_storage()

    def _storage(self):
        return _backing_tensor(self).storage()

    tensor_cls.data_ptr = _data_ptr  # type: ignore[assignment]
    tensor_cls.untyped_storage = _untyped_storage  # type: ignore[assignment]
    tensor_cls.storage = _storage  # type: ignore[assignment]
    tensor_cls._simpletuner_storage_bridge_applied = True  # type: ignore[attr-defined]


_bridge_storage_accessors(WeightQBytesTensor, "_data")
_bridge_storage_accessors(WeightQBitsTensor, "_data")
_bridge_storage_accessors(PackedTensor, "_data")


def _mirror_tensor_data_property(tensor_cls, attrs: tuple[str, ...]) -> None:
    if getattr(tensor_cls, "_simpletuner_data_bridge_applied", False):
        return

    def _data_get(self):
        return _TORCH_TENSOR_DATA_DESCRIPTOR.__get__(self, type(self))

    def _data_set(self, value):
        _TORCH_TENSOR_DATA_DESCRIPTOR.__set__(self, value)
        for attr in attrs:
            if hasattr(value, attr) and hasattr(self, attr):
                setattr(self, attr, getattr(value, attr))

    tensor_cls.data = property(_data_get, _data_set)  # type: ignore[assignment]
    tensor_cls._simpletuner_data_bridge_applied = True  # type: ignore[attr-defined]


_mirror_tensor_data_property(WeightQBytesTensor, ("_data", "_scale", "activation_qtype", "_axis", "_qtype"))
_mirror_tensor_data_property(WeightQBitsTensor, ("_data", "_scale", "_shift", "_axis", "_qtype"))
