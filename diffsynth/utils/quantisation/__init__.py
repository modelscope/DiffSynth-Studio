import logging
import torch

logger = logging.getLogger(__name__)


def _quanto_type_map(model_precision: str):
    if model_precision is None or model_precision == "no_change":
        return None
    from optimum.quanto import qfloat8, qfloat8_e4m3fnuz, qint2, qint4, qint8

    mp = model_precision.lower()
    if mp == "int2-quanto":
        return qint2
    elif mp == "int4-quanto":
        return qint4
    elif mp == "int8-quanto":
        return qint8
    elif mp in ("fp8-quanto", "fp8uz-quanto"):
        if torch.backends.mps.is_available():
            logger.warning(
                "MPS doesn't support dtype float8, please use bf16/fp16/int8-quanto instead."
            )
            return None
        return qfloat8 if mp == "fp8-quanto" else qfloat8_e4m3fnuz
    else:
        raise ValueError(f"Invalid quantisation level: {model_precision}")


def _quanto_model(
    model,
    model_precision,
    base_model_precision=None,
    quantize_activations: bool = False,
):
    try:
        from optimum.quanto import quantize, freeze  # noqa
        # 仅仅 import，就会触发 quanto_workarounds 里的 monkeypatch
        from diffsynth.utils.quantisation import quanto_workarounds  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "To use Quanto, please install the optimum library: `pip install \"optimum[quanto]\"`"
        ) from e

    if model is None:
        return model
    if model_precision is None:
        model_precision = base_model_precision
    if model_precision in (None, "no_change"):
        logger.info("...No quantisation applied to %s.", model.__class__.__name__)
        return model

    logger.info("Quantising %s. Using %s.", model.__class__.__name__, model_precision)
    weight_quant = _quanto_type_map(model_precision)
    if weight_quant is None:
        logger.info("Quantisation level %s resolved to None, skipping.", model_precision)
        return model

    extra_quanto_args = {}
    if quantize_activations:
        logger.info("Quanto: Freezing model weights and activations")
        extra_quanto_args["activations"] = weight_quant
    else:
        logger.info("Quanto: Freezing model weights only")

    quantize(model, weights=weight_quant, **extra_quanto_args)
    freeze(model)
    return model


def get_quant_fn(base_model_precision):
    if base_model_precision is None:
        return None
    precision = base_model_precision.lower()
    if precision == "no_change":
        return None
    if "quanto" in precision:
        return _quanto_model
    # 这里先不支持 torchao
    return None


def quantise_model(
    model=None,
    text_encoders: list = None,
    controlnet=None,
    ema=None,
    args=None,
    return_dict: bool = False,
):
    # 展开 text_encoders，最多支持 4 个以兼容 SimpleTuner 的接口
    te1 = te2 = te3 = te4 = None
    if text_encoders is not None:
        if len(text_encoders) > 0:
            te1 = text_encoders[0]
        if len(text_encoders) > 1:
            te2 = text_encoders[1]
        if len(text_encoders) > 2:
            te3 = text_encoders[2]
        if len(text_encoders) > 3:
            te4 = text_encoders[3]

    models = [
        (
            model,
            {
                "quant_fn": get_quant_fn(args.base_model_precision),
                "model_precision": args.base_model_precision,
                "quantize_activations": getattr(args, "quantize_activations", False),
            },
        ),
        (
            controlnet,
            {
                "quant_fn": get_quant_fn(args.base_model_precision),
                "model_precision": args.base_model_precision,
                "quantize_activations": getattr(args, "quantize_activations", False),
            },
        ),
        (
            te1,
            {
                "quant_fn": get_quant_fn(args.text_encoder_1_precision),
                "model_precision": args.text_encoder_1_precision,
                "base_model_precision": args.base_model_precision,
            },
        ),
        (
            te2,
            {
                "quant_fn": get_quant_fn(args.text_encoder_2_precision),
                "model_precision": args.text_encoder_2_precision,
                "base_model_precision": args.base_model_precision,
            },
        ),
        (
            te3,
            {
                "quant_fn": get_quant_fn(args.text_encoder_3_precision),
                "model_precision": args.text_encoder_3_precision,
                "base_model_precision": args.base_model_precision,
            },
        ),
        (
            te4,
            {
                "quant_fn": get_quant_fn(args.text_encoder_4_precision),
                "model_precision": args.text_encoder_4_precision,
                "base_model_precision": args.base_model_precision,
            },
        ),
        (
            ema,
            {
                "quant_fn": get_quant_fn(args.base_model_precision),
                "model_precision": args.base_model_precision,
                "quantize_activations": getattr(args, "quantize_activations", False),
            },
        ),
    ]

    for i, (m, qargs) in enumerate(models):
        quant_fn = qargs["quant_fn"]
        if m is None or quant_fn is None:
            continue
        quant_args_combined = {
            "model_precision": qargs["model_precision"],
            "base_model_precision": qargs.get("base_model_precision", args.base_model_precision),
            "quantize_activations": qargs.get(
                "quantize_activations", getattr(args, "quantize_activations", False)
            ),
        }
        logger.info("Quantising %s with %s", m.__class__.__name__, quant_args_combined)
        models[i] = (quant_fn(m, **quant_args_combined), qargs)

    # 解包
    model, controlnet, te1, te2, te3, te4, ema = [m for (m, _) in models]

    # 重新打包 text_encoders
    new_text_encoders = []
    if te1 is not None:
        new_text_encoders.append(te1)
    if te2 is not None:
        new_text_encoders.append(te2)
    if te3 is not None:
        new_text_encoders.append(te3)
    if te4 is not None:
        new_text_encoders.append(te4)
    if len(new_text_encoders) == 0:
        new_text_encoders = None

    if return_dict:
        return {
            "model": model,
            "text_encoders": new_text_encoders,
            "controlnet": controlnet,
            "ema": ema,
        }

    return model, new_text_encoders, controlnet, ema