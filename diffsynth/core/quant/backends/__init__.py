import importlib

_LAZY_BACKENDS = {
    "bitsandbytes": ".bitsandbytes",
    "torchao": ".torchao",
    "comfy_kitchen": ".comfy_kitchen",
}
_loaded = set()


def load_backend(name):
    module = _LAZY_BACKENDS.get(name)
    if module is not None and name not in _loaded:
        importlib.import_module(module, __name__)
        _loaded.add(name)


def load_all_backends():
    for name in _LAZY_BACKENDS:
        load_backend(name)


def load_backend_for_method(method):
    from ..config import QUANT_METHODS
    for name in _LAZY_BACKENDS:
        load_backend(name)
        if method in QUANT_METHODS:
            return
