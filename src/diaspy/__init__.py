"""Diaspy's public modules, loaded only when requested."""
from importlib import import_module

__version__ = "0.1.0"
__all__ = ["signatures", "agents", "responders", "utils"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
