"""Backbone registry."""
from importlib import import_module
from ..config import BackboneConfig

_MODULES = {"TranAD": "tranad", "TimesNet": "timesnet", "DTAAD": "dtaad",
            "AnomalyTransformer": "anomaly_transformer", "PatchTST": "patchtst"}
_ALIASES = {"AnomalyTransformerAD": "AnomalyTransformer", "PatchTSTAD": "PatchTST"}


def available_backbones():
    return tuple(_MODULES)


def build_backbone(name, config=None, **kwargs):
    """Construct an untrained backbone; caller loads its own pretrained weights."""
    name = _ALIASES.get(name, name)
    if name not in _MODULES:
        raise ValueError(f"Unknown backbone {name!r}; choose from {available_backbones()}")
    if config is not None and kwargs:
        raise ValueError("Supply config or keyword configuration, not both")
    config = config if config is not None else BackboneConfig(**kwargs)
    if name == "PatchTST" and config.seq_len < 16:
        raise ValueError("The bundled PatchTST uses patch_len=16; seq_len must be >=16")
    if name == "TimesNet" and not 1 <= config.top_k <= config.seq_len // 2:
        raise ValueError("TimesNet top_k must fit the available non-DC FFT bins")
    module = import_module(f".{_MODULES[name]}", __name__)
    return module.Model(config)
