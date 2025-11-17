"""Simple registry for custom (non-native) Ultralytics models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import torch.nn as nn

from ultralytics.models import faster_rcnn as faster_rcnn_module

ModelBuilder = Callable[..., nn.Module]


@dataclass
class ModelEntry:
    """Metadata describing a custom model builder."""

    name: str
    builder: ModelBuilder
    description: str = ""
    default_kwargs: Dict[str, Any] = field(default_factory=dict)

    def build(self, **kwargs) -> nn.Module:
        params = {**self.default_kwargs, **kwargs}
        return self.builder(**params)


_CUSTOM_MODELS: Dict[str, ModelEntry] = {}


def register_model(name: str, builder: ModelBuilder, *, description: str = "", default_kwargs: Dict[str, Any] | None = None) -> None:
    entry = ModelEntry(name=name, builder=builder, description=description, default_kwargs=default_kwargs or {})
    _CUSTOM_MODELS[name] = entry


def get_model_entry(name: str) -> ModelEntry:
    try:
        return _CUSTOM_MODELS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_CUSTOM_MODELS)) or "<none>"
        raise KeyError(f"Unknown model '{name}'. Available models: {available}") from exc


def available_models() -> Dict[str, ModelEntry]:
    return {k: v for k, v in _CUSTOM_MODELS.items()}


# Register built-in custom models here.
register_model(
    "faster_rcnn",
    builder=faster_rcnn_module.build_model,
    description="TorchVision Faster R-CNN (ResNet-50 FPN) baseline",
    default_kwargs={"trainable_backbone_layers": 3},
)
