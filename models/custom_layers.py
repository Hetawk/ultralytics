"""Custom model glue for TorchVision backbones, stored alongside YAML configs."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn

from ultralytics.nn.modules import C2f, Conv, SPPF
from ultralytics.nn.modules.head import Detect

__all__ = [
    "FeatureSelect",
    "YOLOv8BackboneAdapter",
    "YOLOv8VGGBackboneAdapter",
    "YOLOv8DetectHead",
    "YOLOv8VGGDetect",
]


class FeatureSelect(nn.Module):
    """Select a single feature map from a list produced by a TorchVision backbone."""

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        """Return the feature tensor at the configured index."""
        if not isinstance(x, (list, tuple)):
            raise TypeError(
                f"FeatureSelect expected a sequence of tensors but received type {type(x).__name__}. "
                "Ensure the preceding TorchVision module was created with split=True."
            )
        try:
            return x[self.index]
        except IndexError as exc:  # pragma: no cover - defensive guard for misconfigured YAML
            raise RuntimeError(
                f"FeatureSelect could not access index {self.index} from a sequence of length {len(x)}."
            ) from exc


class YOLOv8BackboneAdapter(nn.Module):
    """Project TorchVision feature maps into YOLO-friendly channels with optional CSP blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 2,
        use_spp: bool = False,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1 for YOLOv8BackboneAdapter")
        self.out_channels = out_channels
        stages: list[nn.Module] = [Conv(in_channels, out_channels, k=1)]
        stages.append(C2f(out_channels, out_channels, n=depth, shortcut=False, e=expansion))
        if use_spp:
            stages.append(SPPF(out_channels, out_channels))
        self.net = nn.Sequential(*stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the projection/neck stack to a single feature map."""
        return self.net(x)


class YOLOv8VGGBackboneAdapter(YOLOv8BackboneAdapter):
    """Slightly deeper refinement stack tailored for VGG backbones."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 3,
        expansion: float = 0.5,
        use_spp: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            depth=depth,
            use_spp=use_spp,
            expansion=expansion,
        )


class YOLOv8DetectHead(Detect):
    """Wrapper around Detect head to expose a distinct YAML name and optional reg_max override."""

    def __init__(self, nc: int, ch: Iterable[int], reg_max: int | None = None) -> None:
        ch_tuple = tuple(ch)
        super().__init__(nc=nc, ch=ch_tuple)
        if reg_max is not None and reg_max != self.reg_max:
            self.reg_max = reg_max
            self.no = nc + self.reg_max * 4
            self.cv2 = nn.ModuleList(
                nn.Sequential(Conv(x, x, 3), Conv(x, x, 3), nn.Conv2d(x, 4 * self.reg_max, 1)) for x in ch_tuple
            )
            self.cv3 = nn.ModuleList(
                nn.Sequential(Conv(x, x, 3), Conv(x, x, 3), nn.Conv2d(x, nc, 1)) for x in ch_tuple
            )
            self.dfl = nn.Identity() if self.reg_max <= 1 else self.dfl


class YOLOv8VGGDetect(YOLOv8DetectHead):
    """Alias for clarity when wiring the VGG-backed configuration."""

    pass
