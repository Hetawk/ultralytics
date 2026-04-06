# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
MedDef Model Builder
====================
Single responsibility: resolve a YAML config → return the right
MedDef2 nn.Module instance (correct variant + scale).

Keeping this logic here means ``ultralytics/nn/tasks.py`` stays thin —
``MedDefModel._from_yaml`` just calls ``build_meddef_model()``.

Variant map
-----------
YAML field ``variant``   → meddef2 class
─────────────────────────────────────────
full     (default)       → MedDef2_T
no_def                   → MedDef2_T_NoDef
no_freq                  → MedDef2_T_NoFreq
no_patch                 → MedDef2_T_NoPatch
no_cbam                  → MedDef2_T_NoCBAM
baseline                 → MedDef2_T_Baseline

Scale map  (YAML ``scales`` table: [depth_scale, width_scale, patch_size, embed_dim, num_heads])
-----------
n  →  6 blocks,  384-dim,  6 heads   (nano)
s  →  9 blocks,  576-dim,  9 heads   (small)
m  → 12 blocks,  768-dim, 12 heads   (medium / base)
l  → 15 blocks,  960-dim, 15 heads   (large)
x  → 18 blocks, 1152-dim, 18 heads   (xlarge)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    pass   # avoid circular at runtime

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Variant → class-name table  (resolved at call-time via getattr to avoid
# importing the whole meddef2 package at module load)
# ─────────────────────────────────────────────────────────────────────────────
VARIANT_MAP: dict[str, str] = {
    "full":     "MedDef2_T",
    "no_def":   "MedDef2_T_NoDef",
    "no_freq":  "MedDef2_T_NoFreq",
    "no_patch": "MedDef2_T_NoPatch",
    "no_cbam":  "MedDef2_T_NoCBAM",
    "baseline": "MedDef2_T_Baseline",
}

# Default arch params when no scale table is present in the YAML
_DEFAULTS = dict(depth_scale=1.0, patch_size=16, embed_dim=768, num_heads=12)


def _parse_scale(yaml_cfg: dict) -> dict:
    """Return arch kwargs derived from the YAML ``scales`` table.

    Args:
        yaml_cfg: Raw YAML dict (already loaded).

    Returns:
        dict with keys: depth_scale, patch_size, embed_dim, num_heads.
    """
    scales = yaml_cfg.get("scales", {})
    # Ultralytics may inject scale="" when the model filename has no scale suffix.
    # Treat empty/None scale as the default "n" variant.
    scale = yaml_cfg.get("scale") or "n"

    if scales and scale in scales:
        entry = scales[scale]
        if len(entry) >= 5:
            depth_scale, _, patch_size, embed_dim, num_heads = entry[:5]
        else:
            depth_scale = entry[0]
            patch_size, embed_dim, num_heads = 16, 768, 12
    else:
        depth_scale, patch_size, embed_dim, num_heads = (
            _DEFAULTS["depth_scale"],
            _DEFAULTS["patch_size"],
            _DEFAULTS["embed_dim"],
            _DEFAULTS["num_heads"],
        )

    return dict(
        depth_scale=depth_scale,
        patch_size=int(patch_size),
        embed_dim=int(embed_dim),
        num_heads=int(num_heads),
    )


def build_meddef_model(
    yaml_cfg: dict,
    ch: int,
    nc: int,
) -> nn.Module:
    """Construct and return the appropriate MedDef2 nn.Module.

    This is the **only** place that knows how to translate a parsed YAML dict
    into a concrete model instance.  ``MedDefModel._from_yaml`` and any
    other caller should go through here.

    Args:
        yaml_cfg: Fully resolved YAML dict (``nc`` already overridden if needed).
        ch:       Number of input channels.
        nc:       Number of output classes.

    Returns:
        Instantiated MedDef2 variant as an ``nn.Module``.

    Raises:
        ValueError: If the variant name is unrecognised.
    """
    import ultralytics.models.meddef.meddef2 as _pkg  # lazy — avoids circular

    variant  = yaml_cfg.get("variant", "full")
    cls_name = VARIANT_MAP.get(variant)
    if cls_name is None:
        raise ValueError(
            f"Unknown MedDef variant '{variant}'. "
            f"Valid options: {list(VARIANT_MAP)}"
        )

    ModelCls = getattr(_pkg, cls_name)
    scale_kw = _parse_scale(yaml_cfg)
    depth    = max(1, int(12 * scale_kw["depth_scale"]))

    effective_scale = yaml_cfg.get("scale") or "n"
    logger.info(
        "MedDef build: variant='%s' → %s | scale=%s | "
        "embed_dim=%d | depth=%d | heads=%d",
        variant, cls_name,
        effective_scale,
        scale_kw["embed_dim"], depth, scale_kw["num_heads"],
    )

    return ModelCls(
        img_size=yaml_cfg.get("imgsz", 224),
        patch_size=scale_kw["patch_size"],
        in_channels=ch,
        num_classes=nc,
        embed_dim=scale_kw["embed_dim"],
        depth=depth,
        num_heads=scale_kw["num_heads"],
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    )


__all__ = ["VARIANT_MAP", "build_meddef_model"]
