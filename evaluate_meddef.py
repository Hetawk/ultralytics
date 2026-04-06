#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
MedDef2 Evaluation & Visualization  (backward-compatibility wrapper)
=====================================================================

This script delegates entirely to ``evaluate.py`` with MedDef-specific
defaults (variant names, depth scale map, attack presets).

All new code should use ``evaluate.py`` directly — it works with any
Ultralytics model including MedDef.

Usage mirrors evaluate.py exactly; legacy ``--variant`` / ``--depth``
flags are accepted for convenience and silently ignored by the engine.

Examples::

    python evaluate_meddef.py --model runs/classify/train/tbcr/full/weights/best.pt \\
                              --data /data2/enoch/tbcr --visualize --saliency
"""

from __future__ import annotations

import sys

# Re-export the canonical entry point
from evaluate import main, parse_args  # noqa: F401


VARIANTS = ["full", "no_def", "no_freq", "no_patch", "no_cbam", "baseline"]
DEPTH_SCALE = {"tiny": "n", "small": "s", "base": "m", "large": "l"}

ATTACK_PRESETS = {
    "fgsm": {"name": "FGSM", "eps_range": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]},
    "pgd": {"name": "PGD", "eps_range": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]},
    "bim": {"name": "BIM", "eps_range": [0.01, 0.02, 0.05, 0.1, 0.15]},
    "jsma": {"name": "JSMA", "eps_range": [0.05, 0.1, 0.15, 0.2]},
    "cw": {"name": "C&W", "eps_range": [0.01, 0.05, 0.1]},
}


if __name__ == "__main__":
    main()
