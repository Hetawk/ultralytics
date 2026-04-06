"""
MedDef Enhanced Metrics Module  (backward-compatibility wrapper)
================================================================

This module re-exports everything from the model-agnostic
``classify_metrics`` module.  All new code should import from::

    from ultralytics.utils.classify_metrics import ClassificationMetrics

The ``MedDefMetrics`` alias still works for existing code.
"""

# Re-export everything from the generic module
from ultralytics.utils.classify_metrics import (  # noqa: F401
    ClassificationMetrics,
    ClassificationMetrics as MedDefMetrics,
    compute_metrics,
    register_metric,
    _to_numpy,
    _safe_div,
)

__all__ = ["MedDefMetrics", "ClassificationMetrics", "compute_metrics"]


# See classify_metrics.py.bak or meddef_metrics.py.bak for original code.
