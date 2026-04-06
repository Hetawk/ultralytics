"""
MedDef Visualization — backward-compatibility wrapper
======================================================

All visualization logic now lives in the model-agnostic modules:

* ``classify_visualize.py`` — plots, ClassificationVisualizer, registry
* ``saliency.py``           — Grad-CAM / attribution methods

This file re-exports every public name so existing ``import`` statements
continue to work unchanged.
"""

# Classification visualizations (canonical home)
from ultralytics.utils.classify_visualize import (  # noqa: F401
    ClassificationVisualizer,
    register_plot,
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_auc,
    plot_per_class_metrics,
    plot_radar_chart,
    plot_tsne,
    plot_pca,
    plot_epsilon_sensitivity,
    plot_saliency_map,
    plot_ablation_heatmap,
    plot_asr_heatmap,
    plot_metrics_summary,
    plot_calibration_diagram,
)

# Grad-CAM convenience function (canonical home: saliency.py)
from ultralytics.utils.saliency import compute_gradcam  # noqa: F401

# Backward-compatible alias
MedDefVisualizer = ClassificationVisualizer
