# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.engine.model import Model
from ultralytics.models import meddef
from ultralytics.nn.tasks import MedDefModel
from ultralytics.utils import ROOT


class MedDef(Model):
    """MedDef (Medical Defense) classification model.

    This class provides a unified interface for MedDef models, which are Vision Transformer-based
    classification models with defense mechanisms against adversarial attacks. It supports various
    MedDef architectures including MedDef2_T, MedDef2_T_NoDefense, and variants with CBAM and
    frequency domain features.

    Attributes:
        model: The loaded MedDef model instance.
        task: The task type (always 'classify' for MedDef).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize a MedDef model.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.

    Examples:
        Load a pretrained MedDef2_T model
        >>> model = MedDef("meddef2_t.pt")

        Initialize from a configuration
        >>> model = MedDef("meddef2_t.yaml")

        Train on a custom dataset
        >>> model = MedDef("meddef2_t.yaml")
        >>> model.train(data="path/to/dataset", epochs=100, imgsz=224)
    """

    def __init__(self, model: str | Path = "meddef2_t.pt", task: str | None = None, verbose: bool = False):
        """Initialize a MedDef model.

        This constructor automatically detects whether you're using MedDef1 (ResNet-based) or
        MedDef2 (Transformer-based) models based on the model filename.

        Args:
            model (str | Path): Model name or path to model file.
                Examples:
                - 'meddef1.pt', 'meddef1.yaml' - ResNet-based with full defense
                - 'meddef1_no_afd.yaml' - ResNet without AFD
                - 'meddef2_t.pt' - Transformer-based with full defense
                - 'meddef2_t_no_defense.yaml' - Transformer without defense
                - 'meddef2_t_cbam.yaml' - Transformer with CBAM attention
            task (str, optional): MedDef task specification. Always defaults to 'classify' for MedDef models.
            verbose (bool): Display model info on load.

        Examples:
            >>> from ultralytics import MedDef
            >>> # ResNet-based MedDef1
            >>> model = MedDef("meddef1.yaml")
            >>> model.train(data="imagenet10", epochs=100)
            >>> 
            >>> # Transformer-based MedDef2
            >>> model = MedDef("meddef2_t.yaml")
            >>> model.train(data="imagenet10", epochs=100)
        """
        # Force task to be classify for MedDef models
        if task is None:
            task = "classify"
        elif task != "classify":
            raise ValueError(f"MedDef models only support 'classify' task, got '{task}'")
        
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes for MedDef."""
        return {
            "classify": {
                "model": MedDefModel,
                "trainer": meddef.train.MedDefTrainer,
                "validator": meddef.val.MedDefValidator,
                "predictor": meddef.predict.MedDefPredictor,
            },
        }
