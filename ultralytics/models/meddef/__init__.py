# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
MedDef (Medical Defense) Models Package

This package provides implementations of MedDef classification models with defense mechanisms
against adversarial attacks. The package is organized into two main families:

- meddef1: ResNet-based models with defense mechanisms (AFD, MFE, MSF)
- meddef2: Vision Transformer-based models with defense mechanisms (CBAM, Frequency, Patch)

Each family shares common training, validation, and prediction infrastructure while maintaining
model-specific implementations.

Example:
    >>> from ultralytics import MedDef
    >>> model = MedDef("meddef1.yaml")  # Load ResNet-based MedDef1
    >>> model = MedDef("meddef2_t.yaml")  # Load Transformer-based MedDef2
    >>> model.train(data="dataset.yaml", epochs=100)
"""

from ultralytics.models.meddef.model import MedDef
from ultralytics.models.meddef.predict import MedDefPredictor
from ultralytics.models.meddef.train import MedDefTrainer
from ultralytics.models.meddef.val import MedDefValidator
from ultralytics.models.meddef.model_builder import build_meddef_model, VARIANT_MAP
from ultralytics.models.meddef.distillation import DistillationMixin

# Import model families for internal use
from ultralytics.models.meddef import meddef1, meddef2

__all__ = (
    "MedDef",
    "MedDefTrainer",
    "MedDefValidator",
    "MedDefPredictor",
    "DistillationMixin",
    "build_meddef_model",
    "VARIANT_MAP",
    "meddef1",
    "meddef2",
)
