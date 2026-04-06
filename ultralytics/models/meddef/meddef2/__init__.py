# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
MedDef2: Vision Transformer-based Classification Models with Defense Mechanisms

Model Variants for Ablation Studies:
- MedDef2_T: Full model (CBAM + Freq + Patch + Defense)
- MedDef2_T_NoDef: No DefenseModule
- MedDef2_T_NoFreq: No FrequencyDefense
- MedDef2_T_NoPatch: No PatchConsistency
- MedDef2_T_NoCBAM: No CBAM attention
- MedDef2_T_Baseline: Standard ViT (no defenses)

Size Variants: tiny, small, base, large
"""

from .meddef2_t import MedDef2_T, get_meddef2_t, meddef2_t_0, meddef2_t_1, meddef2_t_2, meddef2_t_3
from .transformer import VisionTransformer
from .defense import (
    CBAM, DefenseModule, MultiScaleFeatures,
    FrequencyDefense, PatchConsistency,
    # NOTE: TRADESLoss / MARTLoss / AdversarialWeightPerturbation are NOT imported here.
    # MedDef2 training uses only defensive distillation — no adversarial training.
    RobustDefenseWrapper,
)
from .variants import (
    MedDef2_T_NoDef, MedDef2_T_NoFreq, MedDef2_T_NoPatch, MedDef2_T_NoCBAM, MedDef2_T_Baseline,
    get_variant, meddef2_full, meddef2_no_def, meddef2_no_freq, meddef2_no_patch, meddef2_no_cbam, meddef2_baseline,
)

__all__ = [
    'MedDef2_T', 'get_meddef2_t', 'meddef2_t_0', 'meddef2_t_1', 'meddef2_t_2', 'meddef2_t_3', 'VisionTransformer',
    'MedDef2_T_NoDef', 'MedDef2_T_NoFreq', 'MedDef2_T_NoPatch', 'MedDef2_T_NoCBAM', 'MedDef2_T_Baseline',
    'get_variant', 'meddef2_full', 'meddef2_no_def', 'meddef2_no_freq', 'meddef2_no_patch', 'meddef2_no_cbam', 'meddef2_baseline',
    'CBAM', 'DefenseModule', 'MultiScaleFeatures', 'FrequencyDefense', 'PatchConsistency', 'RobustDefenseWrapper',
]
