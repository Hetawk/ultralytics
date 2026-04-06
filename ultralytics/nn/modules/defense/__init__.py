# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Defense Modules for Robust Medical Image Models
# Includes: Frequency Domain Defense, Patch Consistency, CBAM, Distillation
#
# This module provides PyTorch-native defense mechanisms that can integrate with
# the Adversarial Robustness Toolbox (ART) framework. These defenses are designed
# for medical imaging applications requiring robustness against diverse attack types.
#
# =============================================================================
# ATTACK TAXONOMY (Referenced from ART - Adversarial Robustness Toolbox)
# =============================================================================
#
# 1. EVASION ATTACKS (art.attacks.evasion)
#    =====================================
#    Goal: Modify inputs at inference time to cause misclassification
#
#    A. Gradient-Based (White-Box) Attacks:
#       - FGSM (FastGradientMethod): Single-step gradient attack
#       - PGD (ProjectedGradientDescent): Iterative gradient attack with projection
#       - BIM (BasicIterativeMethod): Iterative FGSM
#       - MIM (MomentumIterativeMethod): Momentum-enhanced iterative attack
#       - C&W (CarliniL2Method, CarliniLInfMethod, CarliniL0Method): Optimization-based
#       - DeepFool: Minimal perturbation attack
#       - ElasticNet: L1+L2 regularized attack
#       - JSMA (SaliencyMapMethod): Jacobian-based saliency map attack
#       - NewtonFool: Newton's method-based attack
#       - VAT (VirtualAdversarialMethod): Virtual adversarial training attack
#
#    B. Score-Based (Gray-Box) Attacks:
#       - AutoAttack: Ensemble of diverse attacks
#       - AutoPGD (AutoProjectedGradientDescent): Auto-tuned PGD
#       - SquareAttack: Query-efficient score-based attack
#       - SimBA: Simple Black-box Attack
#
#    C. Decision-Based (Black-Box) Attacks:
#       - BoundaryAttack: Decision boundary walking
#       - HopSkipJump: Improved boundary attack
#       - GeoDA (GeometricDecisionBasedAttack): Geometric decision-based
#       - SignOPT (SignOPTAttack): Sign-based optimization
#       - ZOO (ZooAttack): Zeroth-order optimization
#
#    D. Transfer-Based Attacks:
#       - UniversalPerturbation: Image-agnostic perturbations
#       - TargetedUniversalPerturbation: Targeted universal perturbations
#       - FeatureAdversaries: Feature-space attacks
#
#    E. Physical-World Attacks:
#       - AdversarialPatch: Printable adversarial patches
#       - AdversarialTexture: 3D adversarial textures
#       - DPatch/RobustDPatch: Object detector patches
#       - LaserAttack: Physical laser pointer attacks
#       - SpatialTransformation: Geometric transformations
#
# 2. POISONING ATTACKS (art.attacks.poisoning)
#    ==========================================
#    Goal: Corrupt training data to compromise model behavior
#
#    A. Backdoor Attacks:
#       - PoisoningAttackBackdoor: Classic trigger-based backdoor
#       - CleanLabelBackdoor: No label modification required
#       - HiddenTriggerBackdoor: Hidden pattern triggers
#       - SleeperAgentAttack: Delayed activation backdoors
#       - BadDet (RMA, GMA, OGA, ODA): Object detection backdoors
#
#    B. Data Poisoning:
#       - FeatureCollisionAttack: Feature-space collision
#       - GradientMatchingAttack: Gradient-based poisoning
#       - BullseyePolytopeAttack: Polytope-based targeting
#       - PoisoningAttackSVM: SVM-specific poisoning
#
# 3. EXTRACTION ATTACKS (art.attacks.extraction)
#    ============================================
#    Goal: Steal model functionality or parameters
#
#    - CopycatCNN: Model cloning via query access
#    - KnockoffNets: API-based model extraction
#    - FunctionallyEquivalentExtraction: Exact model recovery
#
# 4. INFERENCE ATTACKS (art.attacks.inference)
#    ==========================================
#    Goal: Extract information about training data or model
#
#    A. Membership Inference: Determine if sample was in training set
#    B. Attribute Inference: Infer sensitive attributes
#    C. Model Inversion: Reconstruct training samples
#    D. Reconstruction: Recover input from representations
#
# =============================================================================
# DEFENSE EFFECTIVENESS MATRIX
# =============================================================================
#
# | Defense Module          | FGSM | PGD | C&W | Black-Box | Patch | Poison |
# |-------------------------|------|-----|-----|-----------|-------|--------|
# | FrequencyDefense        | +++  | ++  | ++  | +         | ++    | -      |
# | PatchConsistency        | +    | +   | +   | +         | +++   | -      |
# | CBAMAttention           | ++   | ++  | ++  | +         | ++    | +      |
# | DefensiveDistillation   | ++   | +   | ++  | +         | +     | +      |
# | InputTransformDefense   | ++   | ++  | +   | ++        | +     | -      |
# | AdversarialDetector     | ++   | ++  | +   | +++       | ++    | +      |
# | GradientMasking         | +++  | +   | +   | ++        | +     | -      |
# | EnsembleDefense         | ++   | ++  | ++  | ++        | ++    | +      |
#
# Legend: +++ Very Effective, ++ Effective, + Somewhat Effective, - Not Applicable
#
# =============================================================================

# Core defenses
from .frequency import FrequencyDefense
from .patch import PatchConsistency
from .attention import ChannelAttention, CBAMAttention
from .distillation import DefensiveDistillationLoss, RobustnessRegularizer

# Robust optimization
from .robust_training import (
    TRADESLoss,
    AdversarialWeightPerturbation,
    MARTLoss,
    RobustMinMaxLoss,
)

# Data augmentation
from .augmentation import (
    MedicalDataAugmentation,
    CutMixMedical,
    MixUpMedical,
)

# Randomization
from .randomization import (
    InputRandomization,
    FeatureNoiseInjection,
    StochasticDepthDefense,
)

# Ensemble methods
from .ensemble import (
    EnsembleDefense,
    DiversityLoss,
    SnapshotEnsemble,
)

# Regularization
from .regularization import (
    SpectralNormRegularization,
    JacobianRegularization,
    AdversarialDropout,
    LabelSmoothing,
    GradientPenalty,
)

# Wrapper
from .wrapper import RobustDefenseWrapper

# Input-space preprocessing defences (ART-inspired, pure PyTorch)
from .preprocessing import (
    SpatialSmoothing,
    FeatureSqueezing,
    JpegCompression,
    GaussianAugmentation,
    PixelDefend,
    VarianceMinimization,
    InputTransformPipeline,
)

__all__ = [
    # Core defenses
    'FrequencyDefense',
    'PatchConsistency',
    'ChannelAttention',
    'CBAMAttention',
    'DefensiveDistillationLoss',
    'RobustnessRegularizer',
    
    # NOTE: TRADESLoss, MARTLoss, RobustMinMaxLoss, AdversarialWeightPerturbation are
    # intentionally excluded from __all__.  MedDef NEVER uses adversarial training.
    # They remain importable via `from ...defense.robust_training import TRADESLoss`
    # for research / evaluation purposes ONLY.

    # Data augmentation
    'MedicalDataAugmentation',
    'CutMixMedical',
    'MixUpMedical',
    
    # Randomization
    'InputRandomization',
    'FeatureNoiseInjection',
    'StochasticDepthDefense',
    
    # Ensemble methods
    'EnsembleDefense',
    'DiversityLoss',
    'SnapshotEnsemble',
    
    # Regularization
    'SpectralNormRegularization',
    'JacobianRegularization',
    'AdversarialDropout',
    'LabelSmoothing',
    'GradientPenalty',
    
    # Wrapper
    'RobustDefenseWrapper',

    # Input-space preprocessing (ART-inspired)
    'SpatialSmoothing',
    'FeatureSqueezing',
    'JpegCompression',
    'GaussianAugmentation',
    'PixelDefend',
    'VarianceMinimization',
    'InputTransformPipeline',
]
