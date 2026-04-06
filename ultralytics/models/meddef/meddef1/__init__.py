# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
MedDef1: ResNet-based Classification Models with Defense Mechanisms

This module contains ResNet-based MedDef models that incorporate various defense mechanisms:
- meddef1.py: Full defense (AFD + MFE + MSF)
- meddef1_no_afd.py: Without Adversarial Feature Distillation
- meddef1_no_afd_mfe.py: Without AFD and Multi-scale Feature Extraction
- meddef1_no_afd_mfe_msf.py: Base ResNet without defense mechanisms

All models use self-attention mechanisms and can be configured for different datasets.
"""

# Import model architectures - these will be copied/linked from meddef_winlab
# from .meddef1 import ResNetSelfAttention as MedDef1
# from .meddef1_no_afd import ResNetSelfAttentionNoAFD as MedDef1NoAFD
# from .meddef1_no_afd_mfe import ResNetSelfAttentionNoAFDMFE as MedDef1NoAFDMFE
# from .meddef1_no_afd_mfe_msf import ResNetSelfAttentionNoAFDMFEMSF as MedDef1NoAFDMFEMSF

__all__ = []  # Will be populated when model files are added
