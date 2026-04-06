# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Combined Defense Wrapper - Unified interface for all defense mechanisms."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from .frequency import FrequencyDefense
from .randomization import InputRandomization, FeatureNoiseInjection
from .regularization import LabelSmoothing, GradientPenalty
from .robust_training import TRADESLoss, MARTLoss, RobustMinMaxLoss


class RobustDefenseWrapper(nn.Module):
    """
    Unified Wrapper for All Defense Mechanisms
    
    Combines multiple defense strategies into a single cohesive module:
    - Input preprocessing (randomization, frequency filtering)
    - Feature-level defense (noise injection, CBAM)
    - Output-level defense (ensemble, label smoothing)
    - Training-time defense (adversarial training, regularization)
    
    Usage:
        wrapper = RobustDefenseWrapper(model, config)
        output = wrapper(x)  # Inference
        loss = wrapper.compute_robust_loss(x, y)  # Training
    """

    def __init__(self, 
                 model: nn.Module,
                 use_frequency_defense: bool = True,
                 use_input_randomization: bool = True,
                 use_feature_noise: bool = True,
                 use_patch_consistency: bool = True,
                 use_label_smoothing: bool = True,
                 use_gradient_penalty: bool = True,
                 robust_training: str = 'trades',  # 'trades', 'mart', 'pgd', 'none'
                 config: Optional[Dict] = None):
        """
        Args:
            model: Base model to wrap
            use_*: Whether to use each defense mechanism
            robust_training: Training method ('trades', 'mart', 'pgd', 'none')
            config: Configuration dictionary for defense parameters
        """
        super().__init__()
        self.model = model
        config = config or {}
        
        # Input-level defenses
        if use_frequency_defense:
            self.frequency_defense = FrequencyDefense(
                cutoff_ratio=config.get('freq_cutoff', 0.5)
            )
        else:
            self.frequency_defense = None
        
        if use_input_randomization:
            self.input_randomization = InputRandomization(
                resize_range=config.get('resize_range', (0.9, 1.1)),
                padding_range=config.get('padding_range', (0, 4)),
                use_blur=config.get('use_blur', True)
            )
        else:
            self.input_randomization = None
        
        # Feature-level defenses
        if use_feature_noise:
            self.feature_noise = FeatureNoiseInjection(
                noise_type=config.get('noise_type', 'gaussian'),
                noise_level=config.get('noise_level', 0.1)
            )
        else:
            self.feature_noise = None
        
        # Output-level defenses
        if use_label_smoothing:
            self.label_smoothing = LabelSmoothing(
                smoothing=config.get('smoothing', 0.1)
            )
        else:
            self.label_smoothing = None
        
        # Training-time defenses
        self.robust_training = robust_training
        if robust_training == 'trades':
            self.training_defense = TRADESLoss(
                beta=config.get('trades_beta', 6.0),
                epsilon=config.get('epsilon', 8/255),
                num_steps=config.get('num_steps', 10)
            )
        elif robust_training == 'mart':
            self.training_defense = MARTLoss(
                beta=config.get('mart_beta', 6.0),
                epsilon=config.get('epsilon', 8/255),
                num_steps=config.get('num_steps', 10)
            )
        elif robust_training == 'pgd':
            self.training_defense = RobustMinMaxLoss(
                epsilon=config.get('epsilon', 8/255),
                num_inner_steps=config.get('num_steps', 7)
            )
        else:
            self.training_defense = None
        
        # Regularization
        if use_gradient_penalty:
            self.gradient_penalty = GradientPenalty(
                lambda_gp=config.get('lambda_gp', 10.0)
            )
        else:
            self.gradient_penalty = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with input defenses applied.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Model output
        """
        # Apply input-level defenses during inference
        if self.frequency_defense is not None:
            x = self.frequency_defense(x)
        
        if self.input_randomization is not None and self.training:
            x = self.input_randomization(x)
        
        # Forward through model
        output = self.model(x)
        
        # Apply feature noise during training
        if self.feature_noise is not None and self.training:
            output = self.feature_noise(output)
        
        return output

    def compute_robust_loss(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute robust training loss with all applicable defenses.
        
        Args:
            x: Input batch
            y: Labels
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_dict = {}
        
        # Main training loss (robust or standard)
        if self.training_defense is not None:
            if isinstance(self.training_defense, TRADESLoss):
                main_loss, trades_dict = self.training_defense(self.model, x, y)
                loss_dict.update(trades_dict)
            else:
                main_loss = self.training_defense(self.model, x, y)
                loss_dict['main_loss'] = main_loss.item()
        else:
            # Standard forward with defenses
            logits = self.forward(x)
            if self.label_smoothing is not None:
                main_loss = self.label_smoothing(logits, y)
            else:
                main_loss = F.cross_entropy(logits, y)
            loss_dict['ce_loss'] = main_loss.item()
        
        # Add gradient penalty if enabled
        if self.gradient_penalty is not None:
            gp_loss = self.gradient_penalty(self.model, x, y)
            main_loss = main_loss + gp_loss
            loss_dict['gp_loss'] = gp_loss.item()
        
        loss_dict['total_loss'] = main_loss.item()
        
        return main_loss, loss_dict


__all__ = ['RobustDefenseWrapper']
