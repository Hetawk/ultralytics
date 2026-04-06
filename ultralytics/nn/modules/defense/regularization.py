# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Advanced Regularization Techniques for Adversarial Robustness."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class SpectralNormRegularization(nn.Module):
    """
    Spectral Normalization for Robust Training
    
    Controls the Lipschitz constant of the network by normalizing
    weight matrices by their spectral norm.
    
    Reference: Miyato et al. "Spectral Normalization for Generative Adversarial Networks"
    
    For adversarial robustness, smaller spectral norms = more robust.
    """

    def __init__(self, model: nn.Module, target_layers: Optional[List[str]] = None,
                 power_iterations: int = 1):
        """
        Args:
            model: Model to apply spectral normalization to
            target_layers: List of layer names (applies to all Conv2d/Linear if None)
            power_iterations: Number of power iterations for spectral norm estimation
        """
        super().__init__()
        self.model = model
        self.power_iterations = power_iterations
        self._apply_spectral_norm(target_layers)

    def _apply_spectral_norm(self, target_layers: Optional[List[str]]):
        """Apply spectral normalization to target layers."""
        from torch.nn.utils import spectral_norm
        
        for name, module in self.model.named_modules():
            if target_layers is not None and name not in target_layers:
                continue
            
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    spectral_norm(module, n_power_iterations=self.power_iterations)
                except Exception:
                    pass  # Already applied or incompatible

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spectrally normalized model."""
        return self.model(x)


class JacobianRegularization(nn.Module):
    """
    Jacobian Regularization for Input-Output Sensitivity Control
    
    Penalizes large Jacobian norms to make the model less sensitive
    to input perturbations.
    
    Reference: Hoffman et al. "Robust Learning with Jacobian Regularization"
    
    J(x) = ∂f(x)/∂x
    Loss += λ * ||J(x)||²
    """

    def __init__(self, lambda_reg: float = 0.01, num_projections: int = 1):
        """
        Args:
            lambda_reg: Regularization strength
            num_projections: Number of random projections for Jacobian estimation
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.num_projections = num_projections

    def forward(self, model: nn.Module, x: torch.Tensor, 
                logits: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian regularization loss.
        
        Args:
            model: Target model
            x: Input batch
            logits: Model output logits
            
        Returns:
            Jacobian regularization loss
        """
        B, C = logits.shape
        total_reg = 0.0
        
        for _ in range(self.num_projections):
            # Random projection vector
            v = torch.randn(B, C, device=logits.device)
            v = v / torch.norm(v, dim=1, keepdim=True)
            
            # Compute Jacobian-vector product efficiently
            grad_outputs = v
            jvp = torch.autograd.grad(
                outputs=logits,
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # Frobenius norm of JVP
            total_reg += torch.norm(jvp.view(B, -1), dim=1).mean()
        
        return self.lambda_reg * total_reg / self.num_projections


class AdversarialDropout(nn.Module):
    """
    Adversarial Dropout
    
    Learns which neurons to drop to maximize robustness.
    Drops neurons that are most vulnerable to adversarial perturbations.
    
    Reference: Park & Kwak "Adversarial Dropout for Robust Training"
    """

    def __init__(self, p: float = 0.5, adversarial: bool = True):
        """
        Args:
            p: Base dropout probability
            adversarial: Whether to use adversarial (learned) dropout
        """
        super().__init__()
        self.p = p
        self.adversarial = adversarial
        self.drop_mask = None

    def _compute_adversarial_mask(self, x: torch.Tensor, 
                                  grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute adversarial dropout mask based on gradients."""
        if grad is None:
            # Random dropout if no gradient available
            return torch.bernoulli(torch.ones_like(x) * (1 - self.p))
        
        # Drop neurons with highest gradient magnitude (most vulnerable)
        grad_magnitude = grad.abs()
        
        # Flatten per sample
        B = x.shape[0]
        flat_grad = grad_magnitude.view(B, -1)
        
        # Get threshold for top-p percentile
        k = int(flat_grad.shape[1] * self.p)
        if k > 0:
            threshold, _ = torch.kthvalue(flat_grad, flat_grad.shape[1] - k + 1, dim=1, keepdim=True)
            threshold = threshold.view(B, *([1] * (x.ndim - 1)))
            
            # Create mask (keep low gradient neurons)
            mask = (grad_magnitude < threshold).float()
        else:
            mask = torch.ones_like(x)
        
        return mask

    def forward(self, x: torch.Tensor, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply adversarial dropout.
        
        Args:
            x: Input features
            grad: Gradient of loss w.r.t. x (for adversarial selection)
            
        Returns:
            Dropped features
        """
        if not self.training:
            return x
        
        if self.adversarial and grad is not None:
            mask = self._compute_adversarial_mask(x, grad)
        else:
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p))
        
        self.drop_mask = mask
        
        # Apply dropout with scaling
        return x * mask / (1 - self.p + 1e-12)


class LabelSmoothing(nn.Module):
    """
    Label Smoothing Regularization
    
    Prevents overconfident predictions which can be exploited by adversaries.
    
    Reference: Szegedy et al. "Rethinking the Inception Architecture for Computer Vision"
    """

    def __init__(self, smoothing: float = 0.1, num_classes: Optional[int] = None):
        """
        Args:
            smoothing: Label smoothing factor (0 = no smoothing, 1 = uniform)
            num_classes: Number of classes (inferred if None)
        """
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            logits: Model predictions [B, C]
            target: Ground truth labels [B]
            
        Returns:
            Smoothed cross-entropy loss
        """
        num_classes = self.num_classes or logits.shape[-1]
        
        # Create smoothed labels
        with torch.no_grad():
            smooth_target = torch.zeros_like(logits)
            smooth_target.fill_(self.smoothing / (num_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Cross-entropy with soft targets
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_target * log_probs).sum(dim=-1).mean()
        
        return loss


class GradientPenalty(nn.Module):
    """
    Gradient Penalty Regularization
    
    Penalizes large input gradients to encourage smooth decision boundaries.
    Particularly effective against gradient-based attacks.
    
    Reference: Drucker & Le Cun "Improving generalization performance using double backpropagation"
    """

    def __init__(self, lambda_gp: float = 10.0, norm_type: str = 'l2'):
        """
        Args:
            lambda_gp: Gradient penalty weight
            norm_type: 'l2' or 'l1'
        """
        super().__init__()
        self.lambda_gp = lambda_gp
        self.norm_type = norm_type

    def forward(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient penalty.
        
        Args:
            model: Target model
            x: Input batch
            y: Labels
            
        Returns:
            Gradient penalty loss
        """
        x.requires_grad_(True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        
        # Compute gradient w.r.t. input
        grad = torch.autograd.grad(
            outputs=loss,
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]
        
        grad = grad.view(grad.shape[0], -1)
        
        if self.norm_type == 'l2':
            penalty = torch.norm(grad, p=2, dim=1).mean()
        elif self.norm_type == 'l1':
            penalty = torch.norm(grad, p=1, dim=1).mean()
        else:
            penalty = torch.norm(grad, p=2, dim=1).mean()
        
        return self.lambda_gp * penalty


__all__ = [
    'SpectralNormRegularization',
    'JacobianRegularization',
    'AdversarialDropout',
    'LabelSmoothing',
    'GradientPenalty',
]
