# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Robust Training Optimization Techniques — RESEARCH REFERENCE ONLY.

WARNING
-------
The classes in this module (TRADESLoss, MARTLoss, RobustMinMaxLoss,
AdversarialWeightPerturbation) are adversarial-training methods.

MedDef's robustness strategy does NOT use adversarial training.
These classes are kept here for research comparison / evaluation purposes
and must NEVER be passed as the criterion to MedDefTrainer.

MedDef robustness comes exclusively from:
  • Defensive distillation  (DefensiveDistillationLoss, T=4.0, α=0.5)
  • Frequency domain defense (FrequencyDefense — low-pass filtering)
  • Patch consistency         (PatchConsistency)
  • CBAM attention            (CBAMTransformerBlock)
  • Multi-scale feature defense (DefenseModule)
  • Input-space preprocessing  (InputTransformPipeline — spatial smoothing,
                                feature squeezing, Gaussian augmentation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class TRADESLoss(nn.Module):
    """
    TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)
    
    Combines natural accuracy with robustness by balancing two objectives:
    1. Natural cross-entropy loss on clean examples
    2. KL divergence between clean and adversarial predictions
    
    Reference: Zhang et al. "Theoretically Principled Trade-off between Robustness and Accuracy"
    https://arxiv.org/abs/1901.08573
    
    Effective against:
    - PGD attacks (primary target)
    - FGSM and iterative attacks
    - AutoAttack ensemble
    """

    def __init__(self, beta: float = 6.0, epsilon: float = 8/255, 
                 alpha: float = 2/255, num_steps: int = 10):
        """
        Args:
            beta (float): Trade-off parameter between natural and robust loss.
                         Higher beta = more emphasis on robustness
            epsilon (float): Maximum perturbation magnitude (L_inf)
            alpha (float): Step size for PGD attack
            num_steps (int): Number of PGD steps
        """
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def _pgd_attack(self, model: nn.Module, x: torch.Tensor, 
                    natural_logits: torch.Tensor) -> torch.Tensor:
        """Generate PGD adversarial examples for TRADES."""
        x_adv = x.detach() + 0.001 * torch.randn_like(x)
        x_adv = torch.clamp(x_adv, 0, 1)
        
        for _ in range(self.num_steps):
            x_adv.requires_grad_(True)
            adv_logits = model(x_adv)
            
            # Maximize KL divergence
            loss = self.kl_loss(
                F.log_softmax(adv_logits, dim=-1),
                F.softmax(natural_logits.detach(), dim=-1)
            )
            
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + self.alpha * torch.sign(grad)
            
            # Project back to epsilon ball
            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv

    def forward(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute TRADES loss.
        
        Args:
            model: Target model
            x: Clean input batch [B, C, H, W]
            y: Target labels [B]
            
        Returns:
            Tuple[loss, dict]: Total loss and component breakdown
        """
        model.train()
        
        # Natural predictions
        natural_logits = model(x)
        natural_loss = self.ce_loss(natural_logits, y)
        
        # Generate adversarial examples
        x_adv = self._pgd_attack(model, x, natural_logits)
        
        # Adversarial predictions
        adv_logits = model(x_adv)
        
        # Robust loss (KL divergence)
        robust_loss = self.kl_loss(
            F.log_softmax(adv_logits, dim=-1),
            F.softmax(natural_logits.detach(), dim=-1)
        )
        
        # Combined TRADES loss
        total_loss = natural_loss + self.beta * robust_loss
        
        return total_loss, {
            'natural_loss': natural_loss.item(),
            'robust_loss': robust_loss.item(),
            'total_loss': total_loss.item()
        }


class AdversarialWeightPerturbation(nn.Module):
    """
    Adversarial Weight Perturbation (AWP)
    
    Perturbs model weights adversarially during training to find flat minima,
    which improves generalization and robustness.
    
    Reference: Wu et al. "Adversarial Weight Perturbation Helps Robust Generalization"
    https://arxiv.org/abs/2004.05884
    
    Effective against:
    - Transfer attacks
    - Adaptive attacks
    - Improves worst-case robustness
    """

    def __init__(self, model: nn.Module, gamma: float = 0.01, 
                 awp_warmup: int = 0):
        """
        Args:
            model: Target model
            gamma (float): Weight perturbation magnitude
            awp_warmup (int): Number of epochs before applying AWP
        """
        super().__init__()
        self.model = model
        self.gamma = gamma
        self.awp_warmup = awp_warmup
        self.backup = {}
        self.backup_eps = {}

    def _save_weights(self):
        """Save current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.backup[name] = param.data.clone()

    def _restore_weights(self):
        """Restore saved model weights."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def _calc_awp(self):
        """Calculate adversarial weight perturbation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.detach()
                grad_norm = torch.norm(grad)
                if grad_norm > 0:
                    # Scale perturbation by gradient magnitude
                    awp = self.gamma * grad / (grad_norm + 1e-12)
                    self.backup_eps[name] = awp
                    param.data.add_(awp)

    def perturb(self, epoch: int = 0):
        """Apply adversarial weight perturbation."""
        if epoch < self.awp_warmup:
            return
        self._save_weights()
        self._calc_awp()

    def restore(self):
        """Restore original weights after perturbation."""
        self._restore_weights()
        self.backup_eps = {}


class MARTLoss(nn.Module):
    """
    MART (Misclassification Aware adveRsarial Training)
    
    Emphasizes misclassified examples during adversarial training,
    leading to better robustness on hard examples.
    
    Reference: Wang et al. "Improving Adversarial Robustness Requires Revisiting Misclassified Examples"
    https://openreview.net/forum?id=rklOg6EFwS
    """

    def __init__(self, beta: float = 6.0, epsilon: float = 8/255,
                 alpha: float = 2/255, num_steps: int = 10):
        """
        Args:
            beta: Weight for the boundary loss
            epsilon: Perturbation budget
            alpha: Step size
            num_steps: Number of attack iterations
        """
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps

    def forward(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute MART loss."""
        model.train()
        
        # Generate adversarial examples with PGD
        x_adv = x.detach() + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
        
        for _ in range(self.num_steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + self.alpha * torch.sign(grad)
            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + delta, 0, 1)
        
        # Clean and adversarial predictions
        clean_logits = model(x)
        adv_logits = model(x_adv)
        
        # MART loss components
        probs_clean = F.softmax(clean_logits, dim=-1)
        probs_adv = F.softmax(adv_logits, dim=-1)
        
        # BCE loss for misclassification awareness
        y_onehot = F.one_hot(y, num_classes=clean_logits.shape[-1]).float()
        
        # Adversarial cross-entropy with boosting for misclassified
        adv_loss = F.cross_entropy(adv_logits, y, reduction='none')
        
        # Weight by how wrong the clean prediction is
        clean_correct_prob = (probs_clean * y_onehot).sum(dim=-1)
        boost_weight = 1 - clean_correct_prob
        
        # KL divergence for boundary regularization
        kl_loss = F.kl_div(
            F.log_softmax(adv_logits, dim=-1),
            probs_clean.detach(),
            reduction='none'
        ).sum(dim=-1)
        
        # Combined MART loss
        loss = (adv_loss + self.beta * boost_weight * kl_loss).mean()
        
        return loss


class RobustMinMaxLoss(nn.Module):
    """
    Min-Max Robust Optimization Loss
    
    Trains model to minimize the worst-case loss over a perturbation set.
    Implements the inner maximization (attack) and outer minimization (training).
    
    Effective for:
    - Certified robustness bounds
    - Worst-case performance guarantees
    - Distribution shift robustness
    """

    def __init__(self, epsilon: float = 8/255, alpha: float = 2/255,
                 num_inner_steps: int = 7, norm: str = 'linf'):
        """
        Args:
            epsilon: Perturbation budget
            alpha: Inner step size
            num_inner_steps: Number of inner maximization steps
            norm: Norm type ('linf', 'l2')
        """
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_inner_steps = num_inner_steps
        self.norm = norm

    def _inner_maximize(self, model: nn.Module, x: torch.Tensor, 
                       y: torch.Tensor) -> torch.Tensor:
        """Inner maximization: find worst-case perturbation."""
        delta = torch.zeros_like(x, requires_grad=True)
        
        for _ in range(self.num_inner_steps):
            logits = model(x + delta)
            loss = F.cross_entropy(logits, y)
            
            grad = torch.autograd.grad(loss, delta)[0]
            
            if self.norm == 'linf':
                delta = delta.detach() + self.alpha * torch.sign(grad)
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            elif self.norm == 'l2':
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1, keepdim=True)
                grad_norm = grad_norm.view(-1, 1, 1, 1)
                delta = delta.detach() + self.alpha * grad / (grad_norm + 1e-12)
                delta_norm = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1, keepdim=True)
                delta_norm = delta_norm.view(-1, 1, 1, 1)
                delta = delta * torch.clamp(self.epsilon / (delta_norm + 1e-12), max=1.0)
            
            delta.requires_grad_(True)
        
        return delta.detach()

    def forward(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute min-max robust loss.
        
        Args:
            model: Target model
            x: Clean inputs
            y: Labels
            
        Returns:
            Worst-case loss
        """
        model.train()
        
        # Inner maximization
        worst_delta = self._inner_maximize(model, x, y)
        x_worst = torch.clamp(x + worst_delta, 0, 1)
        
        # Outer minimization
        logits = model(x_worst)
        loss = F.cross_entropy(logits, y)
        
        return loss


__all__ = [
    'TRADESLoss',
    'AdversarialWeightPerturbation',
    'MARTLoss',
    'RobustMinMaxLoss',
]
