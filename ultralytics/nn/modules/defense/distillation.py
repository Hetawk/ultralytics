# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Defensive Distillation and Robustness Regularization Modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DefensiveDistillationLoss(nn.Module):
    """
    Defensive Distillation Loss for training robust models.
    
    Combines standard cross-entropy loss with distillation loss from teacher model.
    The distillation encourages the student to match soft targets from teacher,
    which provides regularization for robustness.
    
    Reference: Papernot et al. "Distillation as a Defense to Adversarial Perturbations"
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        """
        Args:
            temperature (float): Distillation temperature. Higher values = softer targets
            alpha (float): Weight for distillation loss (0-1). 
                          Loss = alpha * distill_loss + (1-alpha) * ce_loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits: torch.Tensor,
                batch_or_target,
                teacher_logits: Optional[torch.Tensor] = None):
        """
        Args:
            student_logits (torch.Tensor): Student model logits [B, num_classes]
            batch_or_target (dict | torch.Tensor): Ultralytics batch dict (containing
                ``'cls'`` key) **or** a plain label tensor [B].  Both forms are
                accepted so the criterion works whether called from the Ultralytics
                training loop (passes the full batch dict) or standalone code.
            teacher_logits (torch.Tensor, optional): Teacher model logits [B, num_classes]

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (loss, loss.detach()) matching the
                Ultralytics criterion convention expected by the training loop.
        """
        # Accept either a batch dict (Ultralytics training loop) or a plain tensor
        if isinstance(batch_or_target, dict):
            target = batch_or_target["cls"]
        else:
            target = batch_or_target

        # Standard cross-entropy loss
        ce = self.ce_loss(student_logits, target)

        if teacher_logits is None:
            return ce, ce.detach()

        # Distillation loss using KL divergence
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=-1)
        distill = self.kl_loss(soft_predictions, soft_targets)

        loss = self.alpha * distill + (1.0 - self.alpha) * ce
        return loss, loss.detach()


class RobustnessRegularizer(nn.Module):
    """
    Regularization term to encourage robust representations.
    Encourages local smoothness of model predictions.
    """

    def __init__(self, epsilon: float = 0.03):
        """
        Args:
            epsilon (float): Perturbation budget for adversarial regularization
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            model (nn.Module): Model to regularize
            x (torch.Tensor): Input batch
            y (torch.Tensor): Target labels
            
        Returns:
            torch.Tensor: Regularization loss
        """
        # Create small adversarial perturbation
        x_adv = x.detach().clone()
        x_adv.requires_grad_(True)
        
        # Compute loss to maximize (towards wrong class)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        
        # Compute gradient
        grad = torch.autograd.grad(loss, x_adv, create_graph=True)[0]
        
        # Create FGSM perturbation
        perturbation = self.epsilon * torch.sign(grad).detach()
        x_adv_perturbed = torch.clamp(x + perturbation, 0, 1)
        
        # Regularization encourages consistency
        logits_original = model(x)
        logits_perturbed = model(x_adv_perturbed)
        
        reg_loss = F.kl_div(
            F.log_softmax(logits_perturbed, dim=-1),
            F.softmax(logits_original, dim=-1),
            reduction='batchmean'
        )
        
        return reg_loss


__all__ = ['DefensiveDistillationLoss', 'RobustnessRegularizer']
