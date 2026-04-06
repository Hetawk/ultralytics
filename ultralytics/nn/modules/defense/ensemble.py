# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ensemble Defense Methods."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np


class EnsembleDefense(nn.Module):
    """
    Model Ensemble Defense
    
    Combines multiple models to improve robustness through:
    1. Prediction averaging
    2. Voting
    3. Stacking
    
    Adversarial examples crafted for one model often don't transfer
    well to different architectures.
    """

    def __init__(self, models: List[nn.Module], 
                 ensemble_method: str = 'average',
                 temperature: float = 1.0):
        """
        Args:
            models: List of models to ensemble
            ensemble_method: 'average', 'vote', or 'weighted'
            temperature: Temperature for softmax (only for 'average')
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.temperature = temperature
        
        if ensemble_method == 'weighted':
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensemble forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Ensembled predictions [B, num_classes]
        """
        predictions = []
        
        for model in self.models:
            logits = model(x)
            predictions.append(logits)
        
        predictions = torch.stack(predictions, dim=0)  # [num_models, B, num_classes]
        
        if self.ensemble_method == 'average':
            # Average logits with temperature
            probs = F.softmax(predictions / self.temperature, dim=-1)
            ensemble_probs = probs.mean(dim=0)
            return torch.log(ensemble_probs + 1e-12)  # Return log-probs
        
        elif self.ensemble_method == 'vote':
            # Hard voting
            votes = predictions.argmax(dim=-1)  # [num_models, B]
            ensemble_pred = torch.mode(votes, dim=0).values
            # Convert to one-hot like logits
            return F.one_hot(ensemble_pred, num_classes=predictions.shape[-1]).float()
        
        elif self.ensemble_method == 'weighted':
            # Weighted average
            weights = F.softmax(self.weights, dim=0)
            weighted_logits = (predictions * weights.view(-1, 1, 1)).sum(dim=0)
            return weighted_logits
        
        else:
            return predictions.mean(dim=0)


class DiversityLoss(nn.Module):
    """
    Diversity Loss for Ensemble Training
    
    Encourages diversity among ensemble members by penalizing
    similar predictions, improving ensemble robustness.
    
    Reference: Pang et al. "Improving Adversarial Robustness via Promoting Ensemble Diversity"
    """

    def __init__(self, diversity_weight: float = 0.5):
        """
        Args:
            diversity_weight: Weight for diversity loss
        """
        super().__init__()
        self.diversity_weight = diversity_weight

    def forward(self, predictions: List[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity loss.
        
        Args:
            predictions: List of model predictions [B, num_classes] each
            y: Ground truth labels [B]
            
        Returns:
            Combined loss (task + diversity)
        """
        num_models = len(predictions)
        
        # Task loss (average CE across models)
        task_loss = sum(F.cross_entropy(pred, y) for pred in predictions) / num_models
        
        # Diversity loss (negative pairwise KL divergence)
        diversity_loss = 0.0
        count = 0
        
        for i in range(num_models):
            for j in range(i + 1, num_models):
                kl_ij = F.kl_div(
                    F.log_softmax(predictions[i], dim=-1),
                    F.softmax(predictions[j], dim=-1),
                    reduction='batchmean'
                )
                kl_ji = F.kl_div(
                    F.log_softmax(predictions[j], dim=-1),
                    F.softmax(predictions[i], dim=-1),
                    reduction='batchmean'
                )
                diversity_loss += (kl_ij + kl_ji) / 2
                count += 1
        
        if count > 0:
            diversity_loss /= count
        
        # We want to maximize diversity (minimize negative diversity)
        total_loss = task_loss - self.diversity_weight * diversity_loss
        
        return total_loss


class SnapshotEnsemble(nn.Module):
    """
    Snapshot Ensemble
    
    Creates an ensemble by saving model checkpoints during training
    with cyclic learning rate schedules.
    
    Reference: Huang et al. "Snapshot Ensembles: Train 1, get M for free"
    """

    def __init__(self, base_model: nn.Module, num_snapshots: int = 5):
        """
        Args:
            base_model: Model architecture to snapshot
            num_snapshots: Number of snapshots to collect
        """
        super().__init__()
        self.base_model = base_model
        self.num_snapshots = num_snapshots
        self.snapshots = []

    def save_snapshot(self):
        """Save current model state as a snapshot."""
        import copy
        snapshot = copy.deepcopy(self.base_model.state_dict())
        self.snapshots.append(snapshot)
        
        # Keep only the last num_snapshots
        if len(self.snapshots) > self.num_snapshots:
            self.snapshots.pop(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensemble prediction using all snapshots.
        
        Args:
            x: Input tensor
            
        Returns:
            Averaged predictions
        """
        if len(self.snapshots) == 0:
            return self.base_model(x)
        
        predictions = []
        original_state = self.base_model.state_dict()
        
        for snapshot in self.snapshots:
            self.base_model.load_state_dict(snapshot)
            self.base_model.eval()
            with torch.no_grad():
                predictions.append(self.base_model(x))
        
        # Restore original state
        self.base_model.load_state_dict(original_state)
        
        return torch.stack(predictions).mean(dim=0)

    @staticmethod
    def cyclic_lr_schedule(epoch: int, epochs_per_cycle: int, 
                          lr_min: float, lr_max: float) -> float:
        """
        Cyclic learning rate for snapshot collection.
        
        Args:
            epoch: Current epoch
            epochs_per_cycle: Epochs per LR cycle
            lr_min: Minimum learning rate
            lr_max: Maximum learning rate
            
        Returns:
            Current learning rate
        """
        cycle = epoch // epochs_per_cycle
        epoch_in_cycle = epoch % epochs_per_cycle
        
        # Cosine annealing within cycle
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch_in_cycle / epochs_per_cycle))
        return lr


__all__ = [
    'EnsembleDefense',
    'DiversityLoss',
    'SnapshotEnsemble',
]
