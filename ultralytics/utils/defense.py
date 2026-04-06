# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# ART (Adversarial Robustness Toolbox) Integration Utilities
# Provides wrappers for ART attacks and defenses for use in ultralytics framework

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, List
from abc import ABC, abstractmethod

# ART (Adversarial Robustness Toolbox) — optional but strongly recommended
try:
    from art.estimators.classification import PyTorchClassifier as _ARTClassifier
    from art.attacks.evasion import (
        DeepFool as _DeepFool,
        AutoProjectedGradientDescent as _AutoPGD,
        SquareAttack as _SquareAttack,
    )
    _ART_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ART_AVAILABLE = False


def _make_art_classifier(
    model: nn.Module,
    input_shape: tuple,
    nb_classes: int,
    device: torch.device,
    epsilon: float = 8 / 255,
):
    """
    Wrap a PyTorch model in an ART PyTorchClassifier.

    The classifier is used internally by ART-backed attacks. The model must
    produce logits (not softmax) for the loss to be computed correctly.
    """
    if not _ART_AVAILABLE:
        raise ImportError(
            "The Adversarial Robustness Toolbox (ART) is required for this attack.\n"
            "Install it with: pip install adversarial-robustness-toolbox"
        )
    return _ARTClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        input_shape=input_shape,
        nb_classes=nb_classes,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )


class AdversarialAttack(ABC):
    """Base class for adversarial attacks"""

    @abstractmethod
    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples"""
        pass


class PGDAttack(AdversarialAttack):
    """
    Projected Gradient Descent (PGD) Attack
    
    One of the strongest iterative adversarial attacks.
    Reference: Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks"
    """

    def __init__(self, model: nn.Module, epsilon: float = 8/255, alpha: float = 2/255,
                 num_iter: int = 20, norm: str = 'linf', device: torch.device = None):
        """
        Args:
            model (nn.Module): Target model
            epsilon (float): Maximum perturbation magnitude
            alpha (float): Step size per iteration
            num_iter (int): Number of iterations
            norm (str): Norm type ('linf', 'l2')
            device (torch.device): Device to use
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.norm = norm
        self.device = device or torch.device('cpu')

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate PGD adversarial examples"""
        self.model.eval()
        
        x_adv = x.clone().detach()
        
        # Random initialization
        if self.norm == 'linf':
            x_adv += torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
        elif self.norm == 'l2':
            noise = torch.randn_like(x_adv)
            noise = noise / torch.norm(noise, p=2, dim=(1, 2, 3), keepdim=True)
            x_adv = x_adv + noise * self.epsilon
        
        x_adv = torch.clamp(x_adv, 0, 1)
        
        # PGD iterations
        for _ in range(self.num_iter):
            x_adv.requires_grad_(True)
            logits = self.model(x_adv)
            loss = F.cross_entropy(logits, y)
            
            grad = torch.autograd.grad(loss, x_adv)[0]
            
            # Update perturbation
            if self.norm == 'linf':
                x_adv = x_adv.detach() + self.alpha * torch.sign(grad)
            elif self.norm == 'l2':
                grad_norm = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
                x_adv = x_adv.detach() + self.alpha * (grad / (grad_norm + 1e-8))
            
            # Project back to epsilon ball
            if self.norm == 'linf':
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            elif self.norm == 'l2':
                delta = x_adv - x
                delta_norm = torch.norm(delta, p=2, dim=(1, 2, 3), keepdim=True)
                delta = delta / (delta_norm + 1e-8) * min(self.epsilon, delta_norm.max().item())
            
            x_adv = torch.clamp(x + delta, 0, 1)
        
        return x_adv


class FGSMAttack(AdversarialAttack):
    """Fast Gradient Sign Method (FGSM) Attack"""

    def __init__(self, model: nn.Module, epsilon: float = 8/255, device: torch.device = None):
        """
        Args:
            model (nn.Module): Target model
            epsilon (float): Perturbation magnitude
            device (torch.device): Device to use
        """
        self.model = model
        self.epsilon = epsilon
        self.device = device or torch.device('cpu')

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate FGSM adversarial examples"""
        self.model.eval()
        
        x_adv = x.clone().detach()
        x_adv.requires_grad_(True)
        
        logits = self.model(x_adv)
        loss = F.cross_entropy(logits, y)
        
        grad = torch.autograd.grad(loss, x_adv)[0]
        
        x_adv = x_adv.detach() + self.epsilon * torch.sign(grad)
        x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv


class CWAttack(AdversarialAttack):
    """Carlini & Wagner (C&W) Attack - stronger but computationally expensive"""

    def __init__(self, model: nn.Module, epsilon: float = 8/255, 
                 learning_rate: float = 0.01, max_iter: int = 100, device: torch.device = None):
        """
        Args:
            model (nn.Module): Target model
            epsilon (float): Maximum perturbation
            learning_rate (float): Optimization learning rate
            max_iter (int): Maximum iterations
            device (torch.device): Device to use
        """
        self.model = model
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.device = device or torch.device('cpu')

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate C&W adversarial examples"""
        self.model.eval()
        
        # Initialize perturbation
        delta = torch.zeros_like(x, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.learning_rate)
        
        for _ in range(self.max_iter):
            x_adv = torch.clamp(x + delta, 0, 1)
            logits = self.model(x_adv)
            
            # C&W loss: minimize perturbation while maximizing loss
            loss = F.cross_entropy(logits, y) + torch.norm(delta, p=2) * 0.01
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Project to epsilon ball
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
        
        return torch.clamp(x + delta.detach(), 0, 1)


class BIMAttack(AdversarialAttack):
    """
    Basic Iterative Method (BIM / I-FGSM)

    Iterative extension of FGSM with smaller step size. Clips the perturbation
    after every step to stay within the epsilon-ball.

    Reference: Kurakin et al. “Adversarial examples in the physical world”
    """

    def __init__(self, model: nn.Module, epsilon: float = 8 / 255,
                 alpha: float = 2 / 255, num_iter: int = 10,
                 device: torch.device = None):
        """
        Args:
            model (nn.Module): Target model
            epsilon (float): Maximum perturbation magnitude (L∞)
            alpha (float): Step size per iteration
            num_iter (int): Number of iterations
            device (torch.device): Device to use
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.device = device or torch.device('cpu')

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate BIM adversarial examples."""
        self.model.eval()
        x_adv = x.clone().detach()

        for _ in range(self.num_iter):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(self.model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]

            x_adv = x_adv.detach() + self.alpha * torch.sign(grad)
            # Clip to epsilon-ball and valid pixel range
            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + delta, 0, 1)

        return x_adv


class MIMAttack(AdversarialAttack):
    """
    Momentum Iterative Method (MI-FGSM / MIM)

    Adds momentum to the iterative FGSM gradient update. The accumulated
    momentum stabilises gradient direction, making it harder for defences
    that rely on gradient masking.

    Reference: Dong et al. “Boosting Adversarial Attacks with Momentum” (CVPR 2018)
    """

    def __init__(self, model: nn.Module, epsilon: float = 8 / 255,
                 alpha: float = 2 / 255, num_iter: int = 10,
                 decay: float = 1.0, device: torch.device = None):
        """
        Args:
            model (nn.Module): Target model
            epsilon (float): Maximum perturbation magnitude (L∞)
            alpha (float): Step size per iteration
            num_iter (int): Number of iterations
            decay (float): Momentum decay factor (1.0 = full carry-over)
            device (torch.device): Device to use
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.decay = decay
        self.device = device or torch.device('cpu')

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate MI-FGSM adversarial examples."""
        self.model.eval()
        x_adv = x.clone().detach()
        momentum = torch.zeros_like(x)

        for _ in range(self.num_iter):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(self.model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]

            # Normalise gradient and accumulate momentum
            grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8)
            momentum = self.decay * momentum + grad

            x_adv = x_adv.detach() + self.alpha * torch.sign(momentum)
            delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + delta, 0, 1)

        return x_adv


class DeepFoolAttack(AdversarialAttack):
    """
    DeepFool Attack (ART-backed)

    Finds the *minimal* perturbation that crosses the nearest decision
    boundary. Does not require an epsilon bound — the resulting perturbation
    is very tight, making it ideal for measuring the model’s true robustness
    margin.

    Reference: Moosavi-Dezfooli et al. “DeepFool: a simple and accurate method
    to fool deep neural networks” (CVPR 2016)
    """

    def __init__(self, model: nn.Module, max_iter: int = 50,
                 epsilon: float = 1e-6, nb_grads: int = 10,
                 input_shape: tuple = (3, 224, 224), nb_classes: int = 2,
                 device: torch.device = None):
        """
        Args:
            model (nn.Module): Target model (must output logits)
            max_iter (int): Maximum number of iterations
            epsilon (float): Overshoot parameter to push past the boundary
            nb_grads (int): Number of top-class gradients to compute per step
            input_shape (tuple): CHW shape of a single input
            nb_classes (int): Number of output classes
            device (torch.device): Device to use
        """
        self.model = model
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.nb_grads = nb_grads
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.device = device or torch.device('cpu')

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate DeepFool adversarial examples via ART."""
        self.model.eval()
        art_clf = _make_art_classifier(
            self.model, self.input_shape, self.nb_classes, self.device
        )
        attack = _DeepFool(
            classifier=art_clf,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
            nb_grads=self.nb_grads,
            batch_size=x.shape[0],
        )
        x_np = x.detach().cpu().numpy()
        x_adv_np = attack.generate(x=x_np)
        return torch.tensor(x_adv_np, dtype=x.dtype, device=x.device)


class AutoPGDAttack(AdversarialAttack):
    """
    Auto Projected Gradient Descent (APGD / AutoPGD) — ART-backed

    An adaptive PGD variant that automatically tunes its step size and
    uses both CE and DLR loss terms. It is a component of the industry-
    standard **AutoAttack** benchmark.

    Reference: Croce & Hein “Reliable evaluation of adversarial robustness
    with an ensemble of diverse parameter-free attacks” (ICML 2020)
    """

    def __init__(self, model: nn.Module, epsilon: float = 8 / 255,
                 eps_step: float = 2 / 255, max_iter: int = 100,
                 norm: Union[int, float, str] = np.inf,
                 loss_type: str = 'cross_entropy',
                 nb_random_init: int = 5,
                 input_shape: tuple = (3, 224, 224), nb_classes: int = 2,
                 device: torch.device = None):
        """
        Args:
            model (nn.Module): Target model (must output logits)
            epsilon (float): Maximum perturbation (L-inf or L2 radius)
            eps_step (float): Initial step size
            max_iter (int): Number of attack iterations
            norm: Perturbation norm — np.inf, 1 or 2
            loss_type (str): 'cross_entropy' | 'difference_logits_ratio'
            nb_random_init (int): Number of random restarts
            input_shape (tuple): CHW input shape
            nb_classes (int): Number of output classes
            device (torch.device): Device to use
        """
        self.model = model
        self.epsilon = epsilon
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.norm = norm
        self.loss_type = loss_type
        self.nb_random_init = nb_random_init
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.device = device or torch.device('cpu')

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate AutoPGD adversarial examples via ART."""
        self.model.eval()
        art_clf = _make_art_classifier(
            self.model, self.input_shape, self.nb_classes, self.device, self.epsilon
        )
        attack = _AutoPGD(
            estimator=art_clf,
            norm=self.norm,
            eps=self.epsilon,
            eps_step=self.eps_step,
            max_iter=self.max_iter,
            targeted=False,
            nb_random_init=self.nb_random_init,
            batch_size=x.shape[0],
            loss_type=self.loss_type,
        )
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        x_adv_np = attack.generate(x=x_np, y=y_np)
        return torch.tensor(x_adv_np, dtype=x.dtype, device=x.device)


class SquareAttackART(AdversarialAttack):
    """
    Square Attack (black-box, score-based) — ART-backed

    Queries the model using *only* output scores — no gradients needed.
    This makes it effective even against defences that obfuscate gradients
    (e.g. input transformations, stochastic defences). It is included in the
    AutoAttack benchmark as the canonical black-box sub-attack.

    Reference: Andriushchenko et al. “Square Attack: a query-efficient
    black-box adversarial attack via random search” (ECCV 2020)
    """

    def __init__(self, model: nn.Module, epsilon: float = 8 / 255,
                 max_iter: int = 5000,
                 norm: Union[int, float, str] = np.inf,
                 input_shape: tuple = (3, 224, 224), nb_classes: int = 2,
                 device: torch.device = None):
        """
        Args:
            model (nn.Module): Target model
            epsilon (float): Maximum perturbation
            max_iter (int): Query budget
            norm: Perturbation norm — np.inf or 2
            input_shape (tuple): CHW input shape
            nb_classes (int): Number of output classes
            device (torch.device): Device to use
        """
        self.model = model
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.norm = norm
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.device = device or torch.device('cpu')

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Generate Square Attack adversarial examples via ART."""
        self.model.eval()
        art_clf = _make_art_classifier(
            self.model, self.input_shape, self.nb_classes, self.device, self.epsilon
        )
        attack = _SquareAttack(
            estimator=art_clf,
            norm=self.norm,
            max_iter=self.max_iter,
            eps=self.epsilon,
            batch_size=x.shape[0],
        )
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        x_adv_np = attack.generate(x=x_np, y=y_np)
        return torch.tensor(x_adv_np, dtype=x.dtype, device=x.device)


class AdversarialTraining:
    """Adversarial training procedure for robust model training"""

    def __init__(self, model: nn.Module, attack: AdversarialAttack, 
                 epsilon: float = 8/255, device: torch.device = None):
        """
        Args:
            model (nn.Module): Model to train
            attack (AdversarialAttack): Attack to use for generating adversarial examples
            epsilon (float): Perturbation budget
            device (torch.device): Device to use
        """
        self.model = model
        self.attack = attack
        self.epsilon = epsilon
        self.device = device or torch.device('cpu')

    def training_step(self, batch: torch.Tensor, labels: torch.Tensor, 
                     optimizer: torch.optim.Optimizer, criterion: nn.Module,
                     attack_prob: float = 0.5) -> float:
        """
        Single adversarial training step
        
        Args:
            batch (torch.Tensor): Input batch
            labels (torch.Tensor): Target labels
            optimizer (torch.optim.Optimizer): Optimizer
            criterion (nn.Module): Loss function
            attack_prob (float): Probability of using adversarial examples
            
        Returns:
            float: Loss value
        """
        self.model.train()
        
        # Adversarial examples with probability
        if np.random.rand() < attack_prob:
            x_adv = self.attack.generate(batch, labels)
            logits = self.model(x_adv)
        else:
            logits = self.model(batch)
        
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


class RobustnessEvaluator:
    """Evaluate model robustness against adversarial attacks"""

    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Args:
            model (nn.Module): Model to evaluate
            device (torch.device): Device to use
        """
        self.model = model
        self.device = device or torch.device('cpu')
        
        self.attacks = {
            'fgsm': FGSMAttack,
            'pgd':  PGDAttack,
            'cw':   CWAttack,
            'bim':       BIMAttack,
            'mim':       MIMAttack,
            'deepfool':  DeepFoolAttack,
            'apgd':      AutoPGDAttack,
            'square':    SquareAttackART,
        }

    def evaluate(self, dataloader, attack_name: str = 'pgd', 
                 attack_kwargs: Optional[dict] = None) -> dict:
        """
        Evaluate robustness against specified attack
        
        Args:
            dataloader: Data loader with (x, y) batches
            attack_name (str): Name of attack ('pgd', 'fgsm', 'cw')
            attack_kwargs (dict, optional): Attack parameters
            
        Returns:
            dict: Evaluation results with clean and robust accuracy
        """
        if attack_name not in self.attacks:
            raise ValueError(f"Attack must be one of {list(self.attacks.keys())}")
        
        attack_kwargs = attack_kwargs or {}
        attack_cls = self.attacks[attack_name]
        attack = attack_cls(self.model, device=self.device, **attack_kwargs)
        
        self.model.eval()
        total_correct = 0
        total_robust_correct = 0
        total_samples = 0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            # ── Clean accuracy (no gradients needed) ─────────────────────────
            with torch.no_grad():
                logits_clean = self.model(x)
                clean_pred = logits_clean.argmax(dim=1)
                total_correct += (clean_pred == y).sum().item()

            # ── Generate adversarial examples (needs gradients for white-box) ─
            # Enable grad context so that gradient-based attacks (FGSM, PGD,
            # BIM, MIM, C&W) can compute ∂L/∂x.  ART-backed and black-box
            # attacks (DeepFool, APGD, Square) handle their own grad context.
            with torch.enable_grad():
                x_adv = attack.generate(x, y)

            # ── Robust accuracy (no gradients needed) ─────────────────────────
            with torch.no_grad():
                logits_adv = self.model(x_adv)
                robust_pred = logits_adv.argmax(dim=1)
                total_robust_correct += (robust_pred == y).sum().item()

            total_samples += x.shape[0]
        
        return {
            'clean_accuracy': 100.0 * total_correct / total_samples,
            'robust_accuracy': 100.0 * total_robust_correct / total_samples,
            'attack': attack_name,
            'samples': total_samples
        }

    def evaluate_multiple_attacks(self, dataloader, attacks: List[str] = None,
                                  attack_kwargs_dict: Optional[dict] = None) -> dict:
        """
        Evaluate robustness against multiple attacks
        
        Args:
            dataloader: Data loader
            attacks (List[str]): List of attack names
            attack_kwargs_dict (dict, optional): Dict mapping attack names to kwargs
            
        Returns:
            dict: Results for all attacks
        """
        attacks = attacks or list(self.attacks.keys())
        attack_kwargs_dict = attack_kwargs_dict or {}
        
        results = {}
        for attack in attacks:
            kwargs = attack_kwargs_dict.get(attack, {})
            results[attack] = self.evaluate(dataloader, attack, kwargs)
        
        return results


class CertifiedDefense:
    """Certified defenses guarantee robustness for bounded perturbations"""

    @staticmethod
    def randomized_smoothing(model: nn.Module, x: torch.Tensor, num_samples: int = 100,
                            sigma: float = 0.25, num_classes: int = 10) -> Tuple[int, float]:
        """
        Randomized smoothing for certified robustness
        
        Args:
            model (nn.Module): Base model
            x (torch.Tensor): Input
            num_samples (int): Number of noise samples
            sigma (float): Noise standard deviation
            num_classes (int): Number of classes
            
        Returns:
            Tuple[int, float]: Predicted class and certified radius
        """
        model.eval()
        
        # Get base prediction
        with torch.no_grad():
            logits_base = model(x)
            class_counts = logits_base.argmax(dim=1)
        
        # Sample with noise
        votes = torch.zeros(num_classes, device=x.device)
        
        for _ in range(num_samples):
            noise = torch.randn_like(x) * sigma
            x_noisy = torch.clamp(x + noise, 0, 1)
            
            with torch.no_grad():
                logits = model(x_noisy)
                votes += (logits.argmax(dim=1) == class_counts).float()
        
        # Certified radius (lower bound)
        certified_radius = 2 * sigma * (votes.max() / num_samples - 0.5)
        
        return class_counts.item(), certified_radius.item()


# Export public API
__all__ = [
    # Abstract base
    'AdversarialAttack',

    # Gradient-based white-box attacks (pure PyTorch, no extra dependencies)
    'FGSMAttack',          # Fast Gradient Sign Method
    'PGDAttack',           # Projected Gradient Descent
    'CWAttack',            # Carlini & Wagner
    'BIMAttack',           # Basic Iterative Method (I-FGSM)
    'MIMAttack',           # Momentum Iterative Method (MI-FGSM)

    # ART-backed attacks (require: pip install adversarial-robustness-toolbox)
    'DeepFoolAttack',      # Minimal-perturbation decision-boundary crossing
    'AutoPGDAttack',       # Auto-PGD (adaptive, used in AutoAttack benchmark)
    'SquareAttackART',     # Square Attack — black-box, score-based, no gradients

    # Training / evaluation utilities
    'AdversarialTraining',
    'RobustnessEvaluator',
    'CertifiedDefense',
]
