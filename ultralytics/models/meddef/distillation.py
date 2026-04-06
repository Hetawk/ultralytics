# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Defensive Distillation Mixin for MedDef trainers
=================================================
Single responsibility: attach teacher-model management and the distillation
loss loop to any ``BaseTrainer`` subclass — without touching the core
trainer code.

How to use
----------
Inherit *before* ``BaseTrainer`` so the MRO resolves correctly:

    class MedDefTrainer(DistillationMixin, BaseTrainer):
        ...

The mixin adds:
  * ``_teacher``              — frozen teacher ``nn.Module`` (or ``None``)
  * ``_distillation_enabled`` — reads ``model.distillation_config``
  * ``_init_teacher``         — deep-copies the initial student weights
  * ``_setup_train``          — calls parent then initialises teacher
  * ``criterion``             — forwards to ``DefensiveDistillationLoss``
                                or plain CE depending on config

Loss formula (when enabled)
---------------------------
  L = α · KL(σ(student/T) ‖ σ(teacher/T))  +  (1−α) · CE(student, y)

where T = distillation temperature, α = distillation weight.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import is_parallel

if TYPE_CHECKING:
    pass   # keep circular-safe

logger = logging.getLogger(__name__)

__all__ = ["DistillationMixin"]


class DistillationMixin:
    """Mixin that adds defensive distillation to any BaseTrainer subclass.

    Attributes:
        _teacher (nn.Module | None): Frozen teacher model; ``None`` when
            distillation is disabled or before ``_setup_train`` is called.
    """

    # set in __init__ of the concrete trainer
    _teacher: nn.Module | None

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _unwrap_model(self) -> nn.Module:
        """Return the raw model, unwrapping DDP if necessary."""
        return self.model.module if is_parallel(self.model) else self.model  # type: ignore[attr-defined]

    def _get_distillation_config(self) -> dict:
        """Return the loaded distillation config as a plain dict."""
        cfg = getattr(self._unwrap_model(), "distillation_config", {}) or {}
        return cfg if isinstance(cfg, dict) else {}

    def _distillation_enabled(self) -> bool:
        """Return ``True`` when the loaded YAML enables distillation."""
        return bool(self._get_distillation_config().get("enabled", False))

    # ──────────────────────────────────────────────────────────────────────
    # Teacher initialisation
    # ──────────────────────────────────────────────────────────────────────

    def _clone_teacher_from_student(self) -> nn.Module:
        """Deep-copy the current student weights as a frozen fallback teacher."""
        inner = getattr(self._unwrap_model(), "model", self._unwrap_model())
        teacher = copy.deepcopy(inner)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.to(self.device)  # type: ignore[attr-defined]
        LOGGER.info(
            "MedDef distillation: frozen teacher created from student init "
            f"({sum(p.numel() for p in teacher.parameters()):,} params)"
        )
        return teacher

    def _resolve_teacher_path(self, teacher_model: str | Path | None) -> Path | None:
        """Resolve a teacher checkpoint path from config or CLI input."""
        if not teacher_model:
            return None

        teacher_path = Path(teacher_model).expanduser()
        if teacher_path.is_absolute():
            return teacher_path

        search_roots = [
            Path.cwd(),
            Path(getattr(self, "save_dir", Path.cwd())),
            Path(getattr(self, "save_dir", Path.cwd())).parent,
        ]
        for root in search_roots:
            candidate = (root / teacher_path).resolve()
            if candidate.exists():
                return candidate

        return teacher_path.resolve()

    def _load_teacher_from_checkpoint(self, teacher_model: str | Path) -> nn.Module:
        """Load a frozen teacher from an already-trained checkpoint."""
        teacher_path = self._resolve_teacher_path(teacher_model)
        if teacher_path is None or not teacher_path.exists():
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {teacher_model}")

        from ultralytics.nn.tasks import load_checkpoint

        teacher_model_obj, _ = load_checkpoint(
            str(teacher_path), device=self.device, fuse=False)
        teacher = getattr(teacher_model_obj, "model", teacher_model_obj)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.to(self.device)  # type: ignore[attr-defined]
        LOGGER.info(
            f"MedDef distillation: loaded frozen teacher from {teacher_path}")
        return teacher

    def _init_teacher(self) -> None:
        """Initialise the frozen teacher for defensive distillation.

        Preference order:
          1. reuse an explicitly provided trained teacher checkpoint
          2. fall back to a copy of the initial student weights
        """
        cfg = self._get_distillation_config()
        teacher_model = cfg.get("teacher_model")

        if teacher_model:
            try:
                self._teacher = self._load_teacher_from_checkpoint(
                    teacher_model)
                return
            except Exception as exc:
                LOGGER.warning(
                    f"MedDef distillation: failed to load teacher checkpoint '{teacher_model}': {exc}. "
                    "Falling back to the student's initial weights."
                )

        self._teacher = self._clone_teacher_from_student()

    # ──────────────────────────────────────────────────────────────────────
    # Training setup hook
    # ──────────────────────────────────────────────────────────────────────

    def _setup_train(self) -> None:  # type: ignore[override]
        """Extend parent ``_setup_train`` to initialise the teacher."""
        super()._setup_train()  # type: ignore[misc]
        if not hasattr(self, "_teacher"):
            self._teacher = None
        if self._distillation_enabled():
            self._init_teacher()

    # ──────────────────────────────────────────────────────────────────────
    # Criterion (overrides BaseTrainer.criterion)
    # ──────────────────────────────────────────────────────────────────────

    def criterion(self, preds: torch.Tensor, batch: dict) -> torch.Tensor:
        """Compute distillation or plain CE loss for a batch.

        When a teacher is present:
            L = α·KL(σ(preds/T)‖σ(teacher/T)) + (1−α)·CE(preds, y)
        Otherwise falls back to the model's own ``init_criterion()``.

        Args:
            preds: Student logits ``[B, nc]``.
            batch: Dict with keys ``img`` and ``cls``.

        Returns:
            Scalar loss tensor.
        """
        wrapped = self._unwrap_model()
        loss_fn = getattr(wrapped, "criterion", None)
        if loss_fn is None:
            loss_fn = wrapped.init_criterion()

        # Hard guard: MedDef NEVER uses adversarial training losses.
        _FORBIDDEN_LOSSES = frozenset({"TRADESLoss", "MARTLoss", "RobustMinMaxLoss",
                                       "AdversarialWeightPerturbation"})
        if type(loss_fn).__name__ in _FORBIDDEN_LOSSES:
            raise RuntimeError(
                f"MedDef training must NOT use adversarial training. "
                f"'{type(loss_fn).__name__}' was set as criterion. "
                f"Use DefensiveDistillationLoss (via distillation.enabled=true in YAML) "
                f"or the default v8ClassificationLoss."
            )

        targets = batch["cls"]
        if targets.ndim > 1:
            targets = targets.squeeze(-1)
        targets = targets.long()

        if self._teacher is not None:
            with torch.no_grad():
                teacher_logits = self._teacher(batch["img"].float())
            return loss_fn(preds, targets, teacher_logits)

        return loss_fn(preds, targets)
