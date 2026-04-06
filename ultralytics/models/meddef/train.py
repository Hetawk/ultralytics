# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from typing import Any

import torch

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import meddef
from ultralytics.models.meddef.distillation import DistillationMixin
from ultralytics.nn.tasks import MedDefModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images
from ultralytics.utils.torch_utils import is_parallel, torch_distributed_zero_first


class MedDefTrainer(DistillationMixin, BaseTrainer):
    """A trainer class extending BaseTrainer for training MedDef classification models.

    This trainer handles the training process for MedDef (Medical Defense) classification tasks,
    which are Vision Transformer-based models with defense mechanisms. It supports comprehensive
    dataset handling and validation similar to YOLO classification training.

    Attributes:
        model (MedDefModel): The MedDef classification model to be trained.
        data (dict[str, Any]): Dictionary containing dataset information including class names and number of classes.
        loss_names (list[str]): Names of the loss functions used during training.
        validator (MedDefValidator): Validator instance for model evaluation.

    Methods:
        set_model_attributes: Set the model's class names from the loaded dataset.
        get_model: Return a modified PyTorch model configured for training.
        setup_model: Load, create or download model for classification.
        build_dataset: Create a ClassificationDataset instance.
        get_dataloader: Return PyTorch DataLoader with transforms for image preprocessing.
        preprocess_batch: Preprocess a batch of images and classes.
        progress_string: Return a formatted string showing training progress.
        get_validator: Return an instance of MedDefValidator.
        label_loss_items: Return a loss dict with labeled training loss items.
        final_eval: Evaluate trained model and save validation results.
        plot_training_samples: Plot training samples with their annotations.

    Examples:
        Initialize and train a MedDef model
        >>> from ultralytics.models.meddef import MedDefTrainer
        >>> args = dict(model="meddef2_t.pt", data="imagenet10", epochs=3)
        >>> trainer = MedDefTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize a MedDefTrainer object.

        Args:
            cfg (dict[str, Any], optional): Default configuration dictionary containing training parameters.
            overrides (dict[str, Any], optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list[Any], optional): List of callback functions to be executed during training.
        """
        if overrides is None:
            overrides = {}
        # Pop 'scale' BEFORE Ultralytics cfg validation — Ultralytics expects scale to be
        # a numeric augmentation factor, but MedDef uses letter codes ('n','s','m','l','x')
        # for architecture size. We restore it into the model YAML dict in get_model().
        self._arch_scale: str = overrides.pop("scale", "n")
        # Pop 'use_class_weights' — not a standard Ultralytics cfg key; we restore it on
        # self.args after super().__init__() so get_dataloader() can read it.
        self._use_class_weights: bool = bool(overrides.pop("use_class_weights", False))
        # If the caller passed a pre-parsed YAML dict as `model` (e.g. from train_meddef.py
        # after overriding distillation config), stash it and replace with the YAML filename
        # string so that BaseTrainer.__init__ can call Path(model) without crashing.
        if isinstance(overrides.get("model"), dict):
            self._model_cfg_override: dict | None = dict(overrides["model"])
            overrides["model"] = overrides["model"].get("yaml_file", "meddef2.yaml")
        else:
            self._model_cfg_override = None
        overrides["task"] = "classify"
        # Disable AMP before super().__init__() so self.args.amp is False when
        # _setup_train() reads it at line 300 of trainer.py.  Setting self.amp=False
        # *after* super().__init__() is too late — _setup_train re-reads self.args.amp.
        # MedDef2 ViT variants have explicit .float() casts throughout forward_features
        # so the whole model must run in fp32.
        overrides["amp"] = False
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224
        # When --resume is used, BaseTrainer.__init__() replaces self.args with the
        # checkpoint's saved args, which includes the original epoch count (e.g. 20).
        # Capture the user's desired epochs *before* super().__init__() so we can
        # restore a higher value afterwards (e.g. extending a 20-epoch run to 100).
        # Same for patience — the checkpoint may have patience=20 from stage-1 but
        # the caller passes patience=0 to disable early stopping for distillation.
        _desired_epochs = overrides.get("epochs")
        _desired_patience = overrides.get("patience")
        super().__init__(cfg, overrides, _callbacks)
        # Restore user's higher epoch count if a resume capped it from the checkpoint.
        if _desired_epochs is not None and self.epochs < _desired_epochs:
            self.args.epochs = _desired_epochs
            self.epochs = _desired_epochs
        # Restore user's patience if the checkpoint's saved args overwrote it.
        if _desired_patience is not None:
            self.args.patience = _desired_patience
        # Belt-and-suspenders: also clear self.amp after init.
        self.amp = False
        self._teacher = None  # initialised by DistillationMixin._setup_train
        # NOTE: do NOT restore use_class_weights onto self.args — it is not a valid
        # Ultralytics cfg key and check_dict_alignment() in get_validator() will
        # raise SyntaxError if it appears there.  Use self._use_class_weights instead.

    def set_model_attributes(self):
        """Set the MedDef model's class names from the loaded dataset."""
        self.model.names = self.data["names"]

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Return a modified PyTorch model configured for training MedDef classification.

        Args:
            cfg (Any, optional): Model configuration.
            weights (Any, optional): Pre-trained model weights.
            verbose (bool, optional): Whether to display model information.

        Returns:
            (MedDefModel): Configured PyTorch model for classification.
        """
        # Inject architecture scale into the YAML dict so build_meddef_model picks
        # the right depth/width.  cfg may be a path string or an already-parsed dict.
        arch_scale = getattr(self, "_arch_scale", "n")
        # Prefer the pre-parsed + distillation-overridden dict stored in __init__.
        if getattr(self, "_model_cfg_override", None) is not None:
            cfg = dict(self._model_cfg_override)
        elif isinstance(cfg, str):
            from ultralytics.nn.tasks import yaml_model_load
            cfg = yaml_model_load(cfg)
        if isinstance(cfg, dict):
            cfg = dict(cfg)  # shallow copy — don't mutate the cached original
            cfg["scale"] = arch_scale
        nc = self.data["nc"]
        ch = self.data.get("channels", 3)
        model = MedDefModel(cfg, nc=nc, ch=ch, verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training
        return model

    def setup_model(self):
        """Load, create or download model for MedDef classification tasks.

        Returns:
            (Any): Model checkpoint if applicable, otherwise None.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model = str(self.model)
        ckpt = None  # preserve checkpoint dict for resume_training()

        # Load a YOLO model locally, from torchvision, or from HUB
        if model.endswith(".pt"):
            self.model, ckpt = attempt_load_one_weight(model, device="cpu")
            for p in self.model.parameters():
                p.requires_grad = True  # for training
        elif model.endswith(".yaml"):
            self.model = self.get_model(cfg=model)
        elif model.split(".")[-1] in {"py", ""}:  # MedDef architecture name
            self.model = self.get_model(cfg=model)
        else:
            raise FileNotFoundError(f"ERROR ❌ Model file {model} not found")

        # When resuming, the model was already loaded from last.pt above with the
        # correct trained weights + optimizer state in `ckpt`.  Do NOT overwrite
        # with pretrained (teacher) weights — that would erase all training progress.
        if not self.resume:
            pretrained = self.args.pretrained
            # Only load weights when pretrained is an actual path/weights object.
            # A bare bool (True) just means "accept pretrained init" — MedDef2 YAML
            # builds have no external checkpoint to load, so skip it in that case.
            if pretrained and not isinstance(pretrained, bool):
                # MedDefModel.load() expects an nn.Module or weights dict, not a raw
                # string path.  When given a .pt path, load the checkpoint first.
                # Ultralytics saves the best model under 'ema'; 'model' key may be None.
                if isinstance(pretrained, str) and pretrained.endswith(".pt"):
                    pt_ckpt = torch.load(pretrained, map_location="cpu")
                    if isinstance(pt_ckpt, dict):
                        weights = pt_ckpt.get("ema") or pt_ckpt.get("model") or pt_ckpt
                    else:
                        weights = pt_ckpt
                    if weights is not None:
                        self.model.load(weights)
                else:
                    self.model.load(pretrained)

        return ckpt

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """Create a ClassificationDataset instance.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): Dataset mode, either 'train' or 'val'.
            batch (int, optional): Batch size. Defaults to None.

        Returns:
            (ClassificationDataset): Classification dataset instance.
        """
        return ClassificationDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Return PyTorch DataLoader with transforms for image preprocessing.

        When ``use_class_weights=True`` is set in the training overrides, the train
        loader uses an inverse-frequency WeightedRandomSampler so every class is
        sampled with equal probability per epoch — this prevents the model collapsing
        to the dominant class on highly imbalanced datasets (e.g. SCISIC 9-class).

        Args:
            dataset_path (str): Path to the dataset folder.
            batch_size (int): Batch size.
            rank (int): Rank of the process in distributed training.
            mode (str): Dataset mode, either 'train' or 'val'.

        Returns:
            (DataLoader): PyTorch DataLoader.
        """
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode)

        # ── Weighted sampler for class imbalance (train only, single-GPU) ──────
        if mode == "train" and self._use_class_weights and rank == -1:
            import os
            from torch.utils.data import WeightedRandomSampler
            from ultralytics.data.build import InfiniteDataLoader, seed_worker
            from ultralytics.utils import LOGGER

            # ClassificationDataset stores samples as [file, class_idx, npy, im]
            targets = torch.tensor([s[1] for s in dataset.samples], dtype=torch.long)
            class_counts = torch.bincount(targets)
            class_weights = 1.0 / class_counts.float().clamp(min=1)
            sample_weights = class_weights[targets]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            nd = torch.cuda.device_count()
            nw = min(os.cpu_count() // max(nd, 1), self.args.workers)
            LOGGER.info(
                f"[MedDefTrainer] WeightedRandomSampler enabled\n"
                f"  class counts : {class_counts.tolist()}\n"
                f"  class weights: {[f'{w:.4f}' for w in class_weights.tolist()]}"
            )
            loader = InfiniteDataLoader(
                dataset=dataset,
                batch_size=min(batch_size, len(dataset)),
                shuffle=False,  # sampler provides the ordering
                num_workers=nw,
                sampler=sampler,
                prefetch_factor=4 if nw > 0 else None,
                pin_memory=nd > 0,
                collate_fn=getattr(dataset, "collate_fn", None),
                worker_init_fn=seed_worker,
                drop_last=self.args.compile,
            )
            return loader
        # ── Standard loader ────────────────────────────────────────────────────

        loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank, drop_last=self.args.compile)
        # Attach inference transforms
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess a batch of images and classes.

        During training, applies an input-space defense pipeline (spatial smoothing,
        feature squeezing, randomised Gaussian noise) *before* the forward pass.
        These are ART-inspired, pure-PyTorch, adversarial-training-free defenses that
        make the model more robust by reducing the attack surface at the pixel level.

        During validation/inference the raw (normalised) pixels are used unchanged
        so metrics are not artificially inflated.

        Args:
            batch (dict[str, Any]): Batch dictionary containing 'img' and 'cls' keys.

        Returns:
            (dict[str, Any]): Preprocessed batch with images moved to device.
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda")
        batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")

        # Apply input-space defenses only during training (model.training == True)
        if getattr(self.model, "training", False):
            if not hasattr(self, "_input_defense_pipeline"):
                from ultralytics.nn.modules.defense import InputTransformPipeline
                self._input_defense_pipeline = (
                    InputTransformPipeline.recommended_training().to(self.device)
                )
            with torch.no_grad():
                batch["img"] = self._input_defense_pipeline(batch["img"].float())

        return batch

    def progress_string(self) -> str:
        """Return a formatted string showing training progress.

        Returns:
            (str): Formatted string with epoch, GPU memory, loss, accuracy, and metrics.
        """
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def get_validator(self):
        """Return an instance of MedDefValidator for validation.

        Returns:
            (MedDefValidator): Validator instance.
        """
        self.loss_names = ["loss"]
        return meddef.val.MedDefValidator(
            self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix: str = "train"):
        """Return a loss dict with labeled training loss items.

        Args:
            loss_items (torch.Tensor, optional): Loss tensor.
            prefix (str): Prefix for loss item names.

        Returns:
            (dict): Dictionary with loss items.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def final_eval(self):
        """Evaluate trained model, compute enhanced metrics, and generate visualizations."""
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    continue  # validation already run
                self.validator.args.plots = self.args.plots
                self.metrics = self.validator(model=f)

        # Run comprehensive metrics + visualizations after final validation
        try:
            enhanced = self.validator.compute_enhanced_metrics(epoch=self.epoch + 1)
            self._enhanced_metrics = enhanced
        except Exception as e:
            from ultralytics.utils import LOGGER
            LOGGER.warning(f"Enhanced metrics computation failed: {e}")

        self.run_callbacks("on_train_end")

    def plot_training_samples(self, batch: dict[str, Any], ni: int):
        """Plot training samples with their annotations.

        Args:
            batch (dict[str, Any]): Batch dictionary.
            ni (int): Number of integrated batches.
        """
        batch["batch_idx"] = torch.arange(batch["img"].shape[0])  # add batch index for plotting
        plot_images(
            labels=batch,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            names=self.data["names"],
            on_plot=self.on_plot,
        )


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Load a single MedDef checkpoint and return (model, ckpt).

    Ultralytics saves best models under the 'ema' key; 'model' may be None.
    This shim replaces the broken tasks.py import with a direct torch.load call.
    """
    import torch as _torch
    ckpt = _torch.load(weight, map_location=device or "cpu")
    if isinstance(ckpt, dict):
        model = ckpt.get("ema") or ckpt.get("model") or ckpt
    else:
        model = ckpt
    if hasattr(model, "float"):
        model = model.float()
    if fuse and hasattr(model, "fuse"):
        model.fuse()
    return model, ckpt
