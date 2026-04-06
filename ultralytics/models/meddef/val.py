# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.utils.plotting import plot_images


class MedDefValidator(BaseValidator):
    """A class extending the BaseValidator class for validation based on a MedDef classification model.

    This validator handles the validation process for MedDef models, including metrics calculation,
    confusion matrix generation, and visualization of results.

    Attributes:
        targets (list[torch.Tensor]): Ground truth class labels.
        pred (list[torch.Tensor]): Model predictions.
        metrics (ClassifyMetrics): Object to calculate and store classification metrics.
        names (dict): Mapping of class indices to class names.
        nc (int): Number of classes.
        confusion_matrix (ConfusionMatrix): Matrix to evaluate model performance across classes.

    Methods:
        get_desc: Return a formatted string summarizing classification metrics.
        init_metrics: Initialize confusion matrix, class names, and tracking containers.
        preprocess: Preprocess input batch by moving data to device.
        update_metrics: Update running metrics with model predictions and batch targets.
        finalize_metrics: Finalize metrics including confusion matrix and processing speed.
        postprocess: Extract the primary prediction from model output.
        get_stats: Calculate and return a dictionary of metrics.
        build_dataset: Create a ClassificationDataset instance for validation.
        get_dataloader: Build and return a data loader for classification validation.
        print_results: Print evaluation metrics for the classification model.
        plot_val_samples: Plot validation image samples with their ground truth labels.
        plot_predictions: Plot images with their predicted class labels.

    Examples:
        >>> from ultralytics.models.meddef import MedDefValidator
        >>> args = dict(model="meddef2_t.pt", data="imagenet10")
        >>> validator = MedDefValidator(args=args)
        >>> validator()

    Notes:
        MedDef models are Vision Transformer-based classification models with defense mechanisms.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize MedDefValidator with dataloader, save directory, and other parameters.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (str | Path, optional): Directory to save results.
            args (dict, optional): Arguments containing model and validation configuration.
            _callbacks (list, optional): List of callback functions to be called during validation.

        Examples:
            >>> from ultralytics.models.meddef import MedDefValidator
            >>> args = dict(model="meddef2_t.pt", data="imagenet10")
            >>> validator = MedDefValidator(args=args)
            >>> validator()
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.targets = None
        self.pred = None
        self.probs = None  # store softmax probabilities for enhanced metrics
        self.args.task = "classify"
        self.metrics = ClassifyMetrics()

    def get_desc(self) -> str:
        """Return a formatted string summarizing classification metrics."""
        return ("%22s" + "%11s" * 2) % ("classes", "top1_acc", "top5_acc")

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize confusion matrix, class names, and tracking containers for predictions and targets."""
        self.names = model.names
        self.nc = len(model.names)
        self.pred = []
        self.targets = []
        self.probs = []  # softmax probabilities per batch
        self.confusion_matrix = ConfusionMatrix(names=model.names)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess input batch by moving data to device and converting to appropriate dtype."""
        batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def update_metrics(self, preds: torch.Tensor, batch: dict[str, Any]) -> None:
        """Update running metrics with model predictions and batch targets.

        Args:
            preds (torch.Tensor): Model predictions, typically logits or probabilities for each class.
            batch (dict[str, Any]): Input batch containing 'cls' key with ground truth labels.
        """
        targets = batch["cls"]
        n5 = min(len(self.names), 5)
        self.pred.append(preds.argsort(1, descending=True)[:, :n5].type(torch.int32).cpu())
        self.targets.append(targets.type(torch.int32).cpu())
        # Store softmax probabilities for ROC-AUC, calibration, and enhanced metrics
        self.probs.append(torch.softmax(preds.float(), dim=1).cpu())

    def finalize_metrics(self, *args, **kwargs) -> None:
        """Finalize metrics including confusion matrix plotting and processing speed."""
        # Process confusion matrix from predictions
        self.confusion_matrix.process_cls_preds(self.pred, self.targets)
        
        # Plot confusion matrices (normalized and non-normalized)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
        
        # Update metrics object
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir
        self.metrics.confusion_matrix = self.confusion_matrix

    def compute_enhanced_metrics(self, epoch: int | None = None) -> dict:
        """Run comprehensive metrics (MCC, Cohen's kappa, ECE, per-class, etc.) and
        save results as CSV + TXT + JSON. Also generates visualization plots.

        Call this AFTER finalize_metrics() / get_stats() so self.pred, self.targets,
        and self.probs are populated.

        Uses the model-agnostic ClassificationMetrics and ClassificationVisualizer
        modules — works with ANY classifier, not tied to a specific architecture.

        Returns:
            dict: Full metrics dictionary from ClassificationMetrics.compute().
        """
        import numpy as np
        from ultralytics.utils.classify_metrics import ClassificationMetrics
        from ultralytics.utils.classify_visualize import ClassificationVisualizer

        # Flatten accumulated tensors
        all_targets = torch.cat(self.targets).numpy().ravel()
        all_preds_topk = torch.cat(self.pred).numpy()  # (N, 5)
        all_pred_ids = all_preds_topk[:, 0]             # top-1 class

        class_names = list(self.names.values()) if isinstance(self.names, dict) else list(self.names)

        # Build ClassificationMetrics (model-agnostic)
        mm = ClassificationMetrics(class_names=class_names)

        if self.probs:
            all_probs = torch.cat(self.probs).numpy()  # (N, C)
            mm.update(
                preds=torch.from_numpy(all_probs),
                targets=torch.from_numpy(all_targets.astype(np.int64)),
                probs=torch.from_numpy(all_probs),
            )
        else:
            mm.update(
                preds=torch.from_numpy(all_pred_ids.astype(np.int64)),
                targets=torch.from_numpy(all_targets.astype(np.int64)),
            )

        # Compute all metrics
        results = mm.compute()

        # Save CSV + TXT + JSON
        mm.save(self.save_dir, epoch=epoch)
        LOGGER.info(
            f"Enhanced metrics: F1={results.get('f1_macro', 0):.4f}  "
            f"MCC={results.get('mcc', 0):.4f}  "
            f"Kappa={results.get('cohen_kappa', 0):.4f}  "
            f"ECE={results.get('ece', 'N/A')}"
        )

        # Generate visualizations (model-agnostic)
        try:
            viz = ClassificationVisualizer(self.save_dir)

            # Confusion matrix
            cm = np.array(results["confusion_matrix"])
            viz.confusion_matrix(cm, class_names)

            # Per-class metrics
            if results.get("per_class"):
                viz.per_class_metrics(results["per_class"])

            # ROC / AUC (needs probabilities)
            if self.probs:
                viz.roc_auc(all_targets, all_probs, class_names)
                # Calibration reliability diagram
                viz.calibration_diagram(all_targets, all_probs)

            # Metrics summary panel
            viz.metrics_summary(results)

            # Training curves (if results.csv exists)
            csv_path = self.save_dir / "results.csv"
            if csv_path.exists():
                viz.training_curves(csv_path)

            LOGGER.info(f"Visualizations saved → {viz.viz_dir}")
        except Exception as e:
            LOGGER.warning(f"Visualization generation failed: {e}")

        return results

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Extract the primary prediction from model output.

        Args:
            preds (torch.Tensor): Model output predictions.

        Returns:
            (torch.Tensor): Processed predictions.
        """
        return preds[0] if isinstance(preds, (list, tuple)) else preds

    def gather_stats(self) -> None:
        """Gather stats from all GPUs for distributed training."""
        if RANK == 0:
            gathered_preds = [None] * dist.get_world_size()
            gathered_targets = [None] * dist.get_world_size()
            dist.gather_object(self.pred, gathered_preds, dst=0)
            dist.gather_object(self.targets, gathered_targets, dst=0)
            self.pred = [pred for rank in gathered_preds for pred in rank]
            self.targets = [targets for rank in gathered_targets for targets in rank]
        elif RANK > 0:
            dist.gather_object(self.pred, None, dst=0)
            dist.gather_object(self.targets, None, dst=0)

    def get_stats(self) -> dict[str, Any]:
        """Calculate and return a dictionary of metrics.

        Returns:
            (dict): Dictionary containing top-1 and top-5 accuracy metrics.
        """
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def build_dataset(self, img_path: str, mode: str = "val", batch=None):
        """Create a ClassificationDataset instance for validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): Dataset mode, typically 'val' or 'test'.
            batch (int, optional): Batch size. Defaults to None.

        Returns:
            (ClassificationDataset): Classification dataset instance.
        """
        return ClassificationDataset(root=img_path, args=self.args, augment=False, prefix=mode)

    def get_dataloader(self, dataset_path: str, batch_size: int):
        """Build and return a data loader for classification validation.

        Args:
            dataset_path (str): Path to the dataset folder.
            batch_size (int): Batch size.

        Returns:
            (DataLoader): PyTorch DataLoader.
        """
        dataset = self.build_dataset(dataset_path, mode=self.args.split)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self) -> None:
        """Print evaluation metrics for the MedDef classification model."""
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5))

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot validation image samples with their ground truth labels.

        Args:
            batch (dict[str, Any]): Batch dictionary containing images and labels.
            ni (int): Number of integrated batches.
        """
        batch["batch_idx"] = torch.arange(batch["img"].shape[0])  # add batch index for plotting
        plot_images(
            labels=batch,
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch: dict[str, Any], preds: torch.Tensor, ni: int) -> None:
        """Plot images with their predicted class labels.

        Args:
            batch (dict[str, Any]): Batch dictionary containing images.
            preds (torch.Tensor): Model predictions.
            ni (int): Number of integrated batches.
        """
        batched_preds = dict(
            img=batch["img"],
            batch_idx=torch.arange(batch["img"].shape[0]),
            cls=torch.argmax(preds, dim=1),
            conf=torch.amax(preds, dim=1),
        )
        plot_images(
            batched_preds,
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred
