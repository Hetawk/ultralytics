"""Generate Grad-CAM overlays for Ultralytics YOLO detectors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import select_device

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class FeatureHook:
    """Capture forward activations and backward gradients from a target module."""

    def __init__(self) -> None:
        self.activation: Optional[torch.Tensor] = None
        self.gradient: Optional[torch.Tensor] = None
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def attach(self, module: torch.nn.Module) -> None:
        self.detach()
        self._handle = module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output: Union[torch.Tensor, Sequence, Dict]) -> None:  # noqa: ANN001
        tensor = self._select_tensor(output)
        if tensor is None:
            raise RuntimeError("Grad-CAM target layer did not return a tensor output.")
        self.activation = tensor
        self.gradient = None
        if tensor.requires_grad:
            tensor.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradient = grad.detach()

    @staticmethod
    def _select_tensor(output: Union[torch.Tensor, Sequence, Dict]) -> Optional[torch.Tensor]:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)):
            for item in reversed(output):
                if isinstance(item, torch.Tensor):  # prefer deepest tensor
                    return item
        if isinstance(output, dict):
            for item in reversed(list(output.values())):
                if isinstance(item, torch.Tensor):
                    return item
        return None

    def reset_gradient(self) -> None:
        self.gradient = None

    def clear(self) -> None:
        self.activation = None
        self.gradient = None

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays for YOLO models.")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--source", type=str, help="Image or directory/glob of images")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device (cpu, 0, 0,1, etc.)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--layer",
        type=int,
        default=-2,
        help="Index of the layer within model.model to target for Grad-CAM (supports negative indices)",
    )
    parser.add_argument("--cls", type=int, default=None, help="Class index to target (default: top scoring class)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for target selection")
    parser.add_argument("--alpha", type=float, default=0.4, help="Overlay alpha for heatmap blending")
    parser.add_argument("--output", type=str, help="Explicit directory to save Grad-CAM outputs (overrides auto layout)")
    parser.add_argument("--project", type=str, default="runs", help="Root directory for auto-generated outputs")
    parser.add_argument("--task", type=str, help="Task subdirectory used in auto layout (defaults to inferred task)")
    parser.add_argument("--mode", type=str, default="gradcam", help="Mode subdirectory name for auto layout")
    parser.add_argument("--dataset-name", type=str, help="Override dataset segment for auto layout")
    parser.add_argument("--run-name", type=str, help="Override run/model segment for auto layout")
    parser.add_argument(
        "--max-images",
        type=int,
        default=16,
        help="Maximum number of images to process (-1 processes all)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1,
        help="Number of top detections per image to visualize (<=0 keeps all passing the IoU/score filters)",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Print indexed list of model.model modules and exit",
    )
    parser.add_argument(
        "--iou-filter",
        type=float,
        default=0.5,
        help="IoU threshold used to suppress duplicate anchors (helps keep one box per object)",
    )
    return parser.parse_args()


def resolve_images(source: str) -> List[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    # treat as glob pattern
    matches = [Path(p) for p in Path().glob(source)]
    return sorted(p for p in matches if p.suffix.lower() in IMAGE_EXTENSIONS)


def prepare_image(
    img_path: Path,
    imgsz: int,
    stride: int,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray, Dict[str, float]]:
    img0 = cv2.imread(str(img_path))
    if img0 is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=stride)
    lb_result = letterbox(image=img0)
    pad_info = compute_padding_info(img0.shape[:2], imgsz)

    img = lb_result.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = torch.from_numpy(img).to(device)
    img = img.unsqueeze(0)  # add batch dim
    img.requires_grad_(True)
    return img, img0, pad_info


def compute_padding_info(shape: Tuple[int, int], imgsz: int) -> Dict[str, float]:
    h, w = shape
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = imgsz - new_w, imgsz - new_h
    dw /= 2
    dh /= 2
    top = round(dh - 0.1)
    bottom = round(dh + 0.1)
    left = round(dw - 0.1)
    right = round(dw + 0.1)
    return {
        "ratio": r,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "new_w": new_w,
        "new_h": new_h,
    }


def overlay_heatmap(heatmap: np.ndarray, original: np.ndarray, alpha: float) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(heatmap_color, alpha, original_bgr, 1 - alpha, 0)
    return overlay


def infer_task_from_model_path(model_path: Path, default: str = "detect") -> str:
    parts = model_path.parts
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return default


def infer_dataset_from_model_path(model_path: Path) -> Optional[str]:
    parts = model_path.parts
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 3 < len(parts):
            return parts[idx + 3]
    return None


def infer_model_name(model_path: Path) -> str:
    parts = model_path.parts
    if "weights" in parts:
        idx = parts.index("weights")
        if idx - 1 >= 0:
            return parts[idx - 1]
    return model_path.stem


def strip_glob_prefix(path_str: str) -> str:
    for token in ("*", "?", "["):
        loc = path_str.find(token)
        if loc != -1:
            return path_str[:loc]
    return path_str


def infer_dataset_from_source(source: Optional[str]) -> str:
    if not source:
        return "dataset"
    trimmed = strip_glob_prefix(source).rstrip("/ ")
    if not trimmed:
        return "dataset"
    path = Path(trimmed)
    parts = path.parts
    for anchor in ("dataset", "datasets", "data"):
        if anchor in parts:
            idx = parts.index(anchor)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    # fallback to last meaningful directory
    if path.is_dir():
        return path.name or "dataset"
    if path.is_file():
        return path.parent.name or "dataset"
    return path.name or "dataset"


def build_save_directory(args: argparse.Namespace, source: Optional[str]) -> Path:
    if args.output:
        return Path(args.output)

    model_path = Path(args.model)
    task = args.task or infer_task_from_model_path(model_path)
    mode = args.mode
    dataset = args.dataset_name or infer_dataset_from_model_path(model_path) or infer_dataset_from_source(source)
    run_name = args.run_name or infer_model_name(model_path)
    return Path(args.project) / task / mode / dataset / run_name


def ensure_feature_gradients(score: torch.Tensor, hook: FeatureHook) -> Optional[torch.Tensor]:
    """Return gradients for the hooked activation, falling back to autograd if hook capture failed."""

    if hook.gradient is not None:
        return hook.gradient
    activation = hook.activation
    if activation is None:
        return None
    try:
        return torch.autograd.grad(score, activation, retain_graph=True, allow_unused=True)[0]
    except RuntimeError:
        return None


def compute_input_saliency(
    score: torch.Tensor,
    image_tensor: torch.Tensor,
    pad: Dict[str, float],
    original_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Fallback saliency map via gradients w.r.t. network input."""

    try:
        grads = torch.autograd.grad(score, image_tensor, retain_graph=True, allow_unused=True)[0]
    except RuntimeError:
        grads = None
    if grads is None:
        return None
    saliency = grads.abs().mean(dim=1).squeeze().detach().cpu().numpy()
    saliency -= saliency.min()
    if saliency.max() > 0:
        saliency /= saliency.max()
    saliency = crop_and_resize_cam(saliency, pad, original_shape)
    return saliency


def get_target_module(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    modules = list(model.model)
    if layer_idx < 0:
        layer_idx += len(modules)
    if layer_idx < 0 or layer_idx >= len(modules):
        raise IndexError(f"Layer index {layer_idx} out of range (model has {len(modules)} modules)")
    return modules[layer_idx]


def compute_cam(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    target_size: Tuple[int, int],
) -> np.ndarray:
    grads = gradients.mean(dim=(2, 3), keepdim=True)
    weighted = (grads * activations).sum(dim=1, keepdim=True)
    cam = F.relu(weighted)
    if torch.all(cam <= 0):
        cam = weighted.abs()
    cam = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def crop_and_resize_cam(cam: np.ndarray, pad: Dict[str, float], original_shape: Tuple[int, int]) -> np.ndarray:
    h, w = cam.shape
    top, bottom, left, right = pad["top"], pad["bottom"], pad["left"], pad["right"]
    cropped = cam[int(top) : h - int(bottom), int(left) : w - int(right)] if (top or bottom or left or right) else cam
    resized = cv2.resize(cropped, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
    return resized


def xywh_tensor_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    x, y, w, h = xywh.unbind(-1)
    return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)


def main() -> None:
    args = parse_args()

    device = select_device(args.device)
    yolo = YOLO(args.model)
    det_model = yolo.model.to(device)
    det_model.eval()

    if args.list_layers:
        for idx, module in enumerate(det_model.model):
            LOGGER.info("%3d: %s", idx, module.__class__.__name__)
        return

    if not args.source:
        raise ValueError("--source is required unless --list-layers is used.")

    images = resolve_images(args.source)
    if not images:
        raise FileNotFoundError(f"No images found for source: {args.source}")

    if args.max_images > 0:
        images = images[: args.max_images]
        LOGGER.info("Limiting Grad-CAM generation to %d images", len(images))

    layer = get_target_module(det_model, args.layer)
    feature_hook = FeatureHook()
    feature_hook.attach(layer)

    save_dir = build_save_directory(args, args.source)
    save_dir.mkdir(parents=True, exist_ok=True)

    stride = int(det_model.stride.max().item()) if hasattr(det_model, "stride") else 32
    try:
        for img_path in images:
            img_tensor, original_rgb, pad_info = prepare_image(img_path, args.imgsz, stride, device)

            det_model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                preds = det_model(img_tensor)
                preds = preds[0] if isinstance(preds, (tuple, list)) else preds

            if feature_hook.activation is None:
                raise RuntimeError("Target layer did not produce activations. Use --list-layers to pick a backbone block.")

            pred_transposed = preds.transpose(1, 2)  # (B, anchors, 4+nc)
            scores = pred_transposed[..., 4:]
            boxes = pred_transposed[..., :4]
            class_scores = scores

            per_detection_outputs = []

            if args.cls is not None and (args.cls < 0 or args.cls >= class_scores.shape[-1]):
                raise ValueError(f"Class index {args.cls} outside valid range [0, {class_scores.shape[-1] - 1}].")

            if args.cls is not None:
                cls_scores = class_scores[0, :, args.cls]
            else:
                cls_scores = class_scores[0].max(dim=1).values

            boxes_xyxy = xywh_tensor_to_xyxy(boxes[0])
            order = torch.argsort(cls_scores, descending=True)
            selected_indices: List[int] = []
            selected_scores: List[torch.Tensor] = []
            limit = args.topk if args.topk > 0 else None
            for idx in order.tolist():
                score_tensor = cls_scores[idx]
                if args.conf > 0 and float(score_tensor) < args.conf:
                    break
                if selected_indices:
                    prev_boxes = boxes_xyxy[selected_indices]
                    if prev_boxes.ndim == 1:
                        prev_boxes = prev_boxes.unsqueeze(0)
                    ious = box_iou(boxes_xyxy[idx].unsqueeze(0), prev_boxes)
                    if float(ious.max()) > args.iou_filter:
                        continue
                selected_indices.append(idx)
                selected_scores.append(score_tensor)
                if limit is not None and len(selected_indices) >= limit:
                    break

            if not selected_indices:
                best_score, best_idx = cls_scores.max(dim=0)
                selected_indices = [int(best_idx.item())]
                selected_scores = [best_score]

            num_selected = len(selected_indices)
            for rank, (anchor_idx, score_tensor) in enumerate(zip(selected_indices, selected_scores)):
                det_model.zero_grad(set_to_none=True)
                feature_hook.reset_gradient()
                if img_tensor.grad is not None:
                    img_tensor.grad.zero_()

                retain_graph = rank < num_selected - 1
                score_tensor.backward(retain_graph=retain_graph)
                grads = ensure_feature_gradients(score_tensor, feature_hook)
                if grads is None:
                    saliency = compute_input_saliency(score_tensor, img_tensor, pad_info, original_rgb.shape[:2])
                    if saliency is None:
                        LOGGER.warning("Failed to capture gradients for %s detection %d; skipping.", img_path.name, rank)
                        continue
                    cam = saliency
                    LOGGER.warning(
                        "Falling back to input-saliency for %s detection %d (no feature gradients).",
                        img_path.name,
                        rank,
                    )
                else:
                    activation = feature_hook.activation
                    if activation is None:
                        raise RuntimeError("Activation buffer cleared before backward. Re-run with different --layer.")
                    cam = compute_cam(activation, grads, img_tensor.shape[2:])
                    cam = crop_and_resize_cam(cam, pad_info, original_rgb.shape[:2])

                if not np.any(cam):
                    LOGGER.warning(
                        "Grad-CAM map for %s is all zeros; try a deeper layer or lower --conf threshold.",
                        img_path.name,
                    )

                overlay = overlay_heatmap(cam, original_rgb, args.alpha)

                anchor_box = boxes[0, anchor_idx].detach().cpu().numpy()
                box_xyxy = xywh_to_xyxy(anchor_box)
                x1, y1, x2, y2 = rescale_box_to_original(box_xyxy, pad_info, original_rgb.shape[:2])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
                per_detection_outputs.append(
                    {
                        "cam": cam,
                        "box": (x1, y1, x2, y2),
                        "score": float(score_tensor.detach().item()),
                        "index": int(anchor_idx),
                    }
                )

                suffix = f"_{rank}" if num_selected > 1 else ""
                overlay_path = save_dir / f"{img_path.stem}_gradcam{suffix}.jpg"
                heatmap_path = save_dir / f"{img_path.stem}_heatmap{suffix}.npy"
                cv2.imwrite(str(overlay_path), overlay)
                np.save(heatmap_path, cam)

            feature_hook.clear()
            if device.type != "cpu":
                torch.cuda.empty_cache()
            if per_detection_outputs:
                combined_cam = np.maximum.reduce([entry["cam"] for entry in per_detection_outputs])
                combined_overlay = overlay_heatmap(combined_cam, original_rgb, args.alpha)
                for entry in per_detection_outputs:
                    x1, y1, x2, y2 = entry["box"]
                    label = f"{entry['score']:.2f}"
                    cv2.rectangle(combined_overlay, (x1, y1), (x2, y2), (0, 215, 255), 2)
                    cv2.putText(
                        combined_overlay,
                        label,
                        (x1, max(y1 - 5, 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 215, 255),
                        1,
                        cv2.LINE_AA,
                    )
                combined_overlay_path = save_dir / f"{img_path.stem}_gradcam_combined.jpg"
                combined_heatmap_path = save_dir / f"{img_path.stem}_heatmap_combined.npy"
                cv2.imwrite(str(combined_overlay_path), combined_overlay)
                np.save(combined_heatmap_path, combined_cam)
            LOGGER.info(f"Saved Grad-CAM visualizations for {img_path.name} to {save_dir}")

    finally:
        feature_hook.detach()

def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=np.float32)


def rescale_box_to_original(box_xyxy: np.ndarray, pad: Dict[str, float], original_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    r = pad["ratio"]
    left, top = pad["left"], pad["top"]
    x1, y1, x2, y2 = box_xyxy
    x1 = (x1 - left) / r
    x2 = (x2 - left) / r
    y1 = (y1 - top) / r
    y2 = (y2 - top) / r
    h, w = original_shape
    x1, y1 = int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1))
    x2, y2 = int(np.clip(x2, 0, w - 1)), int(np.clip(y2, 0, h - 1))
    return x1, y1, x2, y2


if __name__ == "__main__":
    main()
