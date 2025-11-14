"""Generate Grad-CAM overlays for Ultralytics YOLO detectors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import select_device

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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
    parser.add_argument("--output", type=str, default="runs/gradcam", help="Directory to save Grad-CAM outputs")
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
        help="Number of top detections per image to visualize (backprop run per detection)",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Print indexed list of model.model modules and exit",
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
    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}

    def forward_hook(_, __, output):
        activations["value"] = output

    def backward_hook(_, grad_input, grad_output):  # noqa: ARG001
        gradients["value"] = grad_output[0]

    fwd_handle = layer.register_forward_hook(forward_hook)
    bwd_handle = layer.register_full_backward_hook(backward_hook)

    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    stride = int(det_model.stride.max().item()) if hasattr(det_model, "stride") else 32
    try:
        for img_path in images:
            img_tensor, original_rgb, pad_info = prepare_image(img_path, args.imgsz, stride, device)
            activations.clear()
            gradients.clear()

            det_model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                preds = det_model(img_tensor)
                preds = preds[0] if isinstance(preds, (tuple, list)) else preds

            pred_transposed = preds.transpose(1, 2)  # (B, anchors, 4+nc)
            scores = pred_transposed[..., 4:]
            boxes = pred_transposed[..., :4]
            class_scores = scores

            if args.cls is not None and (args.cls < 0 or args.cls >= class_scores.shape[-1]):
                raise ValueError(f"Class index {args.cls} outside valid range [0, {class_scores.shape[-1] - 1}].")

            if args.cls is not None:
                cls_scores = class_scores[0, :, args.cls]
            else:
                cls_scores = class_scores[0].max(dim=1).values

            conf_mask = cls_scores > args.conf
            if conf_mask.any():
                filtered_scores = cls_scores[conf_mask]
                filtered_idx = torch.arange(cls_scores.shape[0], device=device)[conf_mask]
                topk = min(args.topk, filtered_scores.numel())
                top_scores, top_indices = filtered_scores.topk(topk)
                selected_indices = filtered_idx[top_indices]
            else:
                top_scores, selected_indices = cls_scores.topk(min(args.topk, cls_scores.numel()))

            for idx, (score, anchor_idx) in enumerate(zip(top_scores, selected_indices)):
                det_model.zero_grad(set_to_none=True)
                if activations.get("value") is None:
                    raise RuntimeError("Forward hook did not capture activations. Check layer index.")

                retain_graph = idx < len(selected_indices) - 1
                score.backward(retain_graph=retain_graph)
                if gradients.get("value") is None:
                    raise RuntimeError("Backward hook did not capture gradients. Check layer index.")

                cam = compute_cam(activations["value"], gradients["value"], img_tensor.shape[2:])
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

                suffix = f"_{idx}" if len(selected_indices) > 1 else ""
                overlay_path = save_dir / f"{img_path.stem}_gradcam{suffix}.jpg"
                heatmap_path = save_dir / f"{img_path.stem}_heatmap{suffix}.npy"
                cv2.imwrite(str(overlay_path), overlay)
                np.save(heatmap_path, cam)

                gradients.clear()

            LOGGER.info(f"Saved Grad-CAM visualizations for {img_path.name} to {save_dir}")

    finally:
        fwd_handle.remove()
        bwd_handle.remove()

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
