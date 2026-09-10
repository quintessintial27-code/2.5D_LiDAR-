"""
metrics.py
───────────
Evaluation metrics for 3D object detection and height map quality.

Detection:
  - Precision, Recall, F1 per class
  - mAP (mean Average Precision) at IoU 0.5

Height Map:
  - RMSE (Root Mean Squared Error)
  - MAE  (Mean Absolute Error)
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
import torch

from .box_utils import bev_iou


# ─── Detection Metrics ────────────────────────────────────────────────────────

class DetectionMetrics:
    """
    Accumulates predictions and GT boxes across batches,
    then computes AP per class and mAP.
    """

    def __init__(self, num_classes: int = 3, iou_threshold: float = 0.5) -> None:
        self.num_classes = num_classes
        self.iou_thresh = iou_threshold
        self.reset()

    def reset(self) -> None:
        self.pred_boxes: List[np.ndarray] = []
        self.pred_scores: List[np.ndarray] = []
        self.pred_labels: List[np.ndarray] = []
        self.gt_boxes: List[np.ndarray] = []
        self.gt_labels: List[np.ndarray] = []

    def update(
        self,
        pred_boxes: np.ndarray,    # [K, 7]
        pred_scores: np.ndarray,   # [K]
        pred_labels: np.ndarray,   # [K]
        gt_boxes: np.ndarray,      # [G, 7]
        gt_labels: np.ndarray,     # [G]
    ) -> None:
        self.pred_boxes.append(pred_boxes)
        self.pred_scores.append(pred_scores)
        self.pred_labels.append(pred_labels)
        self.gt_boxes.append(gt_boxes)
        self.gt_labels.append(gt_labels)

    def compute(self) -> Dict[str, float]:
        """Compute AP per class and mAP."""
        aps = {}
        for cls in range(self.num_classes):
            ap = self._compute_ap_for_class(cls)
            aps[f"AP_cls{cls}"] = ap

        mAP = float(np.mean(list(aps.values())))
        aps["mAP"] = mAP
        return aps

    def _compute_ap_for_class(self, cls: int) -> float:
        """Compute AP for a single class using all accumulated data."""
        all_pred_scores = []
        all_tp = []
        n_gt_total = 0

        for i in range(len(self.pred_boxes)):
            pred_mask = self.pred_labels[i] == cls
            gt_mask = self.gt_labels[i] == cls

            pred_b = self.pred_boxes[i][pred_mask]
            pred_s = self.pred_scores[i][pred_mask]
            gt_b = self.gt_boxes[i][gt_mask]

            n_gt_total += gt_b.shape[0]

            if pred_b.shape[0] == 0:
                continue

            # Sort by score descending
            order = np.argsort(-pred_s)
            pred_b = pred_b[order]
            pred_s = pred_s[order]

            matched = np.zeros(gt_b.shape[0], dtype=bool)
            tp_flags = np.zeros(pred_b.shape[0], dtype=bool)

            for pi in range(pred_b.shape[0]):
                if gt_b.shape[0] == 0:
                    break
                iou = bev_iou(
                    pred_b[pi:pi+1, [0,1,3,4]],
                    gt_b[:, [0,1,3,4]],
                ).squeeze(0)
                best_gt = int(np.argmax(iou))
                if iou[best_gt] >= self.iou_thresh and not matched[best_gt]:
                    tp_flags[pi] = True
                    matched[best_gt] = True

            all_pred_scores.extend(pred_s.tolist())
            all_tp.extend(tp_flags.tolist())

        if not all_pred_scores or n_gt_total == 0:
            return 0.0

        # Sort globally by score
        order = np.argsort(-np.array(all_pred_scores))
        tp_arr = np.array(all_tp)[order]

        # Precision-Recall curve
        tp_cumsum = np.cumsum(tp_arr)
        fp_cumsum = np.cumsum(~tp_arr)
        recalls = tp_cumsum / n_gt_total
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

        # Compute AP via trapezoidal integration
        return float(np.trapz(precisions, recalls))


# ─── Height Map Metrics ───────────────────────────────────────────────────────

def height_map_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute RMSE and MAE for height map prediction.

    Args:
        pred  : [B, 1, H, W]  predicted heights
        target: [B, 1, H, W]  ground truth heights

    Returns:
        dict with 'rmse', 'mae', 'coverage'
    """
    # Normalise shapes: accept [B,1,H,W] or [B,H,W]
    if pred.dim() == 4:
        pred = pred.squeeze(1)    # [B, H, W]
    if target.dim() == 4:
        target = target.squeeze(1)  # [B, H, W]

    occupied = target > 0.0
    if not occupied.any():
        return {"rmse": 0.0, "mae": 0.0, "coverage": 0.0}

    diff = (pred - target)[occupied]
    rmse = float(torch.sqrt((diff ** 2).mean()))
    mae = float(diff.abs().mean())

    # Coverage: fraction of occupied cells predicted above 0
    coverage = float((pred[occupied] > 0.0).float().mean())

    return {"rmse": rmse, "mae": mae, "coverage": coverage}


# ─── Voxelizer Metrics ────────────────────────────────────────────────────────

def voxelizer_stats(sparse_tensor) -> Dict[str, float]:
    """Print stats about a SparseTensor (for debugging voxelizer output)."""
    D, H, W = sparse_tensor.spatial_shape
    total = sparse_tensor.batch_size * D * H * W
    density = sparse_tensor.num_voxels / total
    return {
        "num_voxels": sparse_tensor.num_voxels,
        "grid_volume": total,
        "density_pct": density * 100,
        "C": sparse_tensor.num_channels,
    }
