"""
detection_head.py
─────────────────
Anchor-Free 3D Object Detection Head (CenterPoint-style).

From the dense BEV feature map, predicts per-object:
  - Heatmap (Gaussian-blurred class-center map) — [B, n_classes, H, W]
  - Offsets (sub-voxel x, y refinement)         — [B, 2, H, W]
  - Height  (center z)                           — [B, 1, H, W]
  - Dimensions (length, width, height)           — [B, 3, H, W]
  - Yaw (sin, cos encoding)                      — [B, 2, H, W]
  - Velocity (vx, vy from frame diff)            — [B, 2, H, W]

Total regression channels = 2 + 1 + 3 + 2 + 2 = 10

Loss:
  - Heatmap: Modified Focal Loss (α=2, β=4) from CenterPoint paper
  - Regression: L1 on positive cells only

Decoding:
  - Peak detection on heatmap (local max, threshold filter)
  - Gather regression values at peak locations
  - Convert to 7-DOF box: [x, y, z, l, w, h, yaw]
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """
    Anchor-free CenterPoint-style detection head.

    Args:
        in_channels  : BEV feature channels (from BEVCollapse)
        num_classes  : number of object classes
        hidden_ch    : intermediate channels for head convolutions
        reg_channels : total regression outputs per location (default 10)
    """

    # Regression channel layout
    REG_OFFSET = slice(0, 2)   # (dx, dy) sub-cell offset
    REG_Z      = slice(2, 3)   # z centre
    REG_DIM    = slice(3, 6)   # (l, w, h)
    REG_YAW    = slice(6, 8)   # (sin θ, cos θ)
    REG_VEL    = slice(8, 10)  # (vx, vy)

    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = 3,
        hidden_ch: int = 64,
        reg_channels: int = 10,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_channels = reg_channels

        # Shared feature trunk
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
        )

        # Heatmap sub-head (one channel per class)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, num_classes, kernel_size=1),
        )

        # Regression sub-head
        self.reg_head = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, reg_channels, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Bias init for heatmap: prior prob = 0.01 (avoid initial loss explosion)
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)   # log(0.01/0.99)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.bias is None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        bev_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            bev_feat: [B, C, H, W]

        Returns:
            {
              'heatmap'  : [B, n_cls, H, W]  — sigmoid probabilities
              'reg'      : [B, 10,   H, W]  — raw regression outputs
            }
        """
        x = self.trunk(bev_feat)
        heatmap = torch.sigmoid(self.heatmap_head(x))  # [B, n_cls, H, W]
        reg = self.reg_head(x)                          # [B, 10, H, W]
        return {"heatmap": heatmap, "reg": reg}

    # ─── Loss ─────────────────────────────────────────────────────────────────

    @staticmethod
    def focal_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 2.0,
        beta: float = 4.0,
    ) -> torch.Tensor:
        """
        Modified Focal Loss for Gaussian heatmaps (CornerNet / CenterPoint).
        Handles both positive (centre) and negative (surroundings) cells.

        Args:
            pred  : [B, C, H, W]  sigmoid probabilities
            target: [B, C, H, W]  Gaussian-blurred ground-truth heatmap ∈ [0,1]

        Returns:
            scalar loss
        """
        pos_mask = (target == 1.0).float()
        neg_mask = (target < 1.0).float()

        pos_loss = (
            -torch.log(pred.clamp(min=1e-6))
            * (1 - pred) ** alpha
            * pos_mask
        )
        neg_loss = (
            -torch.log((1 - pred).clamp(min=1e-6))
            * pred ** alpha
            * (1 - target) ** beta
            * neg_mask
        )

        n_pos = pos_mask.sum()
        loss = (pos_loss + neg_loss).sum()
        return loss / n_pos.clamp(min=1.0)

    @staticmethod
    def regression_loss(
        pred_reg: torch.Tensor,
        target_reg: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        L1 regression loss on positive (object-centre) cells only.

        Args:
            pred_reg  : [B, 10, H, W]
            target_reg: [B, 10, H, W]
            pos_mask  : [B, H, W]  bool — True at object centres

        Returns:
            scalar loss
        """
        if pos_mask.sum() == 0:
            return pred_reg.sum() * 0.0

        # Normalise: ensure pos_mask is [B, H, W]
        if pos_mask.dim() == 2:           # [H, W] → add batch dim
            pos_mask = pos_mask.unsqueeze(0)
        # Now pos_mask is [B, H, W] → expand to [B, 10, H, W]
        mask = pos_mask.unsqueeze(1).expand_as(pred_reg)
        return F.l1_loss(pred_reg[mask], target_reg[mask])

    def loss(
        self,
        predictions: Dict[str, torch.Tensor],
        gt_heatmaps: torch.Tensor,
        gt_regs: torch.Tensor,
        gt_pos_masks: torch.Tensor,
        heatmap_weight: float = 1.0,
        reg_weight: float = 2.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined detection loss.

        Args:
            predictions : output of forward()
            gt_heatmaps : [B, n_cls, H, W]  Gaussian heatmaps
            gt_regs     : [B, 10, H, W]     regression targets
            gt_pos_masks: [B, H, W]         bool, True at centres

        Returns:
            dict with 'total', 'heatmap', 'regression' losses
        """
        l_heat = self.focal_loss(predictions["heatmap"], gt_heatmaps)
        l_reg = self.regression_loss(predictions["reg"], gt_regs, gt_pos_masks)
        total = heatmap_weight * l_heat + reg_weight * l_reg
        return {"total": total, "heatmap": l_heat, "regression": l_reg}

    # ─── Post-process / Decode ────────────────────────────────────────────────

    def decode(
        self,
        predictions: Dict[str, torch.Tensor],
        pc_range: Tuple[float, ...],
        voxel_size: float,
        score_threshold: float = 0.3,
        max_detections: int = 50,
        nms_iou_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Decode raw predictions → 3D bounding boxes.

        Returns:
            List (one per batch item) of dicts:
              'boxes'  : [K, 7]  (x, y, z, l, w, h, yaw)
              'scores' : [K]
              'labels' : [K]
        """
        heatmap = predictions["heatmap"]   # [B, n_cls, H, W]
        reg = predictions["reg"]           # [B, 10, H, W]
        B, _, H, W = heatmap.shape
        x_min, y_min, z_min = pc_range[0], pc_range[1], pc_range[2]

        results = []
        for b in range(B):
            boxes_list, scores_list, labels_list = [], [], []

            for cls in range(self.num_classes):
                heat = heatmap[b, cls]    # [H, W]

                # Local-maximum peak detection (3×3 window)
                heat_max = F.max_pool2d(
                    heat.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1
                ).squeeze()
                peak_mask = (heat == heat_max) & (heat > score_threshold)
                ys, xs = peak_mask.nonzero(as_tuple=True)

                if ys.numel() == 0:
                    continue

                scores = heat[ys, xs]

                # Gather regression values at peak locations
                r = reg[b, :, ys, xs]     # [10, K]

                # Decode absolute coordinates
                xs_f = xs.float()
                ys_f = ys.float()
                cx = (xs_f + r[0]) * voxel_size + x_min
                cy = (ys_f + r[1]) * voxel_size + y_min
                cz = r[2] + z_min
                L  = r[3].exp()
                W_ = r[4].exp()
                H_ = r[5].exp()
                yaw = torch.atan2(r[6], r[7])

                boxes = torch.stack([cx, cy, cz, L, W_, H_, yaw], dim=1)  # [K, 7]

                boxes_list.append(boxes)
                scores_list.append(scores)
                labels_list.append(torch.full((boxes.shape[0],), cls, dtype=torch.long))

            if boxes_list:
                all_boxes = torch.cat(boxes_list)
                all_scores = torch.cat(scores_list)
                all_labels = torch.cat(labels_list)

                # Score-based top-K (simple NMS alternative for CPU speed)
                if all_scores.numel() > max_detections:
                    topk = all_scores.topk(max_detections)[1]
                    all_boxes = all_boxes[topk]
                    all_scores = all_scores[topk]
                    all_labels = all_labels[topk]

                results.append({
                    "boxes": all_boxes,
                    "scores": all_scores,
                    "labels": all_labels,
                })
            else:
                results.append({
                    "boxes": torch.zeros(0, 7),
                    "scores": torch.zeros(0),
                    "labels": torch.zeros(0, dtype=torch.long),
                })

        return results
