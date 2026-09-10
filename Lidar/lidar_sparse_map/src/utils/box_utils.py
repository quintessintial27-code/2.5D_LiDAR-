"""
box_utils.py
─────────────
Utilities for 3D bounding box operations.

Includes:
  - Gaussian radius calculation (for CenterPoint heatmap generation)
  - IoU calculation (BEV and 3D)
  - Box encoding/decoding
  - NMS
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import torch


def get_gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """
    Compute Gaussian radius for a bounding box of given size.
    Formula from CornerNet paper.

    Args:
        height, width: object extent in BEV grid cells
        min_overlap  : minimum IoU overlap

    Returns:
        radius (float) — use int(radius) in practice
    """
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)

    return min(r1, r2, r3)


def bev_iou(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """
    Compute BEV IoU between two sets of boxes.

    Args:
        boxes_a : [M, 5]  (x, y, l, w, yaw) — LiDAR frame, BEV
        boxes_b : [N, 5]

    Returns:
        iou     : [M, N]  pairwise IoU
    """
    # Approximate BEV IoU using axis-aligned bounding boxes
    # (fast, sufficient for NMS)
    M = boxes_a.shape[0]
    N = boxes_b.shape[0]
    iou = np.zeros((M, N), dtype=np.float32)

    for i in range(M):
        ax, ay, al, aw = boxes_a[i, 0], boxes_a[i, 1], boxes_a[i, 2], boxes_a[i, 3]
        for j in range(N):
            bx, by, bl, bw = boxes_b[j, 0], boxes_b[j, 1], boxes_b[j, 2], boxes_b[j, 3]

            # Axis-aligned bounding box of each rotated box (approximation)
            a_x1, a_y1 = ax - al / 2, ay - aw / 2
            a_x2, a_y2 = ax + al / 2, ay + aw / 2
            b_x1, b_y1 = bx - bl / 2, by - bw / 2
            b_x2, b_y2 = bx + bl / 2, by + bw / 2

            inter_x1 = max(a_x1, b_x1)
            inter_y1 = max(a_y1, b_y1)
            inter_x2 = min(a_x2, b_x2)
            inter_y2 = min(a_y2, b_y2)

            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                iou[i, j] = 0.0
            else:
                inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                union = al * aw + bl * bw - inter
                iou[i, j] = inter / union if union > 0 else 0.0

    return iou


def nms_bev(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """
    BEV Non-Maximum Suppression.

    Args:
        boxes  : [N, 7]  (x,y,z,l,w,h,yaw)
        scores : [N]
        iou_threshold: suppress if IoU > threshold

    Returns:
        keep: indices of kept boxes
    """
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        rest = order[1:]
        iou = bev_iou(
            boxes[i:i+1, [0,1,3,4]],    # x,y,l,w
            boxes[rest[:, None].squeeze() if rest.ndim > 1 else rest, :][:, [0,1,3,4]]
        ).squeeze()

        order = rest[iou <= iou_threshold]

    return np.array(keep, dtype=np.int64)


def boxes_to_heatmap(
    boxes: np.ndarray,        # [N, 8]  (x,y,z,l,w,h,yaw,cls)
    heatmap_shape: Tuple[int, int],
    pc_range: Tuple,
    voxel_size: float,
) -> np.ndarray:
    """
    Convert 3D boxes to Gaussian heatmap [H, W].
    Returns the heatmap (no class dimension — caller handles per-class).
    """
    H, W = heatmap_shape
    heatmap = np.zeros((H, W), dtype=np.float32)
    x_min, y_min = pc_range[0], pc_range[1]

    for box in boxes:
        x, y, l, w = box[0], box[1], box[3], box[4]
        cx = int((x - x_min) / voxel_size)
        cy = int((y - y_min) / voxel_size)
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        radius = max(0, int(get_gaussian_radius(l / voxel_size, w / voxel_size)))
        # Simple Gaussian around centre
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < W and 0 <= ny < H:
                    val = np.exp(-(dx**2 + dy**2) / (2 * (radius + 1e-4) ** 2 / 9))
                    heatmap[ny, nx] = max(heatmap[ny, nx], val)

    return heatmap


def encode_boxes(
    boxes: np.ndarray,
    pc_range: Tuple,
    voxel_size: float,
    bev_h: int,
    bev_w: int,
) -> np.ndarray:
    """
    Encode GT boxes as regression targets in BEV grid space.

    Returns [N, 10]: dx, dy, z, log_l, log_w, log_h, sin_yaw, cos_yaw, vx, vy
    """
    x_min, y_min, z_min = pc_range[0], pc_range[1], pc_range[2]
    out = np.zeros((boxes.shape[0], 10), dtype=np.float32)

    for i, box in enumerate(boxes):
        x, y, z, l, w, h, yaw = box[:7]
        cx = (x - x_min) / voxel_size
        cy = (y - y_min) / voxel_size
        dx = cx - int(cx)
        dy = cy - int(cy)
        out[i] = [
            dx, dy,
            z - z_min,
            np.log(max(l, 0.01)), np.log(max(w, 0.01)), np.log(max(h, 0.01)),
            np.sin(yaw), np.cos(yaw),
            0.0, 0.0,   # velocity unknown from single-frame KITTI
        ]

    return out
