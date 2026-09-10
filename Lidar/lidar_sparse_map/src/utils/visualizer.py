"""
visualizer.py
─────────────
Visualisation utilities for point clouds, BEV maps, height maps,
and detection results. Works CPU-only with matplotlib.

open3d is used only for the optional 3D point cloud viewer.
It has no Python 3.13 wheel yet — the 3D view is silently skipped
if open3d is not installed. All other features work without it.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
# open3d is imported lazily inside show_pointcloud_3d() — no top-level import


class LidarVisualizer:
    """All-in-one visualiser for LiDAR perception outputs."""

    # Class colours (BGR for open3d, RGB for matplotlib)
    CLASS_COLORS = {
        0: (0.2, 0.6, 1.0),    # Car — blue
        1: (1.0, 0.4, 0.1),    # Pedestrian — orange
        2: (0.2, 0.9, 0.2),    # Cyclist — green
    }
    CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

    # ─── BEV + Detection Visualisation ───────────────────────────────────────

    @staticmethod
    def plot_bev_detections(
        height_map: np.ndarray,       # [H, W] or [1, H, W]
        detections: Optional[Dict],   # {'boxes':[K,7], 'scores':[K], 'labels':[K]}
        gt_boxes: Optional[np.ndarray] = None,   # [G, 8]
        pc_range: Tuple = (-40., -40., -3., 40., 40., 3.),
        voxel_size: float = 0.20,
        title: str = "BEV Map",
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot bird's-eye-view height map with detection bounding boxes.

        Height map rendered as grayscale; predicted boxes in solid color;
        GT boxes in dashed white.
        """
        if height_map.ndim == 3:
            height_map = height_map[0]   # [1,H,W] → [H,W]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Height map as grayscale background
        ax.imshow(
            height_map,
            cmap="inferno",
            origin="lower",
            extent=[pc_range[0], pc_range[3], pc_range[1], pc_range[4]],
            vmin=0, vmax=height_map.max() + 0.1,
        )
        ax.set_facecolor("black")

        # Draw predicted boxes
        if detections is not None and detections["boxes"].shape[0] > 0:
            boxes = detections["boxes"].numpy() if isinstance(detections["boxes"], torch.Tensor) else detections["boxes"]
            scores = detections["scores"].numpy() if isinstance(detections["scores"], torch.Tensor) else detections["scores"]
            labels = detections["labels"].numpy() if isinstance(detections["labels"], torch.Tensor) else detections["labels"]

            for box, score, label in zip(boxes, scores, labels):
                x, y, z, l, w, h, yaw = box
                color = LidarVisualizer.CLASS_COLORS.get(int(label), (1, 1, 1))
                _draw_box_bev(ax, x, y, l, w, yaw, color, score, solid=True)

        # Draw GT boxes (dashed white)
        if gt_boxes is not None and gt_boxes.shape[0] > 0:
            for box in gt_boxes:
                x, y, z, l, w, h, yaw = box[:7]
                _draw_box_bev(ax, x, y, l, w, yaw, (1, 1, 1), label_text=None, solid=False)

        # Legend
        for cls_id, name in LidarVisualizer.CLASS_NAMES.items():
            ax.plot([], [], color=LidarVisualizer.CLASS_COLORS[cls_id],
                    linewidth=2, label=name)
        ax.plot([], [], color="white", linewidth=1, linestyle="--", label="GT")
        ax.legend(loc="upper right", framealpha=0.5)

        ax.set_title(title, color="white", fontsize=14)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("white")
        fig.patch.set_facecolor("#1a1a2e")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # ─── Adaptive Voxel Grid Visualisation ────────────────────────────────────

    @staticmethod
    def plot_adaptive_grid(
        pts: np.ndarray,
        fine_zone: np.ndarray,
        pc_range: Tuple = (-40., -40., -3., 40., 40., 3.),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Visualise which points are in the fine-resolution zone vs coarse zone.
        Red = fine (dense/dynamic), blue = coarse (sparse/static).
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor("#0d0d1a")
        ax.set_facecolor("#0d0d1a")

        coarse_pts = pts[~fine_zone]
        fine_pts = pts[fine_zone]

        if coarse_pts.shape[0] > 0:
            ax.scatter(coarse_pts[:, 0], coarse_pts[:, 1],
                       c="#4a90d9", s=0.3, alpha=0.4, label="Coarse zone")
        if fine_pts.shape[0] > 0:
            ax.scatter(fine_pts[:, 0], fine_pts[:, 1],
                       c="#ff6b35", s=0.8, alpha=0.8, label="Fine zone (dynamic/dense)")

        ax.set_xlim(pc_range[0], pc_range[3])
        ax.set_ylim(pc_range[1], pc_range[4])
        ax.set_title("Adaptive Voxel Resolution Map", color="white", fontsize=14)
        ax.set_xlabel("X (m)", color="white")
        ax.set_ylabel("Y (m)", color="white")
        ax.tick_params(colors="white")
        ax.legend(framealpha=0.3, labelcolor="white")
        ax.spines[:].set_color("#333355")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # ─── Training Curve ───────────────────────────────────────────────────────

    @staticmethod
    def plot_training_curves(
        train_losses: List[float],
        val_losses: List[float],
        val_maps: List[float],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Plot loss and mAP curves over epochs."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#1a1a2e")

        colors = ["#00d4ff", "#ff6b35"]
        for ax in axes:
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#333355")

        axes[0].plot(train_losses, color=colors[0], linewidth=1.5, label="Train loss")
        axes[0].plot(val_losses, color=colors[1], linewidth=1.5, label="Val loss")
        axes[0].set_title("Loss", color="white")
        axes[0].set_xlabel("Epoch", color="white")
        axes[0].legend(labelcolor="white", framealpha=0.3)

        axes[1].plot(val_maps, color="#7fff7f", linewidth=2, label="Val mAP")
        axes[1].set_title("Validation mAP", color="white")
        axes[1].set_xlabel("Epoch", color="white")
        axes[1].legend(labelcolor="white", framealpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # ─── 3D point cloud (open3d) ──────────────────────────────────────────────

    @staticmethod
    def show_pointcloud_3d(
        pts: np.ndarray,
        boxes: Optional[np.ndarray] = None,
    ) -> None:
        """
        Display 3D point cloud with optional bounding boxes.
        Requires open3d to be installed.
        """
        try:
            import open3d as o3d
        except ImportError:
            print("[Visualizer] open3d not installed — skipping 3D view")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3])

        # Colour by height
        z = pts[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
        colors = plt.cm.plasma(z_norm)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        geometries = [pcd]

        if boxes is not None:
            for box in boxes:
                x, y, z, l, w, h, yaw = box[:7]
                cls = int(box[7]) if box.shape[0] > 7 else 0
                center = np.array([x, y, z + h / 2])
                extent = np.array([l, w, h])
                R = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, yaw])
                obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
                col = LidarVisualizer.CLASS_COLORS.get(cls, (1, 1, 1))
                obb.color = col
                geometries.append(obb)

        o3d.visualization.draw_geometries(
            geometries,
            window_name="LiDAR 3D View",
            width=1024, height=768,
        )


# ─── Helper ───────────────────────────────────────────────────────────────────

def _draw_box_bev(
    ax,
    x: float, y: float,
    l: float, w: float,
    yaw: float,
    color: tuple,
    score: Optional[float] = None,
    label_text: Optional[str] = None,
    solid: bool = True,
) -> None:
    """Draw a rotated 2D box on a matplotlib axis."""
    corners = np.array([
        [-l/2, -w/2],
        [ l/2, -w/2],
        [ l/2,  w/2],
        [-l/2,  w/2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners = corners @ R.T + np.array([x, y])

    poly = plt.Polygon(
        corners,
        closed=True,
        fill=False,
        edgecolor=color,
        linewidth=1.5 if solid else 0.8,
        linestyle="-" if solid else "--",
    )
    ax.add_patch(poly)

    if score is not None:
        ax.text(x, y, f"{score:.2f}", color=color, fontsize=6, ha="center")
