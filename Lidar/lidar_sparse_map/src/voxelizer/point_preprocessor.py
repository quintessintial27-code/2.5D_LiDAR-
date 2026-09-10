"""
point_preprocessor.py
─────────────────────
Raw point cloud preprocessing pipeline before voxelization.

Steps (in order):
  1. Range clip         – drop points outside sensor range
  2. Ground removal     – remove flat ground plane (height threshold method)
  3. Intensity norm     – normalise intensity to [0, 1]
  4. Statistical filter – remove outlier points (too isolated)
  5. Offset encoding    – subtract centroid for translation invariance

All operations are vectorised NumPy — no PyTorch or CUDA needed.
"""

from __future__ import annotations
from typing import Tuple, Optional

import numpy as np


class PointPreprocessor:
    """
    Preprocesses raw LiDAR point clouds before adaptive voxelization.

    Usage:
        preprocessor = PointPreprocessor()
        clean_pts = preprocessor(raw_pts)
    """

    def __init__(
        self,
        # Range limits [metres]
        x_range: Tuple[float, float] = (-40.0, 40.0),
        y_range: Tuple[float, float] = (-40.0, 40.0),
        z_range: Tuple[float, float] = (-3.0, 3.0),
        # Ground removal
        remove_ground: bool = True,
        ground_z_threshold: float = -1.5,   # pts below this = ground
        # Intensity normalisation
        normalise_intensity: bool = True,
        intensity_max: float = 255.0,
        # Statistical outlier removal
        outlier_removal: bool = True,
        outlier_nb_points: int = 5,
        outlier_radius: float = 0.5,
        # Minimum points to keep (safety check)
        min_points: int = 50,
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.remove_ground = remove_ground
        self.ground_z_thresh = ground_z_threshold
        self.norm_intensity = normalise_intensity
        self.int_max = intensity_max
        self.outlier_removal = outlier_removal
        self.outlier_nb = outlier_nb_points
        self.outlier_radius = outlier_radius
        self.min_points = min_points

    # ─── Main entry point ────────────────────────────────────────────────────

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Args:
            points: [N, 4+]  columns = (x, y, z, intensity, ...)
        Returns:
            cleaned points [M, 4]  M ≤ N
        """
        if points.shape[0] < self.min_points:
            return points[:, :4] if points.shape[1] >= 4 else np.zeros((0, 4))

        pts = points[:, :4].copy().astype(np.float32)   # keep first 4 cols

        pts = self._range_clip(pts)
        if pts.shape[0] < self.min_points:
            return pts

        if self.remove_ground:
            pts = self._remove_ground(pts)
        if pts.shape[0] < self.min_points:
            return pts

        if self.norm_intensity:
            pts = self._normalise_intensity(pts)

        if self.outlier_removal and pts.shape[0] > self.outlier_nb * 2:
            pts = self._remove_outliers(pts)

        return pts

    # ─── Step 1: Range clip ──────────────────────────────────────────────────

    def _range_clip(self, pts: np.ndarray) -> np.ndarray:
        mask = (
            (pts[:, 0] >= self.x_range[0]) & (pts[:, 0] < self.x_range[1]) &
            (pts[:, 1] >= self.y_range[0]) & (pts[:, 1] < self.y_range[1]) &
            (pts[:, 2] >= self.z_range[0]) & (pts[:, 2] < self.z_range[1])
        )
        return pts[mask]

    # ─── Step 2: Ground removal ──────────────────────────────────────────────

    def _remove_ground(self, pts: np.ndarray) -> np.ndarray:
        """
        Simple height-threshold ground removal.
        Points below ground_z_threshold are classified as ground and removed.
        This is fast (O(N)) and suitable for flat-ground environments (KITTI).
        For uneven terrain, replace with RANSAC plane fitting.
        """
        return pts[pts[:, 2] > self.ground_z_thresh]

    # ─── Step 3: Intensity normalisation ─────────────────────────────────────

    def _normalise_intensity(self, pts: np.ndarray) -> np.ndarray:
        pts[:, 3] = np.clip(pts[:, 3] / self.int_max, 0.0, 1.0)
        return pts

    # ─── Step 4: Statistical outlier removal ─────────────────────────────────

    def _remove_outliers(self, pts: np.ndarray) -> np.ndarray:
        """
        Radius-based outlier removal:
        A point is an outlier if fewer than `outlier_nb` other points
        are within `outlier_radius` metres of it.

        Vectorised via broadcasting — O(N²) but fast for N < 20k.
        For very large point clouds, use a KD-tree (sklearn).
        """
        xyz = pts[:, :3]   # [N, 3]
        N = xyz.shape[0]

        if N > 10_000:
            # Use sklearn for large clouds (faster than O(N²))
            try:
                from sklearn.neighbors import BallTree
                tree = BallTree(xyz)
                counts = tree.query_radius(xyz, r=self.outlier_radius, count_only=True)
                mask = counts >= self.outlier_nb
            except ImportError:
                mask = np.ones(N, dtype=bool)
        else:
            # Pairwise distance (fine for N < 10k)
            diff = xyz[:, None, :] - xyz[None, :, :]   # [N, N, 3]
            dist_sq = (diff ** 2).sum(axis=2)            # [N, N]
            counts = (dist_sq < self.outlier_radius ** 2).sum(axis=1) - 1  # exclude self
            mask = counts >= self.outlier_nb

        return pts[mask]

    # ─── Utilities ───────────────────────────────────────────────────────────

    @staticmethod
    def load_kitti_bin(path: str) -> np.ndarray:
        """
        Load a KITTI .bin point cloud file.
        Returns [N, 4] array (x, y, z, intensity).
        """
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return pts

    def stats(self, pts: np.ndarray) -> dict:
        """Return basic stats of a processed point cloud."""
        return {
            "n_points": pts.shape[0],
            "x_range": (float(pts[:, 0].min()), float(pts[:, 0].max())),
            "y_range": (float(pts[:, 1].min()), float(pts[:, 1].max())),
            "z_range": (float(pts[:, 2].min()), float(pts[:, 2].max())),
            "intensity_range": (float(pts[:, 3].min()), float(pts[:, 3].max())),
        }

    def __repr__(self) -> str:
        return (
            f"PointPreprocessor(\n"
            f"  range=x{self.x_range} y{self.y_range} z{self.z_range}\n"
            f"  ground_removal={self.remove_ground} (z<{self.ground_z_thresh})\n"
            f"  outlier_removal={self.outlier_removal}"
            f" (nb={self.outlier_nb}, r={self.outlier_radius})\n"
            f")"
        )
