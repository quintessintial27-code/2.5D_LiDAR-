"""
adaptive_voxelizer.py
─────────────────────
Adaptive Variable-Resolution Voxelizer — the core algorithmic novelty.

Algorithm:
  1. Divide the XY plane into a coarse grid of "analysis cells".
  2. Count point density in each cell.
  3. If density > threshold → assign FINE voxel size (0.10 m).
     Else → assign COARSE voxel size (0.20 m).
  4. Optionally use frame differencing to boost resolution around
     dynamically moving regions (large point displacement between frames).
  5. Assign each raw point to its voxel at the appropriate resolution.
  6. Compute per-voxel mean features.
  7. Return a SparseTensor of occupied voxels.

Memory note (for 8 GB RAM / Celeron):
  - COARSE (0.20 m) over 80×80×6 m range → 400×400×30 = 4.8M voxel slots
  - FINE   (0.10 m) over dense zone       → only ~5–10% of full volume
  - With max_voxels=15000 cap, we stay well within RAM limits.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict

import numpy as np
import torch

from ..models.sparse_tensor import SparseTensor


class AdaptiveVoxelizer:
    """
    Converts a raw point cloud into a SparseTensor using
    density-aware + motion-aware adaptive voxel resolution.
    """

    def __init__(
        self,
        pc_range: Tuple[float, float, float, float, float, float] = (
            -40.0, -40.0, -3.0, 40.0, 40.0, 3.0
        ),
        coarse_size: float = 0.20,
        fine_size: float = 0.10,
        density_threshold: int = 15,
        max_voxels: int = 15_000,
        max_pts_per_voxel: int = 20,
        # Frame differencing
        use_frame_diff: bool = True,
        diff_threshold: float = 0.15,
    ) -> None:
        self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax = pc_range
        self.coarse = coarse_size
        self.fine = fine_size
        self.density_thresh = density_threshold
        self.max_voxels = max_voxels
        self.max_pts = max_pts_per_voxel
        self.use_frame_diff = use_frame_diff
        self.diff_threshold = diff_threshold

        # Analysis cell = 1 m² grid for density estimation
        self.cell_size = 1.0
        self._prev_points: Optional[np.ndarray] = None   # for frame diff

        # Pre-compute full-range grid dimensions (coarse resolution)
        self._coarse_shape = self._compute_grid_shape(self.coarse)
        self._fine_shape = self._compute_grid_shape(self.fine)

    # ─── Public API ───────────────────────────────────────────────────────────

    def voxelize(
        self,
        points: np.ndarray,            # [N, 4+] x y z intensity ...
        batch_idx: int = 0,
        prev_points: Optional[np.ndarray] = None,
    ) -> SparseTensor:
        """
        Main entry point.

        Args:
            points    : Raw point cloud [N, 4+] (x, y, z, intensity)
            batch_idx : Batch index (for multi-sample batches)
            prev_points: Previous-frame point cloud for frame differencing

        Returns:
            SparseTensor ready for Sparse CNN
        """
        # 1. Filter points to configured range
        pts = self._range_filter(points)
        if pts.shape[0] == 0:
            return self._empty_tensor(batch_idx)

        # 2. Compute density map (1m grid over XY)
        density_map = self._compute_density(pts)

        # 3. Compute dynamic mask (frame differencing)
        dynamic_mask_xy = self._frame_diff_mask(pts, prev_points)

        # 4. Assign resolution per point
        fine_zone = self._assign_fine_zone(pts, density_map, dynamic_mask_xy)

        # 5. Voxelize with mixed resolution
        voxel_feats, voxel_indices = self._mixed_voxelize(pts, fine_zone, batch_idx)

        # 6. Update previous frame cache
        self._prev_points = pts.copy()

        # 7. Build SparseTensor
        #    Use coarse grid shape as the spatial shape (fine voxels map into this)
        return SparseTensor(
            features=torch.from_numpy(voxel_feats).float(),
            indices=torch.from_numpy(voxel_indices).to(torch.int32),
            spatial_shape=self._coarse_shape,
            batch_size=batch_idx + 1,
        )

    def reset(self) -> None:
        """Clear frame history (call at start of new sequence)."""
        self._prev_points = None

    # ─── Step 1: Range filter ─────────────────────────────────────────────────

    def _range_filter(self, pts: np.ndarray) -> np.ndarray:
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        mask = (
            (x >= self.xmin) & (x < self.xmax) &
            (y >= self.ymin) & (y < self.ymax) &
            (z >= self.zmin) & (z < self.zmax)
        )
        return pts[mask]

    # ─── Step 2: Density map ──────────────────────────────────────────────────

    def _compute_density(self, pts: np.ndarray) -> np.ndarray:
        """
        Returns a 2D density array [nx_cells, ny_cells] counting how many
        points fall in each 1 m² analysis cell.
        """
        nx = int((self.xmax - self.xmin) / self.cell_size)
        ny = int((self.ymax - self.ymin) / self.cell_size)
        cx = np.clip(((pts[:, 0] - self.xmin) / self.cell_size).astype(int), 0, nx - 1)
        cy = np.clip(((pts[:, 1] - self.ymin) / self.cell_size).astype(int), 0, ny - 1)
        density = np.zeros((nx, ny), dtype=np.int32)
        np.add.at(density, (cx, cy), 1)
        return density

    # ─── Step 3: Frame differencing ───────────────────────────────────────────

    def _frame_diff_mask(
        self,
        curr: np.ndarray,
        prev: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Returns a 2D bool array [nx_cells, ny_cells] flagging cells
        where point cloud changed significantly between frames.
        Uses a simple occupancy-grid difference: cells that were occupied
        before but not now (or vice versa) are flagged as dynamic.
        """
        if not self.use_frame_diff or prev is None:
            return None

        prev_f = self._range_filter(prev)
        nx = int((self.xmax - self.xmin) / self.cell_size)
        ny = int((self.ymax - self.ymin) / self.cell_size)

        def occupancy(p: np.ndarray) -> np.ndarray:
            cx = np.clip(((p[:, 0] - self.xmin) / self.cell_size).astype(int), 0, nx - 1)
            cy = np.clip(((p[:, 1] - self.ymin) / self.cell_size).astype(int), 0, ny - 1)
            occ = np.zeros((nx, ny), dtype=np.float32)
            np.add.at(occ, (cx, cy), 1)
            return occ

        occ_curr = occupancy(curr)
        occ_prev = occupancy(prev_f)

        # Normalise to density
        occ_curr_n = np.minimum(occ_curr / (self.density_thresh + 1), 1.0)
        occ_prev_n = np.minimum(occ_prev / (self.density_thresh + 1), 1.0)
        diff = np.abs(occ_curr_n - occ_prev_n)
        dynamic_mask = diff > self.diff_threshold
        return dynamic_mask

    # ─── Step 4: Assign fine zone ─────────────────────────────────────────────

    def _assign_fine_zone(
        self,
        pts: np.ndarray,
        density: np.ndarray,
        dynamic_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Returns a bool array [N_pts] — True where fine resolution applies.
        Criteria: high local density OR flagged as dynamic cell.
        """
        nx, ny = density.shape
        cx = np.clip(((pts[:, 0] - self.xmin) / self.cell_size).astype(int), 0, nx - 1)
        cy = np.clip(((pts[:, 1] - self.ymin) / self.cell_size).astype(int), 0, ny - 1)

        high_density = density[cx, cy] >= self.density_thresh
        is_dynamic = (
            dynamic_mask[cx, cy] if dynamic_mask is not None
            else np.zeros(pts.shape[0], dtype=bool)
        )
        return high_density | is_dynamic

    # ─── Step 5: Mixed voxelization ───────────────────────────────────────────

    def _mixed_voxelize(
        self,
        pts: np.ndarray,
        fine_zone: np.ndarray,
        batch_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign each point to a voxel at either fine or coarse resolution.
        Map fine voxel coordinates into the coarse grid space so that
        all voxels share a single unified grid (coarse is base grid;
        fine voxels subdivide each coarse cell 2×2×2).

        Returns:
            voxel_feats   : [V, 4]  mean (x, y, z, intensity) per voxel
            voxel_indices : [V, 4]  (batch, iz, iy, ix) in coarse grid units
        """
        Dg, Hg, Wg = self._coarse_shape   # (depth, height, width) in coarse units

        voxel_dict: Dict[Tuple[int,int,int,int], list] = {}

        for i in range(pts.shape[0]):
            x, y, z = pts[i, 0], pts[i, 1], pts[i, 2]
            intensity = pts[i, 3] if pts.shape[1] > 3 else 0.0
            use_fine = bool(fine_zone[i])

            # Voxel indices in the appropriate resolution
            vsize = self.fine if use_fine else self.coarse
            # Map to coarse grid: fine voxel size = coarse/2, so fine index / 2 = coarse index
            ix_raw = int((x - self.xmin) / vsize)
            iy_raw = int((y - self.ymin) / vsize)
            iz_raw = int((z - self.zmin) / vsize)

            # Convert to coarse grid coordinates (fine: divide by 2)
            scale = 2 if use_fine else 1
            ix = min(ix_raw // scale, Wg - 1)
            iy = min(iy_raw // scale, Hg - 1)
            iz = min(iz_raw // scale, Dg - 1)

            key = (batch_idx, iz, iy, ix)
            if key not in voxel_dict:
                voxel_dict[key] = []
            pts_in_voxel = voxel_dict[key]
            if len(pts_in_voxel) < self.max_pts:
                pts_in_voxel.append([x, y, z, float(intensity)])

        # Cap total voxels (memory guard)
        if len(voxel_dict) > self.max_voxels:
            # Keep voxels with most points (most informative)
            sorted_keys = sorted(voxel_dict, key=lambda k: -len(voxel_dict[k]))
            voxel_dict = {k: voxel_dict[k] for k in sorted_keys[:self.max_voxels]}

        if not voxel_dict:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, 4), dtype=np.int32),
            )

        keys = list(voxel_dict.keys())
        voxel_feats = np.array(
            [np.mean(voxel_dict[k], axis=0) for k in keys],
            dtype=np.float32,
        )   # [V, 4]
        voxel_indices = np.array(keys, dtype=np.int32)   # [V, 4]

        return voxel_feats, voxel_indices

    # ─── Utilities ────────────────────────────────────────────────────────────

    def _compute_grid_shape(self, vsize: float) -> Tuple[int, int, int]:
        """Return (D, H, W) for a given voxel size."""
        W = int((self.xmax - self.xmin) / vsize)
        H = int((self.ymax - self.ymin) / vsize)
        D = int((self.zmax - self.zmin) / vsize)
        return (D, H, W)

    def _empty_tensor(self, batch_idx: int) -> SparseTensor:
        return SparseTensor(
            features=torch.zeros(0, 4),
            indices=torch.zeros(0, 4, dtype=torch.int32),
            spatial_shape=self._coarse_shape,
            batch_size=batch_idx + 1,
        )

    def __repr__(self) -> str:
        D, H, W = self._coarse_shape
        return (
            f"AdaptiveVoxelizer(\n"
            f"  coarse={self.coarse}m  fine={self.fine}m\n"
            f"  grid_shape=({D},{H},{W})  max_voxels={self.max_voxels}\n"
            f"  density_thresh={self.density_thresh}  frame_diff={self.use_frame_diff}\n"
            f")"
        )
