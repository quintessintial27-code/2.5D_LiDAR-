"""
sparse_tensor.py
────────────────
Custom SparseTensor dataclass for CPU-based Sparse CNN operations.
No CUDA or spconv dependency — pure PyTorch.

A SparseTensor stores only the *occupied* voxels:
  features : [N, C]   – feature vector per occupied voxel
  indices  : [N, 4]   – (batch_idx, z, y, x)  int32
  spatial_shape : (D, H, W)  – full grid dimensions
  batch_size    : int

The key invariant: features[i] corresponds to the voxel at indices[i].
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class SparseTensor:
    """Lightweight sparse voxel tensor (CPU-compatible)."""

    features: torch.Tensor      # [N, C]  float32
    indices: torch.Tensor       # [N, 4]  int32  (batch, z, y, x)
    spatial_shape: Tuple[int, int, int]   # (D, H, W)
    batch_size: int
    # Optional: hash map for fast neighbour lookup (built lazily)
    _hash: Optional[dict] = field(default=None, repr=False, compare=False)

    # ─── Properties ──────────────────────────────────────────────────────────

    @property
    def num_voxels(self) -> int:
        return self.features.shape[0]

    @property
    def num_channels(self) -> int:
        return self.features.shape[1]

    @property
    def device(self) -> torch.device:
        return self.features.device

    @property
    def dtype(self) -> torch.dtype:
        return self.features.dtype

    # ─── Hash map for O(1) voxel lookup ──────────────────────────────────────

    def build_hash(self) -> None:
        """Build Python dict mapping (b, z, y, x) → feature row index."""
        idx = self.indices.cpu().tolist()   # [[b,z,y,x], ...]
        self._hash = {tuple(row): i for i, row in enumerate(idx)}

    def get_hash(self) -> dict:
        if self._hash is None:
            self.build_hash()
        return self._hash  # type: ignore[return-value]

    def invalidate_hash(self) -> None:
        self._hash = None

    # ─── Convenience constructors ─────────────────────────────────────────────

    @classmethod
    def from_dense(
        cls,
        dense: torch.Tensor,
        spatial_shape: Tuple[int, int, int],
    ) -> "SparseTensor":
        """
        Convert a dense [B, C, D, H, W] tensor → SparseTensor
        (keeps only non-zero voxels — useful for testing).
        """
        B, C, D, H, W = dense.shape
        # Find non-zero voxels (any channel non-zero)
        mask = dense.abs().sum(dim=1) > 0    # [B, D, H, W]
        b_idx, z_idx, y_idx, x_idx = mask.nonzero(as_tuple=True)
        indices = torch.stack(
            [b_idx, z_idx, y_idx, x_idx], dim=1
        ).to(torch.int32)
        features = dense[b_idx, :, z_idx, y_idx, x_idx]   # [N, C]
        return cls(
            features=features,
            indices=indices,
            spatial_shape=spatial_shape,
            batch_size=B,
        )

    def to_dense(self) -> torch.Tensor:
        """
        Convert SparseTensor → dense [B, C, D, H, W].
        WARNING: can be memory-heavy for large grids — use sparingly.
        """
        D, H, W = self.spatial_shape
        C = self.num_channels
        out = torch.zeros(
            self.batch_size, C, D, H, W,
            dtype=self.dtype, device=self.device,
        )
        b = self.indices[:, 0].long()
        z = self.indices[:, 1].long()
        y = self.indices[:, 2].long()
        x = self.indices[:, 3].long()
        out[b, :, z, y, x] = self.features
        return out

    # ─── Shape utilities ──────────────────────────────────────────────────────

    def replace(
        self,
        features: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        spatial_shape: Optional[Tuple[int, int, int]] = None,
    ) -> "SparseTensor":
        """Return a new SparseTensor with updated fields (immutable-style)."""
        return SparseTensor(
            features=features if features is not None else self.features,
            indices=indices if indices is not None else self.indices,
            spatial_shape=spatial_shape if spatial_shape is not None else self.spatial_shape,
            batch_size=self.batch_size,
        )

    def __repr__(self) -> str:
        D, H, W = self.spatial_shape
        return (
            f"SparseTensor(N={self.num_voxels}, C={self.num_channels}, "
            f"grid=({D},{H},{W}), batch={self.batch_size}, "
            f"density={self.num_voxels / (self.batch_size * D * H * W):.4%})"
        )
