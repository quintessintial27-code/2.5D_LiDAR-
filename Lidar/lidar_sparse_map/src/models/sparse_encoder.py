"""
sparse_encoder.py
─────────────────
3-Stage Sparse CNN Encoder backbone.

Stage 0: SubMConv3d (4 → 16)    — SubManifold, no downsampling
Stage 1: SparseConv3d (16 → 32) — stride=2, halves spatial dims
Stage 2: SparseConv3d (32 → 64) — stride=2, halves again

Outputs a dict of multi-scale SparseTensors:
  {'stage0': ..., 'stage1': ..., 'stage2': ...}

This multi-scale output feeds the BEV FPN for feature pyramid fusion.

Designed for CPU / 8 GB RAM:
  - Small channel widths [16, 32, 64]
  - No 3rd downsampling stage (would create too-small spatial dims)
  - SubM first stage preserves fine-grained voxel structure
"""

from __future__ import annotations
from typing import Dict

import torch.nn as nn

from .sparse_tensor import SparseTensor
from .sparse_ops import SparseConvBlock


class SparseEncoder(nn.Module):
    """
    Hierarchical 3D Sparse CNN Encoder (CPU-native, no CUDA/spconv).

    Args:
        in_channels   : Number of input features per voxel (default 4: x,y,z,i)
        stage_channels: Output channels per stage [C0, C1, C2]
        kernel_size   : Convolution kernel size (3 recommended)
    """

    def __init__(
        self,
        in_channels: int = 4,
        stage_channels: tuple = (16, 32, 64),
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        C0, C1, C2 = stage_channels

        # ── Stage 0: SubManifold — same sparsity as input ────────────────────
        self.stage0 = nn.Sequential(
            SparseConvBlock(in_channels, C0, kernel_size, stride=1, subm=True),
            SparseConvBlock(C0, C0, kernel_size, stride=1, subm=True),
        )

        # ── Stage 1: Sparse downsampling (stride=2) ──────────────────────────
        self.stage1 = nn.Sequential(
            SparseConvBlock(C0, C1, kernel_size, stride=2, subm=False),
            SparseConvBlock(C1, C1, kernel_size, stride=1, subm=True),
        )

        # ── Stage 2: Sparse downsampling (stride=2) ──────────────────────────
        self.stage2 = nn.Sequential(
            SparseConvBlock(C1, C2, kernel_size, stride=2, subm=False),
            SparseConvBlock(C2, C2, kernel_size, stride=1, subm=True),
        )

        self.out_channels = {"stage0": C0, "stage1": C1, "stage2": C2}

    def forward(self, x: SparseTensor) -> Dict[str, SparseTensor]:
        """
        Args:
            x : SparseTensor from AdaptiveVoxelizer

        Returns:
            dict of multi-scale SparseTensors at 3 resolution levels
        """
        out = {}

        # Stage 0 — full resolution (SubM)
        x0 = x
        for layer in self.stage0:
            x0 = layer(x0)
        out["stage0"] = x0

        # Stage 1 — 1/2 resolution
        x1 = x0
        for layer in self.stage1:
            x1 = layer(x1)
        out["stage1"] = x1

        # Stage 2 — 1/4 resolution
        x2 = x1
        for layer in self.stage2:
            x2 = layer(x2)
        out["stage2"] = x2

        return out

    def extra_repr(self) -> str:
        return (
            f"channels={self.out_channels}\n"
            f"stage0: SubM×2  |  stage1: Sparse(s=2)+SubM  |  stage2: Sparse(s=2)+SubM"
        )
