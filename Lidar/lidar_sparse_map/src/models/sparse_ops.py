"""
sparse_ops.py
─────────────
CPU-native Sparse Convolutional operations built on pure PyTorch.
Implements the two core primitives used in sparse 3D CNNs:

  SubMConv3d  – Submanifold Sparse Conv
                Output occupancy = Input occupancy (exact same positions)
                Ideal for: first layers, feature extraction without expansion

  SparseConv3d – General Sparse Conv (with optional stride)
                 Output occupancy can differ from input.
                 Stride > 1 → spatial downsampling.
                 Ideal for: hierarchical downsampling between stages

Both ops are wrapped as torch.nn.Module with learnable weight/bias.

Algorithm (per-op):
  1. For each output voxel, identify which kernel offsets contribute to it.
  2. Gather input features from those positions (hash-table O(1) lookup).
  3. Multiply by the corresponding kernel weight slice.
  4. Sum all contributions → output feature at that voxel.

Complexity: O(N × K³ × C_in × C_out) — feasible on CPU for small N & channels.
"""

from __future__ import annotations
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_tensor import SparseTensor


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kernel_offsets(kernel_size: int) -> List[Tuple[int, int, int]]:
    """Return all (dz, dy, dx) offset tuples for a cubic kernel."""
    half = kernel_size // 2
    offsets = []
    for dz in range(-half, half + 1):
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                offsets.append((dz, dy, dx))
    return offsets


def _get_output_positions_subm(indices: torch.Tensor) -> torch.Tensor:
    """SubM: output positions == input positions."""
    return indices.clone()


def _get_output_positions_strided(
    indices: torch.Tensor,
    stride: int,
    spatial_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Strided sparse conv: each input position (b,z,y,x) contributes to
    output position (b, z//s, y//s, x//s).  Return unique output positions.
    """
    out = indices.clone()
    out[:, 1] = out[:, 1] // stride   # z
    out[:, 2] = out[:, 2] // stride   # y
    out[:, 3] = out[:, 3] // stride   # x
    # Deduplicate
    out_np = out.numpy()
    # Use a structured approach: convert to linear index, unique, convert back
    D, H, W = spatial_shape
    Ds, Hs, Ws = (D + stride - 1) // stride, (H + stride - 1) // stride, (W + stride - 1) // stride
    B = int(out[:, 0].max().item()) + 1
    lin = (
        out[:, 0] * Ds * Hs * Ws
        + out[:, 1] * Hs * Ws
        + out[:, 2] * Ws
        + out[:, 3]
    )
    unique_lin = torch.unique(lin)
    b = unique_lin // (Ds * Hs * Ws)
    rem = unique_lin % (Ds * Hs * Ws)
    z = rem // (Hs * Ws)
    rem = rem % (Hs * Ws)
    y = rem // Ws
    x = rem % Ws
    out_unique = torch.stack([b, z, y, x], dim=1).to(torch.int32)
    return out_unique


# ─────────────────────────────────────────────────────────────────────────────
# Submanifold Sparse Conv 3D
# ─────────────────────────────────────────────────────────────────────────────

class SubMConv3d(nn.Module):
    """
    Submanifold Sparse Convolution 3D (CPU-native).

    Output sparsity = Input sparsity (same occupied positions).
    Computes the convolution only at already-occupied voxels.
    No stride support (stride is always 1 for SubM).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        indice_key: Optional[str] = None,   # kept for API compat, unused
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.offsets = _kernel_offsets(kernel_size)   # K³ offset tuples
        K = len(self.offsets)   # = kernel_size³

        # Weight: [K, C_in, C_out]
        self.weight = nn.Parameter(
            torch.empty(K, in_channels, out_channels)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=0.1, mode="fan_in")

    def forward(self, x: SparseTensor) -> SparseTensor:
        if x.num_voxels == 0:
            return x.replace(features=torch.zeros(
                0, self.out_channels, dtype=x.dtype, device=x.device
            ))

        feat_in = x.features      # [N, C_in]
        idx = x.indices            # [N, 4]
        N = x.num_voxels
        D, H, W = x.spatial_shape

        h = x.get_hash()           # (b,z,y,x) → row index

        # Accumulate contributions from all kernel positions
        out_feat = torch.zeros(N, self.out_channels, dtype=x.dtype, device=x.device)

        for k_pos, (dz, dy, dx) in enumerate(self.offsets):
            W_k = self.weight[k_pos]   # [C_in, C_out]

            # For each output voxel i at (b,z,y,x), look up input at (b,z+dz,y+dy,x+dx)
            in_z = idx[:, 1] + dz
            in_y = idx[:, 2] + dy
            in_x = idx[:, 3] + dx

            # Valid bounds mask
            valid = (
                (in_z >= 0) & (in_z < D) &
                (in_y >= 0) & (in_y < H) &
                (in_x >= 0) & (in_x < W)
            )
            if not valid.any():
                continue

            # Look up which valid input positions exist in sparse tensor
            valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
            gather_rows = []
            out_rows = []
            for vi in valid_idx.tolist():
                key = (
                    int(idx[vi, 0]), int(in_z[vi]),
                    int(in_y[vi]),  int(in_x[vi])
                )
                src = h.get(key, -1)
                if src >= 0:
                    gather_rows.append(src)
                    out_rows.append(vi)

            if not gather_rows:
                continue

            src_t = torch.tensor(gather_rows, dtype=torch.long)
            dst_t = torch.tensor(out_rows, dtype=torch.long)

            # gathered_feat: [M, C_in]
            gathered = feat_in[src_t]         # [M, C_in]
            contribution = gathered @ W_k      # [M, C_out]
            out_feat.index_add_(0, dst_t, contribution)

        if self.bias is not None:
            out_feat = out_feat + self.bias

        return x.replace(features=out_feat)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"k={self.kernel_size}, SubManifold"
        )


# ─────────────────────────────────────────────────────────────────────────────
# General Sparse Conv 3D (supports stride for downsampling)
# ─────────────────────────────────────────────────────────────────────────────

class SparseConv3d(nn.Module):
    """
    General Sparse Convolution 3D with optional stride (CPU-native).

    stride > 1 → spatial downsampling (halves spatial dims per stride=2).
    Output occupancy is computed from input occupancy via stride mapping.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
        indice_key: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.offsets = _kernel_offsets(kernel_size)
        K = len(self.offsets)

        self.weight = nn.Parameter(torch.empty(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=0.1, mode="fan_in")

    def forward(self, x: SparseTensor) -> SparseTensor:
        if x.num_voxels == 0:
            D, H, W = x.spatial_shape
            s = self.stride
            new_shape = (
                (D + s - 1) // s,
                (H + s - 1) // s,
                (W + s - 1) // s,
            )
            return SparseTensor(
                features=torch.zeros(0, self.out_channels, dtype=x.dtype),
                indices=torch.zeros(0, 4, dtype=torch.int32),
                spatial_shape=new_shape,
                batch_size=x.batch_size,
            )

        feat_in = x.features   # [N, C_in]
        idx_in = x.indices     # [N, 4]
        D, H, W = x.spatial_shape
        s = self.stride
        new_shape = (
            (D + s - 1) // s,
            (H + s - 1) // s,
            (W + s - 1) // s,
        )
        Ds, Hs, Ws = new_shape

        # Compute unique output positions
        idx_out = _get_output_positions_strided(idx_in, s, x.spatial_shape)
        N_out = idx_out.shape[0]

        # Build hash for output position → output row index
        out_hash = {
            (int(idx_out[i, 0]), int(idx_out[i, 1]),
             int(idx_out[i, 2]), int(idx_out[i, 3])): i
            for i in range(N_out)
        }

        out_feat = torch.zeros(N_out, self.out_channels, dtype=x.dtype)

        # For each kernel offset, compute which (in → out) pairs are valid
        for k_pos, (dz, dy, dx) in enumerate(self.offsets):
            W_k = self.weight[k_pos]   # [C_in, C_out]

            # Input positions shifted by offset
            in_z = idx_in[:, 1] + dz
            in_y = idx_in[:, 2] + dy
            in_x = idx_in[:, 3] + dx

            valid = (
                (in_z >= 0) & (in_z < D) &
                (in_y >= 0) & (in_y < H) &
                (in_x >= 0) & (in_x < W)
            )
            if not valid.any():
                continue

            valid_idx = valid.nonzero(as_tuple=False).squeeze(1).tolist()
            src_rows = []
            dst_rows = []
            for vi in valid_idx:
                # Which output voxel does this input voxel contribute to?
                o_key = (
                    int(idx_in[vi, 0]),
                    int(in_z[vi]) // s,
                    int(in_y[vi]) // s,
                    int(in_x[vi]) // s,
                )
                dst = out_hash.get(o_key, -1)
                if dst >= 0:
                    src_rows.append(vi)
                    dst_rows.append(dst)

            if not src_rows:
                continue

            src_t = torch.tensor(src_rows, dtype=torch.long)
            dst_t = torch.tensor(dst_rows, dtype=torch.long)
            gathered = feat_in[src_t]         # [M, C_in]
            contribution = gathered @ W_k      # [M, C_out]
            out_feat.index_add_(0, dst_t, contribution)

        if self.bias is not None:
            out_feat = out_feat + self.bias

        return SparseTensor(
            features=out_feat,
            indices=idx_out,
            spatial_shape=new_shape,
            batch_size=x.batch_size,
        )

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"k={self.kernel_size}, stride={self.stride}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sparse BatchNorm + ReLU wrappers
# ─────────────────────────────────────────────────────────────────────────────

class SparseBatchNorm(nn.Module):
    """BatchNorm applied to sparse features [N, C] → [N, C]."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x: SparseTensor) -> SparseTensor:
        if x.num_voxels == 0:
            return x
        return x.replace(features=self.bn(x.features))


class SparseReLU(nn.Module):
    """ReLU applied to sparse features in-place."""

    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.replace(features=F.relu(x.features, inplace=True))


# ─────────────────────────────────────────────────────────────────────────────
# Sparse Conv Block (Conv + BN + ReLU)
# ─────────────────────────────────────────────────────────────────────────────

class SparseConvBlock(nn.Module):
    """Convenience block: [SubM or Sparse Conv] → BN → ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        subm: bool = False,
    ) -> None:
        super().__init__()
        if subm:
            self.conv = SubMConv3d(in_channels, out_channels, kernel_size)
        else:
            self.conv = SparseConv3d(in_channels, out_channels, kernel_size, stride=stride)
        self.bn = SparseBatchNorm(out_channels)
        self.act = SparseReLU()

    def forward(self, x: SparseTensor) -> SparseTensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
