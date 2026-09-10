"""
bev_collapse.py
───────────────
BEV (Bird's Eye View) Collapse + Feature Pyramid Network (FPN).

Takes multi-scale 3D SparseTensors from SparseEncoder and:
  1. Collapses Z-dimension (3D → 2D) via max-pooling over height axis
  2. Builds a lightweight FPN to fuse coarse + fine features
  3. Outputs a single dense 2D BEV feature map [B, C_fpn, H_bev, W_bev]

This is the bridge between 3D sparse processing and 2D task heads.

Why 2.5D?
  "2.5D" means we keep Z-compressed height information in the feature map
  (via max-height collapse), giving us ground-level spatial layout PLUS
  height awareness — richer than pure 2D but cheaper than full 3D.
"""

from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_tensor import SparseTensor


class BEVCollapse(nn.Module):
    """
    Converts multi-scale 3D SparseTensors → single dense 2D BEV feature map.

    Args:
        stage_channels : dict of {stage_name: n_channels} from SparseEncoder
        fpn_channels   : output channels of FPN (unified feature dim)
    """

    def __init__(
        self,
        stage_channels: Dict[str, int],
        fpn_channels: int = 64,
    ) -> None:
        super().__init__()
        self.fpn_channels = fpn_channels

        # 1×1 convs to project each stage to fpn_channels before fusion
        self.proj = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv2d(ch, fpn_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            )
            for name, ch in stage_channels.items()
        })

        # Post-fusion refinement conv
        self.refine = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, stage_features: Dict[str, SparseTensor]) -> torch.Tensor:
        """
        Args:
            stage_features: {'stage0': SparseTensor, 'stage1': ..., 'stage2': ...}

        Returns:
            bev_feat: [B, fpn_channels, H_bev, W_bev]  dense 2D BEV feature map
        """
        bev_maps = {}
        target_h, target_w = None, None

        # Process stages from coarsest to finest (reverse order for FPN top-down)
        for name, sp in stage_features.items():
            # 3D → 2D: max-collapse over Z dimension
            bev = self._collapse_z(sp)         # [B, C, H, W]
            bev = self.proj[name](bev)          # [B, fpn_ch, H, W]
            bev_maps[name] = bev

            # Target resolution = finest stage (stage0)
            if name == "stage0":
                target_h, target_w = bev.shape[2], bev.shape[3]

        # FPN: upsample coarser stages to match stage0 resolution, then add
        fused = bev_maps["stage0"]
        for name in ["stage1", "stage2"]:
            if name in bev_maps:
                upsampled = F.interpolate(
                    bev_maps[name],
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
                fused = fused + upsampled   # element-wise addition (FPN style)

        # Refine fused features
        out = self.refine(fused)            # [B, fpn_ch, H, W]
        return out

    # ─── Z-Collapse ──────────────────────────────────────────────────────────

    def _collapse_z(self, sp: SparseTensor) -> torch.Tensor:
        """
        Collapse the Z dimension of a 3D SparseTensor via max-pooling,
        producing a dense 2D BEV map [B, C, H, W].

        Strategy:
          - Convert sparse → dense [B, C, D, H, W]
          - max-pool over D (height) dimension
          - Result: [B, C, H, W]

        For memory efficiency, we work directly with sparse indices
        rather than materialising the full dense tensor.
        """
        D, H, W = sp.spatial_shape
        B = sp.batch_size
        C = sp.num_channels

        # Initialise output with -inf (so max-pool ignores empty voxels)
        bev = torch.full((B, C, H, W), fill_value=-1e9, dtype=sp.dtype)

        if sp.num_voxels > 0:
            b = sp.indices[:, 0].long()
            # z = sp.indices[:, 1] — we collapse over this
            y = sp.indices[:, 2].long()
            x = sp.indices[:, 3].long()

            # Scatter-max: for each (b, y, x) keep max feature over all z
            # We iterate over batch for simplicity (batch_size=1 for our use case)
            for bi in range(B):
                mask = b == bi
                if not mask.any():
                    continue
                yi = y[mask]
                xi = x[mask]
                fi = sp.features[mask]    # [M, C]

                # Clamp to valid range
                valid = (yi >= 0) & (yi < H) & (xi >= 0) & (xi < W)
                yi, xi, fi = yi[valid], xi[valid], fi[valid]

                # Linear index for scatter
                lin = yi * W + xi         # [M]
                lin_exp = lin.unsqueeze(1).expand(-1, C)   # [M, C]

                out_flat = bev[bi].view(C, H * W).T.contiguous()  # [H*W, C]
                out_flat.scatter_reduce_(
                    0, lin_exp, fi,
                    reduce="amax", include_self=True
                )
                bev[bi] = out_flat.T.view(C, H, W)

        # Replace -inf with 0 (empty BEV cells)
        bev = bev.clamp(min=0.0)
        return bev
