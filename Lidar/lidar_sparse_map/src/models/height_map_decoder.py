"""
height_map_decoder.py
─────────────────────
Static Environment Height Map Head.

Takes the dense 2D BEV feature map from BEVCollapse and predicts
the max-height (z_max) of the static environment for every BEV cell.

Output: height_map [B, 1, H, W] — grayscale-renderable, encodes
the vertical extent of the static scene (buildings, walls, ground).

Loss: Smooth-L1 against ground-truth height maps derived from
raw point clouds by taking the max-Z of all static points per cell.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeightMapDecoder(nn.Module):
    """
    Lightweight CNN head that regresses per-cell max height from BEV features.

    Architecture:
      BEV feats → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → ReLU → Conv(1×1) → height
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),   # output 1 channel
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, bev_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_feat: [B, C, H, W]  BEV feature map from BEVCollapse

        Returns:
            height_map: [B, 1, H, W]  predicted max height per cell (metres)
        """
        return self.head(bev_feat)

    @staticmethod
    def build_gt_height_map(
        points: torch.Tensor,
        batch_size: int,
        spatial_shape: tuple,
        pc_range: tuple,
        voxel_size: float = 0.20,
    ) -> torch.Tensor:
        """
        Build ground-truth height map from raw point cloud.

        Args:
            points   : [N, 5]  (batch_idx, x, y, z, intensity)
            batch_size: B
            spatial_shape: (D, H, W) grid dims
            pc_range : (x_min, y_min, z_min, x_max, y_max, z_max)
            voxel_size: coarse voxel size in metres

        Returns:
            gt_height: [B, 1, H, W]  max-Z per BEV cell (0 = empty)
        """
        D, H, W = spatial_shape
        x_min, y_min, z_min = pc_range[0], pc_range[1], pc_range[2]

        gt = torch.zeros(batch_size, 1, H, W)

        for b in range(batch_size):
            mask = points[:, 0].long() == b
            pts_b = points[mask]    # [M, 5]
            if pts_b.shape[0] == 0:
                continue

            xi = ((pts_b[:, 1] - x_min) / voxel_size).long().clamp(0, W - 1)
            yi = ((pts_b[:, 2] - y_min) / voxel_size).long().clamp(0, H - 1)
            z = pts_b[:, 3]

            for i in range(pts_b.shape[0]):
                c_y, c_x = int(yi[i]), int(xi[i])
                z_val = float(z[i])
                if z_val > gt[b, 0, c_y, c_x]:
                    gt[b, 0, c_y, c_x] = z_val

        return gt

    @staticmethod
    def height_map_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Smooth-L1 loss on occupied cells only (ignore empty BEV cells).

        Args:
            pred  : [B, 1, H, W]  predicted heights
            target: [B, 1, H, W]  ground-truth heights

        Returns:
            scalar loss
        """
        occupied = (target > 0.0)
        if not occupied.any():
            return pred.sum() * 0.0    # zero loss, keep graph alive

        return F.smooth_l1_loss(pred[occupied], target[occupied], beta=0.1)
