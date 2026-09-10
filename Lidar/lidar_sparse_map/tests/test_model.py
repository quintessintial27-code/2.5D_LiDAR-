"""
test_model.py
──────────────
Unit tests for the Sparse CNN model components.

Tests all layers end-to-end with random synthetic SparseTensors.

Run with:
  python -m pytest tests/test_model.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.models.sparse_tensor import SparseTensor
from src.models.sparse_ops import SubMConv3d, SparseConv3d, SparseConvBlock, SparseBatchNorm, SparseReLU
from src.models.sparse_encoder import SparseEncoder
from src.models.bev_collapse import BEVCollapse
from src.models.height_map_decoder import HeightMapDecoder
from src.models.detection_head import DetectionHead
from src.lightning_module import LidarSparseMappingModule


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_sparse(n_vox=200, C=4, spatial=(30, 40, 40), batch=1) -> SparseTensor:
    """Create a synthetic SparseTensor with random features and valid indices."""
    D, H, W = spatial
    indices = torch.zeros(n_vox, 4, dtype=torch.int32)
    indices[:, 0] = 0                                          # batch 0
    indices[:, 1] = torch.randint(0, D, (n_vox,))
    indices[:, 2] = torch.randint(0, H, (n_vox,))
    indices[:, 3] = torch.randint(0, W, (n_vox,))
    return SparseTensor(
        features=torch.randn(n_vox, C),
        indices=indices,
        spatial_shape=spatial,
        batch_size=batch,
    )


# ─── SubMConv3d tests ─────────────────────────────────────────────────────────

class TestSubMConv3d:

    def test_output_num_voxels_unchanged(self):
        """SubM must preserve occupancy pattern (same N voxels)."""
        sp = make_sparse(n_vox=150, C=4)
        conv = SubMConv3d(4, 16, kernel_size=3)
        out = conv(sp)
        assert out.num_voxels == 150, "SubM must not change number of voxels"

    def test_output_channels(self):
        sp = make_sparse(n_vox=100, C=4)
        conv = SubMConv3d(4, 16, kernel_size=3)
        out = conv(sp)
        assert out.num_channels == 16

    def test_indices_unchanged(self):
        sp = make_sparse(n_vox=50, C=4)
        conv = SubMConv3d(4, 8, kernel_size=3)
        out = conv(sp)
        assert (out.indices == sp.indices).all(), "SubM indices must be identical to input"

    def test_empty_input(self):
        empty = SparseTensor(
            features=torch.zeros(0, 4),
            indices=torch.zeros(0, 4, dtype=torch.int32),
            spatial_shape=(10, 10, 10),
            batch_size=1,
        )
        conv = SubMConv3d(4, 8)
        out = conv(empty)
        assert out.num_voxels == 0

    def test_gradient_flow(self):
        sp = make_sparse(n_vox=20, C=4)
        sp.features.requires_grad_(True)
        conv = SubMConv3d(4, 8)
        out = conv(sp)
        loss = out.features.sum()
        loss.backward()
        assert sp.features.grad is not None, "Gradients must flow through SubMConv3d"


# ─── SparseConv3d tests ───────────────────────────────────────────────────────

class TestSparseConv3d:

    def test_output_shape_stride2(self):
        """Stride=2 must halve the spatial dimensions."""
        sp = make_sparse(n_vox=100, C=16, spatial=(30, 40, 40))
        conv = SparseConv3d(16, 32, kernel_size=3, stride=2)
        out = conv(sp)
        D, H, W = sp.spatial_shape
        assert out.spatial_shape == (D//2, H//2, W//2)

    def test_output_channels_stride2(self):
        sp = make_sparse(n_vox=80, C=16, spatial=(30, 40, 40))
        conv = SparseConv3d(16, 32, stride=2)
        out = conv(sp)
        assert out.num_channels == 32

    def test_empty_input_stride2(self):
        empty = SparseTensor(
            features=torch.zeros(0, 16),
            indices=torch.zeros(0, 4, dtype=torch.int32),
            spatial_shape=(30, 40, 40),
            batch_size=1,
        )
        conv = SparseConv3d(16, 32, stride=2)
        out = conv(empty)
        assert out.num_voxels == 0

    def test_no_stride(self):
        sp = make_sparse(n_vox=50, C=8, spatial=(10, 10, 10))
        conv = SparseConv3d(8, 16, stride=1)
        out = conv(sp)
        assert out.spatial_shape == (10, 10, 10)


# ─── SparseEncoder tests ──────────────────────────────────────────────────────

class TestSparseEncoder:

    def test_output_keys(self):
        sp = make_sparse(n_vox=300, C=4, spatial=(30, 40, 40))
        enc = SparseEncoder(in_channels=4, stage_channels=(16, 32, 64))
        out = enc(sp)
        assert "stage0" in out
        assert "stage1" in out
        assert "stage2" in out

    def test_channel_sizes(self):
        sp = make_sparse(n_vox=200, C=4, spatial=(30, 40, 40))
        enc = SparseEncoder(in_channels=4, stage_channels=(16, 32, 64))
        out = enc(sp)
        assert out["stage0"].num_channels == 16
        assert out["stage1"].num_channels == 32
        assert out["stage2"].num_channels == 64

    def test_spatial_downsampling(self):
        sp = make_sparse(n_vox=200, C=4, spatial=(30, 40, 40))
        enc = SparseEncoder(in_channels=4, stage_channels=(16, 32, 64))
        out = enc(sp)
        D0, H0, W0 = out["stage0"].spatial_shape
        D1, H1, W1 = out["stage1"].spatial_shape
        D2, H2, W2 = out["stage2"].spatial_shape
        assert D1 <= D0 and H1 <= H0 and W1 <= W0, "Stage1 must be spatially smaller"
        assert D2 <= D1 and H2 <= H1 and W2 <= W1, "Stage2 must be spatially smaller"


# ─── BEVCollapse tests ────────────────────────────────────────────────────────

class TestBEVCollapse:

    def setup_method(self):
        self.stage_channels = {"stage0": 16, "stage1": 32, "stage2": 64}
        self.bev = BEVCollapse(self.stage_channels, fpn_channels=64)

    def _make_stage_feats(self):
        return {
            "stage0": make_sparse(200, 16, (30, 40, 40)),
            "stage1": make_sparse(80, 32, (15, 20, 20)),
            "stage2": make_sparse(30, 64, (7, 10, 10)),
        }

    def test_output_is_dense(self):
        feats = self._make_stage_feats()
        out = self.bev(feats)
        assert isinstance(out, torch.Tensor), "BEV output must be a dense tensor"
        assert out.ndim == 4, "Output must be [B, C, H, W]"

    def test_output_channels(self):
        feats = self._make_stage_feats()
        out = self.bev(feats)
        assert out.shape[1] == 64

    def test_output_batch_size(self):
        feats = self._make_stage_feats()
        out = self.bev(feats)
        assert out.shape[0] == 1   # batch=1


# ─── HeightMapDecoder tests ───────────────────────────────────────────────────

class TestHeightMapDecoder:

    def test_output_shape(self):
        bev_feat = torch.randn(1, 64, 40, 40)
        dec = HeightMapDecoder(in_channels=64)
        out = dec(bev_feat)
        assert out.shape == (1, 1, 40, 40), "Height map must be [B,1,H,W]"

    def test_loss_positive(self):
        pred = torch.rand(1, 1, 20, 20)
        target = torch.rand(1, 1, 20, 20) * 2.0
        loss = HeightMapDecoder.height_map_loss(pred, target)
        assert loss.item() >= 0.0


# ─── DetectionHead tests ──────────────────────────────────────────────────────

class TestDetectionHead:

    def test_output_keys(self):
        bev_feat = torch.randn(1, 64, 40, 40)
        head = DetectionHead(in_channels=64, num_classes=3)
        out = head(bev_feat)
        assert "heatmap" in out
        assert "reg" in out

    def test_heatmap_range(self):
        bev_feat = torch.randn(1, 64, 40, 40)
        head = DetectionHead(in_channels=64, num_classes=3)
        out = head(bev_feat)
        assert out["heatmap"].min() >= 0.0
        assert out["heatmap"].max() <= 1.0, "Heatmap must be in [0, 1] (sigmoid)"

    def test_heatmap_shape(self):
        bev_feat = torch.randn(1, 64, 40, 40)
        head = DetectionHead(in_channels=64, num_classes=3)
        out = head(bev_feat)
        assert out["heatmap"].shape == (1, 3, 40, 40)
        assert out["reg"].shape == (1, 10, 40, 40)

    def test_focal_loss_positive(self):
        pred = torch.rand(1, 3, 40, 40)
        target = torch.zeros(1, 3, 40, 40)
        target[0, 0, 20, 20] = 1.0
        loss = DetectionHead.focal_loss(pred, target)
        assert loss.item() > 0.0

    def test_decode_no_crash(self):
        bev_feat = torch.randn(1, 64, 40, 40)
        head = DetectionHead(in_channels=64, num_classes=3)
        out = head(bev_feat)
        results = head.decode(
            out,
            pc_range=(-40., -40., -3., 40., 40., 3.),
            voxel_size=0.20,
            score_threshold=0.5,
        )
        assert isinstance(results, list)
        assert len(results) == 1
        assert "boxes" in results[0]


# ─── Full Pipeline smoke test ─────────────────────────────────────────────────

class TestFullPipeline:

    def test_forward_pass_no_crash(self):
        """End-to-end: SparseTensor → predictions without crash."""
        sp = make_sparse(n_vox=500, C=4, spatial=(30, 40, 40))
        model = LidarSparseMappingModule()
        model.eval()

        with torch.no_grad():
            preds = model(sp)

        assert "height_map" in preds
        assert "heatmap" in preds
        assert "reg" in preds

    def test_parameter_count(self):
        """Model must be small enough for 8 GB RAM."""
        model = LidarSparseMappingModule()
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nParameter count: {n:,}")
        assert n < 5_000_000, f"Model too large for CPU: {n:,} params (must be < 5M)"
