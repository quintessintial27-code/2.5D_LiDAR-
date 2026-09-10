"""
lightning_module.py
────────────────────
PyTorch Lightning module wrapping the full pipeline:
  Adaptive Voxelizer → Sparse CNN Encoder → BEV Collapse → Task Heads

Handles:
  - Forward pass
  - Combined loss (detection + height map)
  - Optimizer + LR scheduler
  - Logging (loss, mAP, height RMSE per epoch)
  - Validation metrics accumulation
"""

from __future__ import annotations
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import lightning as L

from .models.sparse_encoder import SparseEncoder
from .models.bev_collapse import BEVCollapse
from .models.height_map_decoder import HeightMapDecoder
from .models.detection_head import DetectionHead
from .utils.metrics import DetectionMetrics, height_map_metrics


class LidarSparseMappingModule(L.LightningModule):
    """
    Full Adaptive 2.5D LiDAR Mapping pipeline as a Lightning module.

    Args:
        cfg: OmegaConf / dict config from hydra
    """

    def __init__(self, cfg: Optional[Any] = None) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg

        # ── Hyper-parameters ─────────────────────────────────────────────────
        encoder_in_ch   = 4
        stage_channels  = (16, 32, 64)
        fpn_ch          = 64
        num_classes     = 3
        self.lr         = 3e-3
        self.weight_decay = 0.01
        self.hm_weight  = 1.0
        self.det_weight = 1.0
        self.pc_range   = (-40., -40., -3., 40., 40., 3.)
        self.voxel_size = 0.20

        if cfg is not None:
            enc_cfg  = getattr(cfg, "encoder", None)
            if enc_cfg:
                encoder_in_ch  = getattr(enc_cfg, "in_channels", encoder_in_ch)
                stage_channels = tuple(getattr(enc_cfg, "stage_channels", stage_channels))
            self.lr = getattr(getattr(cfg, "optimizer", None), "lr", self.lr)
            self.weight_decay = getattr(getattr(cfg, "optimizer", None), "weight_decay", self.weight_decay)

        # ── Model components ──────────────────────────────────────────────────
        self.encoder = SparseEncoder(
            in_channels=encoder_in_ch,
            stage_channels=stage_channels,
        )
        self.bev_collapse = BEVCollapse(
            stage_channels=self.encoder.out_channels,
            fpn_channels=fpn_ch,
        )
        self.height_decoder = HeightMapDecoder(
            in_channels=fpn_ch,
            hidden_channels=fpn_ch // 2,
        )
        self.det_head = DetectionHead(
            in_channels=fpn_ch,
            num_classes=num_classes,
            hidden_ch=fpn_ch,
        )

        # ── Val metrics accumulator ───────────────────────────────────────────
        self._det_metrics = DetectionMetrics(num_classes=num_classes)
        self._val_height_rmse = []

    # ─── Forward ─────────────────────────────────────────────────────────────

    def forward(self, sparse_tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            sparse_tensor: SparseTensor from AdaptiveVoxelizer

        Returns:
            dict with 'height_map', 'heatmap', 'reg'
        """
        # 1. Sparse 3D encoding
        stage_feats = self.encoder(sparse_tensor)

        # 2. Collapse 3D → 2D BEV feature map
        bev = self.bev_collapse(stage_feats)   # [B, fpn_ch, H, W]

        # 3. Height map prediction
        height_map = self.height_decoder(bev)   # [B, 1, H, W]

        # 4. Detection predictions
        det_out = self.det_head(bev)            # {'heatmap': ..., 'reg': ...}

        return {
            "height_map": height_map,
            **det_out,
        }

    # ─── Training step ────────────────────────────────────────────────────────

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        sparse = batch["sparse_tensor"]
        gt_height = batch["gt_height_map"]        # [B, 1, H, W]
        gt_heatmap = batch["gt_heatmap"]          # [B, n_cls, H, W]
        gt_reg = batch["gt_reg"]                  # [B, 10, H, W]
        gt_pos_mask = batch["gt_pos_mask"]        # [B, H, W]

        preds = self(sparse)

        # Height map loss
        loss_height = HeightMapDecoder.height_map_loss(preds["height_map"], gt_height)

        # Detection loss
        det_losses = self.det_head.loss(
            predictions={"heatmap": preds["heatmap"], "reg": preds["reg"]},
            gt_heatmaps=gt_heatmap,
            gt_regs=gt_reg,
            gt_pos_masks=gt_pos_mask,
        )

        total_loss = (
            self.hm_weight * loss_height +
            self.det_weight * det_losses["total"]
        )

        self.log("train/loss_total", total_loss, prog_bar=True)
        self.log("train/loss_height", loss_height)
        self.log("train/loss_heatmap", det_losses["heatmap"])
        self.log("train/loss_regression", det_losses["regression"])

        return total_loss

    # ─── Validation step ──────────────────────────────────────────────────────

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        sparse = batch["sparse_tensor"]
        gt_height = batch["gt_height_map"]
        gt_heatmap = batch["gt_heatmap"]
        gt_reg = batch["gt_reg"]
        gt_pos_mask = batch["gt_pos_mask"]
        gt_boxes = batch["gt_boxes"]

        preds = self(sparse)

        # Height metrics
        hm_metrics = height_map_metrics(preds["height_map"], gt_height)
        self._val_height_rmse.append(hm_metrics["rmse"])

        # Detection: decode predictions
        detections = self.det_head.decode(
            {"heatmap": preds["heatmap"], "reg": preds["reg"]},
            pc_range=self.pc_range,
            voxel_size=self.voxel_size,
        )

        # Accumulate detection metrics
        import numpy as np
        for det in detections:
            boxes_np = det["boxes"].numpy()
            scores_np = det["scores"].numpy()
            labels_np = det["labels"].numpy()
            gt_b = gt_boxes[0].numpy() if isinstance(gt_boxes, list) else gt_boxes.squeeze(0).numpy()
            gt_lb = gt_b[:, 7].astype(int) if gt_b.shape[0] > 0 else np.zeros(0, int)
            gt_bx = gt_b[:, :7] if gt_b.shape[0] > 0 else np.zeros((0, 7))

            self._det_metrics.update(
                pred_boxes=boxes_np,
                pred_scores=scores_np,
                pred_labels=labels_np,
                gt_boxes=gt_bx,
                gt_labels=gt_lb,
            )

    def on_validation_epoch_end(self) -> None:
        import numpy as np

        # mAP
        ap_results = self._det_metrics.compute()
        self.log("val/mAP", ap_results["mAP"], prog_bar=True)
        for k, v in ap_results.items():
            if k != "mAP":
                self.log(f"val/{k}", v)

        # Height RMSE
        if self._val_height_rmse:
            mean_rmse = float(np.mean(self._val_height_rmse))
            self.log("val/height_rmse", mean_rmse, prog_bar=True)

        # Reset accumulators
        self._det_metrics.reset()
        self._val_height_rmse.clear()

    # ─── Optimizer ───────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=80, eta_min=1e-4
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
