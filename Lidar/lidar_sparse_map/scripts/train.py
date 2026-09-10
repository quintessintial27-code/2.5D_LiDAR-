"""
train.py
─────────
Main training entry point.

Usage:
  python scripts/train.py

For smoke test (1 batch only):
  python scripts/train.py fast_dev_run=true

Override config values:
  python scripts/train.py trainer.max_epochs=5 optimizer.lr=0.001
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import lightning as L
from torch.utils.data import DataLoader

from src.datasets.kitti_dataset import KITTIDataset
from src.lightning_module import LidarSparseMappingModule


def collate_fn(batch):
    """Custom collate: SparseTensors can't be stacked normally.
    For batch_size=1, pass SparseTensor as-is and unsqueeze all other tensors.
    """
    item = batch[0]
    out = {}
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0)   # [C,H,W] → [1,C,H,W]  or [H,W] → [1,H,W]
        else:
            out[k] = v               # SparseTensor, str, int — pass through
    return out


def main():
    L.seed_everything(42)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Loading KITTI dataset...")
    train_dataset = KITTIDataset(
        root_dir="data/kitti",
        split="train",
        use_augment=True,
    )
    val_dataset = KITTIDataset(
        root_dir="data/kitti",
        split="val",
        use_augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Building model...")
    model = LidarSparseMappingModule()
    print(model)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_params:,}")

    # ── Trainer ───────────────────────────────────────────────────────────────
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val/mAP",
        mode="max",
        save_top_k=3,
        filename="epoch{epoch:02d}-mAP{val/mAP:.3f}",
        dirpath="checkpoints/",
        verbose=True,
    )
    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")
    progress = L.pytorch.callbacks.TQDMProgressBar(refresh_rate=5)

    trainer = L.Trainer(
        max_epochs=80,
        accelerator="cpu",
        devices=1,
        precision=32,
        gradient_clip_val=10.0,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, lr_monitor, progress],
        enable_model_summary=True,
        default_root_dir="outputs/",
    )

    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    print(f"\nBest model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best val mAP: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
