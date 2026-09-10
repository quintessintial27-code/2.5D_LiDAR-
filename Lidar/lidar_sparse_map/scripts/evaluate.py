"""
evaluate.py
────────────
Evaluate a trained checkpoint on the KITTI val split.

Usage:
  python scripts/evaluate.py --checkpoint checkpoints/epoch10-mAP0.350.ckpt

Outputs:
  - mAP per class (Car, Pedestrian, Cyclist)
  - Height map RMSE
  - Saves a BEV visualisation for each frame to outputs/eval/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.kitti_dataset import KITTIDataset
from src.lightning_module import LidarSparseMappingModule
from src.utils.metrics import DetectionMetrics, height_map_metrics
from src.utils.visualizer import LidarVisualizer


def main():
    parser = argparse.ArgumentParser(description="KITTI Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .ckpt checkpoint file")
    parser.add_argument("--data_root", type=str, default="data/kitti")
    parser.add_argument("--save_viz", action="store_true",
                        help="Save BEV visualisations for each frame")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit evaluation to N frames (for quick check)")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from: {args.checkpoint}")
    model = LidarSparseMappingModule.load_from_checkpoint(args.checkpoint)
    model.eval()

    # ── Dataset ───────────────────────────────────────────────────────────────
    val_dataset = KITTIDataset(
        root_dir=args.data_root,
        split="val",
        use_augment=False,
    )
    if args.max_frames:
        val_dataset.frame_ids = val_dataset.frame_ids[:args.max_frames]

    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=lambda b: b[0])

    os.makedirs("outputs/eval", exist_ok=True)

    det_metrics = DetectionMetrics(num_classes=3)
    all_height_rmse = []

    import numpy as np

    print(f"\nEvaluating on {len(val_dataset)} frames...")

    with torch.no_grad():
        for i, sample in enumerate(tqdm(loader, desc="Eval")):
            sparse = sample["sparse_tensor"]
            gt_height = sample["gt_height_map"]
            gt_boxes = sample["gt_boxes"]
            frame_id = sample["frame_id"]

            preds = model(sparse)

            # Height map metrics
            hm = height_map_metrics(preds["height_map"], gt_height.unsqueeze(0))
            all_height_rmse.append(hm["rmse"])

            # Detection decode
            detections = model.det_head.decode(
                {"heatmap": preds["heatmap"], "reg": preds["reg"]},
                pc_range=model.pc_range,
                voxel_size=model.voxel_size,
            )[0]

            # Metrics update
            gt_b = gt_boxes.numpy()
            det_metrics.update(
                pred_boxes=detections["boxes"].numpy(),
                pred_scores=detections["scores"].numpy(),
                pred_labels=detections["labels"].numpy(),
                gt_boxes=gt_b[:, :7],
                gt_labels=gt_b[:, 7].astype(int),
            )

            # Optional: save visualisation
            if args.save_viz:
                hm_np = preds["height_map"].squeeze().numpy()
                save_p = f"outputs/eval/frame_{frame_id:06d}.png"
                LidarVisualizer.plot_bev_detections(
                    height_map=hm_np,
                    detections=detections,
                    gt_boxes=gt_b,
                    title=f"Frame {frame_id}",
                    save_path=save_p,
                    show=False,
                )

    # ── Results ───────────────────────────────────────────────────────────────
    ap_results = det_metrics.compute()
    mean_rmse = float(np.mean(all_height_rmse))

    print("\n" + "="*50)
    print("  EVALUATION RESULTS")
    print("="*50)
    print(f"  AP Car:         {ap_results.get('AP_cls0', 0):.4f}")
    print(f"  AP Pedestrian:  {ap_results.get('AP_cls1', 0):.4f}")
    print(f"  AP Cyclist:     {ap_results.get('AP_cls2', 0):.4f}")
    print(f"  mAP:            {ap_results['mAP']:.4f}")
    print(f"  Height RMSE:    {mean_rmse:.4f} m")
    print("="*50)


if __name__ == "__main__":
    main()
