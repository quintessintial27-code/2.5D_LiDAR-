"""
infer_live.py
──────────────
Real-time / offline inference demo.

Modes:
  1. From a single .bin file:
       python scripts/infer_live.py --bin path/to/000000.bin

  2. From a directory of .bin files (plays as a sequence):
       python scripts/infer_live.py --dir data/kitti/training/velodyne

  3. From a checkpoint + KITTI val split (random frame):
       python scripts/infer_live.py --kitti_val data/kitti --random

Displays:
  - BEV height map with detected objects
  - Adaptive voxel resolution overlay
  - Console: detected objects + confidence
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time
import glob
import numpy as np
import torch

from src.voxelizer.point_preprocessor import PointPreprocessor
from src.voxelizer.adaptive_voxelizer import AdaptiveVoxelizer
from src.lightning_module import LidarSparseMappingModule
from src.utils.visualizer import LidarVisualizer


CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}


def load_model(checkpoint_path: str) -> LidarSparseMappingModule:
    model = LidarSparseMappingModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def infer_single(
    model: LidarSparseMappingModule,
    pts_raw: np.ndarray,
    prev_pts: np.ndarray,
    preprocessor: PointPreprocessor,
    voxelizer: AdaptiveVoxelizer,
) -> dict:
    """Run full inference on a single frame."""
    t0 = time.perf_counter()

    pts = preprocessor(pts_raw)
    prev = preprocessor(prev_pts) if prev_pts is not None else None
    sparse = voxelizer.voxelize(pts, batch_idx=0, prev_points=prev)

    with torch.no_grad():
        preds = model(sparse)

    detections = model.det_head.decode(
        {"heatmap": preds["heatmap"], "reg": preds["reg"]},
        pc_range=model.pc_range,
        voxel_size=model.voxel_size,
        score_threshold=0.3,
    )[0]

    elapsed = time.perf_counter() - t0

    return {
        "height_map": preds["height_map"].squeeze().numpy(),
        "detections": detections,
        "pts": pts,
        "elapsed_ms": elapsed * 1000,
    }


def print_detections(detections: dict, frame_idx: int, elapsed_ms: float) -> None:
    boxes = detections["boxes"]
    scores = detections["scores"]
    labels = detections["labels"]

    print(f"\n─── Frame {frame_idx:04d}  ({elapsed_ms:.1f} ms) ───")
    if boxes.shape[0] == 0:
        print("  No detections")
        return
    for i in range(boxes.shape[0]):
        b = boxes[i]
        print(
            f"  [{CLASS_NAMES.get(int(labels[i]), '?'):11s}] "
            f"score={scores[i]:.2f}  "
            f"x={b[0]:+.1f}m y={b[1]:+.1f}m z={b[2]:+.1f}m  "
            f"l={b[3]:.1f} w={b[4]:.1f} h={b[5]:.1f}"
        )


def run_on_bin_sequence(
    bin_files: list,
    model: LidarSparseMappingModule,
    save_dir: str = None,
    show: bool = True,
) -> None:
    preprocessor = PointPreprocessor()
    voxelizer = AdaptiveVoxelizer(use_frame_diff=True)

    prev_raw = None
    for i, path in enumerate(bin_files):
        pts_raw = PointPreprocessor.load_kitti_bin(path)
        result = infer_single(model, pts_raw, prev_raw, preprocessor, voxelizer)
        print_detections(result["detections"], i, result["elapsed_ms"])

        save_p = os.path.join(save_dir, f"frame_{i:06d}.png") if save_dir else None
        LidarVisualizer.plot_bev_detections(
            height_map=result["height_map"],
            detections=result["detections"],
            title=f"Frame {i:04d}  |  {os.path.basename(path)}",
            save_path=save_p,
            show=show,
        )
        prev_raw = pts_raw


def main():
    parser = argparse.ArgumentParser(description="LiDAR Sparse Mapping — Inference Demo")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained .ckpt checkpoint")
    parser.add_argument("--bin", type=str, default=None,
                        help="Path to a single .bin file")
    parser.add_argument("--dir", type=str, default=None,
                        help="Path to directory of .bin files")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save output images")
    parser.add_argument("--no_show", action="store_true",
                        help="Don't show matplotlib windows")
    args = parser.parse_args()

    show = not args.no_show
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # ── Build model ───────────────────────────────────────────────────────────
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model = load_model(args.checkpoint)
    else:
        print("[INFO] No checkpoint provided — using random weights (for pipeline testing)")
        model = LidarSparseMappingModule()
        model.eval()

    # ── Gather .bin files ─────────────────────────────────────────────────────
    if args.bin:
        bin_files = [args.bin]
    elif args.dir:
        bin_files = sorted(glob.glob(os.path.join(args.dir, "*.bin")))
        print(f"Found {len(bin_files)} .bin files in {args.dir}")
    else:
        print("No input specified. Use --bin or --dir.")
        print("Example: python scripts/infer_live.py --dir data/kitti/training/velodyne")
        return

    run_on_bin_sequence(bin_files, model, save_dir=args.save_dir, show=show)


if __name__ == "__main__":
    main()
