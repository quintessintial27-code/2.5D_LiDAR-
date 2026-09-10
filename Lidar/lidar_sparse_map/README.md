# Adaptive Variable Resolution 2.5D LiDAR Mapping

A lightweight **Sparse CNN** based LiDAR perception system for **dynamic environment mapping**, built with pure PyTorch — no CUDA required.

## Overview

This project implements **Adaptive Variable Resolution 2.5D LiDAR Mapping** using a custom Sparse Convolutional Neural Network (Sparse CNN). It processes LiDAR point clouds to simultaneously predict:

- **2.5D Height Maps** — ground surface reconstruction
- **3D Object Detection** — Cars, Pedestrians, Cyclists (CenterPoint-style)
- **Adaptive Voxelization** — variable resolution based on point density

## Architecture

```
LiDAR Point Cloud (.bin)
        │
        ▼
┌─────────────────┐
│ AdaptiveVoxelizer│  ← Variable resolution voxel grid
└────────┬────────┘
         │  SparseTensor [N_voxels, 4]
         ▼
┌─────────────────┐
│  SparseEncoder  │  ← 3-stage Sparse CNN (SubMConv3d + SparseConv3d)
│  Stage0: 16ch   │    Channels: 4->16->32->64
│  Stage1: 32ch   │
│  Stage2: 64ch   │
└────────┬────────┘
         │  Multi-scale features
         ▼
┌─────────────────┐
│   BEVCollapse   │  ← 3D -> 2D Bird's Eye View (FPN)
└────────┬────────┘
         │  BEV feature map [B, 64, H, W]
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│ Height │ │Detection │
│Decoder │ │  Head    │
│[B,1,H,W]│ │heatmap+reg│
└────────┘ └──────────┘
```

**Total parameters: 437,838** (~0.4M — runs on CPU with 8GB RAM)

## Features

- Pure PyTorch sparse convolutions (no CUDA-only libraries)
- Runs on CPU (Intel Celeron compatible)
- KITTI dataset support (training, validation)
- CenterPoint-style 3D detection head
- Multi-scale BEV feature pyramid
- Adaptive voxelization with variable resolution
- Synthetic data generator (no 29GB download needed)

## Project Structure

```
lidar_sparse_map/
├── scripts/
│   ├── train.py                         # Main training script
│   ├── evaluate.py                      # Evaluation / mAP
│   ├── infer_live.py                    # Live inference on .bin files
│   └── generate_velodyne_from_labels.py # Synthetic data generator
├── src/
│   ├── lightning_module.py              # PyTorch Lightning module
│   ├── datasets/kitti_dataset.py        # KITTI dataset loader
│   ├── models/
│   │   ├── sparse_tensor.py             # Sparse CNN primitives
│   │   ├── sparse_encoder.py            # Multi-stage encoder
│   │   ├── bev_collapse.py              # 3D->2D BEV projection
│   │   ├── height_map_decoder.py        # Height map head
│   │   └── detection_head.py            # 3D detection head
│   ├── voxelizer/
│   │   ├── adaptive_voxelizer.py        # Variable resolution voxelizer
│   │   └── point_preprocessor.py        # Point cloud preprocessing
│   └── utils/
│       ├── metrics.py                   # mAP, RMSE, height metrics
│       └── visualizer.py                # Open3D visualization
├── configs/
│   ├── train.yaml                       # Training configuration
│   └── dataset/kitti.yaml               # Dataset configuration
├── tests/                               # 43 unit tests (all passing)
└── requirements.txt
```

## Hardware Requirements

| Component | Minimum | Tested On |
|-----------|---------|-----------|
| RAM | 4 GB | 8 GB |
| CPU | Any x86-64 | Intel Celeron |
| GPU | None required | CPU only |
| Storage | ~200 MB | 256 GB SSD |

## Setup

```bash
# Clone the repo
git clone https://github.com/quintessentials67/lidar-sparse-map.git
cd lidar-sparse-map/lidar_sparse_map

# Install dependencies (CPU-only PyTorch)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install lightning numpy pyyaml tqdm scipy

# Generate synthetic KITTI data (no 29GB download needed!)
# First download label_2 and calib from:
# https://www.cvlibs.net/datasets/kitti/eval_object.php
# Place them in data/kitti/training/
python scripts/generate_velodyne_from_labels.py --max_frames 200

# Train
python scripts/train.py
```

## Dataset

Uses [KITTI 3D Object Detection](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

Only **label_2** (~5 MB) and **calib** (~16 MB) files are needed.
The `generate_velodyne_from_labels.py` script creates realistic synthetic LiDAR
point clouds from the labels — no need to download the full 29 GB velodyne dataset.

## Training

```bash
python scripts/train.py
```

Expected output:
```
Epoch 0:  loss_total=75.5
Epoch 1:  loss_total=45.2
Epoch 5:  loss_total=18.3
```

## License

MIT
