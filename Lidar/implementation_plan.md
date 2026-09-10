# Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception

## Overview

Build a real-time LiDAR perception system that:
1. Ingests raw point clouds from a spinning/solid-state LiDAR sensor
2. Adaptively voxelizes the scene at **variable resolution** (fine-grained near dynamic objects, coarse elsewhere)
3. Processes voxels through a **Sparse CNN backbone** (spconv-based)
4. Produces a **2.5D height-map** + **dynamic object detections** as output
5. Separates static background from dynamic foreground (people, vehicles, robots)

---

## Architecture Blueprint

```
Raw Point Cloud (N × 4: x,y,z,intensity)
        │
        ▼
┌─────────────────────────────────┐
│  Adaptive Voxelizer             │  ← density/motion-aware multi-res grid
│  (coarse 0.2m | fine 0.05m)     │
└─────────────────────────────────┘
        │  Sparse Voxel Tensor
        ▼
┌─────────────────────────────────┐
│  Sparse CNN Encoder (spconv)    │  ← SubMConv3d + SparseConv3d
│  4 stages, stride [1,2,2,2]     │
└─────────────────────────────────┘
        │  Multi-scale Features
        ▼
┌─────────────────────────────────┐
│  2.5D BEV Collapse + FPN        │  ← squeeze Z-dim → BEV feature map
└─────────────────────────────────┘
        │
   ┌────┴────┐
   ▼         ▼
[Height Map] [Detection Head]
  (static)   (dynamic objects: bbox + class + velocity)
```

---

## Technology Stack

| Layer | Tool / Library |
|---|---|
| **Language** | Python 3.10+ |
| **Sparse Conv** | `spconv 2.x` (CUDA) |
| **Deep Learning** | `PyTorch 2.x` |
| **Point Cloud I/O** | `open3d`, `numpy`, `pypcd` |
| **Dataset** | KITTI, nuScenes, Waymo Open |
| **Training** | `PyTorch Lightning` |
| **Visualization** | `open3d`, `matplotlib`, `ROS2 Rviz` (optional) |
| **Config** | `hydra` / `omegaconf` |
| **Evaluation** | KITTI eval toolkit, nuScenes devkit |

---

## Project Directory Structure

```
lidar_sparse_map/
├── configs/
│   ├── model/
│   │   ├── backbone.yaml
│   │   └── heads.yaml
│   ├── dataset/
│   │   ├── kitti.yaml
│   │   └── nuscenes.yaml
│   └── train.yaml
│
├── data/
│   ├── kitti/
│   └── nuscenes/
│
├── src/
│   ├── voxelizer/
│   │   ├── __init__.py
│   │   ├── adaptive_voxelizer.py     ← core adaptive multi-res voxelization
│   │   └── point_preprocessor.py    ← noise filter, range clip, intensity norm
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sparse_encoder.py        ← spconv Sparse CNN backbone
│   │   ├── bev_collapse.py          ← 3D→2D BEV projection + FPN
│   │   ├── height_map_decoder.py    ← static map regression head
│   │   └── detection_head.py        ← dynamic objects anchor/anchor-free head
│   │
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── kitti_dataset.py
│   │   └── nuscenes_dataset.py
│   │
│   ├── losses/
│   │   ├── focal_loss.py
│   │   └── map_loss.py
│   │
│   ├── utils/
│   │   ├── box_utils.py
│   │   ├── metrics.py
│   │   └── visualizer.py
│   │
│   └── lightning_module.py          ← PyTorch Lightning trainer wrapper
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── infer_live.py                ← real-time inference demo
│
├── tests/
│   ├── test_voxelizer.py
│   └── test_model.py
│
├── requirements.txt
└── README.md
```

---

## Proposed Changes (Implementation Phases)

### Phase 1 — Environment & Data Pipeline

#### [NEW] `requirements.txt`
Core dependencies: `torch`, `spconv-cu118`, `open3d`, `pytorch-lightning`, `hydra-core`, `numpy`, `nuscenes-devkit`

#### [NEW] `src/voxelizer/adaptive_voxelizer.py`
**The most novel component.** Implements density-aware adaptive voxelization:
- Divides XY plane into regions
- Computes local point density per region
- Assigns **fine resolution (0.05 m)** to high-density/dynamic regions
- Assigns **coarse resolution (0.2 m)** to sparse/static background
- Returns a unified `spconv.SparseConvTensor`

#### [NEW] `src/voxelizer/point_preprocessor.py`
- Range filtering (0.5 m – 80 m)
- Ground plane removal (RANSAC / height threshold)
- Intensity normalization
- Optional: dynamic motion mask from consecutive frame differencing

#### [NEW] `src/datasets/kitti_dataset.py`
- Loads `.bin` point clouds + calibration + labels
- Returns preprocessed tensors + ground-truth boxes

---

### Phase 2 — Sparse CNN Backbone

#### [NEW] `src/models/sparse_encoder.py`
4-stage encoder using `spconv`:

```
Stage 0: SubMConv3d(4, 16, 3) → BatchNorm → ReLU   (no stride, preserves sparsity)
Stage 1: SparseConv3d(16, 32, 3, stride=2) → BN → ReLU
Stage 2: SparseConv3d(32, 64, 3, stride=2) → BN → ReLU
Stage 3: SparseConv3d(64, 128, 3, stride=2) → BN → ReLU
```

- **SubMConv3d**: Submanifold sparse conv — keeps output sparse at same locations as input (for detail preservation)
- **SparseConv3d**: Standard sparse conv — downsamples and expands receptive field

#### [NEW] `src/models/bev_collapse.py`
- Collapses Z-dimension of sparse 3D feature map → dense 2D BEV
- Applies lightweight FPN (Feature Pyramid Network) for multi-scale fusion
- Output: BEV feature map at 1/4, 1/8 scales

---

### Phase 3 — Task Heads

#### [NEW] `src/models/height_map_decoder.py`
- Takes BEV features
- Predicts per-pillar **max height** (static map)
- Loss: Smooth-L1 against ground-truth height maps

#### [NEW] `src/models/detection_head.py`
- Anchor-free CenterPoint-style head
- Predicts: heatmap (class), offset (x,y), z, dimensions (l,w,h), yaw, velocity (vx,vy)
- Loss: Focal Loss (heatmap) + L1 (regression)

---

### Phase 4 — Training & Evaluation

#### [NEW] `src/lightning_module.py`
- Wraps full model in PyTorch Lightning `LightningModule`
- Handles optimizer (AdamW), scheduler (OneCycleLR)
- Combined loss = `λ1 * detection_loss + λ2 * height_map_loss`

#### [NEW] `scripts/train.py`
- Hydra-based config entry point
- Supports single-GPU and multi-GPU (DDP)

#### [NEW] `scripts/evaluate.py`
- Runs model on val/test split
- Computes mAP, ATE, ASE, AOE (nuScenes metrics)

#### [NEW] `scripts/infer_live.py`
- Real-time inference from a live/recorded point cloud stream
- Visualizes BEV map + detections with open3d

---

## Open Questions

> [!IMPORTANT]
> **Q1 — Dataset:** Which dataset will you use for training?
> - `KITTI` (smaller, simpler, widely used for research)
> - `nuScenes` (larger, multi-sweep, includes velocity labels)
> - `Waymo Open Dataset` (largest, most accurate, requires signed agreement)
> - **Or your own custom LiDAR data?**

> [!IMPORTANT]
> **Q2 — Hardware:** Do you have an NVIDIA GPU available?
> - `spconv` requires CUDA. If no GPU → we can use a CPU-only sparse library as fallback (slower).
> - Minimum recommended: RTX 3060 (8 GB VRAM) for training

> [!IMPORTANT]
> **Q3 — Task Scope:** What is the primary output you need?
> - `A` — 2.5D height map only (static environment mapping)
> - `B` — Dynamic object detection only (bounding boxes)
> - `C` — Both A + B (full pipeline as planned) ← recommended

> [!NOTE]
> **Q4 — Real-time requirement:** Is inference speed critical?
> - If yes → we optimize for TensorRT/ONNX export
> - If no → focus on accuracy first

> [!NOTE]
> **Q5 — Dynamic separation method:** Should the system use:
> - Frame differencing (simple, fast, needs 2+ frames)
> - Learned motion segmentation (better accuracy, needs labeled data)

---

## Key Algorithmic Novelty

The **adaptive voxelizer** is what differentiates this from standard 3D detection:

```
Traditional:  uniform 0.1m grid → 1000×1000×40 = 40M voxels (expensive)

Ours:
  - Coarse zone  (0.2m): far/empty areas   →  ~2M  voxels
  - Fine zone    (0.05m): near/dense areas  →  ~5M  voxels  (focused)
  Total:  ~7M voxels — 5× more efficient, same or better detail
```

---

## Verification Plan

### Automated Tests
```bash
python -m pytest tests/test_voxelizer.py -v    # voxelizer correctness
python -m pytest tests/test_model.py -v        # forward pass shapes
python scripts/train.py epochs=1 fast_dev_run=True  # smoke test
```

### Benchmarks
- mAP on KITTI val (Car, Pedestrian, Cyclist)
- FPS measurement: target ≥ 10 Hz on RTX 3060
- Height map RMSE vs. ground truth

### Visual Verification
- BEV visualization of adaptive voxel grid (coarse vs. fine zones)
- Detection box overlay on point cloud
- Height map rendering as grayscale image
