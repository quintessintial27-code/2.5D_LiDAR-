"""
kitti_dataset.py
─────────────────
KITTI 3D Object Detection Dataset loader for PyTorch.

Expected directory structure:
  data/kitti/
    training/
      velodyne/   ← 000000.bin ... 007480.bin
      label_2/    ← 000000.txt ... 007480.txt
      calib/      ← 000000.txt ... 007480.txt
    ImageSets/
      train.txt   ← list of frame indices for training
      val.txt     ← list of frame indices for validation

KITTI label format (per line):
  class truncated occluded alpha x1 y1 x2 y2 h w l x y z ry

For 3D detection we use: class, h, w, l, x, y, z, ry
  (3D bounding box in camera coordinates — we convert to LiDAR frame)

Frame differencing:
  For each sample, we also load the previous frame's point cloud (if it
  exists in the same sequence) to compute dynamic motion masks.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..voxelizer.point_preprocessor import PointPreprocessor
from ..voxelizer.adaptive_voxelizer import AdaptiveVoxelizer
from ..utils.box_utils import boxes_to_heatmap, get_gaussian_radius


# ─── KITTI Label Parser ────────────────────────────────────────────────────────

KITTI_CLASS_MAP = {
    "Car": 0,
    "Van": 0,
    "Pedestrian": 1,
    "Person_sitting": 1,
    "Cyclist": 2,
}
IGNORE_CLASSES = {"Truck", "Tram", "Misc", "DontCare"}


def parse_kitti_label(label_path: str) -> List[Dict]:
    """Parse a KITTI label .txt file → list of object dicts (camera coords)."""
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            cls = parts[0]
            if cls in IGNORE_CLASSES:
                continue
            cls_id = KITTI_CLASS_MAP.get(cls, -1)
            if cls_id < 0:
                continue
            objects.append({
                "class_id": cls_id,
                "class_name": cls,
                "truncated": float(parts[1]),
                "occluded": int(parts[2]),
                "h": float(parts[8]),
                "w": float(parts[9]),
                "l": float(parts[10]),
                "x_cam": float(parts[11]),   # 3D centre in camera frame
                "y_cam": float(parts[12]),
                "z_cam": float(parts[13]),
                "ry": float(parts[14]),       # rotation around Y axis (camera)
            })
    return objects


def parse_kitti_calib(calib_path: str) -> Dict[str, np.ndarray]:
    """Parse KITTI calibration file → dict of matrices."""
    data = {}
    with open(calib_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, vals = line.strip().split(":", 1)
            data[key.strip()] = np.array([float(v) for v in vals.split()])
    # Reshape standard matrices
    if "Tr_velo_to_cam" in data:
        data["Tr_velo_to_cam"] = data["Tr_velo_to_cam"].reshape(3, 4)
    if "R0_rect" in data:
        data["R0_rect"] = data["R0_rect"].reshape(3, 3)
    return data


def cam_to_lidar(objects: List[Dict], calib: Dict) -> List[Dict]:
    """
    Convert 3D bounding box centres from camera frame to LiDAR frame.
    KITTI: Camera (x=right, y=down, z=forward) → LiDAR (x=forward, y=left, z=up)
    """
    Tr = calib.get("Tr_velo_to_cam", np.eye(3, 4))   # [3,4]
    R0 = calib.get("R0_rect", np.eye(3))              # [3,3]

    # Full 4×4 transform from LiDAR to Camera
    Tr_full = np.eye(4)
    Tr_full[:3, :] = Tr
    T_cam_to_lidar = np.linalg.inv(Tr_full)
    R0_inv = np.linalg.inv(R0)

    result = []
    for obj in objects:
        # Camera-frame centre
        p_cam = np.array([obj["x_cam"], obj["y_cam"], obj["z_cam"], 1.0])
        # Undo R0 rectification
        p_rect = np.concatenate([R0_inv @ p_cam[:3], [1.0]])
        # Transform to LiDAR frame
        p_lidar = T_cam_to_lidar @ p_rect
        # Yaw: camera ry → lidar yaw  (ry in camera = -yaw in lidar approx.)
        yaw_lidar = -obj["ry"] - np.pi / 2
        obj_lidar = dict(obj)
        obj_lidar["x"] = float(p_lidar[0])
        obj_lidar["y"] = float(p_lidar[1])
        obj_lidar["z"] = float(p_lidar[2])
        obj_lidar["yaw"] = float(yaw_lidar)
        result.append(obj_lidar)
    return result


# ─── Dataset Class ────────────────────────────────────────────────────────────

class KITTIDataset(Dataset):
    """
    PyTorch Dataset for KITTI 3D Object Detection.

    Each sample returns:
        sparse_tensor   : SparseTensor (from AdaptiveVoxelizer)
        gt_boxes        : [N, 8] (x,y,z,l,w,h,yaw,class_id)
        gt_heatmap      : [n_cls, H_bev, W_bev]
        gt_reg          : [10, H_bev, W_bev]
        gt_pos_mask     : [H_bev, W_bev]
        gt_height_map   : [1, H_bev, W_bev]
        frame_id        : int
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",        # 'train' | 'val'
        pc_range: Tuple = (-40., -40., -3., 40., 40., 3.),
        coarse_voxel: float = 0.20,
        fine_voxel: float = 0.10,
        bev_h: int = 400,            # BEV map height (matches 80m / 0.20m)
        bev_w: int = 400,
        num_classes: int = 3,
        min_pts: int = 5,
        use_augment: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root_dir)
        self.split = split
        self.pc_range = pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_classes = num_classes
        self.min_pts = min_pts
        self.use_augment = use_augment and (split == "train")

        # Load frame index list
        split_file = self.root / "ImageSets" / f"{split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                self.frame_ids = [line.strip() for line in f if line.strip()]
        else:
            # Fallback: discover from velodyne dir
            velo_dir = self.root / "training" / "velodyne"
            self.frame_ids = sorted(
                p.stem for p in velo_dir.glob("*.bin")
            ) if velo_dir.exists() else []

        # Sub-modules
        self.preprocessor = PointPreprocessor(
            x_range=(pc_range[0], pc_range[3]),
            y_range=(pc_range[1], pc_range[4]),
            z_range=(pc_range[2], pc_range[5]),
        )
        self.voxelizer = AdaptiveVoxelizer(
            pc_range=pc_range,
            coarse_size=coarse_voxel,
            fine_size=fine_voxel,
        )

        print(f"[KITTIDataset] split={split}  frames={len(self.frame_ids)}")

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, idx: int) -> Dict:
        frame_id = self.frame_ids[idx]
        prev_frame_id = self.frame_ids[idx - 1] if idx > 0 else None

        # ── Load point cloud ──────────────────────────────────────────────────
        velo_path = self.root / "training" / "velodyne" / f"{frame_id}.bin"
        pts_raw = PointPreprocessor.load_kitti_bin(str(velo_path))
        pts = self.preprocessor(pts_raw)

        # ── Load previous frame for differencing ──────────────────────────────
        prev_pts = None
        if prev_frame_id is not None:
            prev_path = self.root / "training" / "velodyne" / f"{prev_frame_id}.bin"
            if prev_path.exists():
                prev_raw = PointPreprocessor.load_kitti_bin(str(prev_path))
                prev_pts = self.preprocessor(prev_raw)

        # ── Data augmentation ─────────────────────────────────────────────────
        if self.use_augment:
            pts, prev_pts = self._augment(pts, prev_pts)

        # ── Voxelization → SparseTensor ───────────────────────────────────────
        sparse = self.voxelizer.voxelize(pts, batch_idx=0, prev_points=prev_pts)

        # ── Load labels ───────────────────────────────────────────────────────
        label_path = self.root / "training" / "label_2" / f"{frame_id}.txt"
        calib_path = self.root / "training" / "calib" / f"{frame_id}.txt"
        gt_boxes = self._load_boxes(str(label_path), str(calib_path), pts)

        # ── Build GT maps ─────────────────────────────────────────────────────
        gt_heatmap, gt_reg, gt_pos_mask = self._build_gt_maps(gt_boxes)
        gt_height = self._build_height_map(pts)

        return {
            "sparse_tensor": sparse,
            "gt_boxes": torch.from_numpy(gt_boxes).float() if len(gt_boxes) else torch.zeros(0, 8),
            "gt_heatmap": gt_heatmap,
            "gt_reg": gt_reg,
            "gt_pos_mask": gt_pos_mask,
            "gt_height_map": gt_height,
            "frame_id": int(frame_id),
        }

    # ─── Label loading ────────────────────────────────────────────────────────

    def _load_boxes(
        self, label_path: str, calib_path: str, pts: np.ndarray
    ) -> np.ndarray:
        """Returns [N, 8] array: (x, y, z, l, w, h, yaw, class_id)."""
        if not os.path.exists(label_path):
            return np.zeros((0, 8), dtype=np.float32)

        objects = parse_kitti_label(label_path)
        if not objects:
            return np.zeros((0, 8), dtype=np.float32)

        calib = parse_kitti_calib(calib_path)
        objects = cam_to_lidar(objects, calib)

        rows = []
        for obj in objects:
            x, y, z = obj["x"], obj["y"], obj["z"]
            l, w, h = obj["l"], obj["w"], obj["h"]
            yaw = obj["yaw"]
            cls = obj["class_id"]

            # Filter: must be in range and have enough points
            if not (
                self.pc_range[0] < x < self.pc_range[3] and
                self.pc_range[1] < y < self.pc_range[4]
            ):
                continue
            rows.append([x, y, z, l, w, h, yaw, cls])

        return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 8), dtype=np.float32)

    # ─── GT map generation ────────────────────────────────────────────────────

    def _build_gt_maps(
        self, gt_boxes: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build CenterPoint-style GT maps:
          gt_heatmap  [n_cls, H, W]  — Gaussian blobs at object centres
          gt_reg      [10,   H, W]   — regression targets at centres
          gt_pos_mask [H, W]         — bool, True at centres
        """
        H, W = self.bev_h, self.bev_w
        heatmap = torch.zeros(self.num_classes, H, W)
        reg = torch.zeros(10, H, W)
        pos_mask = torch.zeros(H, W, dtype=torch.bool)

        x_min, y_min = self.pc_range[0], self.pc_range[1]
        vsize = (self.pc_range[3] - self.pc_range[0]) / W

        for box in gt_boxes:
            x, y, z, l, w, h, yaw, cls_id = box
            cls_id = int(cls_id)

            # BEV centre in grid coords
            cx = int((x - x_min) / vsize)
            cy = int((y - y_min) / vsize)
            if not (0 <= cx < W and 0 <= cy < H):
                continue

            # Gaussian radius based on box size
            radius = get_gaussian_radius(l / vsize, w / vsize)
            radius = max(0, int(radius))

            # Draw Gaussian on heatmap
            heatmap[cls_id] = _draw_gaussian(heatmap[cls_id], cx, cy, radius)

            # Regression targets at this location
            pos_mask[cy, cx] = True
            dx = (x - x_min) / vsize - cx    # sub-cell offset
            dy = (y - y_min) / vsize - cy
            reg[0, cy, cx] = float(dx)
            reg[1, cy, cx] = float(dy)
            reg[2, cy, cx] = float(z - self.pc_range[2])
            reg[3, cy, cx] = float(np.log(max(l, 0.01)))
            reg[4, cy, cx] = float(np.log(max(w, 0.01)))
            reg[5, cy, cx] = float(np.log(max(h, 0.01)))
            reg[6, cy, cx] = float(np.sin(yaw))
            reg[7, cy, cx] = float(np.cos(yaw))
            reg[8, cy, cx] = 0.0   # velocity (unknown from single-frame KITTI)
            reg[9, cy, cx] = 0.0

        return heatmap, reg, pos_mask

    def _build_height_map(self, pts: np.ndarray) -> torch.Tensor:
        """Build height map [1, H, W] from point cloud Z values."""
        H, W = self.bev_h, self.bev_w
        height = torch.zeros(1, H, W)
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        vsize = (self.pc_range[3] - self.pc_range[0]) / W

        xi = np.clip(((pts[:, 0] - x_min) / vsize).astype(int), 0, W - 1)
        yi = np.clip(((pts[:, 1] - y_min) / vsize).astype(int), 0, H - 1)

        for i in range(pts.shape[0]):
            z_val = float(pts[i, 2])
            if z_val > height[0, yi[i], xi[i]]:
                height[0, yi[i], xi[i]] = z_val

        return height

    # ─── Data augmentation ────────────────────────────────────────────────────

    def _augment(
        self,
        pts: np.ndarray,
        prev_pts: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Simple random flip + rotation augmentation."""
        # Random flip along X axis
        if np.random.random() > 0.5:
            pts[:, 1] = -pts[:, 1]
            if prev_pts is not None:
                prev_pts[:, 1] = -prev_pts[:, 1]

        # Random rotation ±22.5°
        angle = np.random.uniform(-0.3927, 0.3927)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        pts[:, :2] = pts[:, :2] @ R.T
        if prev_pts is not None:
            prev_pts[:, :2] = prev_pts[:, :2] @ R.T

        return pts, prev_pts


# ─── Gaussian drawing utility ─────────────────────────────────────────────────

def _draw_gaussian(
    heatmap: torch.Tensor,
    cx: int,
    cy: int,
    radius: int,
) -> torch.Tensor:
    """Draw a 2D Gaussian blob centred at (cx, cy) on the heatmap."""
    H, W = heatmap.shape
    diameter = 2 * radius + 1
    sigma = diameter / 6.0

    y_range = torch.arange(0, diameter, dtype=torch.float32) - radius
    x_range = torch.arange(0, diameter, dtype=torch.float32) - radius
    yg, xg = torch.meshgrid(y_range, x_range, indexing="ij")
    gaussian = torch.exp(-(xg ** 2 + yg ** 2) / (2 * sigma ** 2))
    gaussian[gaussian < torch.finfo(gaussian.dtype).eps * gaussian.max()] = 0

    # Clip to valid heatmap region
    y1 = int(cy) - radius
    y2 = int(cy) + radius + 1
    x1 = int(cx) - radius
    x2 = int(cx) + radius + 1

    y1c = max(0, -y1)
    x1c = max(0, -x1)
    y2c = min(diameter, diameter - (y2 - H))
    x2c = min(diameter, diameter - (x2 - W))
    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(H, y2)
    x2 = min(W, x2)

    if y2 <= y1 or x2 <= x1 or y2c <= y1c or x2c <= x1c:
        return heatmap

    masked_heatmap = heatmap[y1:y2, x1:x2]
    masked_gaussian = gaussian[y1c:y2c, x1c:x2c]
    torch.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return heatmap
