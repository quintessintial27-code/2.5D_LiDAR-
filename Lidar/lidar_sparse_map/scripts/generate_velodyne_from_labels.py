"""
generate_velodyne_from_labels.py
─────────────────────────────────
Generates realistic synthetic LiDAR .bin point clouds using your
REAL KITTI label_2 and calib files. No 29 GB download needed!

Strategy per frame:
  1. Read the real label_2 (boxes: Car, Pedestrian, Cyclist)
  2. Read the real calib (so coordinate transforms are accurate)
  3. Generate a realistic ground plane (flat + slight slope noise)
  4. For each labelled object: sprinkle realistic LiDAR returns
     inside the 3D bounding box (density proportional to object size)
  5. Add random background clutter points
  6. Save as float32 .bin file  (x, y, z, intensity × N)

Output goes to:
  data/kitti/training/velodyne/*.bin

Usage:
  python scripts/generate_velodyne_from_labels.py
  python scripts/generate_velodyne_from_labels.py --max_frames 100
  python scripts/generate_velodyne_from_labels.py --data_root path/to/kitti
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ─── Point cloud parameters ───────────────────────────────────────────────────
MAX_RANGE   = 50.0      # metres (Velodyne HDL-64E max ~120m, but we cap at 50 for density)
GROUND_PTS  = 8_000     # background ground plane points per frame
CLUTTER_PTS = 2_000     # random clutter (walls, trees, etc.)
LIDAR_HEIGHT = 1.73     # sensor height above ground (KITTI spec)

# Points per object type (approximate returns from a real 64-beam LiDAR)
OBJ_PTS = {
    "Car":        350,
    "Van":        400,
    "Pedestrian": 80,
    "Person_sitting": 60,
    "Cyclist":    120,
}

RNG = np.random.default_rng(42)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_label(path: Path) -> list:
    objects = []
    if not path.exists():
        return objects
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            cls = parts[0]
            if cls in ("DontCare", "Misc", "Tram"):
                continue
            objects.append({
                "cls":   cls,
                "h":     float(parts[8]),
                "w":     float(parts[9]),
                "l":     float(parts[10]),
                "x_cam": float(parts[11]),
                "y_cam": float(parts[12]),
                "z_cam": float(parts[13]),
                "ry":    float(parts[14]),
            })
    return objects


def parse_calib(path: Path) -> dict:
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.strip().split(":", 1)
            data[k.strip()] = np.array([float(x) for x in v.split()])
    if "Tr_velo_to_cam" in data:
        data["Tr_velo_to_cam"] = data["Tr_velo_to_cam"].reshape(3, 4)
    if "R0_rect" in data:
        data["R0_rect"] = data["R0_rect"].reshape(3, 3)
    return data


def cam_to_lidar_box(obj: dict, calib: dict):
    """Convert KITTI camera-frame box centre → LiDAR frame."""
    Tr = calib.get("Tr_velo_to_cam", np.eye(3, 4))
    R0 = calib.get("R0_rect", np.eye(3))

    Tr_full = np.eye(4)
    Tr_full[:3, :] = Tr
    T_cam_to_lidar = np.linalg.inv(Tr_full)
    R0_inv = np.linalg.inv(R0)

    p_cam  = np.array([obj["x_cam"], obj["y_cam"], obj["z_cam"], 1.0])
    p_rect = np.concatenate([R0_inv @ p_cam[:3], [1.0]])
    p_lidar = T_cam_to_lidar @ p_rect
    yaw_lidar = -obj["ry"] - np.pi / 2.0
    return float(p_lidar[0]), float(p_lidar[1]), float(p_lidar[2]), yaw_lidar


def points_in_box(cx, cy, cz, l, w, h, yaw, n_pts, cls) -> np.ndarray:
    """
    Generate n_pts random LiDAR returns inside a 3D oriented bounding box.
    Points cluster on the surfaces visible to the sensor (front / sides).
    """
    # Uniform interior sample
    lx = RNG.uniform(-l / 2, l / 2, n_pts)
    ly = RNG.uniform(-w / 2, w / 2, n_pts)
    lz = RNG.uniform(0, h, n_pts)        # z from bottom (ground = 0 inside box)

    # Project ~70% of points onto the front face (most visible surface)
    n_front = int(n_pts * 0.45)
    lx[:n_front] = l / 2                 # front face
    ly[:n_front] = RNG.uniform(-w / 2, w / 2, n_front)
    lz[:n_front] = RNG.uniform(0, h, n_front)

    n_side = int(n_pts * 0.25)
    ly[n_front:n_front + n_side] = RNG.choice([-w / 2, w / 2], n_side)  # side faces

    # Add Gaussian noise (simulate beam scatter)
    lx += RNG.normal(0, 0.02, n_pts)
    ly += RNG.normal(0, 0.02, n_pts)
    lz += RNG.normal(0, 0.01, n_pts)

    # Rotate by yaw around Z axis
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rx = cos_y * lx - sin_y * ly
    ry = sin_y * lx + cos_y * ly

    # Translate to world position
    pts_x = rx + cx
    pts_y = ry + cy
    pts_z = lz + cz - h / 2   # box centre z, correct for bottom-of-box offset

    # Intensity: bright for metal objects, dim for pedestrians
    intensity_mean = {
        "Car": 0.6, "Van": 0.6,
        "Pedestrian": 0.25, "Person_sitting": 0.2,
        "Cyclist": 0.35,
    }.get(cls, 0.4)
    intensity = RNG.uniform(
        max(0, intensity_mean - 0.15),
        min(1.0, intensity_mean + 0.15),
        n_pts
    ).astype(np.float32)

    pts = np.stack([
        pts_x.astype(np.float32),
        pts_y.astype(np.float32),
        pts_z.astype(np.float32),
        intensity,
    ], axis=1)   # [N, 4]
    return pts


def generate_ground(n: int = GROUND_PTS) -> np.ndarray:
    """
    Generate a realistic ground plane with Velodyne beam pattern.
    Beams emanate from sensor origin in concentric rings.
    """
    # 64 vertical beams, each sweeping 360°
    n_beams = 64
    n_per_beam = n // n_beams

    all_pts = []
    for beam_idx in range(n_beams):
        # Vertical angle: −24.9° to +2° (Velodyne HDL-64E spec)
        vert_deg = -24.9 + beam_idx * (26.9 / 63.0)
        vert_rad = np.deg2rad(vert_deg)

        # Horizontal sweep
        horiz = RNG.uniform(-np.pi, np.pi, n_per_beam)
        # Ground hit distance for this beam angle
        if vert_rad >= 0:
            continue   # upward beams miss the ground
        dist = LIDAR_HEIGHT / np.abs(np.tan(vert_rad))
        dist = np.clip(dist, 0, MAX_RANGE)

        x = dist * np.cos(horiz)
        y = dist * np.sin(horiz)
        z_base = -LIDAR_HEIGHT   # ground level
        z = z_base + RNG.normal(0, 0.03, n_per_beam)  # slight roughness

        intensity = RNG.uniform(0.05, 0.3, n_per_beam).astype(np.float32)
        pts = np.stack([
            x.astype(np.float32),
            y.astype(np.float32),
            z.astype(np.float32),
            intensity,
        ], axis=1)
        all_pts.append(pts)

    return np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 4), np.float32)


def generate_clutter(n: int = CLUTTER_PTS) -> np.ndarray:
    """Random background points: walls, vegetation, parked cars beyond range."""
    # Scatter points in a ring at far distance
    dist  = RNG.uniform(20, MAX_RANGE, n)
    angle = RNG.uniform(-np.pi, np.pi, n)
    x     = (dist * np.cos(angle)).astype(np.float32)
    y     = (dist * np.sin(angle)).astype(np.float32)
    z     = RNG.uniform(-LIDAR_HEIGHT, 2.0, n).astype(np.float32)
    intensity = RNG.uniform(0.05, 0.4, n).astype(np.float32)
    return np.stack([x, y, z, intensity], axis=1)


def generate_frame(label_path: Path, calib_path: Path) -> np.ndarray:
    """Generate a full synthetic LiDAR frame from real labels + calib."""
    objects = parse_label(label_path)
    calib   = parse_calib(calib_path)

    parts = [generate_ground(), generate_clutter()]

    for obj in objects:
        n_pts = OBJ_PTS.get(obj["cls"], 100)
        try:
            cx, cy, cz, yaw = cam_to_lidar_box(obj, calib)
        except np.linalg.LinAlgError:
            continue

        # Only generate if inside LiDAR range
        if np.sqrt(cx**2 + cy**2) > MAX_RANGE:
            continue

        obj_pts = points_in_box(
            cx, cy, cz,
            l=obj["l"], w=obj["w"], h=obj["h"],
            yaw=yaw, n_pts=n_pts, cls=obj["cls"]
        )
        parts.append(obj_pts)

    return np.concatenate(parts, axis=0)   # [N, 4] float32


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str, default="data/kitti",
                        help="Root of KITTI data directory")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit to first N frames (default: all label files found)")
    args = parser.parse_args()

    root      = Path(args.data_root)
    label_dir = root / "training" / "label_2"
    calib_dir = root / "training" / "calib"
    velo_dir  = root / "training" / "velodyne"
    velo_dir.mkdir(parents=True, exist_ok=True)

    if not label_dir.exists():
        print(f"[ERROR] label_2 directory not found: {label_dir}")
        print("        Copy it from: C:\\Users\\HP\\Downloads\\data_object_label_2\\training\\label_2")
        return

    if not calib_dir.exists():
        print(f"[ERROR] calib directory not found: {calib_dir}")
        print("        Extract data_object_calib.zip and copy calib folder into data/kitti/training/")
        return

    label_files = sorted(label_dir.glob("*.txt"))
    if args.max_frames:
        label_files = label_files[:args.max_frames]

    print(f"\nGenerating synthetic Velodyne .bin files from {len(label_files)} real labels...")
    print(f"Output -> {velo_dir}\n")

    n_skipped   = 0
    total_pts   = 0

    for lf in tqdm(label_files, desc="Generating"):
        stem = lf.stem   # e.g. "000000"
        out_path = velo_dir / f"{stem}.bin"

        if out_path.exists():
            continue   # don't regenerate

        calib_path = calib_dir / f"{stem}.txt"
        pts = generate_frame(lf, calib_path)
        total_pts += pts.shape[0]

        pts.astype(np.float32).tofile(out_path)

    existing = list(velo_dir.glob("*.bin"))
    sizes_mb = sum(f.stat().st_size for f in existing) / (1024 ** 2)

    print(f"\n{'='*55}")
    print(f"  DONE!")
    print(f"  Frames generated : {len(existing)}")
    print(f"  Total disk usage : {sizes_mb:.1f} MB  (vs 29,000 MB for real dataset)")
    print(f"  Avg pts/frame    : {total_pts // max(len(label_files), 1):,}")
    print(f"{'='*55}")
    print(f"\nYou can now run:")
    print(f"  python scripts/train.py")
    print(f"  python scripts/infer_live.py --dir {velo_dir}")


if __name__ == "__main__":
    main()
