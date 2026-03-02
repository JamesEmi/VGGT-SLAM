"""
Convert VGGT-SLAM poses.txt output + VBR ground truth to MAC-Loop protocol format.

VBR GT format: {seq}_gt.txt with header '#timestamp tx ty tz qx qy qz qw'
  - timestamps in seconds, poses in body/lidar frame (lidar T_b = identity)
  - quat order: qx qy qz qw

VGGT-SLAM output: poses.txt with N lines of (frame_id tx ty tz qx qy qz qw)
  - frame_id is sequential index into camera_left images
  - poses are in camera frame (OpenCV: X=right, Y=down, Z=forward)

Camera→body transform is applied using T_b from vbr_calib.yaml.
GT poses are kept in body frame (no additional transform needed).

Usage:
    python convert_vbr_to_macloop.py \
        --poses_txt output/poses.txt \
        --vbr_seq /media/airlab-storage/datasets/VBRome/campus_train0 \
        --output_dir results/VGGT-SLAM@campus_train0/
"""

import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

# VBR cam_l T_b (body-to-camera, same across all VBR sequences)
T_B_CAM_L = np.array([
    [0.005617112780309785, -0.0012881145325502978, 0.9999832905520013, 0.07073856167431194],
    [-0.9999833070153535, -0.0013285236613379636, 0.005615378227348316, 0.23435089305558293],
    [0.001321217132003304, -0.9999982616209818, -0.0012955402899334433, -0.6660491439765341],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float64)

T_CAM_L_B = np.linalg.inv(T_B_CAM_L)


def se3_to_mat(tx, ty, tz, qx, qy, qz, qw):
    """Convert translation + quaternion (x,y,z,w) to 4x4 matrix."""
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    mat[:3, 3] = [tx, ty, tz]
    return mat


def mat_to_se3(mat):
    """Convert 4x4 matrix to (tx, ty, tz, qx, qy, qz, qw)."""
    t = mat[:3, 3]
    q = R.from_matrix(mat[:3, :3]).as_quat()  # returns [x, y, z, w]
    return np.array([t[0], t[1], t[2], q[0], q[1], q[2], q[3]])


def transform_pose_cam_to_body(pose_mat):
    """
    Transform a world-from-camera pose to body frame.
    T_world_body = T_world_cam @ inv(T_b)
    where T_b is body-to-camera, so inv(T_b) = camera-to-body.
    """
    return pose_mat @ T_CAM_L_B


def parse_vbr_timestamps(timestamps_txt):
    """
    Parse VBR camera timestamps.txt (ISO 8601 format) to nanoseconds.
    Format: 1970-01-01T00:06:06.660458140
    """
    timestamps_ns = []
    with open(timestamps_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse time part: HH:MM:SS.nnnnnnnnn
            time_part = line.split('T')[1]
            h, m, s = time_part.split(':')
            total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
            timestamps_ns.append(int(total_seconds * 1_000_000_000))
    return np.array(timestamps_ns, dtype=np.int64)


def load_vggt_poses(poses_txt):
    """Load VGGT-SLAM poses.txt: frame_id tx ty tz qx qy qz qw"""
    data = np.loadtxt(poses_txt)
    frame_ids = data[:, 0].astype(int)
    poses_7 = data[:, 1:]  # tx ty tz qx qy qz qw
    return frame_ids, poses_7


def load_vbr_gt(gt_path):
    """
    Load VBR ground truth: #timestamp tx ty tz qx qy qz qw
    Returns timestamps in nanoseconds and poses as (N, 7).
    """
    data = np.loadtxt(gt_path, comments='#')
    time_s = data[:, 0]
    time_ns = (time_s * 1_000_000_000).astype(np.int64)
    poses_7 = data[:, 1:]  # tx ty tz qx qy qz qw
    return time_ns, poses_7


def convert_poses(poses_txt, vbr_seq_path, output_dir, skip_transform=False):
    """
    Full conversion pipeline:
    1. Load VGGT-SLAM poses (camera frame)
    2. Map frame_ids to real camera timestamps
    3. Transform estimated poses from camera frame to body frame
    4. Load VBR GT poses (already in body frame)
    5. Save as .npy in MAC-SLAM Sandbox format (N, 8): [timestamp, tx, ty, tz, qx, qy, qz, qw]
    """
    os.makedirs(output_dir, exist_ok=True)

    seq_name = os.path.basename(vbr_seq_path)

    # Load camera timestamps for frame_id → real timestamp mapping
    timestamps_txt = os.path.join(vbr_seq_path, "camera_left", "timestamps.txt")
    cam_timestamps_ns = parse_vbr_timestamps(timestamps_txt)

    # Load estimated poses
    frame_ids, est_poses_7 = load_vggt_poses(poses_txt)
    N = len(est_poses_7)

    # Map frame_ids to real timestamps
    valid_mask = frame_ids < len(cam_timestamps_ns)
    if not np.all(valid_mask):
        n_invalid = np.sum(~valid_mask)
        print(f"WARNING: {n_invalid} frame_ids exceed camera timestamp count ({len(cam_timestamps_ns)}), discarding")
        frame_ids = frame_ids[valid_mask]
        est_poses_7 = est_poses_7[valid_mask]
        N = len(est_poses_7)

    time_ns = cam_timestamps_ns[frame_ids]

    # Sort by timestamp (submap overlap/loop closure frames can be out of order)
    sort_idx = np.argsort(time_ns)
    time_ns = time_ns[sort_idx]
    est_poses_7 = est_poses_7[sort_idx]

    # Deduplicate timestamps at submap boundaries (keep later occurrence = better optimized)
    _, idx = np.unique(time_ns[::-1], return_index=True)
    keep = np.sort(len(time_ns) - 1 - idx)
    if len(keep) < len(time_ns):
        print(f"Deduplicating: {len(time_ns)} -> {len(keep)} poses (removed {len(time_ns) - len(keep)} duplicates)")
        time_ns = time_ns[keep]
        est_poses_7 = est_poses_7[keep]
        N = len(est_poses_7)

    # Transform estimated poses from camera frame to body frame
    if not skip_transform:
        transformed = []
        for i in range(N):
            pose_mat = se3_to_mat(*est_poses_7[i])
            pose_body = transform_pose_cam_to_body(pose_mat)
            transformed.append(mat_to_se3(pose_body))
        est_poses_7 = np.array(transformed)
        print(f"Transformed {N} estimated poses from camera frame to body frame")

    # Save estimated poses as (N, 8): [timestamp, tx, ty, tz, qx, qy, qz, qw]
    est_path = os.path.join(output_dir, "poses.npy")
    combined_est = np.concatenate([
        time_ns.astype(np.float64).reshape(-1, 1),
        est_poses_7.astype(np.float64)
    ], axis=-1)
    np.save(est_path, combined_est)
    print(f"Saved {N} estimated poses to {est_path}")

    # Load and save ground truth (already in body frame)
    gt_path = os.path.join(vbr_seq_path, f"{seq_name}_gt.txt")
    if os.path.exists(gt_path):
        gt_time_ns, gt_poses_7 = load_vbr_gt(gt_path)
        N_gt = len(gt_poses_7)

        ref_path = os.path.join(output_dir, "ref_poses.npy")
        combined_gt = np.concatenate([
            gt_time_ns.astype(np.float64).reshape(-1, 1),
            gt_poses_7.astype(np.float64)
        ], axis=-1)
        np.save(ref_path, combined_gt)
        print(f"Saved {N_gt} ground truth poses to {ref_path}")
    else:
        print(f"WARNING: GT file not found: {gt_path}")

    # Write config.yaml for MAC-SLAM Sandbox compatibility
    sandbox_name = os.path.basename(output_dir)
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(f"Project: {sandbox_name}\n")
    print(f"Saved config to {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VGGT-SLAM output to MAC-Loop protocol (VBR)")
    parser.add_argument("--poses_txt", type=str, required=True, help="Path to VGGT-SLAM poses.txt")
    parser.add_argument("--vbr_seq", type=str, required=True,
                        help="Path to VBR sequence root (e.g. .../VBRome/campus_train0)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .npy files")
    parser.add_argument("--skip_transform", action="store_true",
                        help="Skip camera→body coordinate frame transform")
    args = parser.parse_args()

    convert_poses(args.poses_txt, args.vbr_seq, args.output_dir, args.skip_transform)
