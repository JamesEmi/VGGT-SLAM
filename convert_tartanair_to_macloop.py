"""
Convert VGGT-SLAM poses.txt output + TartanAir V2 ground truth to MAC-Loop protocol format.

TartanAir V2 GT format: pose_lcam_front.txt with N lines of (tx ty tz qx qy qz qw) in NED frame.
VGGT-SLAM output: poses.txt with N lines of (frame_id tx ty tz qx qy qz qw) in camera frame.

Camera frame → NED frame transform is applied to estimated poses.
TartanAir GT is already in NED and does not need transformation.

Usage:
    python convert_tartanair_to_macloop.py \
        --poses_txt output/poses.txt \
        --gt_poses /path/to/Data_easy/P000/pose_lcam_front.txt \
        --output_dir results/VGGT-SLAM@TA_E_P000/
"""

import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

# Camera frame (X=right, Y=down, Z=forward) → NED frame (X=forward, Y=right, Z=down)
# From TartanAir tools: tartanair_tools/evaluation/trajectory_transform.py
CAM2NED = np.array([
    [0, 0, 1, 0],   # cam-Z (forward) → NED-X (forward/north)
    [1, 0, 0, 0],   # cam-X (right)   → NED-Y (right/east)
    [0, 1, 0, 0],   # cam-Y (down)    → NED-Z (down)
    [0, 0, 0, 1]
], dtype=np.float64)

NED2CAM = np.linalg.inv(CAM2NED)


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


def transform_pose_cam_to_ned(pose_mat):
    """Transform a camera-frame pose to NED frame: T_ned = CAM2NED @ T_cam @ inv(CAM2NED)"""
    return CAM2NED @ pose_mat @ NED2CAM


def load_vggt_poses(poses_txt):
    """Load VGGT-SLAM poses.txt: frame_id tx ty tz qx qy qz qw"""
    data = np.loadtxt(poses_txt)
    frame_ids = data[:, 0]
    poses_7 = data[:, 1:]  # tx ty tz qx qy qz qw
    return frame_ids, poses_7


def load_tartanair_gt(gt_path):
    """Load TartanAir V2 ground truth: N lines of (tx ty tz qx qy qz qw)"""
    return np.loadtxt(gt_path)


def convert_poses(poses_txt, gt_path, output_dir, skip_transform=False):
    """
    Full conversion pipeline:
    1. Load VGGT-SLAM poses (camera frame)
    2. Transform estimated poses from camera frame to NED (TartanAir GT convention)
    3. Load TartanAir GT poses (already in NED)
    4. Synthesize 10Hz timestamps
    5. Save as .npy in MAC-SLAM Sandbox format (N, 8): [timestamp, tx, ty, tz, qx, qy, qz, qw]
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load estimated poses
    frame_ids, est_poses_7 = load_vggt_poses(poses_txt)
    N = len(est_poses_7)

    # Synthesize timestamps at 10Hz (frame_id * 100ms in nanoseconds)
    time_ns = (frame_ids * 100_000_000).astype(np.int64)

    # Sort by timestamp (VGGT-SLAM submap overlap/loop closure frames can be out of order)
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

    # Transform estimated poses from camera frame to NED frame
    if not skip_transform:
        transformed = []
        for i in range(N):
            pose_mat = se3_to_mat(*est_poses_7[i])
            pose_ned = transform_pose_cam_to_ned(pose_mat)
            transformed.append(mat_to_se3(pose_ned))
        est_poses_7 = np.array(transformed)
        print(f"Transformed {N} estimated poses from camera frame to NED")

    # Save estimated poses as (N, 8): [timestamp, tx, ty, tz, qx, qy, qz, qw]
    est_path = os.path.join(output_dir, "poses.npy")
    combined_est = np.concatenate([
        time_ns.astype(np.float64).reshape(-1, 1),
        est_poses_7.astype(np.float64)
    ], axis=-1)
    np.save(est_path, combined_est)
    print(f"Saved {N} estimated poses to {est_path}")

    # Load and save ground truth
    if gt_path and os.path.exists(gt_path):
        gt_poses_7 = load_tartanair_gt(gt_path)
        N_gt = len(gt_poses_7)

        gt_time_ns = (np.arange(N_gt) * 100_000_000).astype(np.int64)

        ref_path = os.path.join(output_dir, "ref_poses.npy")
        combined_gt = np.concatenate([
            gt_time_ns.astype(np.float64).reshape(-1, 1),
            gt_poses_7.astype(np.float64)
        ], axis=-1)
        np.save(ref_path, combined_gt)
        print(f"Saved {N_gt} ground truth poses to {ref_path}")

    # Write config.yaml for MAC-SLAM Sandbox compatibility
    seq_name = os.path.basename(output_dir)
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(f"Project: {seq_name}\n")
    print(f"Saved config to {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VGGT-SLAM output to MAC-Loop protocol (TartanAir)")
    parser.add_argument("--poses_txt", type=str, required=True, help="Path to VGGT-SLAM poses.txt")
    parser.add_argument("--gt_poses", type=str, default=None, help="Path to TartanAir pose_lcam_front.txt")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .npy files")
    parser.add_argument("--skip_transform", action="store_true", help="Skip camera→NED coordinate frame transform")
    args = parser.parse_args()

    convert_poses(args.poses_txt, args.gt_poses, args.output_dir, args.skip_transform)
