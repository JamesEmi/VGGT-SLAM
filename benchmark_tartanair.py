#!/usr/bin/env python3
"""
Benchmark VGGT-SLAM on TartanAir V2 sequences and output results in MAC-Loop protocol format.

Usage:
    # Run all TartanAir V2 sequences:
    python benchmark_tartanair.py --tartanair_root /media/airlab-storage/datasets/TartanAir/tartanair_v2_envs_test

    # Run specific sequences:
    python benchmark_tartanair.py --tartanair_root /media/airlab-storage/datasets/TartanAir/tartanair_v2_envs_test --sequences TA_E_P000 TA_H_P003

    # Convert only (skip SLAM, just convert existing poses.txt):
    python benchmark_tartanair.py --tartanair_root /media/airlab-storage/datasets/TartanAir/tartanair_v2_envs_test --convert_only --raw_results_dir ./raw_output

    # Custom SLAM args:
    python benchmark_tartanair.py --tartanair_root /media/airlab-storage/datasets/TartanAir/tartanair_v2_envs_test --submap_size 16 --min_disparity 50
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
from datetime import datetime

# Generate sequence dict: {"TA_E_P000": ("Data_easy", "P000"), ...}
TARTANAIR_SEQUENCES = {}
for diff in ["easy", "hard"]:
    for i in range(8):
        short = f"TA_{diff[0].upper()}_P{i:03d}"
        TARTANAIR_SEQUENCES[short] = (f"Data_{diff}", f"P{i:03d}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_vggt_slam(image_folder, save_dir, slam_args):
    """Run VGGT-SLAM on a single sequence."""
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "run_vggt_slam.py"),
        "--image_folder", image_folder,
        "--log_results",
        "--skip_dense_log",
        "--save_dir", save_dir,
        "--log_path", "poses.txt",
        "--submap_size", str(slam_args.submap_size),
        "--min_disparity", str(slam_args.min_disparity),
        "--conf_threshold", str(slam_args.conf_threshold),
        "--lc_thres", str(slam_args.lc_thres),
        "--max_loops", str(slam_args.max_loops),
    ]
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"WARNING: VGGT-SLAM exited with code {result.returncode}")
        return False
    return True


def convert_poses(poses_txt, gt_poses_path, output_dir):
    """Convert VGGT-SLAM output to MAC-Loop protocol (TartanAir)."""
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "convert_tartanair_to_macloop.py"),
        "--poses_txt", poses_txt,
        "--gt_poses", gt_poses_path,
        "--output_dir", output_dir,
    ]
    print(f"Converting: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Benchmark VGGT-SLAM on TartanAir V2")
    parser.add_argument("--tartanair_root", type=str, required=True,
                        help="Root directory of TartanAir V2 (contains Data_easy/ and Data_hard/)")
    parser.add_argument("--sequences", nargs="+", default=None,
                        help="Specific sequences to run (e.g. TA_E_P000 TA_H_P003). Default: all")
    parser.add_argument("--results_root", type=str, default="/mnt/data/anyslam/macloop/vggt-slam-results",
                        help="Root directory for MAC-Loop protocol results")
    parser.add_argument("--convert_only", action="store_true",
                        help="Skip SLAM, only convert existing poses.txt files")
    parser.add_argument("--raw_results_dir", type=str, default=None,
                        help="Directory containing raw VGGT-SLAM outputs (for --convert_only)")

    # SLAM hyperparameters
    parser.add_argument("--submap_size", type=int, default=16)
    parser.add_argument("--min_disparity", type=float, default=50)
    parser.add_argument("--conf_threshold", type=float, default=25.0)
    parser.add_argument("--lc_thres", type=float, default=0.95)
    parser.add_argument("--max_loops", type=int, default=1)

    args = parser.parse_args()

    # Determine which sequences to run
    if args.sequences:
        sequences = {k: TARTANAIR_SEQUENCES[k] for k in args.sequences if k in TARTANAIR_SEQUENCES}
        unknown = [k for k in args.sequences if k not in TARTANAIR_SEQUENCES]
        if unknown:
            print(f"WARNING: Unknown sequences: {unknown}")
            print(f"Available: {list(TARTANAIR_SEQUENCES.keys())}")
    else:
        sequences = TARTANAIR_SEQUENCES

    # Create timestamp directory
    timestamp = datetime.now().strftime("%m_%d_%H%M%S")
    protocol_dir = os.path.join(args.results_root, "VGGT-SLAM", timestamp)
    os.makedirs(protocol_dir, exist_ok=True)

    # Raw output directory (where VGGT-SLAM writes poses.txt)
    if args.convert_only and args.raw_results_dir:
        raw_root = args.raw_results_dir
    else:
        raw_root = os.path.join(args.results_root, "raw_output", timestamp)

    print(f"TartanAir root:  {args.tartanair_root}")
    print(f"Sequences:       {list(sequences.keys())}")
    print(f"Protocol output: {protocol_dir}")
    print(f"Raw output:      {raw_root}")
    print()

    results_summary = []
    total_start = time.time()

    for seq_short, (difficulty, env) in sequences.items():
        seq_start = time.time()
        seq_path = os.path.join(args.tartanair_root, difficulty, env)
        image_folder = os.path.join(seq_path, "image_lcam_front")
        gt_poses_path = os.path.join(seq_path, "pose_lcam_front.txt")

        if not os.path.isdir(image_folder):
            print(f"SKIP: Image folder not found: {image_folder}")
            results_summary.append((seq_short, "SKIP", 0))
            continue

        # Directory for raw VGGT-SLAM output
        raw_seq_dir = os.path.join(raw_root, f"VGGT-SLAM@{seq_short}")
        protocol_seq_dir = os.path.join(protocol_dir, f"VGGT-SLAM@{seq_short}")

        # Step 1: Run VGGT-SLAM (unless convert_only)
        if not args.convert_only:
            os.makedirs(raw_seq_dir, exist_ok=True)
            success = run_vggt_slam(image_folder, raw_seq_dir, args)
            if not success:
                results_summary.append((seq_short, "SLAM_FAIL", time.time() - seq_start))
                continue

        # Step 2: Convert to MAC-Loop protocol
        poses_txt = os.path.join(raw_seq_dir, "poses.txt")
        if not os.path.exists(poses_txt):
            print(f"SKIP: poses.txt not found: {poses_txt}")
            results_summary.append((seq_short, "NO_POSES", time.time() - seq_start))
            continue

        success = convert_poses(poses_txt, gt_poses_path, protocol_seq_dir)

        # Copy loop closure records if they exist
        lc_path = os.path.join(raw_seq_dir, "poses_loop_closures.txt")
        if os.path.exists(lc_path):
            shutil.copy2(lc_path, os.path.join(protocol_seq_dir, "loop_closures.txt"))
            num_lc = sum(1 for _ in open(lc_path))
            print(f"  Copied {num_lc} loop closure records")

        elapsed = time.time() - seq_start
        status = "OK" if success else "CONVERT_FAIL"
        results_summary.append((seq_short, status, elapsed))
        print(f"{seq_short}: {status} ({elapsed:.1f}s)")

    total_time = time.time() - total_start

    # Print summary
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for seq, status, elapsed in results_summary:
        print(f"  {seq:12s}  {status:12s}  {elapsed:7.1f}s")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results at: {protocol_dir}")


if __name__ == "__main__":
    main()
