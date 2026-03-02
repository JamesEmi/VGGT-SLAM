#!/usr/bin/env python3
"""
Benchmark VGGT-SLAM on EuRoC sequences and output results in MAC-Loop protocol format.

Usage:
    # Run all EuRoC sequences:
    python benchmark_euroc.py --euroc_root /media/airlab-storage/datasets/EuRoC

    # Run specific sequences:
    python benchmark_euroc.py --euroc_root /media/airlab-storage/datasets/EuRoC --sequences MH01 MH02

    # Convert only (skip SLAM, just convert existing poses.txt):
    python benchmark_euroc.py --euroc_root /media/airlab-storage/datasets/EuRoC --convert_only --raw_results_dir ./raw_output

    # Custom SLAM args:
    python benchmark_euroc.py --euroc_root /media/airlab-storage/datasets/EuRoC --submap_size 16 --min_disparity 50
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

EUROC_SEQUENCES = {
    "MH01": "MH_01_easy",
    "MH02": "MH_02_easy",
    "MH03": "MH_03_medium",
    "MH04": "MH_04_difficult",
    "MH05": "MH_05_difficult",
    "V101": "V1_01_easy",
    "V102": "V1_02_medium",
    "V103": "V1_03_difficult",
    "V201": "V2_01_easy",
    "V202": "V2_02_medium",
    "V203": "V2_03_difficult",
}

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


def convert_poses(poses_txt, euroc_seq_path, output_dir):
    """Convert VGGT-SLAM output to MAC-Loop protocol."""
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "convert_to_macloop.py"),
        "--poses_txt", poses_txt,
        "--euroc_seq", euroc_seq_path,
        "--output_dir", output_dir,
    ]
    print(f"Converting: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Benchmark VGGT-SLAM on EuRoC")
    parser.add_argument("--euroc_root", type=str, required=True,
                        help="Root directory of EuRoC dataset (contains MH_01_easy/ etc.)")
    parser.add_argument("--sequences", nargs="+", default=None,
                        help="Specific sequences to run (e.g. MH01 MH02). Default: all")
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
        sequences = {k: EUROC_SEQUENCES[k] for k in args.sequences if k in EUROC_SEQUENCES}
        unknown = [k for k in args.sequences if k not in EUROC_SEQUENCES]
        if unknown:
            print(f"WARNING: Unknown sequences: {unknown}")
            print(f"Available: {list(EUROC_SEQUENCES.keys())}")
    else:
        sequences = EUROC_SEQUENCES

    # Create timestamp directory
    timestamp = datetime.now().strftime("%m_%d_%H%M%S")
    protocol_dir = os.path.join(args.results_root, "VGGT-SLAM", timestamp)
    os.makedirs(protocol_dir, exist_ok=True)

    # Raw output directory (where VGGT-SLAM writes poses.txt)
    if args.convert_only and args.raw_results_dir:
        raw_root = args.raw_results_dir
    else:
        raw_root = os.path.join(args.results_root, "raw_output", timestamp)

    print(f"EuRoC root:      {args.euroc_root}")
    print(f"Sequences:       {list(sequences.keys())}")
    print(f"Protocol output: {protocol_dir}")
    print(f"Raw output:      {raw_root}")
    print()

    results_summary = []
    total_start = time.time()

    for seq_short, seq_full in sequences.items():
        seq_start = time.time()
        euroc_seq_path = os.path.join(args.euroc_root, seq_full)
        image_folder = os.path.join(euroc_seq_path, "mav0", "cam0", "data")

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

        success = convert_poses(poses_txt, euroc_seq_path, protocol_seq_dir)
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
        print(f"  {seq:6s}  {status:12s}  {elapsed:7.1f}s")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results at: {protocol_dir}")


if __name__ == "__main__":
    main()
