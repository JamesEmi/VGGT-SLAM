#!/usr/bin/env python3
"""
Benchmark VGGT-SLAM on VBR (Visual Benchmark Rome) sequences and output results in MAC-Loop protocol format.

Usage:
    # Run all VBR train sequences:
    python benchmark_vbr.py --vbr_root /media/airlab-storage/datasets/VBRome

    # Run specific sequences:
    python benchmark_vbr.py --vbr_root /media/airlab-storage/datasets/VBRome --sequences campus_train0 pincio_train0

    # Convert only (skip SLAM, just convert existing poses.txt):
    python benchmark_vbr.py --vbr_root /media/airlab-storage/datasets/VBRome --convert_only --raw_results_dir ./raw_output/03_02_120000

    # Custom SLAM args:
    python benchmark_vbr.py --vbr_root /media/airlab-storage/datasets/VBRome --submap_size 16 --min_disparity 50
"""

import os
import sys
import time
import shutil
import glob
import tempfile
import argparse
import subprocess
from datetime import datetime

# All VBR train sequences (only train sequences have GT poses)
VBR_SEQUENCES = [
    "campus_train0",
    "campus_train1",
    "ciampino_train0",
    "ciampino_train1",
    "colosseo_train0",
    "diag_train0",
    "pincio_train0",
    "spagna_train0",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_downsampled_folder(image_folder, downsample, tmpdir):
    """Create a temp folder with symlinks to every Nth image."""
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    if not images:
        images = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    selected = images[::downsample]
    ds_folder = os.path.join(tmpdir, "downsampled_images")
    os.makedirs(ds_folder, exist_ok=True)
    for i, img_path in enumerate(selected):
        # Use sequential naming so VGGT-SLAM sees frame_id 0,1,2,...
        ext = os.path.splitext(img_path)[1]
        link_name = f"{i:010d}{ext}"
        os.symlink(os.path.abspath(img_path), os.path.join(ds_folder, link_name))
    print(f"Downsampled {len(images)} -> {len(selected)} images (factor {downsample})")
    return ds_folder


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


def convert_poses(poses_txt, vbr_seq_path, output_dir, downsample=1):
    """Convert VGGT-SLAM output to MAC-Loop protocol (VBR)."""
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "convert_vbr_to_macloop.py"),
        "--poses_txt", poses_txt,
        "--vbr_seq", vbr_seq_path,
        "--output_dir", output_dir,
    ]
    if downsample > 1:
        cmd += ["--downsample", str(downsample)]
    print(f"Converting: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Benchmark VGGT-SLAM on VBR")
    parser.add_argument("--vbr_root", type=str, required=True,
                        help="Root directory of VBR dataset (contains campus_train0/, etc.)")
    parser.add_argument("--sequences", nargs="+", default=None,
                        help="Specific sequences to run (e.g. campus_train0 pincio_train0). Default: all train")
    parser.add_argument("--results_root", type=str, default="/mnt/data/anyslam/macloop/vggt-slam-results",
                        help="Root directory for MAC-Loop protocol results")
    parser.add_argument("--raw_output_root", type=str, default="/mnt/data/anyslam/macloop/vggt-slam-results/raw_output",
                        help="Root directory for raw VGGT-SLAM outputs (poses.txt, point clouds)")
    parser.add_argument("--convert_only", action="store_true",
                        help="Skip SLAM, only convert existing poses.txt files")
    parser.add_argument("--raw_results_dir", type=str, default=None,
                        help="Directory containing raw VGGT-SLAM outputs (for --convert_only)")

    # Downsampling
    parser.add_argument("--downsample", type=int, default=2,
                        help="Pick every Nth image (default: 2, i.e. half framerate)")

    # SLAM hyperparameters
    parser.add_argument("--submap_size", type=int, default=16)
    parser.add_argument("--min_disparity", type=float, default=50)
    parser.add_argument("--conf_threshold", type=float, default=25.0)
    parser.add_argument("--lc_thres", type=float, default=0.95)
    parser.add_argument("--max_loops", type=int, default=1)

    args = parser.parse_args()

    # Determine which sequences to run
    if args.sequences:
        sequences = [s for s in args.sequences if s in VBR_SEQUENCES]
        unknown = [s for s in args.sequences if s not in VBR_SEQUENCES]
        if unknown:
            print(f"WARNING: Unknown sequences: {unknown}")
            print(f"Available: {VBR_SEQUENCES}")
    else:
        sequences = VBR_SEQUENCES

    # Create timestamp directory
    timestamp = datetime.now().strftime("%m_%d_%H%M%S")
    protocol_dir = os.path.join(args.results_root, "VGGT-SLAM", timestamp)
    os.makedirs(protocol_dir, exist_ok=True)

    # Raw output directory (where VGGT-SLAM writes poses.txt)
    if args.convert_only and args.raw_results_dir:
        raw_root = args.raw_results_dir
    else:
        raw_root = os.path.join(args.raw_output_root, timestamp)

    print(f"VBR root:        {args.vbr_root}")
    print(f"Sequences:       {sequences}")
    print(f"Protocol output: {protocol_dir}")
    print(f"Raw output:      {raw_root}")
    print()

    results_summary = []
    total_start = time.time()

    for seq_name in sequences:
        seq_start = time.time()
        seq_path = os.path.join(args.vbr_root, seq_name)
        image_folder = os.path.join(seq_path, "camera_left", "data")
        gt_path = os.path.join(seq_path, f"{seq_name}_gt.txt")

        if not os.path.isdir(image_folder):
            print(f"SKIP: Image folder not found: {image_folder}")
            results_summary.append((seq_name, "SKIP", 0))
            continue

        if not os.path.exists(gt_path):
            print(f"SKIP: GT file not found: {gt_path}")
            results_summary.append((seq_name, "NO_GT", 0))
            continue

        # Directory for raw VGGT-SLAM output
        raw_seq_dir = os.path.join(raw_root, f"VGGT-SLAM@{seq_name}")
        protocol_seq_dir = os.path.join(protocol_dir, f"VGGT-SLAM@{seq_name}")

        # Step 1: Run VGGT-SLAM (unless convert_only)
        if not args.convert_only:
            os.makedirs(raw_seq_dir, exist_ok=True)
            # Downsample images if requested
            run_image_folder = image_folder
            tmpdir = None
            if args.downsample > 1:
                tmpdir = tempfile.mkdtemp(prefix=f"vbr_ds_{seq_name}_")
                run_image_folder = create_downsampled_folder(image_folder, args.downsample, tmpdir)
            success = run_vggt_slam(run_image_folder, raw_seq_dir, args)
            # Clean up temp dir
            if tmpdir and os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir)
            if not success:
                results_summary.append((seq_name, "SLAM_FAIL", time.time() - seq_start))
                continue

        # Step 2: Convert to MAC-Loop protocol
        poses_txt = os.path.join(raw_seq_dir, "poses.txt")
        if not os.path.exists(poses_txt):
            print(f"SKIP: poses.txt not found: {poses_txt}")
            results_summary.append((seq_name, "NO_POSES", time.time() - seq_start))
            continue

        success = convert_poses(poses_txt, seq_path, protocol_seq_dir, downsample=args.downsample)

        # Copy loop closure records if they exist
        lc_path = os.path.join(raw_seq_dir, "poses_loop_closures.txt")
        if os.path.exists(lc_path):
            shutil.copy2(lc_path, os.path.join(protocol_seq_dir, "loop_closures.txt"))
            num_lc = sum(1 for _ in open(lc_path))
            print(f"  Copied {num_lc} loop closure records")

        elapsed = time.time() - seq_start
        status = "OK" if success else "CONVERT_FAIL"
        results_summary.append((seq_name, status, elapsed))
        print(f"{seq_name}: {status} ({elapsed:.1f}s)")

    total_time = time.time() - total_start

    # Print summary
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for seq, status, elapsed in results_summary:
        print(f"  {seq:20s}  {status:12s}  {elapsed:7.1f}s")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results at: {protocol_dir}")


if __name__ == "__main__":
    main()
