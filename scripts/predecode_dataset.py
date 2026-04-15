#!/usr/bin/env python3
"""
Pre-decode all video frames from wiscohumanoids/colored_cube_v7 to JPEG on disk.
Run this ONCE before training. Takes ~5-10 minutes on the VM.

Usage:
    python3 predecode_dataset.py
    python3 predecode_dataset.py --out_dir /opt/dlami/nvme/cube_frames
"""

import argparse
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, default="/opt/dlami/nvme/cube_frames",
                    help="Where to write decoded frames. NVMe is fastest on g5.")
parser.add_argument("--workers", type=int, default=4,
                    help="Parallel workers for saving JPEGs (decoding is single-threaded via lerobot).")
args = parser.parse_args()

os.environ["LEROBOT_VIDEO_BACKEND"] = "pyav"

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

DATASET_REPO = "wiscohumanoids/colored_cube_v7"
CAMERAS      = ["head_top", "head_front", "wrist_right"]
OUT_DIR      = Path(args.out_dir)

print(f"Output: {OUT_DIR}")
for cam in CAMERAS:
    (OUT_DIR / cam).mkdir(parents=True, exist_ok=True)

meta = LeRobotDatasetMetadata(DATASET_REPO)
print(f"Dataset: {meta.total_frames:,} frames, {meta.total_episodes} episodes")

# Load with video keys so we get actual frames
delta_timestamps = {
    "observation.images.head_top":    [0.0],
    "observation.images.head_front":  [0.0],
    "observation.images.wrist_right": [0.0],
    "observation.state":              [0.0],
    "action":                         [0.0],
}

print("Opening dataset ...")
dataset = LeRobotDataset(DATASET_REPO, delta_timestamps=delta_timestamps, video_backend="pyav")
print(f"Loaded: {len(dataset):,} items\n")

# Check which frames already exist so we can resume if interrupted
def frame_path(cam, idx):
    return OUT_DIR / cam / f"{idx:06d}.jpg"

already_done = sum(
    1 for idx in range(len(dataset))
    if frame_path(CAMERAS[0], idx).exists()
)
if already_done > 0:
    print(f"Resuming: {already_done:,} frames already decoded, skipping those.")

def save_frame(tensor_chw, path):
    """Save a CHW float32 [0,1] or uint8 tensor as JPEG."""
    arr = tensor_chw.numpy()
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    # CHW -> HWC
    if arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)
    Image.fromarray(arr).save(path, quality=92, optimize=False)

t0 = time.perf_counter()
saved = 0
skipped = already_done

# Use DataLoader with num_workers=0 — _query_videos must not be called from subprocesses
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,   # MUST be 0 for pyav video decoding
    drop_last=False,
)

with ThreadPoolExecutor(max_workers=args.workers) as io_pool:
    futures = []
    for idx, batch in enumerate(loader):
        # Check if already decoded (resume support)
        if frame_path(CAMERAS[0], idx).exists():
            continue

        # Submit JPEG saves to thread pool so disk I/O doesn't block decoding
        for cam in CAMERAS:
            key   = f"observation.images.{cam}"
            frame = batch[key][0]  # squeeze batch dim -> (C, H, W)
            path  = frame_path(cam, idx)
            futures.append(io_pool.submit(save_frame, frame, path))

        saved += 1

        # Drain futures periodically to avoid memory buildup
        if len(futures) > 500:
            for f in futures:
                f.result()
            futures = []

        if (idx + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            rate    = saved / elapsed if elapsed > 0 else 0
            total   = len(dataset) - skipped
            eta     = (total - saved) / rate / 60 if rate > 0 else 0
            print(f"  {idx+1:>6,}/{len(dataset):,}  "
                  f"{saved:,} saved  ({rate:.0f} frames/s  ETA {eta:.1f} min)")

    # Wait for remaining saves
    for f in futures:
        f.result()

elapsed = time.perf_counter() - t0
print(f"\nDone in {elapsed/60:.1f} min.")
print(f"Frames written to: {OUT_DIR}")
print(f"\nNow run training:")
print(f"  python3 train_act_vm.py --frames_dir {OUT_DIR} --push_to_hub --hf_repo wiscohumanoids/act_colored_cube --hf_token YOUR_TOKEN")
