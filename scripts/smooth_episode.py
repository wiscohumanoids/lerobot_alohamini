#!/usr/bin/env python3
"""
Smooth recorded ttt primitive episodes.

The jitter cause: the leader encoder updates at lower resolution than the 30Hz
recording rate, producing staircase steps (many duplicate frames then a sudden
jump). Fix: interpolate through only the frames where the joint actually moved,
producing a smooth curve at the original timestamps.

Only right-arm joints are smoothed (left arm is stationary during ttt).

Usage:
    python3 scripts/smooth_episode.py --square 8
    python3 scripts/smooth_episode.py --all
    python3 scripts/smooth_episode.py --square 8 --dry_run
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter

parser = argparse.ArgumentParser()
parser.add_argument("--square",  type=int, default=None)
parser.add_argument("--all",     action="store_true")
parser.add_argument("--window",  type=int, default=51,   help="Savitzky-Golay window (odd number)")
parser.add_argument("--poly",    type=int, default=3,    help="Polynomial order")
parser.add_argument("--dir",     type=str, default="outputs/ttt_primitives")
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()

SAVE_DIR = Path(args.dir)

RIGHT_ARM_KEYS = [
    "arm_right_shoulder_pan.pos",
    "arm_right_shoulder_lift.pos",
    "arm_right_elbow_flex.pos",
    "arm_right_wrist_flex.pos",
    "arm_right_wrist_roll.pos",
    "arm_right_gripper.pos",
]

def smooth_episode(sq: int):
    bak = SAVE_DIR / f"square_{sq}.json.bak"
    path = SAVE_DIR / f"square_{sq}.json"
    src = bak if bak.exists() else path
    if not src.exists():
        print(f"  Square {sq}: not found, skipping.")
        return

    ep = json.loads(src.read_text())
    ts = np.array([f["timestamp"] for f in ep])
    window = args.window if args.window % 2 == 1 else args.window + 1
    print(f"  Square {sq}: {len(ep)} frames, {ts[-1]:.2f}s  window={window} poly={args.poly}")

    for key in RIGHT_ARM_KEYS:
        raw = np.array([f["action"][key] for f in ep])
        if raw.std() < 1e-6:
            continue

        smoothed = savgol_filter(raw, window_length=window, polyorder=args.poly)
        smoothed[0]  = raw[0]
        smoothed[-1] = raw[-1]

        before = np.abs(np.diff(raw)).max()
        after  = np.abs(np.diff(smoothed)).max()
        print(f"    {key:45s}  max_jump: {before:.3f}→{after:.3f}")

        for i, frame in enumerate(ep):
            frame["action"][key] = float(smoothed[i])

    if not args.dry_run:
        if not bak.exists():
            bak.write_text(path.read_text())
        path.write_text(json.dumps(ep, indent=2))
        print(f"    Saved → {path.name}  (original at {bak.name})\n")
    else:
        print(f"    [dry run — not saved]\n")


squares = list(range(1, 10)) if args.all else ([args.square] if args.square else [])
if not squares:
    print("Specify --square N or --all")
else:
    for sq in squares:
        smooth_episode(sq)
