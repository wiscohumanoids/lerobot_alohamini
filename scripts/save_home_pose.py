#!/usr/bin/env python3
"""
Read current follower arm joint positions and save as HOME pose.

Usage:
    # Save current pose as home
    python3 scripts/save_home_pose.py --remote_ip 10.139.203.203

    # Print saved home pose
    python3 scripts/save_home_pose.py --print

    # Move robot back to saved home pose
    python3 scripts/save_home_pose.py --remote_ip 10.139.203.203 --goto
"""

import argparse
import json
import time
from pathlib import Path

HOME_FILE = Path("home_pose.json")

parser = argparse.ArgumentParser()
parser.add_argument("--remote_ip", type=str, default="10.139.203.203")
parser.add_argument("--print",  action="store_true", help="Print saved home pose and exit")
parser.add_argument("--goto",   action="store_true", help="Move to saved home pose instead of saving")
parser.add_argument("--duration", type=float, default=3.0,
                    help="Seconds to hold the goto command (gives motors time to reach pose)")
args = parser.parse_args()

if args.print:
    if not HOME_FILE.exists():
        print("No home pose saved yet. Run without --print to save one.")
    else:
        pose = json.loads(HOME_FILE.read_text())
        print("Saved HOME pose:")
        for k, v in pose.items():
            print(f"  {k}: {v:.4f}")
    exit(0)

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import precise_sleep

robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini")
robot = LeKiwiClient(robot_config)
robot.connect()

obs = robot.get_observation()

# Pull out arm joint positions only (exclude velocities/base)
arm_keys = [k for k in obs if ".pos" in k]
if not arm_keys:
    # fallback: grab everything except camera images
    arm_keys = [k for k, v in obs.items()
                if not hasattr(v, 'shape') or v.ndim == 0]

pose = {k: float(obs[k]) for k in sorted(arm_keys)}

if args.goto:
    if not HOME_FILE.exists():
        print("No home pose saved. Run without --goto first to save one.")
        robot.disconnect()
        exit(1)

    pose = json.loads(HOME_FILE.read_text())
    print("Moving to HOME pose...")

    # Send the home action repeatedly for `duration` seconds
    # Base velocities all zero so the robot doesn't drive anywhere
    action = {**pose, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.height_mm": 0.0}

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.duration:
        robot.send_action(action)
        precise_sleep(1.0 / 30)

    print("Done.")
else:
    HOME_FILE.write_text(json.dumps(pose, indent=2))
    print(f"HOME pose saved to {HOME_FILE}:")
    for k, v in pose.items():
        print(f"  {k}: {v:.4f}")

robot.disconnect()
