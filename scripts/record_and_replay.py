#!/usr/bin/env python3
"""
Record one episode via leader arms, then replay it back after returning to HOME.

Flow:
  1. Connect leader arms + robot (follower)
  2. Mirror leader -> follower in real time (teleop mode)
  3. Press ENTER to save current pose as HOME
  4. Press ENTER to start recording — move the arms through your task
  5. Press ENTER to stop recording — episode saved locally (+ optionally HF)
  6. Press ENTER to go to HOME, then immediately replay the recorded episode

Server side: just keep lekiwi_host running as usual. No changes needed.

Usage:
    python3 scripts/record_and_replay.py --remote_ip 10.139.203.203
    python3 scripts/record_and_replay.py --remote_ip 10.139.203.203 --hf_dataset wiscohumanoids/replay_test
    python3 scripts/record_and_replay.py --remote_ip 10.139.203.203 --arm_profile am-arm-6dof
"""

import argparse
import json
import time
from pathlib import Path

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.robot_utils import precise_sleep

parser = argparse.ArgumentParser()
parser.add_argument("--remote_ip",   type=str, default="10.139.203.203")
parser.add_argument("--arm_profile", type=str, default="so-arm-5dof",
                    choices=["so-arm-5dof", "am-arm-6dof"])
parser.add_argument("--fps",         type=int, default=30)
parser.add_argument("--leader_id",   type=str, default="so101_leader_bi")
parser.add_argument("--left_port",   type=str, default="/dev/ttyACM0")
parser.add_argument("--right_port",  type=str, default="/dev/ttyACM1")
parser.add_argument("--hf_dataset",  type=str, default=None,
                    help="Optional HF repo to push episode to, e.g. wiscohumanoids/replay_test")
parser.add_argument("--hf_token",    type=str, default=None,
                    help="HuggingFace write token (or set HF_TOKEN env var)")
parser.add_argument("--save_dir",    type=str, default="outputs/replay_episodes",
                    help="Local directory to save the recorded episode")
parser.add_argument("--goto_duration", type=float, default=4.0,
                    help="Seconds to hold HOME position before replaying")
args = parser.parse_args()

SAVE_DIR  = Path(args.save_dir)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
HOME_FILE = SAVE_DIR / "home_pose.json"
EP_FILE   = SAVE_DIR / "episode.json"

# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------
print("Connecting to robot and leader arms...")

robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini")
robot = LeKiwiClient(robot_config)

bi_cfg = BiSOLeaderConfig(
    left_arm_config=SOLeaderConfig(port=args.left_port, arm_profile=args.arm_profile),
    right_arm_config=SOLeaderConfig(port=args.right_port, arm_profile=args.arm_profile),
    id=args.leader_id,
)
leader = BiSOLeader(bi_cfg)

robot.connect()
leader.connect()
print("Connected.\n")

def get_action() -> dict:
    """Read leader arms and return full action dict (arms + zero base)."""
    arm = leader.get_action()
    arm = {f"arm_{k}": v for k, v in arm.items()}
    return {**arm, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.height_mm": 0.0}

def teleop_loop(prompt: str) -> dict:
    """
    Mirror leader->follower continuously until user presses ENTER.
    Returns the last action sent (used to snapshot HOME pose).
    """
    print(prompt)
    import threading
    entered = threading.Event()
    threading.Thread(target=lambda: (input(), entered.set()), daemon=True).start()

    last_action = {}
    while not entered.is_set():
        t0 = time.perf_counter()
        last_action = get_action()
        robot.send_action(last_action)
        precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))
    return last_action

# ---------------------------------------------------------------------------
# STEP 1 — set HOME, press ENTER to save
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Move arms to your HOME position.")
print("        The follower will mirror your leaders in real time.")
print("        Press ENTER when you're at HOME to save it.")
print("=" * 60)

last = teleop_loop("")
# Snapshot current follower state (observation) as HOME — more accurate than leader
obs = robot.get_observation()
home_pose = {k: float(v) for k, v in obs.items()
             if isinstance(v, float) or (hasattr(v, '__len__') is False and hasattr(v, 'item'))}
# Also grab arm keys from the last action as fallback
arm_home = {k: float(v) for k, v in last.items() if ".pos" in k or "arm_" in k}
home_pose = {**arm_home}  # use leader-reported positions (already in motor space)
HOME_FILE.write_text(json.dumps(home_pose, indent=2))
print(f"\nHOME saved ({len(home_pose)} joints):")
for k, v in home_pose.items():
    print(f"  {k}: {v:.3f}")

# ---------------------------------------------------------------------------
# STEP 2 — record episode
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Recording will start on ENTER.")
print("        Move the arms through your task carefully.")
print("        Press ENTER again when done.")
print("=" * 60)

input("  --> Press ENTER to START recording...")
print("  [RECORDING] Move now. Press ENTER to stop.")

episode = []  # list of {timestamp, action}
import threading
stop_event = threading.Event()
threading.Thread(target=lambda: (input(), stop_event.set()), daemon=True).start()

t_start = time.perf_counter()
frame_idx = 0
while not stop_event.is_set():
    t0 = time.perf_counter()
    action = get_action()
    robot.send_action(action)
    episode.append({
        "frame_index": frame_idx,
        "timestamp": round(time.perf_counter() - t_start, 4),
        "action": {k: float(v) for k, v in action.items()},
    })
    frame_idx += 1
    precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))

duration = episode[-1]["timestamp"] if episode else 0
print(f"\n  [STOPPED] Recorded {len(episode)} frames ({duration:.1f}s)")

EP_FILE.write_text(json.dumps(episode, indent=2))
print(f"  Episode saved locally to {EP_FILE}")

# Push to HF if requested
if args.hf_dataset:
    try:
        from huggingface_hub import HfApi
        import os
        token = args.hf_token or os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        api.create_repo(args.hf_dataset, repo_type="dataset", exist_ok=True, private=True)
        api.upload_file(
            path_or_fileobj=str(EP_FILE),
            path_in_repo="episode.json",
            repo_id=args.hf_dataset,
            repo_type="dataset",
        )
        print(f"  Episode also pushed to HF: {args.hf_dataset}")
    except Exception as e:
        print(f"  HF push failed (non-fatal): {e}")

# ---------------------------------------------------------------------------
# STEP 3 — go HOME, then replay
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Ready to replay.")
print("        Place the cube back in starting position.")
print("        Press ENTER to go HOME and then immediately replay.")
print("=" * 60)

input("  --> Press ENTER to go HOME + replay...")

# Go to HOME
print(f"\n  Going to HOME for {args.goto_duration}s...")
home_action = {**json.loads(HOME_FILE.read_text()),
               "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.height_mm": 0.0}
t0 = time.perf_counter()
while time.perf_counter() - t0 < args.goto_duration:
    robot.send_action(home_action)
    precise_sleep(1.0 / args.fps)

# Replay episode
print(f"  Replaying {len(episode)} frames...")
for i, frame in enumerate(episode):
    t0 = time.perf_counter()
    robot.send_action(frame["action"])
    dt = time.perf_counter() - t0
    # Sleep to maintain original timing between frames
    if i + 1 < len(episode):
        next_ts = episode[i + 1]["timestamp"]
        curr_ts = frame["timestamp"]
        target_sleep = (next_ts - curr_ts) - dt
        if target_sleep > 0:
            precise_sleep(target_sleep)

print("\n  Replay complete.")
print("=" * 60)

robot.disconnect()
leader.disconnect()
print("Done.")
