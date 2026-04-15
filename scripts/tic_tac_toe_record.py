#!/usr/bin/env python3
"""
Record 9 motion primitives (one per tic-tac-toe square), then replay on demand.

Layout:
    1 | 2 | 3
    ---------
    4 | 5 | 6
    ---------
    7 | 8 | 9

Flow:
  1. Connect, mirror leader->follower
  2. Press ENTER to save HOME
  3. For each square 1-9:
       - Press ENTER to start recording
       - Perform the motion (go to square, place piece, return to HOME)
       - Press ENTER to stop
       - Automatically returns to HOME
  4. Infinite loop: type 1-9 to replay that square, 'q' to quit

Usage:
    python3 scripts/tic_tac_toe_record.py --remote_ip 10.139.203.203
    python3 scripts/tic_tac_toe_record.py --remote_ip 10.139.203.203 \\
        --hf_dataset wiscohumanoids/ttt_primitives --hf_token hf_...
"""

import argparse
import json
import os
import threading
import time
from pathlib import Path

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.robot_utils import precise_sleep

parser = argparse.ArgumentParser()
parser.add_argument("--remote_ip",     type=str, default="10.139.203.203")
parser.add_argument("--arm_profile",   type=str, default="so-arm-5dof",
                    choices=["so-arm-5dof", "am-arm-6dof"])
parser.add_argument("--fps",           type=int, default=30)
parser.add_argument("--leader_id",     type=str, default="so101_leader_bi")
parser.add_argument("--left_port",     type=str, default="/dev/ttyACM0")
parser.add_argument("--right_port",    type=str, default="/dev/ttyACM1")
parser.add_argument("--hf_dataset",    type=str, default=None)
parser.add_argument("--hf_token",      type=str, default=None)
parser.add_argument("--save_dir",      type=str, default="outputs/ttt_primitives")
parser.add_argument("--goto_duration", type=float, default=3.0,
                    help="Seconds to hold HOME pose before each replay")
parser.add_argument("--resume",        action="store_true",
                    help="Skip already-recorded squares and go straight to replay loop")
args = parser.parse_args()

SAVE_DIR  = Path(args.save_dir)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
HOME_FILE = SAVE_DIR / "home_pose.json"

SQUARE_NAMES = {
    1: "top-left",   2: "top-center",   3: "top-right",
    4: "mid-left",   5: "center",       6: "mid-right",
    7: "bot-left",   8: "bot-center",   9: "bot-right",
}

def ep_file(sq: int) -> Path:
    return SAVE_DIR / f"square_{sq}.json"

# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------
print("Connecting...")
robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini"))
leader = BiSOLeader(BiSOLeaderConfig(
    left_arm_config=SOLeaderConfig(port=args.left_port, arm_profile=args.arm_profile),
    right_arm_config=SOLeaderConfig(port=args.right_port, arm_profile=args.arm_profile),
    id=args.leader_id,
))
robot.connect()
leader.connect()
print("Connected.\n")

def get_action() -> dict:
    arm = leader.get_action()
    arm = {f"arm_{k}": v for k, v in arm.items()}
    return {**arm, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.height_mm": 0.0}

def teleop_until_enter(prompt: str = "") -> None:
    """Mirror leader->follower until user hits ENTER."""
    if prompt:
        print(prompt)
    done = threading.Event()
    threading.Thread(target=lambda: (input(), done.set()), daemon=True).start()
    while not done.is_set():
        t0 = time.perf_counter()
        robot.send_action(get_action())
        precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))

def go_home() -> None:
    if not HOME_FILE.exists():
        return
    home_action = {**json.loads(HOME_FILE.read_text()),
                   "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.height_mm": 0.0}
    print(f"  -> Going HOME ({args.goto_duration}s)...")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.goto_duration:
        robot.send_action(home_action)
        precise_sleep(1.0 / args.fps)

def record_episode() -> list:
    """Record frames until ENTER, return episode list."""
    frames = []
    stop = threading.Event()
    threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
    t_start = time.perf_counter()
    idx = 0
    while not stop.is_set():
        t0 = time.perf_counter()
        action = get_action()
        robot.send_action(action)
        frames.append({
            "frame_index": idx,
            "timestamp":   round(time.perf_counter() - t_start, 4),
            "action":      {k: float(v) for k, v in action.items()},
        })
        idx += 1
        precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))
    return frames

def replay_episode(frames: list) -> None:
    """Replay a recorded episode at original timing."""
    print(f"  -> Replaying {len(frames)} frames...")
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        robot.send_action(frame["action"])
        if i + 1 < len(frames):
            gap = frames[i + 1]["timestamp"] - frame["timestamp"]
            precise_sleep(max(gap - (time.perf_counter() - t0), 0.0))
    print("  -> Replay done.")

def push_to_hf(local_path: Path, repo_path: str) -> None:
    if not args.hf_dataset:
        return
    try:
        from huggingface_hub import HfApi
        token = args.hf_token or os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        api.create_repo(args.hf_dataset, repo_type="dataset", exist_ok=True, private=True)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=args.hf_dataset,
            repo_type="dataset",
        )
        print(f"  -> Pushed to HF: {args.hf_dataset}/{repo_path}")
    except Exception as e:
        print(f"  -> HF push failed (non-fatal): {e}")

# ---------------------------------------------------------------------------
# STEP 1 — HOME
# ---------------------------------------------------------------------------
if args.resume and HOME_FILE.exists():
    print(f"Resuming — using existing HOME from {HOME_FILE}")
else:
    print("=" * 60)
    print("STEP 1: Move arms to HOME position (leader->follower mirroring active).")
    print("        Press ENTER to save HOME.")
    print("=" * 60)
    teleop_until_enter()
    # Save leader-reported arm positions as HOME
    last = get_action()
    home = {k: float(v) for k, v in last.items() if "arm_" in k}
    HOME_FILE.write_text(json.dumps(home, indent=2))
    print(f"HOME saved ({len(home)} joints).\n")

# ---------------------------------------------------------------------------
# STEP 2 — Record 9 squares
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Record one motion per square.")
print("  Board layout:")
print("    1 | 2 | 3")
print("    ---------")
print("    4 | 5 | 6")
print("    ---------")
print("    7 | 8 | 9")
print("=" * 60)

for sq in range(1, 10):
    path = ep_file(sq)

    if args.resume and path.exists():
        print(f"  Square {sq} ({SQUARE_NAMES[sq]}): already recorded, skipping.")
        continue

    print(f"\n--- Square {sq}: {SQUARE_NAMES[sq]} ---")
    print("  Motion should start AND end at HOME.")
    input(f"  Press ENTER to start recording square {sq}...")
    print("  [RECORDING] Perform the motion. Press ENTER to stop.")

    frames = record_episode()
    dur = frames[-1]["timestamp"] if frames else 0
    print(f"  Recorded {len(frames)} frames ({dur:.1f}s)")

    path.write_text(json.dumps(frames, indent=2))
    push_to_hf(path, f"square_{sq}.json")

    go_home()
    print(f"  Square {sq} done.\n")

print("\nAll 9 squares recorded!")
push_to_hf(HOME_FILE, "home_pose.json")

# ---------------------------------------------------------------------------
# STEP 3 — Replay loop
# ---------------------------------------------------------------------------
# Load all episodes into memory
episodes = {}
for sq in range(1, 10):
    path = ep_file(sq)
    if path.exists():
        episodes[sq] = json.loads(path.read_text())

print("\n" + "=" * 60)
print("REPLAY MODE — type a square number (1-9) to replay it.")
print("Board layout:")
print("  1 | 2 | 3")
print("  ---------")
print("  4 | 5 | 6")
print("  ---------")
print("  7 | 8 | 9")
print("Type 'q' to quit.")
print("=" * 60)

while True:
    try:
        raw = input("\nSquare to replay (1-9) or 'q': ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if raw.lower() == "q":
        break

    if not raw.isdigit() or int(raw) not in range(1, 10):
        print("  Invalid — enter a number 1-9.")
        continue

    sq = int(raw)
    if sq not in episodes:
        print(f"  Square {sq} not recorded yet.")
        continue

    print(f"  Square {sq}: {SQUARE_NAMES[sq]}")
    go_home()
    replay_episode(episodes[sq])

print("\nDone. Disconnecting.")
robot.disconnect()
leader.disconnect()
