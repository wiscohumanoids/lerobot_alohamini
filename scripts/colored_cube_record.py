#!/usr/bin/env python3
"""
Record pick/place primitives for tape stacking (3 tapes).

Records 6 episodes in order:
  pick_1, place_1, pick_2, place_2, pick_3, place_3

pick_N  = pick tape from fixed location N (1=top-left, 2=middle-left, 3=bottom-left)
place_N = place tape onto stack at layer N height (1=bottom, 2=middle, 3=top)

Usage:
    python3 scripts/colored_cube_record.py --remote_ip 10.139.203.203
    python3 scripts/colored_cube_record.py --remote_ip 10.139.203.203 --resume
    python3 scripts/colored_cube_record.py --remote_ip 10.139.203.203 --replay_only
"""

import argparse
import json
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
parser.add_argument("--save_dir",      type=str, default="outputs/cube_primitives")
parser.add_argument("--goto_duration", type=float, default=3.0)
parser.add_argument("--resume",        action="store_true",
                    help="Skip already-recorded episodes, go to replay")
parser.add_argument("--replay_only",   action="store_true",
                    help="Skip recording entirely, go straight to replay")
args = parser.parse_args()

SAVE_DIR  = Path(args.save_dir)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
HOME_FILE = SAVE_DIR / "home_pose.json"

EPISODES = [
    "pick_1", "place_1",
    "pick_2", "place_2",
    "pick_3", "place_3",
]

DESCRIPTIONS = {
    "pick_1":  "Pick tape from location 1 (top-left)",
    "place_1": "Place tape onto stack — layer 1 (bottom)",
    "pick_2":  "Pick tape from location 2 (middle-left)",
    "place_2": "Place tape onto stack — layer 2 (middle)",
    "pick_3":  "Pick tape from location 3 (bottom-left)",
    "place_3": "Place tape onto stack — layer 3 (top)",
}

def ep_file(name: str) -> Path:
    return SAVE_DIR / f"{name}.json"

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
    print(f"  -> Replaying {len(frames)} frames...")
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        robot.send_action(frame["action"])
        if i + 1 < len(frames):
            gap = frames[i + 1]["timestamp"] - frame["timestamp"]
            precise_sleep(max(gap - (time.perf_counter() - t0), 0.0))
    print("  -> Done.")

# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if args.replay_only or (args.resume and HOME_FILE.exists()):
    print(f"Using existing HOME from {HOME_FILE}")
else:
    print("=" * 60)
    print("STEP 1: Move arms to HOME position.")
    print("        Press ENTER to save HOME.")
    print("=" * 60)
    teleop_until_enter()
    last = get_action()
    home = {k: float(v) for k, v in last.items() if "arm_" in k}
    HOME_FILE.write_text(json.dumps(home, indent=2))
    print(f"HOME saved.\n")

# ---------------------------------------------------------------------------
# Record episodes
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Recording 6 episodes: pick_1, place_1, pick_2, place_2, pick_3, place_3")
print("=" * 60)

for name in EPISODES:
    path = ep_file(name)

    if args.replay_only or (args.resume and path.exists()):
        print(f"  {name}: skipping.")
        continue

    print(f"\n--- {name}: {DESCRIPTIONS[name]} ---")
    print("  Motion should start AND end at HOME.")
    input(f"  Press ENTER to start recording '{name}'...")
    print("  [RECORDING] Perform the motion. Press ENTER to stop.")

    frames = record_episode()
    dur = frames[-1]["timestamp"] if frames else 0
    print(f"  Recorded {len(frames)} frames ({dur:.1f}s)")

    path.write_text(json.dumps(frames, indent=2))
    go_home()
    print(f"  '{name}' saved.\n")

print("\nAll episodes recorded!")

# ---------------------------------------------------------------------------
# Replay loop
# ---------------------------------------------------------------------------
episodes = {}
for name in EPISODES:
    path = ep_file(name)
    if path.exists():
        episodes[name] = json.loads(path.read_text())

print("\n" + "=" * 60)
print("REPLAY MODE — type episode name to replay.")
print("Available:", ", ".join(episodes.keys()))
print("Type 'q' to quit.")
print("=" * 60)

while True:
    try:
        raw = input("\nEpisode to replay (or 'q'): ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if raw.lower() == "q":
        break

    if raw not in episodes:
        print(f"  Unknown — choose from: {', '.join(episodes.keys())}")
        continue

    go_home()
    replay_episode(episodes[raw])

print("\nDone. Disconnecting.")
robot.disconnect()
leader.disconnect()
