#!/usr/bin/env python3
"""
Tape stacking demo: voice command + camera vision → LLM plans pick/place sequence → robot executes.

Setup:
  - 3 tapes at fixed positions: 1=top-left, 2=middle-left, 3=bottom-left
  - 3 pick primitives (pick_1/2/3) and 3 place primitives (place_1/2/3)
  - place_N = layer N on the stack (1=bottom, 2=middle, 3=top)

Flow:
  1. Press ENTER to start voice capture
  2. Speak your command, press ENTER to stop
  3. Camera image taken, sent to LLM with voice command
  4. LLM outputs pick/place sequence
  5. Robot executes chain

Usage:
    python3 scripts/tape_stack_play.py --remote_ip 10.139.203.203
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import AzureOpenAI

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import precise_sleep

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--remote_ip",      type=str, default="10.139.203.203")
parser.add_argument("--primitives_dir", type=str, default="outputs/cube_primitives")
parser.add_argument("--goto_duration",  type=float, default=3.0)
parser.add_argument("--fps",            type=int, default=30)
parser.add_argument("--sample_rate",    type=int, default=16000)
parser.add_argument("--text",           action="store_true",
                    help="Type commands instead of using voice (for WSL/no-mic environments)")
args = parser.parse_args()

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

for var in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
    if not os.environ.get(var):
        print(f"ERROR: {var} not set.")
        sys.exit(1)

PRIMITIVES_DIR = Path(args.primitives_dir)
HOME_FILE = PRIMITIVES_DIR / "home_pose.json"
DEBUG_DIR = Path("outputs/tape_debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load primitives
# ---------------------------------------------------------------------------
EPISODE_NAMES = ["pick_1", "place_1", "pick_2", "place_2", "pick_3", "place_3"]
missing = [n for n in EPISODE_NAMES if not (PRIMITIVES_DIR / f"{n}.json").exists()]
if missing:
    print(f"ERROR: Missing primitives: {missing}")
    print("Run colored_cube_record.py first.")
    sys.exit(1)
if not HOME_FILE.exists():
    print(f"ERROR: No home pose at {HOME_FILE}")
    sys.exit(1)

episodes = {n: json.loads((PRIMITIVES_DIR / f"{n}.json").read_text()) for n in EPISODE_NAMES}
home_pose = json.loads(HOME_FILE.read_text())
print(f"Loaded 6 primitives + HOME from {PRIMITIVES_DIR}")

# ---------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2025-01-01-preview",
)

# ---------------------------------------------------------------------------
# Connect robot
# ---------------------------------------------------------------------------
print("Connecting to robot...")
robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini", polling_timeout_ms=500))
robot.connect()
print("Connected.\n")

# ---------------------------------------------------------------------------
# Robot helpers
# ---------------------------------------------------------------------------
def go_home() -> None:
    action = {**home_pose, "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0, "lift_axis.height_mm": 0.0}
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.goto_duration:
        robot.send_action(action)
        precise_sleep(1.0 / args.fps)

def replay(name: str) -> None:
    frames = episodes[name]
    print(f"  -> {name} ({len(frames)} frames)...")
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        robot.send_action(frame["action"])
        if i + 1 < len(frames):
            gap = frames[i + 1]["timestamp"] - frame["timestamp"]
            precise_sleep(max(gap - (time.perf_counter() - t0), 0.0))

def capture_image(timeout_s: float = 10.0) -> np.ndarray:
    deadline = time.monotonic() + timeout_s
    last_fresh = robot._last_fresh_obs_at
    while time.monotonic() < deadline:
        obs = robot.get_observation()
        if robot._last_fresh_obs_at > last_fresh:
            for key in ("head_top", "observation.images.head_top"):
                if key in obs and obs[key] is not None:
                    return obs[key]
            last_fresh = robot._last_fresh_obs_at
    raise RuntimeError("No fresh camera frame received.")

# ---------------------------------------------------------------------------
# Voice capture
# ---------------------------------------------------------------------------
def record_voice() -> str:
    """Record audio until ENTER pressed, transcribe with Whisper, return text."""
    if args.text:
        return input("  Type your command: ").strip()

    print("  [Recording voice...] Press ENTER to stop.")
    frames = []
    stop = [False]

    def callback(indata, frame_count, time_info, status):
        if not stop[0]:
            frames.append(indata.copy())

    import threading
    stream = sd.InputStream(samplerate=args.sample_rate, channels=1,
                            dtype="float32", callback=callback)
    stream.start()
    threading.Thread(target=lambda: (input(), stop.__setitem__(0, True)), daemon=True).start()
    while not stop[0]:
        time.sleep(0.05)
    stream.stop()
    stream.close()

    if not frames:
        return ""

    audio = np.concatenate(frames, axis=0)
    audio_path = str(DEBUG_DIR / "voice.wav")
    sf.write(audio_path, audio, args.sample_rate)

    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(model="whisper-1", file=f)
    return result.text.strip()

# ---------------------------------------------------------------------------
# LLM: image + voice → pick/place sequence
# ---------------------------------------------------------------------------
PLAN_PROMPT = """You are controlling a robot arm that can pick and place tapes.

The camera image shows 3 tapes at fixed positions:
  - Position 1: top-left
  - Position 2: middle-left
  - Position 3: bottom-left

The robot has these primitives:
  - pick_1: pick tape from position 1
  - pick_2: pick tape from position 2
  - pick_3: pick tape from position 3
  - place_1: place tape as layer 1 (bottom of stack)
  - place_2: place tape as layer 2 (middle of stack)
  - place_3: place tape as layer 3 (top of stack)

Look at the image to identify which color tape is at which position.
Then read the user's voice command and determine the correct pick/place sequence.

Rules:
- Each pick must be immediately followed by its place before the next pick.
- place_1 always goes first (bottom), place_2 second, place_3 last (top).
- Only include tapes mentioned in the command.

User command: "{command}"

Output ONLY a JSON array of primitive names in order, no explanation. Example:
["pick_3", "place_1", "pick_1", "place_2"]"""

def plan_from_command(command: str, img: np.ndarray) -> list[str]:
    # Save + encode image
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img
    img_path = str(DEBUG_DIR / "scene.jpg")
    cv2.imwrite(img_path, img_bgr)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    prompt = PLAN_PROMPT.format(command=command)
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        max_tokens=100,
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    print(f"  LLM plan: {raw}")

    # Parse JSON array
    import re
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse plan from: {raw}")
    sequence = json.loads(match.group())

    # Validate
    valid = set(EPISODE_NAMES)
    for name in sequence:
        if name not in valid:
            raise ValueError(f"Unknown primitive '{name}' in plan")
    return sequence

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
print("=" * 50)
print("  TAPE STACKING DEMO")
print("=" * 50)
print("Going to HOME...\n")
go_home()

while True:
    try:
        input("\nPress ENTER to start voice command (or Ctrl-C to quit)...")
    except KeyboardInterrupt:
        break

    # Voice
    try:
        command = record_voice()
    except Exception as e:
        print(f"  Voice capture failed: {e}")
        continue

    if not command:
        print("  No speech detected.")
        continue

    print(f"\n  You said: \"{command}\"")
    input("  Press ENTER to confirm and execute (or Ctrl-C to cancel)...")

    # Capture image
    print("  Capturing scene...")
    try:
        img = capture_image()
    except Exception as e:
        print(f"  Camera failed: {e}")
        continue

    # Plan
    print("  Planning sequence...")
    try:
        sequence = plan_from_command(command, img)
    except Exception as e:
        print(f"  Planning failed: {e}")
        continue

    print(f"\n  Executing: {' → '.join(sequence)}")

    # Execute
    go_home()
    for name in sequence:
        replay(name)
    go_home()

    print("  Done!\n")

print("\nDisconnecting.")
robot.disconnect()
print("Done.")
