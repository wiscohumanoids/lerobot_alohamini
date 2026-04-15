#!/usr/bin/env python3
"""
Capture head_top camera image, apply BGR color masks for red and yellow pieces,
save pre-mask and post-mask images to outputs/ttt_vision_test/.

Red   BGR range: B=110-150, G=25-50,   R=30-60
Yellow BGR range: B=130-170, G=130-160, R=60-80

Usage:
    python3 scripts/ttt_vision_test.py --remote_ip 10.139.203.203
"""

import argparse
import base64
import os
import sys
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
from openai import AzureOpenAI

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from tictactoe_detector import detect_board_scipy

parser = argparse.ArgumentParser()
parser.add_argument("--remote_ip", type=str, default="10.139.203.203")
args = parser.parse_args()

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

for var in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
    if not os.environ.get(var):
        print(f"ERROR: {var} not set.")
        sys.exit(1)

llm = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2025-01-01-preview",
)

OUT_DIR = Path(__file__).parent.parent / "outputs" / "ttt_vision_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Connecting to robot...")
robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini"))
robot.connect()
print("Connected.\n")

# BGR ranges
RED_LOWER    = np.array([110, 25,  30])
RED_UPPER    = np.array([150, 50,  60])
YELLOW_LOWER = np.array([130, 130, 60])
YELLOW_UPPER = np.array([170, 160, 80])

# Minimum masked pixels in a cell to count as occupied
RED_THRESH    = 200
YELLOW_THRESH = 200

def capture(flush_frames: int = 5) -> np.ndarray:
    """Flush stale cached frames then return a fresh one."""
    for _ in range(flush_frames):
        robot.get_observation()
        time.sleep(1.0 / 30)
    obs = robot.get_observation()
    for key in ("head_top", "observation.images.head_top"):
        if key in obs and obs[key] is not None:
            return obs[key]
    raise RuntimeError("head_top camera not found in observation.")

def board_state(img: np.ndarray, red_mask: np.ndarray, yellow_mask: np.ndarray) -> dict:
    """
    Divide the image into a 3x3 grid and count red/yellow pixels per cell.
    Returns dict sq (1-9) -> 'X', 'O', or None.
    """
    h, w = img.shape[:2]
    cell_h, cell_w = h // 3, w // 3
    state = {}
    for sq in range(1, 10):
        row = (sq - 1) // 3
        col = (sq - 1) % 3
        r0, r1 = row * cell_h, (row + 1) * cell_h
        c0, c1 = col * cell_w, (col + 1) * cell_w
        pad_h = int(0.2 * cell_h)
        pad_w = int(0.2 * cell_w)

        r0 += pad_h
        r1 -= pad_h
        c0 += pad_w
        c1 -= pad_w
        red_px    = int(red_mask   [r0:r1, c0:c1].sum() // 255)
        yellow_px = int(yellow_mask[r0:r1, c0:c1].sum() // 255)
        MARGIN = 1.3
        if red_px >= RED_THRESH and red_px >= yellow_px * MARGIN:
            state[sq] = "X"
        elif yellow_px >= YELLOW_THRESH and yellow_px > red_px * MARGIN:
            state[sq] = "O"
        else:
            state[sq] = None
    return state

def print_board(state: dict) -> None:
    def cell(sq):
        return state[sq] if state[sq] else "."
    print(f"\n  {cell(1)} | {cell(2)} | {cell(3)}")
    print("  ---------")
    print(f"  {cell(4)} | {cell(5)} | {cell(6)}")
    print("  ---------")
    print(f"  {cell(7)} | {cell(8)} | {cell(9)}\n")

print("Auto-capturing every 5 seconds. Ctrl-C to quit.\n")

cap_idx = 0
while True:
    try:
        img = capture()
    except KeyboardInterrupt:
        break
    except RuntimeError as e:
        print(f"  Capture failed: {e}")
        time.sleep(5.0)
        continue

    BOARD_X0, BOARD_Y0 = 340, 240
    BOARD_X1, BOARD_Y1 = 580, 440

    # Step 1: crop raw
    crop_raw = img[BOARD_Y0:BOARD_Y1, BOARD_X0:BOARD_X1]

    # Step 2: mask on raw crop
    red_mask    = cv2.inRange(crop_raw, RED_LOWER,    RED_UPPER)
    yellow_mask = cv2.inRange(crop_raw, YELLOW_LOWER, YELLOW_UPPER)
    combined    = cv2.bitwise_or(red_mask, yellow_mask)
    masked_crop = cv2.bitwise_and(crop_raw, crop_raw, mask=combined)

    # Step 3: convert to BGR
    crop_bgr   = cv2.cvtColor(crop_raw,    cv2.COLOR_RGB2BGR)
    masked_bgr = cv2.cvtColor(masked_crop, cv2.COLOR_RGB2BGR)

    # Step 4: upscale + sharpen AFTER masking
    def upscale_sharpen(im):
        up = cv2.resize(im, (im.shape[1] * 4, im.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[0, -1, 0], [-1, 5.8, -1], [0, -1, 0]])
        return cv2.filter2D(up, -1, kernel)

    raw_final    = upscale_sharpen(crop_bgr)
    masked_final = upscale_sharpen(masked_bgr)

    # Save
    raw_path    = OUT_DIR / f"cap_{cap_idx}_raw.jpg"
    masked_path = OUT_DIR / f"cap_{cap_idx}_masked.jpg"
    cv2.imwrite(str(raw_path),    raw_final)
    cv2.imwrite(str(masked_path), masked_final)
    print(f"  Saved: {raw_path.name}")
    print(f"         {masked_path.name}")

    # --- LLM: pass raw crop + masked crop ---
    print("\n  Asking GPT-4.1...")
    _, buf_raw = cv2.imencode(".jpg", raw_final,    [cv2.IMWRITE_JPEG_QUALITY, 95])
    _, buf_msk = cv2.imencode(".jpg", masked_final, [cv2.IMWRITE_JPEG_QUALITY, 95])
    b64_raw = base64.b64encode(buf_raw.tobytes()).decode("utf-8")
    b64_msk = base64.b64encode(buf_msk.tobytes()).decode("utf-8")

    llm_prompt = """You are reading a tic-tac-toe board from a top-down camera.

Color mapping:
  RED pieces    = X
  YELLOW pieces = O

Board squares:
  1 | 2 | 3
  ---------
  4 | 5 | 6
  ---------
  7 | 8 | 9

Print ONLY the board state (X, O, or . for empty). No extra text:
. | . | .
---------
. | . | .
---------
. | . | ."""

    response = llm.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_raw}"}},
            {"type": "text",      "text": llm_prompt},
        ]}],
        max_tokens=60,
        temperature=0,
    )
    print("\n  --- GPT board ---")
    print(response.choices[0].message.content.strip())
    print("  -----------------\n")

    cap_idx += 1
    time.sleep(5.0)

print("Disconnecting.")
robot.disconnect()
