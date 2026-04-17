#!/usr/bin/env python3
"""
Tic-tac-toe using recorded motion primitives + GPT-4.1 vision for board perception.

Requires:
  - AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT env vars
  - Recorded primitives from tic_tac_toe_record.py in outputs/ttt_primitives/
  - lekiwi_host running on the robot

Usage:
    python3 scripts/ttt_play.py --remote_ip 10.139.203.203
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
from openai import AzureOpenAI
from ttt_detector import read_board

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import precise_sleep

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--remote_ip",     type=str, default="10.139.203.203")
parser.add_argument("--primitives_dir",type=str, default="outputs/ttt_primitives")
parser.add_argument("--goto_duration", type=float, default=3.0)
parser.add_argument("--fps",           type=int,   default=30)
args = parser.parse_args()

# Load .env from repo root
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

PRIMITIVES_DIR = Path(args.primitives_dir)
HOME_FILE      = PRIMITIVES_DIR / "home_pose.json"
DEBUG_DIR      = Path("outputs/ttt_debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

SQUARE_NAMES = {
    1: "top-left",   2: "top-center",   3: "top-right",
    4: "mid-left",   5: "center",       6: "mid-right",
    7: "bot-left",   8: "bot-center",   9: "bot-right",
}

# ---------------------------------------------------------------------------
# Verify setup
# ---------------------------------------------------------------------------
for var in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
    if not os.environ.get(var):
        print(f"ERROR: {var} env var not set.")
        sys.exit(1)

if not HOME_FILE.exists():
    print(f"ERROR: No home pose found at {HOME_FILE}.")
    print("Run tic_tac_toe_record.py first.")
    sys.exit(1)

missing = [sq for sq in range(1, 10) if not (PRIMITIVES_DIR / f"square_{sq}.json").exists()]
if missing:
    print(f"ERROR: Missing recorded squares: {missing}")
    print("Run tic_tac_toe_record.py first.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load primitives
# ---------------------------------------------------------------------------
episodes = {}
for sq in range(1, 10):
    episodes[sq] = json.loads((PRIMITIVES_DIR / f"square_{sq}.json").read_text())
home_pose = json.loads(HOME_FILE.read_text())
print(f"Loaded 9 primitives + HOME pose from {PRIMITIVES_DIR}")

# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------
client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2025-01-01-preview",
)

# ---------------------------------------------------------------------------
# Connect robot (observation only — no leader needed)
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

def replay(sq: int) -> None:
    frames = episodes[sq]
    print(f"  Replaying square {sq} ({SQUARE_NAMES[sq]}, {len(frames)} frames)...")
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        robot.send_action(frame["action"])
        if i + 1 < len(frames):
            gap = frames[i + 1]["timestamp"] - frame["timestamp"]
            precise_sleep(max(gap - (time.perf_counter() - t0), 0.0))
    print("  Done.")

# Board crop coordinates — same as ttt_vision_test
BOARD_X0, BOARD_Y0 = 340, 240
BOARD_X1, BOARD_Y1 = 580, 440

# Color mask ranges — applied to raw camera image (same as ttt_vision_test)
RED_LOWER    = np.array([110, 25,  30])
RED_UPPER    = np.array([150, 50,  60])
YELLOW_LOWER = np.array([130, 130, 60])
YELLOW_UPPER = np.array([170, 160, 80])

# ---------------------------------------------------------------------------
# Perspective-correct grid annotation (from annotate_grid.py manual clicks)
# Coordinates are in the CROPPED image space (after BOARD crop + 4x upscale)
# ---------------------------------------------------------------------------
# Clicks were made on cap_0_raw.jpg which is already the cropped + 4x upscaled image.
# So these coordinates are directly in upscaled-crop space — use as-is.
GRID_CORNERS = [
    (132, 86),
    (715, 87),
    (203, 648),
    (878, 602),
]
GRID_VERTICAL_X   = [395, 611]
GRID_HORIZONTAL_Y = [251, 412]
GRID_TILT         = 42   # same tilt as draw_annotated_grid.py

def _interp(p1, p2, coord, axis):
    """Linear interpolation along edge p1→p2 at given coord on given axis (0=x,1=y)."""
    span = p2[axis] - p1[axis]
    if abs(span) < 1e-6:
        return (p1[1 - axis] + p2[1 - axis]) / 2
    t = max(0.0, min(1.0, (coord - p1[axis]) / span))
    return p1[1 - axis] + t * (p2[1 - axis] - p1[1 - axis])

def draw_perspective_grid(img: np.ndarray) -> np.ndarray:
    """Draw perspective-correct board outline + grid + square numbers on img."""
    out = img.copy()
    tl, tr, bl, br = GRID_CORNERS
    vx   = GRID_VERTICAL_X
    hy   = GRID_HORIZONTAL_Y
    tilt = GRID_TILT

    # Board outline
    pts = np.array([tl, tr, br, bl], np.int32)
    cv2.polylines(out, [pts], True, (255, 80, 0), 3)

    # Vertical dividers (tilted)
    for x in vx:
        y_top = _interp(tl, tr, x, 0)
        y_bot = _interp(bl, br, x, 0)
        cv2.line(out, (int(x - tilt), int(y_top)), (int(x + tilt), int(y_bot)), (0, 220, 0), 3)

    # Horizontal dividers
    for y in hy:
        x_left  = _interp(tl, bl, y, 1)
        x_right = _interp(tr, br, y, 1)
        cv2.line(out, (int(x_left), int(y)), (int(x_right), int(y)), (0, 220, 0), 3)

    # Square numbers at cell centers
    y_bounds = [min(tl[1], tr[1]), hy[0], hy[1], max(bl[1], br[1])]
    for sq in range(1, 10):
        r = (sq - 1) // 3
        c = (sq - 1) % 3
        y_mid = (y_bounds[r] + y_bounds[r + 1]) / 2
        xl = _interp(tl, bl, y_mid, 1)
        xr = _interp(tr, br, y_mid, 1)
        x_bounds = [xl, vx[0], vx[1], xr]
        x_mid = (x_bounds[c] + x_bounds[c + 1]) / 2
        cv2.putText(out, str(sq), (int(x_mid) - 18, int(y_mid) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 255), 3, cv2.LINE_AA)
    return out

def capture_board_image(timeout_s: float = 10.0) -> np.ndarray:
    """Block until a genuinely fresh frame arrives from the network."""
    deadline = time.monotonic() + timeout_s
    last_fresh = robot._last_fresh_obs_at
    attempt = 0
    while time.monotonic() < deadline:
        obs = robot.get_observation()  # blocks up to polling_timeout_ms per call
        if robot._last_fresh_obs_at > last_fresh:
            for key in ("head_top", "observation.images.head_top"):
                if key in obs and obs[key] is not None:
                    print(f"  Got fresh frame after {attempt} retries")
                    return obs[key]
            last_fresh = robot._last_fresh_obs_at
        attempt += 1
    raise RuntimeError(f"No fresh camera frame received within {timeout_s}s.")

_img_idx = 0

def process_and_save(img: np.ndarray, label: str) -> tuple[str, str]:
    """
    Crop raw image, produce masked + annotated versions.
    Saves both to debug dir. Returns (masked_path, annotated_path) as strings.
    """
    global _img_idx

    # Step 1: crop raw camera image (RGB from camera)
    crop_raw = img[BOARD_Y0:BOARD_Y1, BOARD_X0:BOARD_X1]

    # Step 2: apply color mask on raw RGB crop (same as ttt_vision_test)
    red_mask    = cv2.inRange(crop_raw, RED_LOWER,    RED_UPPER)
    yellow_mask = cv2.inRange(crop_raw, YELLOW_LOWER, YELLOW_UPPER)
    combined    = cv2.bitwise_or(red_mask, yellow_mask)
    masked_raw  = cv2.bitwise_and(crop_raw, crop_raw, mask=combined)

    # Step 3: convert to BGR, upscale + sharpen
    def upscale_sharpen(im_bgr):
        up = cv2.resize(im_bgr, (im_bgr.shape[1] * 4, im_bgr.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[0, -1, 0], [-1, 5.8, -1], [0, -1, 0]])
        return cv2.filter2D(up, -1, kernel)

    crop_bgr   = cv2.cvtColor(crop_raw,  cv2.COLOR_RGB2BGR)
    masked_bgr = cv2.cvtColor(masked_raw, cv2.COLOR_RGB2BGR)
    sharpened        = upscale_sharpen(crop_bgr)
    masked_sharpened = upscale_sharpen(masked_bgr)

    # Step 4: draw perspective grid on raw for debug
    annotated = draw_perspective_grid(sharpened)

    # Save for debugging
    annotated_path = str(DEBUG_DIR / f"img_{_img_idx:03d}_{label}_annotated.jpg")
    masked_path    = str(DEBUG_DIR / f"img_{_img_idx:03d}_{label}_masked.jpg")
    cv2.imwrite(annotated_path, annotated)
    cv2.imwrite(masked_path,    masked_sharpened)
    print(f"  Saved: {Path(annotated_path).name}, {Path(masked_path).name}")
    _img_idx += 1

    return masked_path, annotated_path

# ---------------------------------------------------------------------------
# Board detection via ttt_detector (blue outline + perspective warp + HSV)
# (LLM vision call commented out — replaced by ttt_detector)
# VISION_PROMPT = """..."""
# ---------------------------------------------------------------------------

def read_board_from_image(img: np.ndarray) -> dict:
    """
    Uses ttt_detector on the annotated crop to read board state.
    Returns dict sq(1-9) -> 'X', 'O', or None.
    """
    _, annotated_path = process_and_save(img, "human_turn")

    try:
        cells = read_board(annotated_path)  # list of 9 symbols, row-major
    except Exception as e:
        print(f"  Detector failed ({e})")
        return {sq: None for sq in range(1, 10)}

    board = {i + 1: (v if v in ("X", "O") else None) for i, v in enumerate(cells)}

    def cell(sq): return board[sq] if board[sq] else "."
    print(f"  Detector board read:")
    print(f"    {cell(1)} | {cell(2)} | {cell(3)}")
    print(f"    ---------")
    print(f"    {cell(4)} | {cell(5)} | {cell(6)}")
    print(f"    ---------")
    print(f"    {cell(7)} | {cell(8)} | {cell(9)}")
    return board

# ---------------------------------------------------------------------------
# GPT call 2: strategy — pick best move given text board state
# ---------------------------------------------------------------------------
STRATEGY_PROMPT = """You are a tic-tac-toe expert. Choose the optimal next move.

Current board (. = empty):
{board_str}

Square-by-square state:
{square_listing}

You are playing as {symbol}.
VALID moves (only these squares are empty, you MUST pick from this list): {empty_squares}

Game rules:
- Winning lines: 1-2-3, 4-5-6, 7-8-9 (rows), 1-4-7, 2-5-8, 3-6-9 (columns), 1-5-9, 3-5-7 (diagonals).

Before choosing, work through these steps concisely — only mention relevant lines:

Step 1 - MY WINS: Any line where I have 2 and the third is in {empty_squares}?
Step 2 - OPPONENT WINS: Any line where opponent has 2 and the third is in {empty_squares}? (must block)
Step 3 - MY FORKS: Any square in {empty_squares} that creates 2+ threats for me?
Step 4 - OPPONENT FORKS: Any square in {empty_squares} where opponent could fork?
Step 5 - CHOOSE: Win > Block > Fork > Block Fork > Center(5) > Opposite corner > Empty corner > Empty side.

IMPORTANT: Your final answer MUST be one of the squares in {empty_squares}. Do not pick any other number.

After your reasoning, output ONLY one final line:
  MOVE: <square_number>"""

def ask_gpt_for_move(board: dict, symbol: str) -> str:
    """Pure text strategy call. Returns 'MOVE: N' or 'GAME_OVER: ...'."""
    def cell(sq):
        return board[sq] if board[sq] else "."
    board_str = (
        f"  {cell(1)} | {cell(2)} | {cell(3)}\n"
        f"  ---------\n"
        f"  {cell(4)} | {cell(5)} | {cell(6)}\n"
        f"  ---------\n"
        f"  {cell(7)} | {cell(8)} | {cell(9)}"
    )
    square_listing = "\n".join(
        f"  square {sq}: {board[sq] if board[sq] else 'empty'}"
        for sq in range(1, 10)
    )
    empty_squares = [sq for sq in range(1, 10) if board[sq] is None]
    prompt = STRATEGY_PROMPT.format(
        board_str=board_str,
        square_listing=square_listing,
        symbol=symbol,
        empty_squares=empty_squares,
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0,
        )
        full = response.choices[0].message.content.strip()
        print(f"  GPT reasoning:\n{full}\n")
        for line in reversed(full.splitlines()):
            line = line.strip()
            if line.startswith("MOVE:") or line.startswith("GAME_OVER:"):
                return line
        return full
    except Exception as e:
        print(f"  Strategy call failed ({e})")
        empty = [sq for sq in range(1, 10) if board[sq] is None]
        while True:
            raw = input(f"  Enter robot's move manually (empty squares: {empty}): ").strip()
            if raw.isdigit() and int(raw) in empty:
                return f"MOVE: {raw}"
            print("  Invalid square.")

# ---------------------------------------------------------------------------
# Board display
# ---------------------------------------------------------------------------
def print_board(board: dict) -> None:
    def cell(sq):
        return board[sq] if board[sq] else str(sq)
    print(f"\n  {cell(1)} | {cell(2)} | {cell(3)}")
    print("  ---------")
    print(f"  {cell(4)} | {cell(5)} | {cell(6)}")
    print("  ---------")
    print(f"  {cell(7)} | {cell(8)} | {cell(9)}\n")

def check_winner(board: dict) -> str | None:
    """Returns 'X', 'O', 'DRAW', or None if game ongoing."""
    lines = [
        (1, 2, 3), (4, 5, 6), (7, 8, 9),  # rows
        (1, 4, 7), (2, 5, 8), (3, 6, 9),  # cols
        (1, 5, 9), (3, 5, 7),              # diags
    ]
    for a, b, c in lines:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    if all(board[sq] for sq in range(1, 10)):
        return "DRAW"
    return None

# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------
print("=" * 50)
print("  TIC-TAC-TOE  —  Robot vs Human")
print("=" * 50)

while True:
    order = input("\n1 = You go first (X)   2 = Robot goes first (X): ").strip()
    if order in ("1", "2"):
        break
    print("Enter 1 or 2.")

# Human is always O if robot goes first, X if human goes first
if order == "1":
    human_symbol = "X"
    robot_symbol = "O"
    turn = "human"
else:
    human_symbol = "O"
    robot_symbol = "X"
    turn = "robot"

board = {sq: None for sq in range(1, 10)}

print(f"\nYou are {human_symbol}, Robot is {robot_symbol}.")
print("Starting game. Going to HOME...\n")
go_home()

while True:
    print_board(board)
    winner = check_winner(board)
    if winner:
        if winner == "DRAW":
            print("It's a DRAW!")
        elif winner == human_symbol:
            print(f"You win! ({human_symbol})")
        else:
            print(f"Robot wins! ({robot_symbol})")
        break

    if turn == "human":
        print(f"Your turn ({human_symbol}) — place your piece on the board.")
        input("  Press ENTER when done...")
        print("  Waiting 2s for camera to settle...")
        time.sleep(2.0)
        print("  Reading board from image...")
        curr_img = capture_board_image()
        # Vision call: read full board state from image
        detected = read_board_from_image(curr_img)
        # Find what square the human just played (newly detected piece)
        new_sq = None
        for sq in range(1, 10):
            if detected[sq] == human_symbol and board[sq] is None:
                new_sq = sq
                break
        if new_sq:
            board[new_sq] = human_symbol
            print(f"  Detected your piece at square {new_sq} ({SQUARE_NAMES[new_sq]})")
        else:
            print("  Could not detect your move from image.")
            while True:
                raw = input("  Which square did you play? (1-9): ").strip()
                if raw.isdigit() and int(raw) in range(1, 10) and board[int(raw)] is None:
                    new_sq = int(raw)
                    board[new_sq] = human_symbol
                    print(f"  Recorded your piece at square {new_sq} ({SQUARE_NAMES[new_sq]})")
                    break
                print("  Invalid or occupied square, try again.")
        turn = "robot"

    else:
        print(f"Robot's turn ({robot_symbol}).")
        # Strategy call: pure text, no image
        raw_response = ask_gpt_for_move(board, robot_symbol)

        if raw_response.startswith("GAME_OVER:"):
            # Ignore GPT's early game-over call — let check_winner handle it
            print("  GPT called GAME_OVER early — ignoring, continuing game.")
            empty = [s for s in range(1, 10) if board[s] is None]
            if not empty:
                break
            # Pick any empty square as fallback
            sq = empty[0]
            board[sq] = robot_symbol
            print(f"  Robot plays {robot_symbol} at square {sq} ({SQUARE_NAMES[sq]}) [fallback]")
            print("  Press ENTER when ready for robot to move...", end="", flush=True)
            input()
            go_home()
            replay(sq)
            go_home()
            print("  Waiting 2s for camera to settle...")
            time.sleep(2.0)
            turn = "human"

        elif raw_response.startswith("MOVE:"):
            sq_str = raw_response.split(":", 1)[1].strip()
            if not sq_str.isdigit() or int(sq_str) not in range(1, 10):
                print(f"  Invalid GPT response '{raw_response}', skipping turn.")
                turn = "human"
                continue
            sq = int(sq_str)
            if board[sq] is not None:
                fallback = next((s for s in range(1, 10) if board[s] is None), None)
                if fallback is None:
                    print("Board full.")
                    break
                print(f"  Square {sq} taken, falling back to {fallback}.")
                sq = fallback

            board[sq] = robot_symbol
            print(f"  Robot plays {robot_symbol} at square {sq} ({SQUARE_NAMES[sq]})")
            print("  Press ENTER when ready for robot to move...", end="", flush=True)
            input()
            go_home()
            replay(sq)
            go_home()
            print("  Waiting 2s for camera to settle...")
            time.sleep(2.0)
            turn = "human"

        else:
            print(f"  Unexpected GPT response: '{raw_response}', skipping.")
            turn = "human"

print("\nGame over. Returning to HOME.")
go_home()
robot.disconnect()
print("Done.")
