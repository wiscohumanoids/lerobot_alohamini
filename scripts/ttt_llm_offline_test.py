#!/usr/bin/env python3
"""
Offline test: apply the same BGR color mask from ttt_vision_test to each
saved raw image, then run tictactoe_detector on the masked result.
No robot connection, no LLM.

Usage:
    python3 scripts/ttt_llm_offline_test.py
    python3 scripts/ttt_llm_offline_test.py --dir outputs/ttt_debug
    python3 scripts/ttt_llm_offline_test.py --debug
"""

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from tictactoe_detector import detect_board_scipy

# Same mask ranges as ttt_vision_test (applied to raw camera image, no conversion)
RED_LOWER    = np.array([110, 25,  30])
RED_UPPER    = np.array([150, 50,  60])
YELLOW_LOWER = np.array([130, 130, 60])
YELLOW_UPPER = np.array([170, 160, 80])

parser = argparse.ArgumentParser()
parser.add_argument("--dir",       type=str,   default="outputs/ttt_vision_test")
parser.add_argument("--tolerance", type=float, default=60)
parser.add_argument("--debug",     action="store_true")
args = parser.parse_args()

images = sorted(Path(args.dir).glob("*_raw.jpg"))
if not images:
    print(f"No *_raw.jpg files found in {args.dir}")
    sys.exit(1)

print(f"Found {len(images)} images in {args.dir}\n")

for img_path in images:
    # Use the corresponding masked image — detector needs pieces-only black background
    masked_path = Path(str(img_path).replace("_raw.jpg", "_masked.jpg"))
    if not masked_path.exists():
        print(f"=== {img_path.name} === (no masked file, skipping)")
        continue

    print(f"=== {img_path.name} ===")
    try:
        # masked file is BGR (saved by cv2) — detector uses PIL which reads as RGB
        # so convert BGR->RGB before passing
        img_bgr = cv2.imread(str(masked_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        from PIL import Image as PILImage
        PILImage.fromarray(img_rgb).save(tmp_path)

        # In the masked RGB image: red pieces are red, yellow pieces are yellow
        x_color = (130, 37, 45)   # red in RGB (midpoint of mask range)
        o_color = (150, 145, 70)  # yellow in RGB (midpoint of mask range)

        np_board = detect_board_scipy(
            tmp_path,
            x_color=x_color,
            o_color=o_color,
            tolerance=args.tolerance,
            debug=args.debug,
        )
        for r in range(3):
            row_str = " | ".join(np_board[r][c] if np_board[r][c] != "-" else "." for c in range(3))
            print(f"  {row_str}")
            if r < 2:
                print("  ---------")
    except Exception as e:
        print(f"  Error: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    print()
