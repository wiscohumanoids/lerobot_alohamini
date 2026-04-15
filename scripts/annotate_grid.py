#!/usr/bin/env python3
"""
Step 1: Click 4 corners of the board (top-left, top-right, bottom-left, bottom-right).
Step 2: Click 2 vertical dividers, then 2 horizontal dividers.
Draws the board outline + 3x3 grid with square numbers. Press 's' to save.

Usage:
    python3 scripts/annotate_grid.py
    python3 scripts/annotate_grid.py --image outputs/ttt_vision_test/cap_2_raw.jpg
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default="outputs/ttt_vision_test/cap_0_raw.jpg")
args = parser.parse_args()

img = cv2.imread(args.image)
if img is None:
    print(f"Could not load {args.image}")
    exit(1)

clone = img.copy()
state = {"corners": [], "v": [], "h": [], "phase": "corners"}

PHASE_MSG = {
    "corners": "PHASE 1: Click 4 board corners — top-left, top-right, bottom-left, bottom-right",
    "v":       "PHASE 2: Click 2 vertical dividers (left then right)",
    "h":       "PHASE 3: Click 2 horizontal dividers (top then bottom)",
    "done":    "Done! Press 's' to save, 'r' to reset.",
}

def redraw():
    img[:] = clone[:]
    # Draw corners
    for pt in state["corners"]:
        cv2.circle(img, pt, 6, (255, 0, 0), -1)
    if len(state["corners"]) == 4:
        pts = np.array(state["corners"], np.int32)
        cv2.polylines(img, [pts[[0,1,3,2]]], True, (255, 0, 0), 2)
    # Draw dividers
    h_img, w_img = img.shape[:2]
    for x in state["v"]:
        cv2.line(img, (x, 0), (x, h_img), (0, 255, 0), 2)
    for y in state["h"]:
        cv2.line(img, (0, y), (w_img, y), (0, 255, 0), 2)
    # Draw square numbers when fully done
    if len(state["v"]) == 2 and len(state["h"]) == 2:
        vx = sorted(state["v"])
        hy = sorted(state["h"])
        xs = [0, vx[0], vx[1], w_img]
        ys = [0, hy[0], hy[1], h_img]
        for sq in range(1, 10):
            r = (sq - 1) // 3
            c = (sq - 1) % 3
            cx = (xs[c] + xs[c+1]) // 2 - 10
            cy = (ys[r] + ys[r+1]) // 2 + 10
            cv2.putText(img, str(sq), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("annotate_grid", img)

def click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    phase = state["phase"]
    if phase == "corners" and len(state["corners"]) < 4:
        state["corners"].append((x, y))
        print(f"  Corner {len(state['corners'])}: ({x}, {y})")
        if len(state["corners"]) == 4:
            state["phase"] = "v"
            print(f"\n{PHASE_MSG['v']}")
    elif phase == "v" and len(state["v"]) < 2:
        state["v"].append(x)
        print(f"  Vertical {len(state['v'])}: x={x}")
        if len(state["v"]) == 2:
            state["phase"] = "h"
            print(f"\n{PHASE_MSG['h']}")
    elif phase == "h" and len(state["h"]) < 2:
        state["h"].append(y)
        print(f"  Horizontal {len(state['h'])}: y={y}")
        if len(state["h"]) == 2:
            state["phase"] = "done"
            vx = sorted(state["v"])
            hy = sorted(state["h"])
            print(f"\n  vertical_x   = {vx}")
            print(f"  horizontal_y = {hy}")
            print(f"\n{PHASE_MSG['done']}")

    redraw()

cv2.imshow("annotate_grid", img)
cv2.setMouseCallback("annotate_grid", click)
print(f"Image: {args.image}  ({img.shape[1]}x{img.shape[0]})")
print(f"\n{PHASE_MSG['corners']}\n")

while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord('s'):
        out_path = Path(args.image).parent / (Path(args.image).stem + "_grid.jpg")
        cv2.imwrite(str(out_path), img)
        print(f"  Saved: {out_path}")
    elif key == ord('r'):
        state["corners"].clear()
        state["v"].clear()
        state["h"].clear()
        state["phase"] = "corners"
        redraw()
        print(f"\nReset.\n{PHASE_MSG['corners']}\n")
    else:
        break

cv2.destroyAllWindows()
if state["v"] and state["h"]:
    print(f"\nFinal grid coordinates:")
    print(f"  corners      = {state['corners']}")
    print(f"  vertical_x   = {sorted(state['v'])}")
    print(f"  horizontal_y = {sorted(state['h'])}")
