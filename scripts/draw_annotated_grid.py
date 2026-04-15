#!/usr/bin/env python3
"""
Draw the annotated grid on raw and masked images using manually clicked coordinates.
Uses perspective-correct cell boundaries by interpolating along the board edges.

Usage:
    python3 scripts/draw_annotated_grid.py
"""

import cv2
import numpy as np
from pathlib import Path

# ── Manually annotated coordinates ──────────────────────────────────────────
CORNERS = [(132, 86), (715, 87), (203, 648), (878, 602)]  # TL, TR, BL, BR
VERTICAL_X   = [395, 611]   # x positions of the 2 vertical dividers
HORIZONTAL_Y = [251, 412]   # y positions of the 2 horizontal dividers
VERTICAL_TILT = 42          # pixels: top shifts left by this, bottom shifts right by this

OUT_DIR = Path("outputs/ttt_vision_test")


def perspective_grid_lines(corners, vx, hy):
    """
    Given 4 corners (TL, TR, BL, BR) and divider positions,
    compute perspective-correct grid line endpoints by interpolating
    along the board edges.

    Returns list of (pt1, pt2) line segments.
    """
    tl, tr, bl, br = [np.array(c, dtype=float) for c in corners]

    # Image height/width extents for the board (approximate)
    # Top edge: tl → tr
    # Bottom edge: bl → br
    # Left edge: tl → bl
    # Right edge: tr → br

    img_h = max(bl[1], br[1])   # approx bottom of board
    img_top = min(tl[1], tr[1]) # approx top of board

    # For vertical dividers at x=vx[0], vx[1]:
    # Find where the divider x intersects top and bottom edges of the board.
    # Top edge goes from tl to tr (in x). Bottom edge goes from bl to br.
    def interp_edge(p1, p2, x):
        """Find y on line p1→p2 at given x (linear interp)."""
        if abs(p2[0] - p1[0]) < 1e-6:
            return (p1[1] + p2[1]) / 2
        t = (x - p1[0]) / (p2[0] - p1[0])
        t = max(0.0, min(1.0, t))
        return p1[1] + t * (p2[1] - p1[1])

    lines = []

    # Vertical divider lines: tilted — top shifts left, bottom shifts right
    tilt = VERTICAL_TILT
    for x in vx:
        y_top = interp_edge(tl, tr, x)
        y_bot = interp_edge(bl, br, x)
        lines.append(((int(x - tilt), int(y_top)), (int(x + tilt), int(y_bot))))

    # Horizontal divider lines: from left edge to right edge at each hy
    def interp_left(y):
        # Left edge: tl → bl
        if abs(bl[1] - tl[1]) < 1e-6:
            return tl[0]
        t = (y - tl[1]) / (bl[1] - tl[1])
        t = max(0.0, min(1.0, t))
        return tl[0] + t * (bl[0] - tl[0])

    def interp_right(y):
        # Right edge: tr → br
        if abs(br[1] - tr[1]) < 1e-6:
            return tr[0]
        t = (y - tr[1]) / (br[1] - tr[1])
        t = max(0.0, min(1.0, t))
        return tr[0] + t * (br[0] - tr[0])

    for y in hy:
        x_left  = interp_left(y)
        x_right = interp_right(y)
        lines.append(((int(x_left), int(y)), (int(x_right), int(y))))

    return lines


def cell_center(corners, vx, hy, row, col):
    """Compute center of cell (row, col) in 0-indexed 3x3 grid."""
    tl, tr, bl, br = [np.array(c, dtype=float) for c in corners]

    # x boundaries for columns: left edge x, vx[0], vx[1], right edge x
    # y boundaries for rows: top edge y, hy[0], hy[1], bottom edge y

    def interp_left(y):
        if abs(bl[1] - tl[1]) < 1e-6:
            return tl[0]
        t = (y - tl[1]) / (bl[1] - tl[1])
        t = max(0.0, min(1.0, t))
        return tl[0] + t * (bl[0] - tl[0])

    def interp_right(y):
        if abs(br[1] - tr[1]) < 1e-6:
            return tr[0]
        t = (y - tr[1]) / (br[1] - tr[1])
        t = max(0.0, min(1.0, t))
        return tr[0] + t * (br[0] - tr[0])

    def interp_top(x):
        if abs(tr[0] - tl[0]) < 1e-6:
            return tl[1]
        t = (x - tl[0]) / (tr[0] - tl[0])
        t = max(0.0, min(1.0, t))
        return tl[1] + t * (tr[1] - tl[1])

    def interp_bottom(x):
        if abs(br[0] - bl[0]) < 1e-6:
            return bl[1]
        t = (x - bl[0]) / (br[0] - bl[0])
        t = max(0.0, min(1.0, t))
        return bl[1] + t * (br[1] - bl[1])

    # y boundaries
    y_bounds = [min(tl[1], tr[1]), hy[0], hy[1], max(bl[1], br[1])]
    y_mid = (y_bounds[row] + y_bounds[row + 1]) / 2

    # x boundaries at y_mid (interpolating left/right edges)
    xl = interp_left(y_mid)
    xr = interp_right(y_mid)
    x_bounds = [xl, vx[0], vx[1], xr]
    x_mid = (x_bounds[col] + x_bounds[col + 1]) / 2

    return (int(x_mid), int(y_mid))


def annotate(img_path: Path, out_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Could not load {img_path}")
        return

    tl, tr, bl, br = CORNERS

    # Draw board outline (blue)
    pts = np.array([tl, tr, br, bl], np.int32)
    cv2.polylines(img, [pts], True, (255, 80, 0), 2)

    # Draw corner dots
    for pt in CORNERS:
        cv2.circle(img, pt, 7, (255, 80, 0), -1)

    # Draw grid lines (green)
    lines = perspective_grid_lines(CORNERS, VERTICAL_X, HORIZONTAL_Y)
    for p1, p2 in lines:
        cv2.line(img, p1, p2, (0, 220, 0), 2)

    # Draw square numbers 1-9 at cell centers
    for sq in range(1, 10):
        r = (sq - 1) // 3
        c = (sq - 1) % 3
        cx, cy = cell_center(CORNERS, VERTICAL_X, HORIZONTAL_Y, r, c)
        cv2.putText(img, str(sq), (cx - 12, cy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 255), 3, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    print(f"  Saved: {out_path}")


annotate(OUT_DIR / "cap_0_raw.jpg",    OUT_DIR / "cap_0_annotated_raw.jpg")
annotate(OUT_DIR / "cap_0_masked.jpg", OUT_DIR / "cap_0_annotated_masked.jpg")
