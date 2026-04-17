#!/usr/bin/env python3
"""
Tic-Tac-Toe Board Reader
========================
Reads a physical tic-tac-toe board from a photo and outputs the board state.

Expects an image with:
  - A blue rectangular outline around the 3x3 board
  - Green grid lines dividing the cells (numbered 1-9)
  - Red blocks = X, Yellow/cream blocks = O, Empty = .

Usage:
    python tictactoe_reader.py <image_path>
    python tictactoe_reader.py image1.jpg image2.jpg ...

Output:
    Prints the board in a 3x3 grid format, e.g.:
        O | X | O
        ---------
        . | X | .
        ---------
        . | O | X

Dependencies:
    pip install opencv-python numpy
"""

import sys
import cv2
import numpy as np


def find_board_corners(hsv_image):
    """
    Detect the blue rectangular outline and return its 4 corners
    sorted as [top-left, top-right, bottom-right, bottom-left].
    """
    blue_mask = cv2.inRange(hsv_image, (100, 100, 100), (130, 255, 255))
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No blue outline detected in the image.")

    # Merge all blue contours and find the convex hull
    all_pts = np.vstack(contours)
    hull = cv2.convexHull(all_pts)

    # Approximate the hull to a quadrilateral
    for eps_mult in np.arange(0.01, 0.20, 0.005):
        eps = eps_mult * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, eps, True)
        if len(approx) == 4:
            break

    if len(approx) != 4:
        raise ValueError(
            f"Could not approximate blue outline to 4 corners "
            f"(got {len(approx)} points)."
        )

    # Sort corners: TL, TR, BR, BL
    pts = approx.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_board(image, corners, output_size=600):
    """
    Apply a perspective transform to straighten the board into a square image.
    """
    dst = np.array(
        [[0, 0], [output_size, 0], [output_size, output_size], [0, output_size]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (output_size, output_size))
    return warped


def classify_cell(roi_hsv):
    """
    Given the HSV region of a single cell, classify it as 'X' (red), 'O' (yellow),
    or '.' (empty).

    Strategy:
      1. Mask out annotation overlays (green grid lines, bright yellow numbers,
         blue border remnants).
      2. Check median brightness (Value channel) of remaining pixels.
         - Dark (V < 115) → empty cell (brown wood background).
         - Bright (V >= 115) → a block is present; check hue to determine color.
      3. For bright pixels, count red-hue vs yellow-hue pixels to decide.
    """
    # Build annotation mask to exclude overlaid graphics
    green = cv2.inRange(roi_hsv, (35, 50, 50), (85, 255, 255))
    bright_yellow_text = cv2.inRange(roi_hsv, (20, 180, 210), (35, 255, 255))
    blue_border = cv2.inRange(roi_hsv, (100, 80, 80), (130, 255, 255))
    annotations = green | bright_yellow_text | blue_border

    # Dilate to remove fringe pixels near annotations
    kernel = np.ones((5, 5), np.uint8)
    annotations = cv2.dilate(annotations, kernel, iterations=1)

    valid = annotations == 0
    if not np.any(valid):
        return "."

    h_vals = roi_hsv[:, :, 0][valid]
    s_vals = roi_hsv[:, :, 1][valid]
    v_vals = roi_hsv[:, :, 2][valid]

    v_median = np.median(v_vals)

    # Empty cells are dark brown wood (V typically 50-100)
    # Blocks (red or yellow) are significantly brighter (V typically 140-255)
    if v_median < 115:
        return "."

    # Cell has a block — determine color by counting bright pixels in each hue range
    bright = v_vals > 120
    red_hue = (h_vals < 12) | (h_vals > 158)  # red wraps around 0/180
    yellow_hue = (h_vals >= 15) & (h_vals <= 40)

    red_count = np.sum(bright & red_hue)
    yellow_count = np.sum(bright & yellow_hue)

    if red_count > yellow_count:
        return "X"
    elif yellow_count > red_count:
        return "O"
    else:
        return "."


def read_board(image_path):
    """
    Read a tic-tac-toe board from an annotated image.

    Returns a list of 9 symbols ['X'|'O'|'.'] in row-major order
    (cells 1-9, left-to-right, top-to-bottom).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Step 1: Find and straighten the board
    corners = find_board_corners(hsv)
    warped = warp_board(img, corners, output_size=600)
    warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

    # Step 2: Classify each of the 9 cells
    cell_size = 200  # 600 / 3
    margin = int(cell_size * 0.18)  # inset to avoid grid lines at edges

    board = []
    for cell_idx in range(9):
        row, col = cell_idx // 3, cell_idx % 3
        y1 = row * cell_size + margin
        y2 = (row + 1) * cell_size - margin
        x1 = col * cell_size + margin
        x2 = (col + 1) * cell_size - margin

        roi = warped_hsv[y1:y2, x1:x2]
        symbol = classify_cell(roi)
        board.append(symbol)

    return board


def format_board(board):
    """Pretty-print a 3x3 board."""
    lines = []
    for row in range(3):
        cells = board[row * 3 : row * 3 + 3]
        lines.append(" | ".join(cells))
        if row < 2:
            lines.append("---------")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_path> [image_path ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        try:
            board = read_board(path)
            if len(sys.argv) > 2:
                print(f"\n--- {path} ---")
            print(format_board(board))
        except Exception as e:
            print(f"Error processing {path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()