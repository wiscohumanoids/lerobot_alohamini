import argparse
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
from scipy.cluster.vq import kmeans, vq

# ─────────────────────────────────────────────
#  Pre-processing
# ─────────────────────────────────────────────

def crop_to_board(img_array: np.ndarray) -> np.ndarray:
    """Crops tightly to the bounding box of the colored squares."""
    brightness = img_array.mean(axis=2)
    y_indices, x_indices = np.where((brightness > 30) & (brightness < 240))
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return img_array 
        
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    
    return img_array[y_min:y_max+1, x_min:x_max+1]


# ─────────────────────────────────────────────
#  Color matching
# ─────────────────────────────────────────────

def color_distance(pixel: np.ndarray, target: tuple[int, int, int]) -> float:
    return float(np.sqrt(np.sum((pixel.astype(float) - np.array(target)) ** 2)))

def match_color(mean_pixel: np.ndarray,
                x_color: tuple, o_color: tuple,
                tolerance: float) -> str:
    """Matches a piece to X or O based on color distance."""
    distances = {
        "X": color_distance(mean_pixel, x_color),
        "O": color_distance(mean_pixel, o_color),
    }
    best_label = min(distances, key=distances.get)
    
    if distances[best_label] <= tolerance:
        return best_label
    return "?"


# ─────────────────────────────────────────────
#  Grid Clustering Logic
# ─────────────────────────────────────────────

def cluster_coordinates(coords: list[float], k: int = 3) -> list[int]:
    """
    Uses K-Means clustering to naturally group skewed coordinates into lanes.
    Returns a list of grid indices mapping to the original coords.
    Falls back to sorted rank assignment when there are fewer points than k.
    """
    data = np.array(coords, dtype=float)
    n = len(data)

    if n < k:
        # Not enough points to form k clusters — assign by sorted rank instead
        order = np.argsort(data)
        indices = np.empty(n, dtype=int)
        for rank, idx in enumerate(order):
            indices[idx] = rank
        return indices.tolist()

    # Find k cluster centers (our distinct columns or rows)
    centroids, _ = kmeans(data, k)

    # Sort centroids so 0 is left/top, 1 is center, 2 is right/bottom
    centroids = np.sort(centroids)

    # Map each original coordinate to the index of its nearest centroid
    indices, _ = vq(data, centroids)
    return indices.tolist()


# ─────────────────────────────────────────────
#  Display & Game Logic (Debug Only)
# ─────────────────────────────────────────────

SYMBOLS = {"X": "X", "O": "O", "empty": "-", "?": "?"}

def check_winner(board: list[list[str]]) -> str | None:
    lines = [
        [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)], # rows
        [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)], # cols
        [(0,0),(1,1),(2,2)], [(0,2),(1,1),(2,0)],                      # diagonals
    ]
    for line in lines:
        values = [board[r][c] for r, c in line]
        if values[0] in ("X", "O") and all(v == values[0] for v in values):
            return values[0]
    return None

def print_board(board: list[list[str]]) -> None:
    print("\n  Detected Board:")
    print("  ┌───┬───┬───┐")
    for i, row in enumerate(board):
        cells = " │ ".join(SYMBOLS[c] for c in row)
        print(f"  │ {cells} │")
        if i < 2: print("  ├───┼───┼───┤")
    print("  └───┴───┴───┘\n")


# ─────────────────────────────────────────────
#  Main Centroid Pipeline
# ─────────────────────────────────────────────

def detect_board_scipy(image_path: str,
                       x_color: tuple[int, int, int],
                       o_color: tuple[int, int, int],
                       tolerance: float = 40,
                       debug: bool = False) -> np.ndarray:
                 
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    cropped = crop_to_board(img_array)

    # Binary mask of the colored pieces
    brightness = cropped.mean(axis=2)
    mask = (brightness > 30) & (brightness < 240)

    # Label distinct "islands" of color
    labeled_array, num_features = label(mask)

    if debug: print("\n  ── Piece Diagnostics ──────────────")

    pieces = []
    # Analyze each detected piece
    for i in range(1, num_features + 1):
        piece_mask = labeled_array == i
        if piece_mask.sum() < 50: # Filter dust
            continue
            
        piece_pixels = cropped[piece_mask]
        mean_color = piece_pixels.mean(axis=0)
        cy, cx = center_of_mass(piece_mask)
        piece_label = match_color(mean_color, x_color, o_color, tolerance)
        
        pieces.append({
            "cx": cx, "cy": cy, 
            "mean_color": mean_color, 
            "label": piece_label
        })

    # Initialize empty board
    board = [["empty" for _ in range(3)] for _ in range(3)]

    if pieces:
        # Cluster the coordinates to combat perspective skew
        cxs = [p["cx"] for p in pieces]
        cys = [p["cy"] for p in pieces]
        
        col_indices = cluster_coordinates(cxs)
        row_indices = cluster_coordinates(cys)
        
        # Place pieces on the board
        for idx, p in enumerate(pieces):
            r = row_indices[idx]
            c = col_indices[idx]
            board[r][c] = p["label"]
            
            if debug:
                mc = p["mean_color"]
                print(f"  Piece found at (x:{p['cx']:4.0f}, y:{p['cy']:4.0f}) → Mapped to Grid [{r},{c}]")
                print(f"  Color: R={mc[0]:.0f} G={mc[1]:.0f} B={mc[2]:.0f} → Recognized as {p['label']}\n")

    # Generate the clean NumPy array
    np_board = np.array([[SYMBOLS[cell] for cell in row] for row in board])

    if debug:
        print_board(board)
        winner = check_winner(board)
        if winner:
            print(f"  🏆 Winner: {winner}\n")
        else:
            empty_count = sum(row.count("empty") for row in board)
            print(f"  Status: {'Draw' if empty_count == 0 else 'Game in progress'}\n")
            
        if pieces:
            plt.figure(figsize=(6, 6))
            plt.imshow(cropped)
            for idx, p in enumerate(pieces):
                plt.scatter(p["cx"], p["cy"], color='cyan', s=100, zorder=5, edgecolors='black')
                plt.text(p["cx"] - 15, p["cy"] - 20, f"[{row_indices[idx]}, {col_indices[idx]}]", 
                         color='white', bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))
            plt.title("Blob Centroids & K-Means Grid Mapping")
            plt.axis('off')
            plt.show()
    else:
        # PURE OUTPUT FOR NON-DEBUG RUNS
        print(np_board)

    return np_board


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Centroid-based tic-tac-toe detector.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--x-color", nargs=3, type=int, required=True)
    parser.add_argument("--o-color", nargs=3, type=int, required=True)
    parser.add_argument("--tolerance", type=float, default=40)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.debug:
        print(f"\n  Image    : {args.image}\n  X target : RGB{tuple(args.x_color)}\n  O target : RGB{tuple(args.o_color)}")
    
    try:
        detect_board_scipy(args.image, tuple(args.x_color), tuple(args.o_color), args.tolerance, args.debug)
    except Exception as e:
        if args.debug:
            print(f"\n  ✗ Error: {e}", file=sys.stderr)
        else:
            print(e, file=sys.stderr)

if __name__ == "__main__":
    main()