#!/usr/bin/env python3
"""
Test the strategy prompt in isolation — no robot, no camera.
Enter a board state manually and see GPT's reasoning + move.

Usage:
    python3 scripts/ttt_strategy_test.py
    python3 scripts/ttt_strategy_test.py --board "X.O/.X./..." --symbol O
    python3 scripts/ttt_strategy_test.py --board "X.O/.X./..." --symbol O

Board format (--board): 9 chars, row by row, separated by /
  X = X piece, O = O piece, . = empty
  e.g. "X.O/.X./..." means:
    X | . | O
    ---------
    . | X | .
    ---------
    . | . | .
"""

import argparse
import os
import sys
from pathlib import Path

from openai import AzureOpenAI

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

for var in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
    if not os.environ.get(var):
        print(f"ERROR: {var} not set.")
        sys.exit(1)

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2025-01-01-preview",
)

# Import the exact same prompts from ttt_play
sys.path.insert(0, str(Path(__file__).parent))

# ── Inline the prompts so this script is standalone ──────────────────────────

STRATEGY_PROMPT = """You are a tic-tac-toe expert. Choose the optimal next move.

Current board (. = empty):
{board_str}

You are playing as {symbol}.

Game rules:
- Two players alternate turns placing their symbol on empty squares.
- The first player to get 3 of their symbols in a row (horizontal, vertical, or diagonal) wins.
- Winning lines: 1-2-3, 4-5-6, 7-8-9 (rows), 1-4-7, 2-5-8, 3-6-9 (columns), 1-5-9, 3-5-7 (diagonals).
- If all 9 squares are filled with no winner, it is a draw.
- You may only play on empty squares (marked with .).

Before choosing, work through these steps explicitly:

Be concise — only list lines/squares that are relevant, skip ones with nothing to note.

Step 1 - MY WINS: List any winning line where I have 2 and the third square is empty.
Step 2 - OPPONENT WINS: List any line where opponent has 2 and the third is empty (must block).
Step 3 - MY FORKS: List empty squares that would create 2+ threats for me.
Step 4 - OPPONENT FORKS: List empty squares where opponent could fork.
Step 5 - CHOOSE: Win > Block > Fork > Block Fork > Center(5) > Opposite corner > Empty corner > Empty side.

After your reasoning, output ONLY one final line:
  MOVE: <square_number>
Or if the game is already over:
  GAME_OVER: X
  GAME_OVER: O
  GAME_OVER: DRAW"""


def parse_board(board_str: str) -> dict:
    """Parse 'X.O/.X./...' into sq->val dict."""
    rows = board_str.strip().split("/")
    if len(rows) != 3 or any(len(r) != 3 for r in rows):
        print("ERROR: board must be 3 rows of 3 chars separated by /  e.g. X.O/.X./...")
        sys.exit(1)
    board = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            sq = r * 3 + c + 1
            board[sq] = ch if ch in ("X", "O") else None
    return board


def board_to_str(board: dict) -> str:
    def cell(sq):
        return board[sq] if board[sq] else "."
    return (
        f"  {cell(1)} | {cell(2)} | {cell(3)}\n"
        f"  ---------\n"
        f"  {cell(4)} | {cell(5)} | {cell(6)}\n"
        f"  ---------\n"
        f"  {cell(7)} | {cell(8)} | {cell(9)}"
    )


def print_board(board: dict):
    def cell(sq):
        return board[sq] if board[sq] else str(sq)
    print(f"\n  {cell(1)} | {cell(2)} | {cell(3)}")
    print("  ---------")
    print(f"  {cell(4)} | {cell(5)} | {cell(6)}")
    print("  ---------")
    print(f"  {cell(7)} | {cell(8)} | {cell(9)}\n")


def ask(board: dict, symbol: str):
    board_str = board_to_str(board)
    prompt = STRATEGY_PROMPT.format(board_str=board_str, symbol=symbol)
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        temperature=0,
    )
    full = response.choices[0].message.content.strip()
    print(f"\n--- GPT reasoning ---\n{full}\n---------------------")
    for line in reversed(full.splitlines()):
        line = line.strip()
        if line.startswith("MOVE:") or line.startswith("GAME_OVER:"):
            print(f"\n>>> {line}\n")
            return
    print(f"\n>>> (no MOVE/GAME_OVER found in response)\n")


# ── Main ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--board",  type=str, default=None, help="e.g. X.O/.X./...")
parser.add_argument("--symbol", type=str, default="O",  choices=["X", "O"])
args = parser.parse_args()

if args.board:
    board = parse_board(args.board)
    print_board(board)
    ask(board, args.symbol)
else:
    # Interactive mode
    print("Interactive mode — enter board row by row (X/O/. for each cell, or 'q' to quit)\n")
    while True:
        print("Enter 3 rows (e.g. 'X . .' then '. O .' then '. . X'), or board string (e.g. X../.../.X.), or 'q':")
        raw = input("> ").strip()
        if raw.lower() == "q":
            break
        # Accept either slash-separated or space-separated 9 chars
        raw = raw.replace(" ", "")
        if "/" not in raw and len(raw) == 9:
            raw = raw[:3] + "/" + raw[3:6] + "/" + raw[6:]
        board = parse_board(raw)
        sym = input("Your symbol to play (X or O) [O]: ").strip() or "O"
        print_board(board)
        ask(board, sym)
