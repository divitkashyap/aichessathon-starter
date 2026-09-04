"""Local harness entrypoint for the compiled-search challenger."""

from __future__ import annotations

import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastsearch import search_timed, warm_up

warm_up()


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    fallback = min(legal, key=lambda move: move.uci()).uci()
    try:
        result = search_timed(fen, time_left_ms)
    except Exception as error:
        print(f"fast search fallback: {type(error).__name__}: {error}")
        return fallback
    if chess.Move.from_uci(result.move) not in board.legal_moves:
        print("fast search fallback: illegal move")
        return fallback
    print(
        f"depth={result.depth} score={result.score} nodes={result.nodes} "
        f"time={result.elapsed_ms:.1f}ms"
    )
    return result.move
