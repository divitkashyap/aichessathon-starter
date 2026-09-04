"""Differential-test the Python and Numba NNUE encoders on legal positions."""

from __future__ import annotations

import argparse
import random

import chess
import numpy as np

from fastcore import position_from_fen
from nnue.fast_features import encode_fast
from nnue.features import encode_board


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260904)
    arguments = parser.parse_args()
    rng = random.Random(arguments.seed)
    board = chess.Board()

    for index in range(arguments.positions):
        expected = encode_board(board)
        pieces, state = position_from_fen(board.fen(en_passant="fen"))
        white, black, turn = encode_fast(pieces, state)
        np.testing.assert_array_equal(white, expected.white)
        np.testing.assert_array_equal(black, expected.black)
        if turn != expected.turn:
            raise AssertionError(f"turn mismatch at position {index}: {board.fen()}")

        if board.is_game_over() or board.ply() >= 240:
            board = chess.Board()
        else:
            legal = list(board.legal_moves)
            board.push(legal[rng.randrange(len(legal))])
        if (index + 1) % 10_000 == 0:
            print(f"matched={index + 1:,}")


if __name__ == "__main__":
    main()
