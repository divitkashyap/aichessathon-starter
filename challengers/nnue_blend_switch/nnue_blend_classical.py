"""Private classical evaluator copied for the NNUE blend challenger."""

from __future__ import annotations

from typing import Final

import chess
import numpy as np
from numba import njit

PIECE_VALUES: Final = np.array([0, 100, 320, 335, 500, 900, 0], dtype=np.int32)
PHASE_WEIGHTS: Final = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int8)
MAX_PHASE: Final = 24


def _piece_square_value(piece_type: int, square: int) -> tuple[int, int]:
    file_index = square & 7
    rank_index = square >> 3
    centre = 7 - abs(2 * file_index - 7) - abs(2 * rank_index - 7)
    if piece_type == chess.PAWN:
        return rank_index * 9 + centre * 2, rank_index * 14 + centre
    if piece_type == chess.KNIGHT:
        return centre * 8 - (8 if file_index in (0, 7) else 0), centre * 6
    if piece_type == chess.BISHOP:
        return centre * 5 + rank_index * 2, centre * 4
    if piece_type == chess.ROOK:
        seventh = 18 if rank_index == 6 else 0
        return rank_index * 2 + seventh, centre * 2 + seventh
    if piece_type == chess.QUEEN:
        return centre * 2, centre * 2
    castled_file_bonus = 12 if file_index in (1, 2, 5, 6) else 0
    return -centre * 9 - rank_index * 8 + castled_file_bonus, centre * 10


MG_TABLE: Final = np.array(
    [
        [0] * 64,
        *[
            [_piece_square_value(piece_type, square)[0] for square in range(64)]
            for piece_type in range(1, 7)
        ],
    ],
    dtype=np.int16,
)
EG_TABLE: Final = np.array(
    [
        [0] * 64,
        *[
            [_piece_square_value(piece_type, square)[1] for square in range(64)]
            for piece_type in range(1, 7)
        ],
    ],
    dtype=np.int16,
)


@njit(cache=False)
def evaluate(board: np.ndarray, state: np.ndarray) -> int:
    """Return the classical score from the side-to-move perspective."""
    middlegame = 0
    endgame = 0
    phase = 0
    white_bishops = 0
    black_bishops = 0
    for square in range(64):
        piece = int(board[square])
        if piece == 0:
            continue
        piece_type = abs(piece)
        relative_square = square if piece > 0 else square ^ 56
        sign = 1 if piece > 0 else -1
        value = int(PIECE_VALUES[piece_type])
        middlegame += sign * (value + int(MG_TABLE[piece_type, relative_square]))
        endgame += sign * (value + int(EG_TABLE[piece_type, relative_square]))
        phase += int(PHASE_WEIGHTS[piece_type])
        if piece == chess.BISHOP:
            white_bishops += 1
        elif piece == -chess.BISHOP:
            black_bishops += 1
    if white_bishops >= 2:
        middlegame += 28
        endgame += 38
    if black_bishops >= 2:
        middlegame -= 28
        endgame -= 38
    phase = min(MAX_PHASE, phase)
    white_score = (middlegame * phase + endgame * (MAX_PHASE - phase)) // MAX_PHASE
    return white_score if state[0] > 0 else -white_score
