"""Numba feature encoder for the compact search board."""

from __future__ import annotations

import numpy as np
from numba import njit

from fastcore import KING, SIDE
from nnue.features import FEATURES_PER_BUCKET, MAX_ACTIVE_FEATURES, PADDING_INDEX


@njit(cache=False)
def _fast_perspective_indices(board: np.ndarray, perspective: int) -> np.ndarray:
    king_square = -1
    for square in range(64):
        if board[square] == perspective * KING:
            king_square = square
            break
    if king_square < 0:
        raise ValueError("NNUE positions must contain both kings")

    oriented_king = king_square if perspective > 0 else king_square ^ 56
    mirror_files = (oriented_king & 7) >= 4
    if mirror_files:
        oriented_king ^= 7
    bucket = (oriented_king >> 3) * 4 + (oriented_king & 7)

    indices = np.full(MAX_ACTIVE_FEATURES, PADDING_INDEX, dtype=np.int32)
    count = 0
    for square in range(64):
        piece = int(board[square])
        piece_type = abs(piece)
        if piece == 0 or piece_type == KING:
            continue
        relative_colour = 0 if piece * perspective > 0 else 1
        plane = relative_colour * 5 + piece_type - 1
        normalized_square = square if perspective > 0 else square ^ 56
        if mirror_files:
            normalized_square ^= 7
        indices[count] = bucket * FEATURES_PER_BUCKET + plane * 64 + normalized_square
        count += 1

    # The accumulator sum is order-independent, but canonical ordering makes
    # Python/Numba parity exact and improves compression in training shards.
    for index in range(1, count):
        value = indices[index]
        cursor = index - 1
        while cursor >= 0 and indices[cursor] > value:
            indices[cursor + 1] = indices[cursor]
            cursor -= 1
        indices[cursor + 1] = value
    return indices


@njit(cache=False)
def encode_fast(board: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    return (
        _fast_perspective_indices(board, 1),
        _fast_perspective_indices(board, -1),
        int(state[SIDE]),
    )
