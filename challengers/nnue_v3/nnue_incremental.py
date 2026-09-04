"""Incremental integer NNUE accumulators for the compact search board.

The compiled search core owns the board and undo record.  These helpers update a
pair of ``int64`` accumulators immediately after the corresponding fastcore
make/unmake operation.  Kings are bucket selectors rather than ordinary NNUE
features, so a king move refreshes that king's perspective while the opposite
perspective only needs ordinary piece deltas (for example, the rook in castling).
"""

from __future__ import annotations

import numpy as np
from nnue_core import (
    CASTLE,
    KING,
    ROOK,
    UNDO_CAPTURE_SQUARE,
    UNDO_CAPTURED,
    UNDO_MOVED_PIECE,
    make_move,
    unmake_move,
)
from numba import njit

FEATURES_PER_BUCKET = 10 * 64


@njit(cache=False)
def _find_king(board: np.ndarray, perspective: int) -> int:
    for square in range(64):
        if board[square] == perspective * KING:
            return square
    raise ValueError("NNUE positions must contain both kings")


@njit(cache=False)
def _feature_index(
    piece: int,
    square: int,
    perspective: int,
    king_square: int,
) -> int:
    """Return one feature index, or -1 for a king/empty square."""
    piece_type = abs(piece)
    if piece == 0 or piece_type == KING:
        return -1

    oriented_king = king_square if perspective > 0 else king_square ^ 56
    mirror_files = (oriented_king & 7) >= 4
    if mirror_files:
        oriented_king ^= 7
    bucket = (oriented_king >> 3) * 4 + (oriented_king & 7)

    relative_colour = 0 if piece * perspective > 0 else 1
    plane = relative_colour * 5 + piece_type - 1
    normalized_square = square if perspective > 0 else square ^ 56
    if mirror_files:
        normalized_square ^= 7
    return bucket * FEATURES_PER_BUCKET + plane * 64 + normalized_square


@njit(cache=False)
def _add_piece_delta(
    accumulator: np.ndarray,
    feature: np.ndarray,
    piece: int,
    square: int,
    perspective: int,
    king_square: int,
    sign: int,
) -> None:
    index = _feature_index(piece, square, perspective, king_square)
    if index < 0:
        return
    for channel in range(feature.shape[1]):
        accumulator[channel] += np.int64(sign) * np.int64(feature[index, channel])


@njit(cache=False)
def refresh_perspective(
    board: np.ndarray,
    perspective: int,
    king_square: int,
    feature: np.ndarray,
    feature_bias: np.ndarray,
    accumulator: np.ndarray,
) -> None:
    """Rebuild one accumulator exactly from the current compact board."""
    for channel in range(feature.shape[1]):
        accumulator[channel] = np.int64(feature_bias[channel])
    for square in range(64):
        _add_piece_delta(
            accumulator,
            feature,
            int(board[square]),
            square,
            perspective,
            king_square,
            1,
        )


@njit(cache=False)
def initialize_accumulators(
    board: np.ndarray,
    feature: np.ndarray,
    feature_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create White/Black accumulators and their current king-square metadata."""
    white_king = _find_king(board, 1)
    black_king = _find_king(board, -1)
    white = np.empty(feature.shape[1], dtype=np.int64)
    black = np.empty(feature.shape[1], dtype=np.int64)
    refresh_perspective(board, 1, white_king, feature, feature_bias, white)
    refresh_perspective(board, -1, black_king, feature, feature_bias, black)
    king_squares = np.array([white_king, black_king], dtype=np.int8)
    return white, black, king_squares


@njit(cache=False)
def _castle_rook_move(target: int) -> tuple[int, int]:
    if target == 6:  # White O-O: h1-f1.
        return 7, 5
    if target == 2:  # White O-O-O: a1-d1.
        return 0, 3
    if target == 62:  # Black O-O: h8-f8.
        return 63, 61
    if target == 58:  # Black O-O-O: a8-d8.
        return 56, 59
    raise ValueError("invalid castling target")


@njit(cache=False)
def _apply_move_delta(
    accumulator: np.ndarray,
    perspective: int,
    king_square: int,
    move: int,
    undo: np.ndarray,
    feature: np.ndarray,
    forward: bool,
) -> None:
    """Apply or reverse all non-bucket-changing piece deltas for one view."""
    source = move & 63
    target = (move >> 6) & 63
    promotion = (move >> 12) & 7
    moved_piece = int(undo[UNDO_MOVED_PIECE])
    moving_side = 1 if moved_piece > 0 else -1
    placed_piece = moving_side * promotion if promotion else moved_piece
    captured = int(undo[UNDO_CAPTURED])
    capture_square = int(undo[UNDO_CAPTURE_SQUARE])

    if forward:
        _add_piece_delta(accumulator, feature, moved_piece, source, perspective, king_square, -1)
        _add_piece_delta(accumulator, feature, placed_piece, target, perspective, king_square, 1)
        _add_piece_delta(
            accumulator, feature, captured, capture_square, perspective, king_square, -1
        )
    else:
        _add_piece_delta(accumulator, feature, placed_piece, target, perspective, king_square, -1)
        _add_piece_delta(accumulator, feature, moved_piece, source, perspective, king_square, 1)
        _add_piece_delta(
            accumulator, feature, captured, capture_square, perspective, king_square, 1
        )

    if move & CASTLE:
        rook_source, rook_target = _castle_rook_move(target)
        rook = moving_side * ROOK
        if forward:
            _add_piece_delta(accumulator, feature, rook, rook_source, perspective, king_square, -1)
            _add_piece_delta(accumulator, feature, rook, rook_target, perspective, king_square, 1)
        else:
            _add_piece_delta(accumulator, feature, rook, rook_target, perspective, king_square, -1)
            _add_piece_delta(accumulator, feature, rook, rook_source, perspective, king_square, 1)


@njit(cache=False)
def update_after_make(
    board: np.ndarray,
    move: int,
    undo: np.ndarray,
    feature: np.ndarray,
    feature_bias: np.ndarray,
    white: np.ndarray,
    black: np.ndarray,
    king_squares: np.ndarray,
) -> None:
    """Update accumulators after ``fastcore.make_move`` has changed the board."""
    target = (move >> 6) & 63
    moved_piece = int(undo[UNDO_MOVED_PIECE])
    moving_side = 1 if moved_piece > 0 else -1
    king_move = abs(moved_piece) == KING

    if king_move and moving_side > 0:
        king_squares[0] = target
        refresh_perspective(board, 1, target, feature, feature_bias, white)
    else:
        _apply_move_delta(white, 1, int(king_squares[0]), move, undo, feature, True)

    if king_move and moving_side < 0:
        king_squares[1] = target
        refresh_perspective(board, -1, target, feature, feature_bias, black)
    else:
        _apply_move_delta(black, -1, int(king_squares[1]), move, undo, feature, True)


@njit(cache=False)
def update_after_unmake(
    board: np.ndarray,
    move: int,
    undo: np.ndarray,
    feature: np.ndarray,
    feature_bias: np.ndarray,
    white: np.ndarray,
    black: np.ndarray,
    king_squares: np.ndarray,
) -> None:
    """Reverse accumulators after ``fastcore.unmake_move`` restores the board."""
    source = move & 63
    moved_piece = int(undo[UNDO_MOVED_PIECE])
    moving_side = 1 if moved_piece > 0 else -1
    king_move = abs(moved_piece) == KING

    if king_move and moving_side > 0:
        king_squares[0] = source
        refresh_perspective(board, 1, source, feature, feature_bias, white)
    else:
        _apply_move_delta(white, 1, int(king_squares[0]), move, undo, feature, False)

    if king_move and moving_side < 0:
        king_squares[1] = source
        refresh_perspective(board, -1, source, feature, feature_bias, black)
    else:
        _apply_move_delta(black, -1, int(king_squares[1]), move, undo, feature, False)


@njit(cache=False)
def make_move_with_accumulators(
    board: np.ndarray,
    state: np.ndarray,
    move: int,
    undo: np.ndarray,
    feature: np.ndarray,
    feature_bias: np.ndarray,
    white: np.ndarray,
    black: np.ndarray,
    king_squares: np.ndarray,
) -> None:
    """Apply one compact move and its NNUE delta in the required order."""
    make_move(board, state, move, undo)
    update_after_make(
        board,
        move,
        undo,
        feature,
        feature_bias,
        white,
        black,
        king_squares,
    )


@njit(cache=False)
def unmake_move_with_accumulators(
    board: np.ndarray,
    state: np.ndarray,
    move: int,
    undo: np.ndarray,
    feature: np.ndarray,
    feature_bias: np.ndarray,
    white: np.ndarray,
    black: np.ndarray,
    king_squares: np.ndarray,
) -> None:
    """Restore one compact move and reverse its NNUE delta in the required order."""
    unmake_move(board, state, move, undo)
    update_after_unmake(
        board,
        move,
        undo,
        feature,
        feature_bias,
        white,
        black,
        king_squares,
    )


@njit(cache=False)
def evaluate_accumulator_arrays(
    white: np.ndarray,
    black: np.ndarray,
    turn: int,
    output_weight: np.ndarray,
    tempo: int,
    output_scale_cp: int,
    feature_scale: int,
    weight_scale: int,
) -> int:
    """Evaluate existing accumulators without rebuilding sparse features."""
    us = white if turn > 0 else black
    them = black if turn > 0 else white
    total = np.int64(tempo)
    for channel in range(output_weight.shape[0]):
        us_value = min(feature_scale, max(0, us[channel]))
        them_value = min(feature_scale, max(0, them[channel]))
        total += np.int64(output_weight[channel]) * (us_value - them_value)
    total *= output_scale_cp
    denominator = feature_scale * weight_scale
    if total >= 0:
        return int((total + denominator // 2) // denominator)
    return -int((-total + denominator // 2) // denominator)
