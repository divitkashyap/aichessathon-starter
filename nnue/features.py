"""Deterministic sparse king-bucket features shared by training and inference.

Each perspective sees the board with its own pieces moving towards rank eight.
Positions whose friendly king is on files e--h are mirrored horizontally.  This
folds the king to one of 32 buckets (eight ranks by four files) without losing
left/right context for the remaining pieces.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import chess
import numpy as np

PIECE_PLANES: Final = 10
FEATURES_PER_BUCKET: Final = PIECE_PLANES * 64
KING_BUCKETS: Final = 32
INPUT_FEATURES: Final = KING_BUCKETS * FEATURES_PER_BUCKET
PADDING_INDEX: Final = INPUT_FEATURES
MAX_ACTIVE_FEATURES: Final = 30


@dataclass(frozen=True, slots=True)
class EncodedPosition:
    """Fixed-width sparse indices for both king perspectives."""

    white: np.ndarray
    black: np.ndarray
    turn: np.int8


def _orient_square(square: int, perspective: chess.Color, mirror_files: bool) -> int:
    oriented = square if perspective == chess.WHITE else square ^ 56
    return oriented ^ 7 if mirror_files else oriented


def king_bucket(board: chess.Board, perspective: chess.Color) -> tuple[int, bool]:
    """Return the normalized king bucket and whether files are mirrored."""
    king = board.king(perspective)
    if king is None:
        raise ValueError("NNUE positions must contain both kings")
    oriented = king if perspective == chess.WHITE else king ^ 56
    mirror_files = chess.square_file(oriented) >= 4
    if mirror_files:
        oriented ^= 7
    return chess.square_rank(oriented) * 4 + chess.square_file(oriented), mirror_files


def perspective_indices(board: chess.Board, perspective: chess.Color) -> np.ndarray:
    """Encode all occupied squares for one normalized king perspective."""
    bucket, mirror_files = king_bucket(board, perspective)
    indices = np.full(MAX_ACTIVE_FEATURES, PADDING_INDEX, dtype=np.int32)
    # Kings are represented by the perspective bucket and are therefore not
    # duplicated as ordinary piece-square features.
    pieces = sorted(
        (square, piece)
        for square, piece in board.piece_map().items()
        if piece.piece_type != chess.KING
    )
    if len(pieces) > MAX_ACTIVE_FEATURES:
        raise ValueError(f"position has {len(pieces)} pieces; expected at most 32")

    for offset, (square, piece) in enumerate(pieces):
        relative_colour = 0 if piece.color == perspective else 1
        plane = relative_colour * 5 + piece.piece_type - 1
        normalized_square = _orient_square(square, perspective, mirror_files)
        indices[offset] = (
            bucket * FEATURES_PER_BUCKET + plane * 64 + normalized_square
        )
    indices[: len(pieces)].sort()
    return indices


def encode_board(board: chess.Board) -> EncodedPosition:
    """Encode a legal board; target scores are kept in White's perspective."""
    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        raise ValueError("NNUE positions must contain both kings")
    return EncodedPosition(
        white=perspective_indices(board, chess.WHITE),
        black=perspective_indices(board, chess.BLACK),
        turn=np.int8(1 if board.turn == chess.WHITE else -1),
    )


def encode_fen(fen: str) -> EncodedPosition:
    """Accept four- or six-field FENs, including the Lichess evaluation format."""
    fields = fen.split()
    if len(fields) == 4:
        fen = f"{fen} 0 1"
    elif len(fields) != 6:
        raise ValueError(f"expected a four- or six-field FEN, got {len(fields)} fields")
    return encode_board(chess.Board(fen))


def encode_fens(fens: Iterable[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch helper used by dataset builders and parity checks."""
    encoded = [encode_fen(fen) for fen in fens]
    if not encoded:
        empty = np.empty((0, MAX_ACTIVE_FEATURES), dtype=np.int32)
        return empty, empty.copy(), np.empty(0, dtype=np.int8)
    return (
        np.stack([position.white for position in encoded]),
        np.stack([position.black for position in encoded]),
        np.asarray([position.turn for position in encoded], dtype=np.int8),
    )


def canonical_position_key(fen: str) -> str:
    """Return the four FEN fields that determine legal moves and evaluation."""
    fields = fen.split()
    if len(fields) not in (4, 6):
        raise ValueError(f"expected a four- or six-field FEN, got {len(fields)} fields")
    return " ".join(fields[:4])


def white_cp_to_side_to_move(target_cp: np.ndarray, turn: np.ndarray) -> np.ndarray:
    """Orient White-point-of-view labels to the model's side-to-move output."""
    return target_cp * turn.astype(target_cp.dtype, copy=False)
