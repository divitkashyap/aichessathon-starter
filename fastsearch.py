"""Compiled alpha-beta prototype built on :mod:`fastcore`.

The competition agent does not import this module yet.  It is a challenger that
must pass tactical, clock and paired-game gates before replacing the V1 search.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Final

import chess
import numpy as np
from numba import njit, objmode

from fastcore import (
    CAPTURE,
    EN_PASSANT,
    HALFMOVE,
    KING,
    PAWN,
    SIDE,
    UNDO_SIZE,
    generate_legal_moves,
    is_square_attacked,
    make_move,
    move_to_uci,
    position_from_fen,
    unmake_move,
)

MATE: Final = 1_000_000
INFINITY: Final = 2_000_000
MAX_QDEPTH: Final = 10
TIME_CHECK_MASK: Final = 2_047

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


@dataclass(frozen=True, slots=True)
class FastSearchResult:
    move: str
    score: int
    depth: int
    nodes: int
    elapsed_ms: float


@njit(cache=False)
def evaluate(board: np.ndarray, state: np.ndarray) -> int:
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
    return white_score if state[SIDE] > 0 else -white_score


@njit(cache=False)
def _in_check(board: np.ndarray, state: np.ndarray) -> bool:
    side = int(state[SIDE])
    for square in range(64):
        if board[square] == side * KING:
            return is_square_attacked(board, square, -side)
    return True


@njit(cache=False)
def _move_score(board: np.ndarray, state: np.ndarray, move: int) -> int:
    source = move & 63
    target = (move >> 6) & 63
    promotion = (move >> 12) & 7
    score = 0
    if promotion:
        score += 900_000 + int(PIECE_VALUES[promotion])
    if move & CAPTURE:
        victim = PAWN if move & EN_PASSANT else abs(int(board[target]))
        attacker = abs(int(board[source]))
        score += 1_000_000 + 16 * int(PIECE_VALUES[victim])
        score -= int(PIECE_VALUES[attacker])
    return score


@njit(cache=False)
def _order_moves(board: np.ndarray, state: np.ndarray, moves: np.ndarray) -> None:
    """In-place selection ordering avoids Python objects and sort allocations."""
    count = len(moves)
    for index in range(count - 1):
        best_index = index
        best_score = _move_score(board, state, int(moves[index]))
        for candidate in range(index + 1, count):
            score = _move_score(board, state, int(moves[candidate]))
            if score > best_score:
                best_index = candidate
                best_score = score
        if best_index != index:
            moves[index], moves[best_index] = moves[best_index], moves[index]


@njit(cache=False)
def _out_of_time(stats: np.ndarray, deadline: float) -> bool:
    if stats[1] != 0:
        return True
    if deadline == math.inf or int(stats[0]) & TIME_CHECK_MASK:
        return False
    with objmode(now="float64"):
        now = time.perf_counter()
    if now >= deadline:
        stats[1] = 1
        return True
    return False


@njit(cache=False)
def _quiescence(
    board: np.ndarray,
    state: np.ndarray,
    alpha: int,
    beta: int,
    ply: int,
    qdepth: int,
    stats: np.ndarray,
    deadline: float,
) -> int:
    stats[0] += 1
    if _out_of_time(stats, deadline):
        return 0
    if state[HALFMOVE] >= 100:
        return 0

    in_check = _in_check(board, state)
    if qdepth >= MAX_QDEPTH:
        return evaluate(board, state)
    if not in_check:
        stand_pat = evaluate(board, state)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

    moves = generate_legal_moves(board, state)
    if in_check and len(moves) == 0:
        return -MATE + ply
    _order_moves(board, state, moves)
    undo = np.empty(UNDO_SIZE, dtype=np.int64)
    for move_value in moves:
        move = int(move_value)
        if not in_check and not (move & CAPTURE) and not ((move >> 12) & 7):
            continue
        make_move(board, state, move, undo)
        score = -_quiescence(board, state, -beta, -alpha, ply + 1, qdepth + 1, stats, deadline)
        unmake_move(board, state, move, undo)
        if stats[1] != 0:
            return 0
        if score >= beta:
            return score
        if score > alpha:
            alpha = score
    return alpha


@njit(cache=False)
def _negamax(
    board: np.ndarray,
    state: np.ndarray,
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    stats: np.ndarray,
    deadline: float,
) -> int:
    stats[0] += 1
    if _out_of_time(stats, deadline):
        return 0
    if state[HALFMOVE] >= 100:
        return 0
    if depth <= 0:
        return _quiescence(board, state, alpha, beta, ply, 0, stats, deadline)

    moves = generate_legal_moves(board, state)
    if len(moves) == 0:
        return -MATE + ply if _in_check(board, state) else 0
    _order_moves(board, state, moves)
    best_score = -INFINITY
    undo = np.empty(UNDO_SIZE, dtype=np.int64)
    for move_value in moves:
        move = int(move_value)
        make_move(board, state, move, undo)
        score = -_negamax(board, state, depth - 1, -beta, -alpha, ply + 1, stats, deadline)
        unmake_move(board, state, move, undo)
        if stats[1] != 0:
            return 0
        if score > best_score:
            best_score = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best_score


@njit(cache=False)
def search_root(
    board: np.ndarray,
    state: np.ndarray,
    depth: int,
    deadline: float,
) -> tuple[int, int, int, bool]:
    stats = np.zeros(2, dtype=np.int64)
    moves = generate_legal_moves(board, state)
    if len(moves) == 0:
        return 0, 0, 0, True
    _order_moves(board, state, moves)
    best_move = int(moves[0])
    best_score = -INFINITY
    alpha = -INFINITY
    undo = np.empty(UNDO_SIZE, dtype=np.int64)
    for move_value in moves:
        move = int(move_value)
        make_move(board, state, move, undo)
        score = -_negamax(board, state, depth - 1, -INFINITY, -alpha, 1, stats, deadline)
        unmake_move(board, state, move, undo)
        if stats[1] != 0:
            return best_score, best_move, int(stats[0]), False
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
    return best_score, best_move, int(stats[0]), True


def search_fixed_depth(fen: str, depth: int) -> FastSearchResult:
    board, state = position_from_fen(fen)
    started = time.perf_counter()
    score, move, nodes, completed = search_root(board, state, depth, math.inf)
    if not completed:
        raise RuntimeError("unbounded fixed-depth search stopped unexpectedly")
    return FastSearchResult(
        move=move_to_uci(move),
        score=score,
        depth=depth,
        nodes=nodes,
        elapsed_ms=(time.perf_counter() - started) * 1_000,
    )


def _move_budget_ms(board: chess.Board, time_left_ms: int) -> float:
    remaining = max(1.0, float(time_left_ms))
    reserve = min(5_000.0, max(150.0, remaining * 0.07))
    usable = max(1.0, remaining - reserve)
    if board.fullmove_number < 20:
        moves_to_go = 34
    elif board.fullmove_number < 40:
        moves_to_go = 26
    else:
        moves_to_go = 20
    target = usable / moves_to_go + 325.0
    budget = min(5_000.0, target, remaining * 0.12)
    margin = max(10.0, min(100.0, remaining * 0.02))
    return max(2.0, min(budget, remaining - margin))


def search_timed(fen: str, time_left_ms: int) -> FastSearchResult:
    """Iteratively deepen until the conservative per-move deadline."""
    source = chess.Board(fen)
    board, state = position_from_fen(fen)
    started = time.perf_counter()
    deadline = started + _move_budget_ms(source, time_left_ms) / 1_000
    legal = generate_legal_moves(board, state)
    if len(legal) == 0:
        raise ValueError("search requested from a terminal position")

    best_move = int(legal[0])
    best_score = -INFINITY
    completed_depth = 0
    total_nodes = 0
    for depth in range(1, 64):
        score, move, nodes, completed = search_root(board, state, depth, deadline)
        total_nodes += nodes
        if not completed:
            break
        best_move = move
        best_score = score
        completed_depth = depth
        if abs(score) >= MATE - 100:
            break
    return FastSearchResult(
        move=move_to_uci(best_move),
        score=best_score,
        depth=completed_depth,
        nodes=total_nodes,
        elapsed_ms=(time.perf_counter() - started) * 1_000,
    )


def warm_up() -> None:
    """Spend Numba compilation time during the platform's initialization window."""
    board, state = position_from_fen(chess.STARTING_FEN)
    search_root(board, state, 2, math.inf)
