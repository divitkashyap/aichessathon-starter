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

try:
    from .lmr_core import (
        BISHOP,
        CAPTURE,
        EN_PASSANT,
        HALFMOVE,
        HASH_KEY,
        KING,
        KNIGHT,
        PAWN,
        SIDE,
        UNDO_SIZE,
        generate_legal_moves,
        has_legal_move,
        is_square_attacked,
        make_move,
        move_to_uci,
        position_from_fen,
        unmake_move,
    )
except ImportError:
    from lmr_core import (
        BISHOP,
        CAPTURE,
        EN_PASSANT,
        HALFMOVE,
        HASH_KEY,
        KING,
        KNIGHT,
        PAWN,
        SIDE,
        UNDO_SIZE,
        generate_legal_moves,
        has_legal_move,
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
MAX_SEARCH_PLY: Final = 128
MAX_GAME_HISTORY: Final = 600
MAX_CHECK_EXTENSIONS: Final = 4
EXTENSION_HASH: Final = np.array(
    [0, 0x123456789ABCDE, 0x2468ACE13579BD, 0x3456789ABCDEF1, 0x456789ABCDEF12],
    dtype=np.int64,
)
TT_SIZE: Final = 1 << 18
TT_MASK: Final = TT_SIZE - 1
TT_EXACT: Final = 0
TT_LOWER: Final = 1
TT_UPPER: Final = 2

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
HALFMOVE_HASH: Final = np.array(
    [((index + 1) * 0x4F1BBCDCBFA54001) & ((1 << 63) - 1) for index in range(101)],
    dtype=np.int64,
)


@dataclass(frozen=True, slots=True)
class FastSearchResult:
    move: str
    score: int
    depth: int
    nodes: int
    elapsed_ms: float


@dataclass(slots=True)
class FastSearchMemory:
    """Reusable search tables for one game process."""

    tt_keys: np.ndarray
    tt_scores: np.ndarray
    tt_moves: np.ndarray
    tt_depths: np.ndarray
    tt_bounds: np.ndarray
    killer_moves: np.ndarray
    history_scores: np.ndarray


@njit(cache=False)
def evaluate(board: np.ndarray, state: np.ndarray) -> int:
    middlegame = 0
    endgame = 0
    phase = 0
    white_bishops = 0
    black_bishops = 0
    nonking_count = 0
    bishop_count = 0
    knight_count = 0
    bishops_same_color = True
    bishop_color = -1
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
        if piece_type != KING:
            nonking_count += 1
            if piece_type == BISHOP:
                bishop_count += 1
                square_color = ((square & 7) + (square >> 3)) & 1
                if bishop_color < 0:
                    bishop_color = square_color
                elif bishop_color != square_color:
                    bishops_same_color = False
            elif piece_type == KNIGHT:
                knight_count += 1
    insufficient_material = (
        nonking_count == 0
        or (nonking_count == 1 and (bishop_count == 1 or knight_count == 1))
        or (
            bishop_count > 0
            and bishop_count == nonking_count
            and bishops_same_color
        )
    )
    if insufficient_material:
        return 0
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
def _tactical_move_score(board: np.ndarray, move: int) -> int:
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
def _move_score(
    board: np.ndarray,
    state: np.ndarray,
    move: int,
    preferred: int,
    ply: int,
    killer_moves: np.ndarray,
    history_scores: np.ndarray,
) -> int:
    if move == preferred:
        return 10_000_000
    tactical_score = _tactical_move_score(board, move)
    if tactical_score:
        return tactical_score
    if ply < MAX_SEARCH_PLY:
        if move == killer_moves[ply, 0]:
            return 800_000
        if move == killer_moves[ply, 1]:
            return 700_000
    side_index = 0 if state[SIDE] > 0 else 1
    source = move & 63
    target = (move >> 6) & 63
    return int(history_scores[side_index, source * 64 + target])


@njit(cache=False)
def _order_moves(
    board: np.ndarray,
    state: np.ndarray,
    moves: np.ndarray,
    preferred: int,
    ply: int,
    killer_moves: np.ndarray,
    history_scores: np.ndarray,
) -> None:
    """In-place selection ordering avoids Python objects and sort allocations."""
    count = len(moves)
    for index in range(count - 1):
        best_index = index
        best_score = _move_score(
            board,
            state,
            int(moves[index]),
            preferred,
            ply,
            killer_moves,
            history_scores,
        )
        for candidate in range(index + 1, count):
            score = _move_score(
                board,
                state,
                int(moves[candidate]),
                preferred,
                ply,
                killer_moves,
                history_scores,
            )
            if score > best_score:
                best_index = candidate
                best_score = score
        if best_index != index:
            moves[index], moves[best_index] = moves[best_index], moves[index]


@njit(cache=False)
def _prepare_move_scores(board, state, moves, preferred, ply, killer_moves, history_scores):
    # Freeze scores before searching children: their history updates must not
    # change this node's ordering relative to the eager reference.
    scores = np.empty(len(moves), dtype=np.int64)
    for index in range(len(moves)):
        scores[index] = _move_score(
            board, state, int(moves[index]), preferred, ply,
            killer_moves, history_scores,
        )
    return scores


@njit(cache=False)
def _pick_next_move(moves, scores, index):
    # Identical tie handling and swaps to the original selection sort, but
    # defer each suffix until it is actually searched (many nodes cut off).
    best = index
    for candidate in range(index + 1, len(moves)):
        if scores[candidate] > scores[best]:
            best = candidate
    if best != index:
        moves[index], moves[best] = moves[best], moves[index]
        scores[index], scores[best] = scores[best], scores[index]


@njit(cache=False)
def _order_quiescence_moves(
    board: np.ndarray,
    moves: np.ndarray,
    include_quiets: bool,
) -> int:
    """Put relevant tactical moves first and return the number to search."""
    count = len(moves)
    if not include_quiets:
        tactical_count = 0
        for index in range(count):
            move = int(moves[index])
            if move & CAPTURE or (move >> 12) & 7:
                moves[tactical_count], moves[index] = moves[index], moves[tactical_count]
                tactical_count += 1
        count = tactical_count

    for index in range(count - 1):
        best_index = index
        best_score = _tactical_move_score(board, int(moves[index]))
        for candidate in range(index + 1, count):
            score = _tactical_move_score(board, int(moves[candidate]))
            if score > best_score:
                best_index = candidate
                best_score = score
        if best_index != index:
            moves[index], moves[best_index] = moves[best_index], moves[index]
    return count


@njit(cache=False)
def _record_quiet_cutoff(
    state: np.ndarray,
    move: int,
    depth: int,
    ply: int,
    killer_moves: np.ndarray,
    history_scores: np.ndarray,
) -> None:
    if move & CAPTURE or (move >> 12) & 7:
        return
    if ply < MAX_SEARCH_PLY and move != killer_moves[ply, 0]:
        killer_moves[ply, 1] = killer_moves[ply, 0]
        killer_moves[ply, 0] = move
    side_index = 0 if state[SIDE] > 0 else 1
    index = (move & 63) * 64 + ((move >> 6) & 63)
    history_scores[side_index, index] = min(
        500_000,
        int(history_scores[side_index, index]) + depth * depth,
    )


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
def _is_threefold(
    key: int,
    halfmove_clock: int,
    history: np.ndarray,
    history_count: int,
    path: np.ndarray,
    path_count: int,
) -> bool:
    """Return whether the current node is the third same-side occurrence."""
    current_index = history_count + path_count - 1
    first_index = max(0, current_index - halfmove_clock)
    matches = 0
    index = current_index - 2
    while index >= first_index:
        previous = history[index] if index < history_count else path[index - history_count]
        if previous == key:
            matches += 1
            if matches >= 2:
                return True
        index -= 2
    return False


@njit(cache=False)
def _tt_key(state: np.ndarray) -> np.int64:
    halfmove = min(100, int(state[HALFMOVE]))
    return np.int64(state[HASH_KEY]) ^ HALFMOVE_HASH[halfmove]


@njit(cache=False)
def _score_to_tt(score: int, ply: int) -> int:
    if score >= MATE - MAX_SEARCH_PLY:
        return score + ply
    if score <= -MATE + MAX_SEARCH_PLY:
        return score - ply
    return score


@njit(cache=False)
def _score_from_tt(score: int, ply: int) -> int:
    if score >= MATE - MAX_SEARCH_PLY:
        return score - ply
    if score <= -MATE + MAX_SEARCH_PLY:
        return score + ply
    return score


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
    history: np.ndarray,
    history_count: int,
    path: np.ndarray,
    path_count: int,
) -> int:
    stats[0] += 1
    if _out_of_time(stats, deadline):
        return 0
    if state[HALFMOVE] >= 100:
        if _in_check(board, state) and not has_legal_move(board, state):
            return -MATE + ply
        return 0
    if _is_threefold(
        int(state[HASH_KEY]),
        int(state[HALFMOVE]),
        history,
        history_count,
        path,
        path_count,
    ):
        return 0

    in_check = _in_check(board, state)
    if qdepth >= MAX_QDEPTH:
        if not has_legal_move(board, state):
            return -MATE + ply if in_check else 0
        return evaluate(board, state)
    if not in_check:
        stand_pat = evaluate(board, state)
        if stand_pat >= beta:
            if not has_legal_move(board, state):
                return 0
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

    moves = generate_legal_moves(board, state)
    if len(moves) == 0:
        return -MATE + ply if in_check else 0
    move_count = _order_quiescence_moves(board, moves, in_check)
    undo = np.empty(UNDO_SIZE, dtype=np.int64)
    for move_index in range(move_count):
        move = int(moves[move_index])
        make_move(board, state, move, undo)
        path[path_count] = state[HASH_KEY]
        score = -_quiescence(
            board,
            state,
            -beta,
            -alpha,
            ply + 1,
            qdepth + 1,
            stats,
            deadline,
            history,
            history_count,
            path,
            path_count + 1,
        )
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
    history: np.ndarray,
    history_count: int,
    path: np.ndarray,
    path_count: int,
    tt_keys: np.ndarray,
    tt_scores: np.ndarray,
    tt_moves: np.ndarray,
    tt_depths: np.ndarray,
    tt_bounds: np.ndarray,
    killer_moves: np.ndarray,
    history_scores: np.ndarray,
    extensions_used: int = 0,
) -> int:
    stats[0] += 1
    if _out_of_time(stats, deadline):
        return 0
    if state[HALFMOVE] >= 100:
        if _in_check(board, state) and not has_legal_move(board, state):
            return -MATE + ply
        return 0
    if _is_threefold(
        int(state[HASH_KEY]),
        int(state[HALFMOVE]),
        history,
        history_count,
        path,
        path_count,
    ):
        return 0
    if depth <= 0:
        return _quiescence(
            board,
            state,
            alpha,
            beta,
            ply,
            0,
            stats,
            deadline,
            history,
            history_count,
            path,
            path_count,
        )

    # Spend a bounded amount of extra depth on forcing check sequences.
    # Leaves still enter quiescence; only positive-depth checked nodes extend.
    parent_in_check = _in_check(board, state)
    if parent_in_check and extensions_used < MAX_CHECK_EXTENSIONS:
        depth += 1
        extensions_used += 1
    original_alpha = alpha
    original_beta = beta
    # Different remaining extension budgets have different search semantics.
    key = _tt_key(state) ^ EXTENSION_HASH[extensions_used]
    tt_index = int(key) & TT_MASK
    preferred = 0
    if tt_depths[tt_index] >= 0 and tt_keys[tt_index] == key:
        preferred = int(tt_moves[tt_index])
        if tt_depths[tt_index] >= depth:
            tt_score = _score_from_tt(int(tt_scores[tt_index]), ply)
            if tt_bounds[tt_index] == TT_EXACT:
                return tt_score
            if tt_bounds[tt_index] == TT_LOWER and tt_score > alpha:
                alpha = tt_score
            elif tt_bounds[tt_index] == TT_UPPER and tt_score < beta:
                beta = tt_score
            if alpha >= beta:
                return tt_score

    moves = generate_legal_moves(board, state)
    if len(moves) == 0:
        return -MATE + ply if _in_check(board, state) else 0
    move_scores = _prepare_move_scores(
        board,
        state,
        moves,
        preferred,
        ply,
        killer_moves,
        history_scores,
    )
    best_score = -INFINITY
    best_move = int(moves[0])
    undo = np.empty(UNDO_SIZE, dtype=np.int64)
    for move_index in range(len(moves)):
        _pick_next_move(moves, move_scores, move_index)
        move = int(moves[move_index])
        make_move(board, state, move, undo)
        path[path_count] = state[HASH_KEY]
        if move_index == 0:
            score = -_negamax(
                board,
                state,
                depth - 1,
                -beta,
                -alpha,
                ply + 1,
                stats,
                deadline,
                history,
                history_count,
                path,
                path_count + 1,
                tt_keys,
                tt_scores,
                tt_moves,
                tt_depths,
                tt_bounds,
                killer_moves,
                history_scores,
                extensions_used,
            )
        else:
            # Principal-variation search: later moves first prove they cannot
            # beat alpha using a one-point window.  A conservative late-move
            # reduction applies only to quiet, late moves that are neither
            # killer nor TT-preferred, and only when both positions are safe
            # from check.  A reduced fail-high is verified at full depth before
            # the existing full-window PVS re-search.
            quiet_nonpromotion = (move & CAPTURE) == 0 and ((move >> 12) & 7) == 0
            is_killer = ply < MAX_SEARCH_PLY and (
                move == killer_moves[ply, 0] or move == killer_moves[ply, 1]
            )
            can_reduce = (
                depth >= 3
                and ply > 0
                and move_index >= 4
                and not parent_in_check
                and not _in_check(board, state)
                and quiet_nonpromotion
                and not is_killer
                and move != preferred
            )
            probe_depth = depth - 2 if can_reduce else depth - 1
            score = -_negamax(
                board,
                state,
                probe_depth,
                -alpha - 1,
                -alpha,
                ply + 1,
                stats,
                deadline,
                history,
                history_count,
                path,
                path_count + 1,
                tt_keys,
                tt_scores,
                tt_moves,
                tt_depths,
                tt_bounds,
                killer_moves,
                history_scores,
                extensions_used,
            )
            if can_reduce and stats[1] == 0 and score > alpha:
                score = -_negamax(
                    board,
                    state,
                    depth - 1,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    stats,
                    deadline,
                    history,
                    history_count,
                    path,
                    path_count + 1,
                    tt_keys,
                    tt_scores,
                    tt_moves,
                    tt_depths,
                    tt_bounds,
                    killer_moves,
                    history_scores,
                    extensions_used,
                )
            if stats[1] == 0 and alpha < score < beta:
                score = -_negamax(
                    board,
                    state,
                    depth - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                    stats,
                    deadline,
                    history,
                    history_count,
                    path,
                    path_count + 1,
                    tt_keys,
                    tt_scores,
                    tt_moves,
                    tt_depths,
                    tt_bounds,
                    killer_moves,
                    history_scores,
                    extensions_used,
                )
        unmake_move(board, state, move, undo)
        if stats[1] != 0:
            return 0
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            _record_quiet_cutoff(
                state,
                move,
                depth,
                ply,
                killer_moves,
                history_scores,
            )
            break
    if tt_depths[tt_index] <= depth or tt_keys[tt_index] == key:
        bound = TT_EXACT
        if best_score <= original_alpha:
            bound = TT_UPPER
        elif best_score >= original_beta:
            bound = TT_LOWER
        tt_keys[tt_index] = key
        tt_scores[tt_index] = _score_to_tt(best_score, ply)
        tt_moves[tt_index] = best_move
        tt_depths[tt_index] = depth
        tt_bounds[tt_index] = bound
    return best_score


@njit(cache=False)
def _search_root(
    board: np.ndarray,
    state: np.ndarray,
    depth: int,
    deadline: float,
    history: np.ndarray,
    history_count: int,
    tt_keys: np.ndarray,
    tt_scores: np.ndarray,
    tt_moves: np.ndarray,
    tt_depths: np.ndarray,
    tt_bounds: np.ndarray,
    killer_moves: np.ndarray,
    history_scores: np.ndarray,
) -> tuple[int, int, int, bool]:
    stats = np.zeros(2, dtype=np.int64)
    moves = generate_legal_moves(board, state)
    if len(moves) == 0:
        return 0, 0, 0, True
    root_key = _tt_key(state)
    root_index = int(root_key) & TT_MASK
    preferred = 0
    if tt_depths[root_index] >= 0 and tt_keys[root_index] == root_key:
        preferred = int(tt_moves[root_index])
    move_scores = _prepare_move_scores(
        board,
        state,
        moves,
        preferred,
        0,
        killer_moves,
        history_scores,
    )
    best_move = int(moves[0])
    best_score = -INFINITY
    alpha = -INFINITY
    undo = np.empty(UNDO_SIZE, dtype=np.int64)
    path = np.empty(MAX_SEARCH_PLY, dtype=np.int64)
    for move_index in range(len(moves)):
        _pick_next_move(moves, move_scores, move_index)
        move = int(moves[move_index])
        make_move(board, state, move, undo)
        path[0] = state[HASH_KEY]
        if move_index == 0:
            score = -_negamax(
                board,
                state,
                depth - 1,
                -INFINITY,
                -alpha,
                1,
                stats,
                deadline,
                history,
                history_count,
                path,
                1,
                tt_keys,
                tt_scores,
                tt_moves,
                tt_depths,
                tt_bounds,
                killer_moves,
                history_scores,
            )
        else:
            score = -_negamax(
                board,
                state,
                depth - 1,
                -alpha - 1,
                -alpha,
                1,
                stats,
                deadline,
                history,
                history_count,
                path,
                1,
                tt_keys,
                tt_scores,
                tt_moves,
                tt_depths,
                tt_bounds,
                killer_moves,
                history_scores,
            )
            if stats[1] == 0 and score > alpha:
                score = -_negamax(
                    board,
                    state,
                    depth - 1,
                    -INFINITY,
                    -alpha,
                    1,
                    stats,
                    deadline,
                    history,
                    history_count,
                    path,
                    1,
                    tt_keys,
                    tt_scores,
                    tt_moves,
                    tt_depths,
                    tt_bounds,
                    killer_moves,
                    history_scores,
                )
        unmake_move(board, state, move, undo)
        if stats[1] != 0:
            return best_score, best_move, int(stats[0]), False
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
    tt_keys[root_index] = root_key
    tt_scores[root_index] = _score_to_tt(best_score, 0)
    tt_moves[root_index] = best_move
    tt_depths[root_index] = depth
    tt_bounds[root_index] = TT_EXACT
    return best_score, best_move, int(stats[0]), True


def _history_array(history: list[int] | tuple[int, ...], root_key: int) -> tuple[np.ndarray, int]:
    values = list(history[-MAX_GAME_HISTORY:])
    if not values or values[-1] != root_key:
        values.append(root_key)
    values = values[-MAX_GAME_HISTORY:]
    array = np.zeros(MAX_GAME_HISTORY, dtype=np.int64)
    array[: len(values)] = values
    return array, len(values)


def _new_tt() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros(TT_SIZE, dtype=np.int64),
        np.zeros(TT_SIZE, dtype=np.int32),
        np.zeros(TT_SIZE, dtype=np.int32),
        np.full(TT_SIZE, -1, dtype=np.int8),
        np.zeros(TT_SIZE, dtype=np.int8),
    )


def _new_move_ordering() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((MAX_SEARCH_PLY, 2), dtype=np.int32),
        np.zeros((2, 64 * 64), dtype=np.int32),
    )


def new_search_memory() -> FastSearchMemory:
    tt = _new_tt()
    move_ordering = _new_move_ordering()
    return FastSearchMemory(*tt, *move_ordering)


def search_root(
    board: np.ndarray,
    state: np.ndarray,
    depth: int,
    deadline: float,
    history: list[int] | tuple[int, ...] = (),
) -> tuple[int, int, int, bool]:
    history_array, history_count = _history_array(history, int(state[HASH_KEY]))
    tt = _new_tt()
    move_ordering = _new_move_ordering()
    return _search_root(
        board,
        state,
        depth,
        deadline,
        history_array,
        history_count,
        *tt,
        *move_ordering,
    )


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


def search_timed(
    fen: str,
    time_left_ms: int,
    history: list[int] | tuple[int, ...] = (),
    memory: FastSearchMemory | None = None,
) -> FastSearchResult:
    """Iteratively deepen until the conservative per-move deadline."""
    source = chess.Board(fen)
    board, state = position_from_fen(fen)
    started = time.perf_counter()
    deadline = started + _move_budget_ms(source, time_left_ms) / 1_000
    legal = generate_legal_moves(board, state)
    if len(legal) == 0:
        raise ValueError("search requested from a terminal position")
    history_array, history_count = _history_array(history, int(state[HASH_KEY]))
    if memory is None:
        memory = new_search_memory()
    tt = (
        memory.tt_keys,
        memory.tt_scores,
        memory.tt_moves,
        memory.tt_depths,
        memory.tt_bounds,
    )
    move_ordering = (memory.killer_moves, memory.history_scores)

    best_move = int(legal[0])
    best_score = -INFINITY
    completed_depth = 0
    total_nodes = 0
    for depth in range(1, 64):
        score, move, nodes, completed = _search_root(
            board,
            state,
            depth,
            deadline,
            history_array,
            history_count,
            *tt,
            *move_ordering,
        )
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
