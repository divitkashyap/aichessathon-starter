"""A clock-safe classical chess search used by the competition entrypoint.

This is the first strength baseline.  It deliberately uses python-chess for move
generation while the search design and evaluation settle.  The hot board/search
core can then move to numba without changing the public agent contract.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Final, Literal

import chess

MATE: Final = 1_000_000
MATE_WINDOW: Final = MATE - 1_000
INFINITY: Final = 2_000_000
MAX_PLY: Final = 96
MAX_QPLY: Final = 12
TIME_CHECK_MASK: Final = 127
TT_MAX_ENTRIES: Final = 250_000
ASPIRATION_WINDOW: Final = 45
FUTILITY_MARGIN: Final = 120

# Development experiments are isolated behind environment switches so each
# heuristic can be measured against the frozen champion. Defaults stay off until
# a paired match demonstrates a gain.
ENABLE_ASPIRATION: Final = os.environ.get("EINSTEIN_ASPIRATION") == "1"
ENABLE_NULL_MOVE: Final = os.environ.get("EINSTEIN_NULL_MOVE") == "1"
ENABLE_LMR: Final = os.environ.get("EINSTEIN_LMR") == "1"
ENABLE_FUTILITY: Final = os.environ.get("EINSTEIN_FUTILITY") == "1"
ENABLE_CHECK_ORDERING: Final = os.environ.get("EINSTEIN_CHECK_ORDERING") == "1"

EXACT: Final = 0
LOWER: Final = 1
UPPER: Final = 2

PIECE_VALUE: Final = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}
PHASE_WEIGHT: Final = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}
MAX_PHASE: Final = 24
MOBILITY_WEIGHT: Final = {
    chess.PAWN: 0,
    chess.KNIGHT: 4,
    chess.BISHOP: 4,
    chess.ROOK: 2,
    chess.QUEEN: 1,
    chess.KING: 0,
}

Bound = Literal[0, 1, 2]
PositionKey = tuple[object, int]


class SearchTimeout(Exception):
    """Raised internally when a hard move deadline has been reached."""


@dataclass(slots=True)
class TTEntry:
    depth: int
    score: int
    bound: Bound
    move: chess.Move | None


@dataclass(frozen=True, slots=True)
class SearchResult:
    move: chess.Move
    score: int
    depth: int
    nodes: int
    elapsed_ms: float
    principal_variation: tuple[chess.Move, ...]


def _piece_square_value(piece_type: chess.PieceType, square: chess.Square) -> tuple[int, int]:
    """Generate small, original middlegame/endgame piece-square terms."""
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    centre = int(7 - abs(2 * file_index - 7) - abs(2 * rank_index - 7))

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
    # In the middlegame the king prefers the back rank and files near castled
    # positions. In the endgame it should centralise.
    castled_file_bonus = 12 if file_index in (1, 2, 5, 6) else 0
    return -centre * 9 - rank_index * 8 + castled_file_bonus, centre * 10


MG_TABLE: Final = {
    piece_type: tuple(_piece_square_value(piece_type, square)[0] for square in chess.SQUARES)
    for piece_type in chess.PIECE_TYPES
}
EG_TABLE: Final = {
    piece_type: tuple(_piece_square_value(piece_type, square)[1] for square in chess.SQUARES)
    for piece_type in chess.PIECE_TYPES
}


def _relative_square(square: chess.Square, color: chess.Color) -> chess.Square:
    return square if color == chess.WHITE else chess.square_mirror(square)


def _pawn_structure(board: chess.Board, color: chess.Color) -> tuple[int, int]:
    pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    counts = [0] * 8
    for square in pawns:
        counts[chess.square_file(square)] += 1

    middlegame = 0
    endgame = 0
    for count in counts:
        if count > 1:
            middlegame -= 14 * (count - 1)
            endgame -= 18 * (count - 1)

    for square in pawns:
        file_index = chess.square_file(square)
        relative_rank = chess.square_rank(_relative_square(square, color))
        neighbours = (
            (file_index > 0 and counts[file_index - 1] > 0)
            or (file_index < 7 and counts[file_index + 1] > 0)
        )
        if not neighbours:
            middlegame -= 11
            endgame -= 9

        passed = True
        for enemy_square in enemy_pawns:
            if abs(chess.square_file(enemy_square) - file_index) > 1:
                continue
            enemy_relative_rank = chess.square_rank(_relative_square(enemy_square, color))
            if enemy_relative_rank > relative_rank:
                passed = False
                break
        if passed:
            middlegame += (relative_rank * relative_rank) * 3
            endgame += (relative_rank * relative_rank) * 7

    return middlegame, endgame


def _king_safety(board: chess.Board, color: chess.Color) -> int:
    king = board.king(color)
    if king is None:
        return -MATE // 2

    relative = _relative_square(king, color)
    file_index = chess.square_file(relative)
    rank_index = chess.square_rank(relative)
    score = 0
    if rank_index == 0 and file_index in (1, 2, 6):
        score += 24

    shield_rank = chess.square_rank(king) + (1 if color == chess.WHITE else -1)
    if 0 <= shield_rank < 8:
        first_file = max(0, chess.square_file(king) - 1)
        last_file = min(7, chess.square_file(king) + 1)
        for shield_file in range(first_file, last_file + 1):
            piece = board.piece_at(chess.square(shield_file, shield_rank))
            if piece == chess.Piece(chess.PAWN, color):
                score += 11

    attacked_ring = 0
    for square in board.attacks(king):
        if board.is_attacked_by(not color, square):
            attacked_ring += 1
    return score - attacked_ring * 8


def evaluate(board: chess.Board) -> int:
    """Return a tapered position score from the side-to-move perspective."""
    middlegame = 0
    endgame = 0
    phase = 0

    for piece_type in chess.PIECE_TYPES:
        for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
            squares = board.pieces(piece_type, color)
            phase += PHASE_WEIGHT[piece_type] * len(squares)
            for square in squares:
                relative = _relative_square(square, color)
                value = PIECE_VALUE[piece_type]
                middlegame += sign * (value + MG_TABLE[piece_type][relative])
                endgame += sign * (value + EG_TABLE[piece_type][relative])
                middlegame += sign * MOBILITY_WEIGHT[piece_type] * len(board.attacks(square))

    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            middlegame += sign * 28
            endgame += sign * 38

        pawn_mg, pawn_eg = _pawn_structure(board, color)
        middlegame += sign * pawn_mg
        endgame += sign * pawn_eg
        middlegame += sign * _king_safety(board, color)

        friendly_pawns = board.pieces(chess.PAWN, color)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        for rook in board.pieces(chess.ROOK, color):
            file_mask = chess.BB_FILES[chess.square_file(rook)]
            if not friendly_pawns & file_mask:
                middlegame += sign * (18 if not enemy_pawns & file_mask else 9)

    phase = min(MAX_PHASE, phase)
    white_score = (middlegame * phase + endgame * (MAX_PHASE - phase)) // MAX_PHASE
    return white_score if board.turn == chess.WHITE else -white_score


class ChessEngine:
    """Stateful iterative-deepening chess engine for one competition game."""

    def __init__(self) -> None:
        self.tt: dict[PositionKey, TTEntry] = {}
        self.history: dict[tuple[bool, int, int, int | None], int] = {}
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
        self.deadline = math.inf
        self.nodes = 0
        self.pv: dict[PositionKey, chess.Move] = {}

    def choose_move(self, board: chess.Board, time_left_ms: int) -> SearchResult:
        started = time.perf_counter()
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("get_move called on a terminal position")

        if len(self.tt) > TT_MAX_ENTRIES:
            self.tt.clear()
        if len(self.history) > 100_000:
            self.history.clear()

        budget_ms = self._move_budget_ms(board, time_left_ms)
        self.deadline = started + budget_ms / 1000.0
        self.nodes = 0

        ordered = self._ordered_moves(board, legal_moves, self.pv.get(self._key(board)), 0)
        best_move = ordered[0]
        best_score = -INFINITY
        completed_depth = 0

        for depth in range(1, MAX_PLY):
            if depth > 1 and time.perf_counter() >= self.deadline:
                break
            try:
                score, move = self._aspiration_search(
                    board, depth, best_move, best_score, completed_depth > 0
                )
            except SearchTimeout:
                break
            best_move = move
            best_score = score
            completed_depth = depth
            self.pv[self._key(board)] = move
            if abs(score) >= MATE_WINDOW:
                break

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return SearchResult(
            move=best_move,
            score=best_score,
            depth=completed_depth,
            nodes=self.nodes,
            elapsed_ms=elapsed_ms,
            principal_variation=self._principal_variation(board, completed_depth),
        )

    def _aspiration_search(
        self,
        board: chess.Board,
        depth: int,
        previous_best: chess.Move,
        previous_score: int,
        use_window: bool,
    ) -> tuple[int, chess.Move]:
        if not ENABLE_ASPIRATION or not use_window or abs(previous_score) >= MATE_WINDOW:
            return self._search_root(board, depth, previous_best, -INFINITY, INFINITY)

        window = ASPIRATION_WINDOW
        while True:
            alpha = max(-INFINITY, previous_score - window)
            beta = min(INFINITY, previous_score + window)
            score, move = self._search_root(board, depth, previous_best, alpha, beta)
            if alpha < score < beta:
                return score, move
            window *= 2
            if window >= INFINITY:
                return self._search_root(board, depth, previous_best, -INFINITY, INFINITY)

    def _move_budget_ms(self, board: chess.Board, time_left_ms: int) -> float:
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

    def _search_root(
        self,
        board: chess.Board,
        depth: int,
        previous_best: chess.Move,
        alpha: int,
        beta: int,
    ) -> tuple[int, chess.Move]:
        self._check_time(force=True)
        moves = self._ordered_moves(board, list(board.legal_moves), previous_best, 0)
        best_score = -INFINITY
        best_move = moves[0]

        for index, move in enumerate(moves):
            self._check_time(force=True)
            board.push(move)
            try:
                if index == 0:
                    score = -self._search(board, depth - 1, -beta, -alpha, 1)
                else:
                    score = -self._search(board, depth - 1, -alpha - 1, -alpha, 1)
                    if alpha < score < beta:
                        score = -self._search(board, depth - 1, -beta, -alpha, 1)
            finally:
                board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return best_score, best_move

    def _search(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        *,
        allow_null: bool = True,
    ) -> int:
        self.nodes += 1
        self._check_time()

        if (
            board.halfmove_clock >= 100
            or board.is_insufficient_material()
            or (ply >= 4 and board.is_repetition(3))
        ):
            return 0
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply, 0)

        selective_search = ENABLE_NULL_MOVE or ENABLE_LMR or ENABLE_FUTILITY
        in_check = board.is_check() if selective_search else False
        if (
            ENABLE_NULL_MOVE
            and allow_null
            and depth >= 3
            and not in_check
            and abs(beta) < MATE_WINDOW
            and self._has_non_pawn_material(board, board.turn)
        ):
            reduction = 2 + depth // 5
            board.push(chess.Move.null())
            try:
                null_score = -self._search(
                    board,
                    max(0, depth - 1 - reduction),
                    -beta,
                    -beta + 1,
                    ply + 1,
                    allow_null=False,
                )
            finally:
                board.pop()
            if null_score >= beta:
                return null_score

        key = self._key(board)
        original_alpha = alpha
        original_beta = beta
        entry = self.tt.get(key)
        if entry is not None and entry.depth >= depth:
            score = self._score_from_tt(entry.score, ply)
            if entry.bound == EXACT:
                return score
            if entry.bound == LOWER:
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)
            if alpha >= beta:
                return score

        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if board.is_check() else 0

        tt_move = entry.move if entry is not None else None
        moves = self._ordered_moves(board, moves, tt_move, ply)
        best_score = -INFINITY
        best_move: chess.Move | None = None
        static_eval = (
            evaluate(board) if ENABLE_FUTILITY and depth <= 2 and not in_check else None
        )

        for index, move in enumerate(moves):
            quiet = not board.is_capture(move) and move.promotion is None
            gives_check = (
                board.gives_check(move) if ENABLE_LMR or ENABLE_FUTILITY else False
            )
            if (
                ENABLE_FUTILITY
                and index > 0
                and depth == 1
                and quiet
                and not gives_check
                and static_eval is not None
                and static_eval + FUTILITY_MARGIN <= alpha
            ):
                continue
            board.push(move)
            try:
                if index == 0:
                    score = -self._search(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    reduction = 0
                    if (
                        ENABLE_LMR
                        and depth >= 3
                        and index >= 4
                        and quiet
                        and not in_check
                        and not gives_check
                    ):
                        reduction = 1 + int(depth >= 6 and index >= 10)
                    score = -self._search(
                        board,
                        max(0, depth - 1 - reduction),
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                    )
                    if alpha < score < beta:
                        score = -self._search(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if quiet:
                    self._record_killer(move, ply)
                    history_key = (board.turn, move.from_square, move.to_square, move.promotion)
                    self.history[history_key] = self.history.get(history_key, 0) + depth * depth
                break

        bound: Bound
        if best_score <= original_alpha:
            bound = UPPER
        elif best_score >= original_beta:
            bound = LOWER
        else:
            bound = EXACT
        self.tt[key] = TTEntry(depth, self._score_to_tt(best_score, ply), bound, best_move)
        if best_move is not None:
            self.pv[key] = best_move
        return best_score

    def _quiescence(
        self, board: chess.Board, alpha: int, beta: int, ply: int, qdepth: int
    ) -> int:
        self.nodes += 1
        self._check_time()

        if (
            board.halfmove_clock >= 100
            or board.is_insufficient_material()
            or (ply >= 4 and board.is_repetition(3))
        ):
            return 0
        in_check = board.is_check()
        if qdepth >= MAX_QPLY:
            return evaluate(board)

        if in_check:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
        else:
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
            moves = [
                move
                for move in board.legal_moves
                if board.is_capture(move) or move.promotion is not None
            ]

        moves = self._ordered_moves(board, moves, None, min(ply, MAX_PLY - 1))
        for move in moves:
            board.push(move)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1, qdepth + 1)
            finally:
                board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def _ordered_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        preferred: chess.Move | None,
        ply: int,
    ) -> list[chess.Move]:
        def move_score(move: chess.Move) -> int:
            if move == preferred:
                return 10_000_000
            score = 0
            if move.promotion is not None:
                score += 900_000 + PIECE_VALUE[move.promotion]
            if board.is_capture(move):
                victim = board.piece_type_at(move.to_square)
                if victim is None and board.is_en_passant(move):
                    victim = chess.PAWN
                attacker = board.piece_type_at(move.from_square)
                if victim is not None and attacker is not None:
                    score += 1_000_000 + 16 * PIECE_VALUE[victim] - PIECE_VALUE[attacker]
            elif ply < MAX_PLY:
                first, second = self.killers[ply]
                if move == first:
                    score += 800_000
                elif move == second:
                    score += 700_000
                key = (board.turn, move.from_square, move.to_square, move.promotion)
                score += min(600_000, self.history.get(key, 0))
            if ENABLE_CHECK_ORDERING and board.gives_check(move):
                score += 100_000
            return score

        return sorted(moves, key=move_score, reverse=True)

    def _record_killer(self, move: chess.Move, ply: int) -> None:
        if ply >= MAX_PLY or self.killers[ply][0] == move:
            return
        self.killers[ply][1] = self.killers[ply][0]
        self.killers[ply][0] = move

    @staticmethod
    def _has_non_pawn_material(board: chess.Board, color: chess.Color) -> bool:
        return bool(
            board.occupied_co[color]
            & ~int(board.pieces(chess.PAWN, color))
            & ~int(board.pieces(chess.KING, color))
        )

    def _principal_variation(self, board: chess.Board, depth: int) -> tuple[chess.Move, ...]:
        line: list[chess.Move] = []
        copy = board.copy(stack=False)
        seen: set[PositionKey] = set()
        for _ in range(depth):
            key = self._key(copy)
            if key in seen:
                break
            seen.add(key)
            move = self.pv.get(key)
            if move is None or move not in copy.legal_moves:
                break
            line.append(move)
            copy.push(move)
        return tuple(line)

    def _check_time(self, *, force: bool = False) -> None:
        if (force or not self.nodes & TIME_CHECK_MASK) and time.perf_counter() >= self.deadline:
            raise SearchTimeout

    @staticmethod
    def _key(board: chess.Board) -> PositionKey:
        return board._transposition_key(), min(board.halfmove_clock, 100)

    @staticmethod
    def _score_to_tt(score: int, ply: int) -> int:
        if score >= MATE_WINDOW:
            return score + ply
        if score <= -MATE_WINDOW:
            return score - ply
        return score

    @staticmethod
    def _score_from_tt(score: int, ply: int) -> int:
        if score >= MATE_WINDOW:
            return score - ply
        if score <= -MATE_WINDOW:
            return score + ply
        return score
