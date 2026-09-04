"""Differential stress test for the experimental Numba board core."""

from __future__ import annotations

import argparse
import random
import time

import chess
import numpy as np

from fastcore import (
    CASTLE,
    EN_PASSANT,
    UNDO_SIZE,
    generate_legal_moves,
    make_move,
    move_to_uci,
    position_from_fen,
    unmake_move,
)


def _assert_equal(actual: np.ndarray, expected: np.ndarray, message: str) -> None:
    if not np.array_equal(actual, expected):
        raise AssertionError(
            f"{message}\nactual={actual.tolist()}\nexpected={expected.tolist()}"
        )


def verify_position(board: chess.Board, rng: random.Random) -> tuple[bool, bool, bool]:
    """Check legal moves, one state transition and its exact reversal."""
    fen = board.fen(en_passant="fen")
    pieces, state = position_from_fen(fen)
    original_pieces = pieces.copy()
    original_state = state.copy()
    encoded_moves = generate_legal_moves(pieces, state)
    actual = {move_to_uci(int(move)): int(move) for move in encoded_moves}
    expected_moves = list(board.legal_moves)
    expected = {move.uci() for move in expected_moves}
    if actual.keys() != expected:
        missing = sorted(expected - actual.keys())
        extra = sorted(actual.keys() - expected)
        raise AssertionError(f"legal mismatch at {fen}\nmissing={missing}\nextra={extra}")
    if not expected_moves:
        return False, False, False

    special = [
        move
        for move in expected_moves
        if board.is_castling(move) or board.is_en_passant(move) or move.promotion
    ]
    reference_move = rng.choice(special or expected_moves)
    move_uci = reference_move.uci()
    encoded = actual[move_uci]
    was_castle = bool(encoded & CASTLE)
    was_en_passant = bool(encoded & EN_PASSANT)
    was_promotion = reference_move.promotion is not None

    undo = np.empty(UNDO_SIZE, dtype=np.int16)
    make_move(pieces, state, encoded, undo)
    board.push(reference_move)
    expected_pieces, expected_state = position_from_fen(board.fen(en_passant="fen"))
    _assert_equal(pieces, expected_pieces, f"board mismatch after {move_uci} from {fen}")
    _assert_equal(state, expected_state, f"state mismatch after {move_uci} from {fen}")

    unmake_move(pieces, state, encoded, undo)
    _assert_equal(pieces, original_pieces, f"board not restored after {move_uci} from {fen}")
    _assert_equal(state, original_state, f"state not restored after {move_uci} from {fen}")
    return was_castle, was_en_passant, was_promotion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--progress", type=int, default=10_000)
    arguments = parser.parse_args()

    rng = random.Random(arguments.seed)
    board = chess.Board()
    castles = en_passants = promotions = resets = 0
    started = time.perf_counter()

    # Compile before timing the campaign.
    generate_legal_moves(*position_from_fen(board.fen()))
    for index in range(1, arguments.positions + 1):
        was_castle, was_en_passant, was_promotion = verify_position(board, rng)
        castles += int(was_castle)
        en_passants += int(was_en_passant)
        promotions += int(was_promotion)

        if board.is_game_over() or board.ply() >= 250:
            board.reset()
            resets += 1
        if arguments.progress > 0 and index % arguments.progress == 0:
            elapsed = time.perf_counter() - started
            print(f"verified {index:,} positions ({index / elapsed:,.0f}/s)", flush=True)

    elapsed = time.perf_counter() - started
    print(
        f"PASS positions={arguments.positions:,} seed={arguments.seed} "
        f"elapsed={elapsed:.2f}s rate={arguments.positions / elapsed:,.0f}/s "
        f"castles={castles} en_passants={en_passants} promotions={promotions} "
        f"game_resets={resets}"
    )


if __name__ == "__main__":
    main()
