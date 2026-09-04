"""Correctness tests for the compiled search challenger."""

import time
import unittest

import chess
import numpy as np

from fastcore import HASH_KEY, move_to_uci, position_from_fen
from fastsearch import evaluate, search_fixed_depth, search_root, search_timed


class FastSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        search_fixed_depth(chess.STARTING_FEN, 1)

    def test_starting_evaluation_is_symmetric(self) -> None:
        board, state = position_from_fen(chess.STARTING_FEN)
        self.assertEqual(evaluate(board, state), 0)

    def test_mate_in_one_is_taken(self) -> None:
        fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
        result = search_fixed_depth(fen, 2)
        board = chess.Board(fen)
        board.push_uci(result.move)
        self.assertTrue(board.is_checkmate())

    def test_search_restores_board_and_state(self) -> None:
        pieces, state = position_from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        )
        original_pieces = pieces.copy()
        original_state = state.copy()
        search_root(pieces, state, 3, float("inf"))
        np.testing.assert_array_equal(pieces, original_pieces)
        np.testing.assert_array_equal(state, original_state)

    def test_deadline_interrupt_restores_state(self) -> None:
        pieces, state = position_from_fen(chess.STARTING_FEN)
        original_pieces = pieces.copy()
        original_state = state.copy()
        started = time.perf_counter()
        _, _, _, completed = search_root(pieces, state, 12, started + 0.05)
        elapsed = time.perf_counter() - started
        self.assertFalse(completed)
        self.assertLess(elapsed, 0.1)
        np.testing.assert_array_equal(pieces, original_pieces)
        np.testing.assert_array_equal(state, original_state)

    def test_timed_search_returns_a_legal_move_under_low_clock(self) -> None:
        started = time.perf_counter()
        result = search_timed(chess.STARTING_FEN, 100)
        elapsed = time.perf_counter() - started
        self.assertIn(chess.Move.from_uci(result.move), chess.Board().legal_moves)
        self.assertGreaterEqual(result.depth, 1)
        self.assertLess(elapsed, 0.1)

    def test_search_avoids_third_repetition_when_a_win_remains(self) -> None:
        fen = "7k/8/8/8/8/8/r7/Q6K w - - 0 1"
        source = chess.Board(fen)
        repeated = source.copy(stack=False)
        repeated.push_uci("a1a2")
        _, root_state = position_from_fen(fen)
        _, repeated_state = position_from_fen(repeated.fen(en_passant="fen"))
        repeated_key = int(repeated_state[HASH_KEY])
        history = [11, repeated_key, 22, repeated_key, int(root_state[HASH_KEY])]

        pieces, state = position_from_fen(fen)
        _, move, _, completed = search_root(
            pieces, state, 2, float("inf"), history=history
        )
        self.assertTrue(completed)
        self.assertNotEqual(move_to_uci(move), "a1a2")

    def test_round_five_horizon_position_avoids_ne4_at_depth_four(self) -> None:
        fen = "r4rk1/3b1p2/1pnqpn2/p2pN2p/3P2p1/P1PB1N2/1P1Q1PPP/R3R1K1 b - - 1 16"
        result = search_fixed_depth(fen, 4)
        self.assertNotEqual(result.move, "f6e4")

    def test_round_five_forced_mate_is_visible_at_depth_five(self) -> None:
        fen = "r4rk1/3b1p2/1pnqp3/p2pN2p/3PB3/P1P2p2/1P1Q1PPP/R3R1K1 w - - 0 18"
        result = search_fixed_depth(fen, 5)
        self.assertEqual(result.move, "d2g5")
        self.assertGreaterEqual(result.score, 999_900)


if __name__ == "__main__":
    unittest.main()
