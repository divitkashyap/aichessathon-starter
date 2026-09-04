"""Correctness tests for the isolated late-move-reduction challenger."""

import time
import unittest

import chess
import numpy as np

from challengers.lmr.lmr_core import position_from_fen
from challengers.lmr.lmr_search import search_fixed_depth, search_root, search_timed


class LMRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Keep Numba compilation out of the deadline-restoration assertion.
        search_fixed_depth(chess.STARTING_FEN, 1)

    def test_timed_search_returns_a_legal_move(self) -> None:
        result = search_timed(chess.STARTING_FEN, 250)
        self.assertIn(chess.Move.from_uci(result.move), chess.Board().legal_moves)

    def test_mate_fixture_is_preserved(self) -> None:
        fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
        result = search_fixed_depth(fen, 2)
        board = chess.Board(fen)
        board.push_uci(result.move)
        self.assertTrue(board.is_checkmate())

    def test_timeout_restores_board_and_state(self) -> None:
        pieces, state = position_from_fen(chess.STARTING_FEN)
        original_pieces = pieces.copy()
        original_state = state.copy()
        started = time.perf_counter()
        _, _, _, completed = search_root(pieces, state, 12, started + 0.02)
        self.assertFalse(completed)
        np.testing.assert_array_equal(pieces, original_pieces)
        np.testing.assert_array_equal(state, original_state)

    def test_known_round_five_tactics(self) -> None:
        before = "r4rk1/3b1p2/1pnqpn2/p2pN2p/3P2p1/P1PB1N2/1P1Q1PPP/R3R1K1 b - - 1 16"
        self.assertNotEqual(search_fixed_depth(before, 5).move, "f6e4")
        attack = "r4rk1/3b1p2/1pnqp3/p2pN2p/3PB3/P1P2p2/1P1Q1PPP/R3R1K1 w - - 0 18"
        result = search_fixed_depth(attack, 5)
        self.assertEqual(result.move, "d2g5")
        self.assertGreaterEqual(result.score, 999900)


if __name__ == "__main__":
    unittest.main()
