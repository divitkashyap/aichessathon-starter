"""Correctness tests for the compiled search challenger."""

import time
import unittest

import chess
import numpy as np

from fastcore import position_from_fen
from fastsearch import evaluate, search_fixed_depth, search_root


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


if __name__ == "__main__":
    unittest.main()
