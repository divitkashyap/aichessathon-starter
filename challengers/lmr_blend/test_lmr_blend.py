"""Legality, tactic, and restoration checks for the LMR blend challenger."""

from __future__ import annotations

import time
import unittest

import chess
import numpy as np

from nnue_blend_core import position_from_fen
from nnue_blend_search import search_fixed_depth, search_root


class LMRBlendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        search_fixed_depth(chess.STARTING_FEN, 1)

    def test_returns_legal_move(self) -> None:
        result = search_fixed_depth(chess.STARTING_FEN, 2)
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


if __name__ == "__main__":
    unittest.main()
