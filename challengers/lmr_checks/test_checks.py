import math
import time
import unittest

import chess
import numpy as np

from challengers.lmr_checks import lmr_core as core
from challengers.lmr_checks import lmr_search as search
from challengers.lmr_terminal import lmr_search as reference

BEFORE_BD7 = "r1b3k1/ppp2r2/1b1p3B/3N1p2/q1P5/5P2/PP1Q2PP/4RR1K b - - 4 22"


class CheckExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        search.search_fixed_depth(chess.STARTING_FEN, 2)

    def test_round12_forcing_mate(self):
        board = chess.Board(BEFORE_BD7)
        board.push_uci("c8d7")
        result = search.search_fixed_depth(board.fen(), 4)
        self.assertEqual(result.move, "d5f6")
        self.assertEqual(result.score, search.MATE - 9)
        for move in "d5f6 g8h8 h6g7 h8g7 d2g5 g7h8 g5h6 f7h7 h6h7".split():
            board.push_uci(move)
        self.assertTrue(board.is_checkmate())

    def test_round12_avoids_recorded_losing_move(self):
        self.assertNotEqual(search.search_fixed_depth(BEFORE_BD7, 4).move, "c8d7")

    def test_timeout_restores_state(self):
        board, state = core.position_from_fen(chess.STARTING_FEN)
        old_board, old_state = board.copy(), state.copy()
        _, _, _, completed = search.search_root(board, state, 12, time.perf_counter() + 0.02)
        self.assertFalse(completed)
        np.testing.assert_array_equal(board, old_board)
        np.testing.assert_array_equal(state, old_state)

    def test_exhausted_extension_budget_matches_reference(self):
        fen = "4k3/4r3/8/8/8/8/4R3/4K3 b - - 0 1"
        scores = []
        for engine in (reference, search):
            board, state = core.position_from_fen(fen)
            before_board, before_state = board.copy(), state.copy()
            args = (board, state, 2, -search.INFINITY, search.INFINITY, 1,
                    np.zeros(2, dtype=np.int64), math.inf,
                    np.zeros(600, dtype=np.int64), 0, np.zeros(128, dtype=np.int64), 0,
                    *engine._new_tt(), *engine._new_move_ordering())
            result = engine._negamax(*args, search.MAX_CHECK_EXTENSIONS) if engine is search else engine._negamax(*args)
            scores.append(result)
            np.testing.assert_array_equal(board, before_board)
            np.testing.assert_array_equal(state, before_state)
        self.assertEqual(*scores)

    def test_extension_budget_keys_are_distinct(self):
        self.assertEqual(len(set(map(int, search.EXTENSION_HASH))), search.MAX_CHECK_EXTENSIONS + 1)
        self.assertEqual(search.EXTENSION_HASH[0], 0)
