import math
import unittest

import chess
import numpy as np

from challengers.lmr_null import lmr_core as core
from challengers.lmr_null import lmr_search as search
from challengers.lmr_terminal import lmr_search as reference


class NullProbeTests(unittest.TestCase):
    def test_null_probe_restores_state_and_ancestor_history(self):
        source = chess.Board()
        source.push_uci("e2e4")
        board, state = core.position_from_fen(source.fen(en_passant="fen"))
        before_board, before_state = board.copy(), state.copy()
        path = np.zeros(128, dtype=np.int64)
        path[:3] = [11, 22, int(state[core.HASH_KEY])]
        score = search._negamax(
            board,
            state,
            6,
            -1001,
            -1000,
            3,
            np.zeros(2, dtype=np.int64),
            math.inf,
            np.zeros(600, dtype=np.int64),
            0,
            path,
            3,
            *search._new_tt(),
            *search._new_move_ordering(),
            True,
        )
        self.assertEqual(score, -1000)
        np.testing.assert_array_equal(board, before_board)
        np.testing.assert_array_equal(state, before_state)
        np.testing.assert_array_equal(path[:3], [11, 22, int(state[core.HASH_KEY])])
        self.assertEqual(core.compute_hash(board, state), state[core.HASH_KEY])

    def test_sparse_endings_match_unpruned_variant(self):
        for fen in ("8/4k3/8/4K3/4P3/8/8/8 w - - 0 1", "8/8/4k3/8/4K3/8/4P3/8 b - - 0 1"):
            self.assertTrue(chess.Board(fen).is_valid())
            old = reference.search_fixed_depth(fen, 5)
            new = search.search_fixed_depth(fen, 5)
            self.assertEqual((old.move, old.score, old.nodes), (new.move, new.score, new.nodes))

    def test_known_tactic_and_timed_legal_move(self):
        fen = "r4rk1/3b1p2/1pnqp3/p2pN2p/3PB3/P1P2p2/1P1Q1PPP/R3R1K1 w - - 0 18"
        result = search.search_fixed_depth(fen, 5)
        self.assertEqual(result.move, "d2g5")
        self.assertGreaterEqual(result.score, 999900)
        timed = search.search_timed(chess.STARTING_FEN, 100)
        self.assertIn(chess.Move.from_uci(timed.move), chess.Board().legal_moves)

    def test_expired_search_unwinds(self):
        search.warm_up()
        board, state = core.position_from_fen(chess.STARTING_FEN)
        before_board, before_state = board.copy(), state.copy()
        _, _, _, complete = search.search_root(board, state, 30, 0.0)
        self.assertFalse(complete)
        np.testing.assert_array_equal(board, before_board)
        np.testing.assert_array_equal(state, before_state)
