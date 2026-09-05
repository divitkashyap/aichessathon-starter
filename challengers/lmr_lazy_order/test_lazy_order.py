"""The ordering optimization must preserve v7's exact search decisions."""

import math
import random
import unittest

import chess
import numpy as np

from challengers.lmr_checks_draws import lmr_search as reference
from challengers.lmr_lazy_order import lmr_search as candidate
from tools.openings import opening_fens


class LazyOrderingTests(unittest.TestCase):
    def test_order_and_ties_match_reference(self):
        rng = random.Random(59127)
        board = chess.Board()
        for _ in range(256):
            if board.is_game_over():
                board = chess.Board()
            board.push(rng.choice(list(board.legal_moves)))
            pieces, state = candidate.position_from_fen(board.fen())
            original = candidate.generate_legal_moves(pieces, state)
            if not len(original):
                continue
            eager, lazy = original.copy(), original.copy()
            killers, history = candidate._new_move_ordering()
            # Include equal scores, preferred moves and killer/history bonuses.
            history[:] = np.random.default_rng(_).integers(0, 4, history.shape)
            killers[0, 0] = original[-1]
            preferred = int(original[0]) if _ % 2 else 0
            reference._order_moves(pieces, state, eager, preferred, 0, killers, history)
            scores = candidate._prepare_move_scores(
                pieces, state, lazy, preferred, 0, killers, history
            )
            # Child searches may mutate history after the scores are frozen.
            history[:] = 500000
            for index in range(len(lazy)):
                candidate._pick_next_move(lazy, scores, index)
                self.assertEqual(int(lazy[index]), int(eager[index]))

    def test_fixed_depth_exact_parity(self):
        for fen in opening_fens('legacy') + opening_fens('validation'):
            old = reference.search_fixed_depth(fen, 4)
            new = candidate.search_fixed_depth(fen, 4)
            self.assertEqual((new.move, new.score, new.nodes),
                             (old.move, old.score, old.nodes), fen)

    def test_timeout_restores_board(self):
        board, state = candidate.position_from_fen(chess.STARTING_FEN)
        saved_board, saved_state = board.copy(), state.copy()
        result = candidate.search_root(board, state, 6, -math.inf)
        self.assertFalse(result[3])
        np.testing.assert_array_equal(board, saved_board)
        np.testing.assert_array_equal(state, saved_state)

    def test_round12_forced_mate_preserved(self):
        fen = 'r5k1/pppb1r2/1b1p3B/3N1p2/q1P5/5P2/PP1Q2PP/4RR1K w - - 5 23'
        result = candidate.search_fixed_depth(fen, 4)
        self.assertEqual((result.move, result.score), ('d5f6', candidate.MATE - 9))


if __name__ == '__main__':
    unittest.main()
