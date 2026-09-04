import math
import unittest

import chess
import numpy as np

from challengers.lmr_terminal import lmr_core as core
from challengers.lmr_terminal import lmr_search as search


class TerminalTests(unittest.TestCase):
    def qscore(self, fen, qdepth, beta=search.INFINITY):
        board, state = core.position_from_fen(fen)
        original_board, original_state = board.copy(), state.copy()
        score = search._quiescence(
            board,
            state,
            -search.INFINITY,
            beta,
            7,
            qdepth,
            np.zeros(2, dtype=np.int64),
            math.inf,
            np.zeros(600, dtype=np.int64),
            0,
            np.zeros(128, dtype=np.int64),
            0,
        )
        np.testing.assert_array_equal(board, original_board)
        np.testing.assert_array_equal(state, original_state)
        return score

    def test_mate_at_quiescence_horizon(self):
        fen = "8/8/8/8/8/2k5/1q6/K7 w - - 0 1"
        self.assertTrue(chess.Board(fen).is_checkmate())
        self.assertEqual(self.qscore(fen, search.MAX_QDEPTH), -search.MATE + 7)

    def test_mate_precedes_fifty_move_draw(self):
        fen = "8/8/8/8/8/2k5/1q6/K7 w - - 100 1"
        self.assertEqual(self.qscore(fen, 0), -search.MATE + 7)
        board, state = core.position_from_fen(fen)
        self.assertEqual(
            search._negamax(
                board,
                state,
                1,
                -search.INFINITY,
                search.INFINITY,
                7,
                np.zeros(2, dtype=np.int64),
                math.inf,
                np.zeros(600, dtype=np.int64),
                0,
                np.zeros(128, dtype=np.int64),
                0,
                *search._new_tt(),
                *search._new_move_ordering(),
            ),
            -search.MATE + 7,
        )

    def test_stalemate_precedes_stand_pat_and_horizon(self):
        fen = "8/8/8/8/8/1q6/2k5/K7 w - - 0 1"
        self.assertTrue(chess.Board(fen).is_stalemate())
        self.assertEqual(self.qscore(fen, 0, -1000), 0)
        self.assertEqual(self.qscore(fen, search.MAX_QDEPTH), 0)

    def test_early_legal_probe_matches_oracle_and_restores_state(self):
        source = chess.Board()
        for ply in range(128):
            board, state = core.position_from_fen(source.fen(en_passant="fen"))
            before_board, before_state = board.copy(), state.copy()
            moves = sorted(source.legal_moves, key=lambda m: m.uci())
            self.assertEqual(core.has_legal_move(board, state), bool(moves))
            np.testing.assert_array_equal(board, before_board)
            np.testing.assert_array_equal(state, before_state)
            if not moves:
                break
            source.push(moves[(ply * 41 + 7) % len(moves)])
