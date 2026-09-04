import unittest

import chess

from challengers.lmr_shelter import lmr_core as core
from challengers.lmr_shelter import lmr_search as search
from challengers.lmr_terminal import lmr_search as reference


class ShelterTests(unittest.TestCase):
    def test_full_pawn_shield_and_missing_files(self):
        board, _ = core.position_from_fen("6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1")
        self.assertEqual(search._shelter_penalty(board, chess.G1, 1), 0)
        self.assertEqual(search._shelter_penalty(board, chess.G8, -1), 0)
        board[chess.G2] = 0
        self.assertEqual(search._shelter_penalty(board, chess.G1, 1), 20)
        board[chess.G3] = core.PAWN
        self.assertEqual(search._shelter_penalty(board, chess.G1, 1), 8)

    def test_no_queen_means_no_change(self):
        for fen in ("6k1/5ppp/8/8/8/8/5P1P/6K1 w - - 0 1",
                    "8/8/3k4/8/5K2/8/6P1/8 w - - 0 1"):
            board, state = core.position_from_fen(fen)
            self.assertEqual(search.evaluate(board, state), reference.evaluate(board, state))

    def test_exposed_king_penalized_and_color_symmetric(self):
        original = chess.Board("3q2k1/5ppp/8/8/8/8/5P1P/3Q2K1 w - - 0 1")
        for source in (original, original.mirror()):
            board, state = core.position_from_fen(source.fen())
            delta = search.evaluate(board, state) - reference.evaluate(board, state)
            self.assertLess(delta, 0)
            self.assertGreaterEqual(delta, -60)
        board, state = core.position_from_fen(original.fen())
        opposite = state.copy()
        opposite[core.SIDE] *= -1
        self.assertEqual(search.evaluate(board, state), -search.evaluate(board, opposite))

    def test_known_mating_tactic(self):
        fen = "r4rk1/3b1p2/1pnqp3/p2pN2p/3PB3/P1P2p2/1P1Q1PPP/R3R1K1 w - - 0 18"
        result = search.search_fixed_depth(fen, 5)
        self.assertEqual(result.move, "d2g5")
        self.assertGreater(result.score, 999900)
