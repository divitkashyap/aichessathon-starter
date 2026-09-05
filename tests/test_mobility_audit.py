"""Offline feature invariants; these are not playing-strength tests."""

import unittest

import chess

from tools.audit_mobility import mobility


class MobilityTests(unittest.TestCase):
    def test_symmetric_start(self):
        self.assertEqual(mobility(chess.Board()), 0)

    def test_turn_negates(self):
        board = chess.Board()
        board.push_uci("e2e4")
        score = mobility(board)
        board.turn = not board.turn
        self.assertEqual(mobility(board), -score)

    def test_color_mirror_preserves_relative_score(self):
        board = chess.Board()
        for move in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5"):
            board.push_uci(move)
        self.assertEqual(mobility(board), mobility(board.mirror()))

    def test_bare_kings(self):
        self.assertEqual(mobility(chess.Board("7k/8/8/8/8/8/8/K7 w - - 0 1")), 0)


if __name__ == "__main__":
    unittest.main()
