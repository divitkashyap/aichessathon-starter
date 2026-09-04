"""Differential correctness tests for the experimental compiled move generator."""

import random
import unittest

import chess

from fastcore import legal_moves_uci


class FastCoreTests(unittest.TestCase):
    def assert_matches_python_chess(self, board: chess.Board) -> None:
        expected = {move.uci() for move in board.legal_moves}
        self.assertSetEqual(legal_moves_uci(board.fen()), expected, board.fen())

    def test_edge_case_positions(self) -> None:
        fens = (
            chess.STARTING_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/2P5/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1",
            "8/P6k/8/8/8/8/6K1/8 w - - 0 1",
            "8/8/8/3pP3/8/8/6K1/7k w - d6 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
            "4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1",
        )
        for fen in fens:
            with self.subTest(fen=fen):
                self.assert_matches_python_chess(chess.Board(fen))

    def test_random_legal_positions(self) -> None:
        rng = random.Random(20260904)
        board = chess.Board()
        checked = 0
        while checked < 1_000:
            self.assert_matches_python_chess(board)
            checked += 1
            if board.is_game_over() or board.ply() >= 180:
                board.reset()
                continue
            board.push(rng.choice(list(board.legal_moves)))


if __name__ == "__main__":
    unittest.main()

