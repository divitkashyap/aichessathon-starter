"""Differential correctness tests for the experimental compiled move generator."""

import random
import unittest

import chess
import numpy as np

from fastcore import (
    UNDO_SIZE,
    generate_legal_moves,
    legal_moves_uci,
    make_move,
    perft,
    position_from_fen,
    unmake_move,
)


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

    def test_make_unmake_restores_every_field(self) -> None:
        board = chess.Board(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        )
        pieces, state = position_from_fen(board.fen())
        original_pieces = pieces.copy()
        original_state = state.copy()
        for move in generate_legal_moves(pieces, state):
            undo = np.empty(UNDO_SIZE, dtype=np.int16)
            make_move(pieces, state, int(move), undo)
            unmake_move(pieces, state, int(move), undo)
            np.testing.assert_array_equal(pieces, original_pieces)
            np.testing.assert_array_equal(state, original_state)

    def test_standard_perft_positions(self) -> None:
        cases = (
            (chess.STARTING_FEN, (20, 400, 8_902, 197_281)),
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                (48, 2_039, 97_862),
            ),
            (
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                (14, 191, 2_812, 43_238),
            ),
        )
        for fen, expected_by_depth in cases:
            for depth, expected in enumerate(expected_by_depth, start=1):
                with self.subTest(fen=fen, depth=depth):
                    pieces, state = position_from_fen(fen)
                    original_pieces = pieces.copy()
                    original_state = state.copy()
                    self.assertEqual(int(perft(pieces, state, depth)), expected)
                    np.testing.assert_array_equal(pieces, original_pieces)
                    np.testing.assert_array_equal(state, original_state)


if __name__ == "__main__":
    unittest.main()
