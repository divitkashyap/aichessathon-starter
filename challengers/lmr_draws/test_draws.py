"""Tests for narrow insufficient-material recognition in the draw challenger."""

import random
import unittest

import chess

from challengers.lmr_draws import lmr_core as core
from challengers.lmr_draws import lmr_search as draws
from challengers.lmr_terminal import lmr_search as baseline

RECOGNIZED = (
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",  # K v K
    "4k3/8/8/8/8/8/2B5/4K3 w - - 0 1",  # K+B v K
    "4k3/8/8/8/8/8/2N5/4K3 w - - 0 1",  # K+N v K
    "4k3/8/8/8/4b3/8/2B5/4K3 w - - 0 1",  # same-color bishops
)
NOT_AUTOMATIC = (
    "4k3/8/8/8/8/1N6/2N5/4K3 w - - 0 1",  # K+NN v K
    "4k3/8/8/8/8/1n6/2B5/4K3 w - - 0 1",  # K+B v K+N
    "4k3/8/8/4b3/8/8/2B5/4K3 w - - 0 1",  # opposite-color bishops
    "4k3/8/8/8/8/8/P7/4K3 w - - 0 1",  # any pawn
    "4k3/8/8/8/8/8/R7/4K3 w - - 0 1",  # any rook
    "4k3/8/8/8/8/8/Q7/4K3 w - - 0 1",  # any queen
)


def evaluate(fen: str, turn: chess.Color) -> int:
    board = chess.Board(fen)
    board.turn = turn
    pieces, state = core.position_from_fen(board.fen())
    return int(draws.evaluate(pieces, state))


class DrawRecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        draws.evaluate(*core.position_from_fen(chess.STARTING_FEN))

    def test_recognized_positions_match_python_chess_and_mirror(self) -> None:
        for fen in RECOGNIZED:
            board = chess.Board(fen)
            self.assertTrue(board.is_insufficient_material())
            for turn in (chess.WHITE, chess.BLACK):
                board.turn = turn
                self.assertEqual(evaluate(fen, turn), 0)
                mirrored = board.mirror()
                self.assertEqual(evaluate(mirrored.fen(), mirrored.turn), 0)

    def test_nonrecognized_positions_match_baseline_evaluation(self) -> None:
        for fen in NOT_AUTOMATIC:
            board = chess.Board(fen)
            self.assertFalse(board.is_insufficient_material())
            for turn in (chess.WHITE, chess.BLACK):
                board.turn = turn
                current = board.fen()
                self.assertEqual(evaluate(current, turn), evaluate_baseline(current))

    def test_no_search_or_make_unmake_contract_changes(self) -> None:
        fen = RECOGNIZED[0]
        pieces, state = core.position_from_fen(fen)
        before_pieces, before_state = pieces.copy(), state.copy()
        self.assertEqual(draws.evaluate(pieces, state), 0)
        self.assertEqual(pieces.tolist(), before_pieces.tolist())
        self.assertEqual(state.tolist(), before_state.tolist())

    def test_sparse_positions_differential(self) -> None:
        rng = random.Random(20260904)
        checked = 0
        for _ in range(1024):
            board = chess.Board(None)
            squares = rng.sample(range(64), 7)
            board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
            for square in squares[2:2 + rng.randrange(6)]:
                board.set_piece_at(square, chess.Piece(rng.choice([chess.BISHOP, chess.KNIGHT, chess.PAWN, chess.ROOK, chess.QUEEN]), rng.choice([chess.WHITE, chess.BLACK])))
            board.turn = rng.choice([chess.WHITE, chess.BLACK])
            if not board.is_valid():
                continue
            current = evaluate(board.fen(), board.turn)
            expected = 0 if board.is_insufficient_material() else evaluate_baseline(board.fen())
            self.assertEqual(current, expected, board.fen())
            checked += 1
        self.assertGreater(checked, 300)


def evaluate_baseline(fen: str) -> int:
    pieces, state = baseline.position_from_fen(fen)
    return int(baseline.evaluate(pieces, state))


if __name__ == "__main__":
    unittest.main()
