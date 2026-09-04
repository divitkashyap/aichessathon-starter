"""Focused evaluator tests for the guarded NNUE blend."""

import unittest

import chess
from nnue_blend_classical import evaluate as evaluate_classical
from nnue_blend_core import position_from_fen
from nnue_blend_incremental import initialize_accumulators
from nnue_blend_search import (
    MIN_BLEND_PIECES,
    NNUE_FEATURE,
    NNUE_FEATURE_BIAS,
    evaluate_blended,
)


def _scores(fen: str) -> tuple[int, int, int]:
    board, state = position_from_fen(fen)
    white, black, _ = initialize_accumulators(board, NNUE_FEATURE, NNUE_FEATURE_BIAS)
    classical = evaluate_classical(board, state)
    blended = evaluate_blended(board, state, white, black)
    return int(board.astype(bool).sum()), classical, blended


class BlendTests(unittest.TestCase):
    def test_sparse_positions_are_pure_classical(self):
        for fen in ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "7k/5Kn1/6P1/8/2B5/8/8/8 b - - 0 1"):
            self.assertTrue(chess.Board(fen).is_valid())
            occupied, classical, blended = _scores(fen)
            self.assertLess(occupied, MIN_BLEND_PIECES)
            self.assertEqual(blended, classical)

    def test_dense_blend_correction_is_bounded(self):
        board = chess.Board()
        for ply in range(64):
            _occupied, classical, blended = _scores(board.fen())
            self.assertLessEqual(abs(blended - classical), 25)
            legal = sorted(board.legal_moves, key=lambda m: m.uci())
            if not legal:
                break
            board.push(legal[(ply * 37 + 11) % len(legal)])
