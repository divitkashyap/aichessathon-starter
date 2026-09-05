"""Focused tests for the isolated pawn-safety evaluator adjustment."""

from __future__ import annotations

import unittest
import random

import chess
import numpy as np

from challengers.lmr_checks_draws import lmr_search as reference
from challengers.lmr_pawn_safety import lmr_core as core
from challengers.lmr_pawn_safety import lmr_search as candidate


class PawnSafetyTests(unittest.TestCase):
    def test_random_positions_match_independent_numeric_reference(self) -> None:
        rng = random.Random(9052026)
        for _ in range(256):
            position = chess.Board()
            for _ply in range(rng.randrange(100)):
                if position.is_game_over():
                    break
                position.push(rng.choice(list(position.legal_moves)))
            pieces, state = core.position_from_fen(position.fen())
            if position.is_insufficient_material():
                self.assertEqual(candidate.evaluate(pieces, state), 0)
                continue
            mg = eg = phase = 0
            for square, piece in position.piece_map().items():
                sign = 1 if piece.color == chess.WHITE else -1
                relative = square if piece.color == chess.WHITE else chess.square_mirror(square)
                mg += sign * (int(reference.PIECE_VALUES[piece.piece_type]) + int(reference.MG_TABLE[piece.piece_type, relative]))
                eg += sign * (int(reference.PIECE_VALUES[piece.piece_type]) + int(reference.EG_TABLE[piece.piece_type, relative]))
                phase += int(reference.PHASE_WEIGHTS[piece.piece_type])
            for color in [chess.WHITE, chess.BLACK]:
                sign = 1 if color == chess.WHITE else -1
                if len(position.pieces(chess.BISHOP, color)) >= 2:
                    mg += sign * 28
                    eg += sign * 38
                king = position.king(color)
                rank = chess.square_rank(king)
                relative_rank = rank if color else 7 - rank
                file = chess.square_file(king)
                if relative_rank > 1 or file in [3, 4] or not position.pieces(chess.QUEEN, not color):
                    continue
                shield_files = {file - 1, file, file + 1}
                for pawn in position.pieces(chess.PAWN, color):
                    pawn_rank = chess.square_rank(pawn)
                    relative_pawn_rank = pawn_rank if color else 7 - pawn_rank
                    if chess.square_file(pawn) in shield_files and relative_pawn_rank > max(1, relative_rank):
                        mg -= sign * 9 * (relative_pawn_rank - 1)
            phase = min(24, phase)
            expected = (mg * phase + eg * (24 - phase)) // 24
            expected *= 1 if position.turn == chess.WHITE else -1
            self.assertEqual(candidate.evaluate(pieces, state), expected, position.fen())

    def assert_same_as_reference(self, fen: str) -> None:
        board, state = core.position_from_fen(fen)
        ref_board, ref_state = reference.position_from_fen(fen)
        self.assertEqual(candidate.evaluate(board, state), reference.evaluate(ref_board, ref_state))

    def test_queen_absent_is_exact_reference(self) -> None:
        self.assert_same_as_reference("6k1/8/8/7P/8/8/8/6K1 w - - 0 1")

    def test_unadvanced_shield_is_unchanged_with_queen(self) -> None:
        self.assert_same_as_reference("3q2k1/8/8/8/8/8/7P/6K1 w - - 0 1")

    def test_advanced_shield_bonus_is_removed(self) -> None:
        fen = "3q2k1/8/8/8/7P/8/8/6K1 w - - 0 1"
        self.assertTrue(chess.Board(fen).is_valid())
        board, state = core.position_from_fen(fen)
        ref_board, ref_state = reference.position_from_fen(fen)
        self.assertLess(candidate.evaluate(board, state), reference.evaluate(ref_board, ref_state))

    def test_advanced_own_file_pawn_is_a_shield_pawn(self) -> None:
        fen = "3q2k1/8/8/8/6P1/8/8/6K1 w - - 0 1"
        self.assertTrue(chess.Board(fen).is_valid())
        board, state = core.position_from_fen(fen)
        ref_board, ref_state = reference.position_from_fen(fen)
        self.assertLess(candidate.evaluate(board, state), reference.evaluate(ref_board, ref_state))

    def test_mirrored_adjustment_matches_side_to_move_perspective(self) -> None:
        white_fen = "3q2k1/8/8/8/7P/8/8/6K1 w - - 0 1"
        black_fen = "6k1/8/8/7p/8/8/8/3Q2K1 b - - 0 1"
        self.assertTrue(chess.Board(white_fen).is_valid())
        self.assertTrue(chess.Board(black_fen).is_valid())
        white_board, white_state = core.position_from_fen(white_fen)
        black_board, black_state = core.position_from_fen(black_fen)
        white_ref_board, white_ref_state = reference.position_from_fen(white_fen)
        black_ref_board, black_ref_state = reference.position_from_fen(black_fen)
        self.assertEqual(candidate.evaluate(white_board, white_state), reference.evaluate(white_ref_board, white_ref_state) - 3)
        self.assertEqual(candidate.evaluate(black_board, black_state), reference.evaluate(black_ref_board, black_ref_state) - 3)

    def test_unrelated_wing_pawn_is_unchanged(self) -> None:
        self.assert_same_as_reference("3q2k1/8/8/P7/8/8/8/6K1 w - - 0 1")

    def test_evaluate_does_not_mutate_inputs(self) -> None:
        board, state = core.position_from_fen("3q2k1/8/8/7P/8/8/8/6K1 w - - 0 1")
        board_before = board.copy()
        state_before = state.copy()
        candidate.evaluate(board, state)
        np.testing.assert_array_equal(board, board_before)
        np.testing.assert_array_equal(state, state_before)


if __name__ == "__main__":
    unittest.main()
