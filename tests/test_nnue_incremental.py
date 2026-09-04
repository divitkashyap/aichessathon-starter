"""Exact differential tests for incremental quantized NNUE accumulators."""

from __future__ import annotations

import unittest

import numpy as np

from fastcore import (
    CAPTURE,
    CASTLE,
    EN_PASSANT,
    UNDO_SIZE,
    generate_legal_moves,
    move_to_uci,
    position_from_fen,
)
from nnue.fast_features import encode_fast
from nnue.features import INPUT_FEATURES, PADDING_INDEX
from nnue.incremental import (
    evaluate_accumulator_arrays,
    initialize_accumulators,
    make_move_with_accumulators,
    unmake_move_with_accumulators,
)
from nnue.quantize import QuantizedNNUE
from nnue.runtime import evaluate_quantized_arrays


class IncrementalNNUETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(20260904)
        cls.weights = QuantizedNNUE(
            feature=rng.integers(
                -96, 97, size=(INPUT_FEATURES, 16), dtype=np.int16
            ),
            feature_bias=rng.integers(-256, 257, size=16, dtype=np.int32),
            output_weight=rng.integers(-64, 65, size=16, dtype=np.int16),
            tempo=np.int64(17_123),
            output_scale_cp=400,
        )

    def _full_accumulator(self, indices: np.ndarray) -> np.ndarray:
        active = indices[indices != PADDING_INDEX]
        return self.weights.feature_bias.astype(np.int64) + self.weights.feature[
            active
        ].sum(axis=0, dtype=np.int64)

    def _assert_exact(
        self,
        board: np.ndarray,
        state: np.ndarray,
        white: np.ndarray,
        black: np.ndarray,
        king_squares: np.ndarray,
    ) -> None:
        white_indices, black_indices, turn = encode_fast(board, state)
        np.testing.assert_array_equal(white, self._full_accumulator(white_indices))
        np.testing.assert_array_equal(black, self._full_accumulator(black_indices))

        expected = evaluate_quantized_arrays(
            white_indices,
            black_indices,
            turn,
            self.weights.feature,
            self.weights.feature_bias,
            self.weights.output_weight,
            int(self.weights.tempo),
            self.weights.output_scale_cp,
            self.weights.feature_scale,
            self.weights.weight_scale,
        )
        actual = evaluate_accumulator_arrays(
            white,
            black,
            turn,
            self.weights.output_weight,
            int(self.weights.tempo),
            self.weights.output_scale_cp,
            self.weights.feature_scale,
            self.weights.weight_scale,
        )
        self.assertEqual(actual, expected)

        self.assertEqual(int(board[int(king_squares[0])]), 6)
        self.assertEqual(int(board[int(king_squares[1])]), -6)

    def _encoded_move(self, board: np.ndarray, state: np.ndarray, uci: str) -> int:
        for move in generate_legal_moves(board, state):
            if move_to_uci(int(move)) == uci:
                return int(move)
        self.fail(f"{uci} is not legal in the compact position")

    def _exercise_move(self, fen: str, uci: str, required_flag: int = 0) -> None:
        board, state = position_from_fen(fen)
        original_board = board.copy()
        original_state = state.copy()
        white, black, king_squares = initialize_accumulators(
            board, self.weights.feature, self.weights.feature_bias
        )
        original_white = white.copy()
        original_black = black.copy()
        original_kings = king_squares.copy()
        self._assert_exact(board, state, white, black, king_squares)

        move = self._encoded_move(board, state, uci)
        if required_flag:
            self.assertTrue(move & required_flag)
        undo = np.empty(UNDO_SIZE, dtype=np.int64)
        make_move_with_accumulators(
            board,
            state,
            move,
            undo,
            self.weights.feature,
            self.weights.feature_bias,
            white,
            black,
            king_squares,
        )
        self._assert_exact(board, state, white, black, king_squares)

        unmake_move_with_accumulators(
            board,
            state,
            move,
            undo,
            self.weights.feature,
            self.weights.feature_bias,
            white,
            black,
            king_squares,
        )
        self._assert_exact(board, state, white, black, king_squares)
        np.testing.assert_array_equal(board, original_board)
        np.testing.assert_array_equal(state, original_state)
        np.testing.assert_array_equal(white, original_white)
        np.testing.assert_array_equal(black, original_black)
        np.testing.assert_array_equal(king_squares, original_kings)

    def test_deterministic_playout_and_complete_unmake(self) -> None:
        board, state = position_from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        )
        original_board = board.copy()
        original_state = state.copy()
        white, black, king_squares = initialize_accumulators(
            board, self.weights.feature, self.weights.feature_bias
        )
        original_white = white.copy()
        original_black = black.copy()
        original_kings = king_squares.copy()
        history: list[tuple[int, np.ndarray]] = []

        self._assert_exact(board, state, white, black, king_squares)
        for ply in range(192):
            moves = generate_legal_moves(board, state)
            if len(moves) == 0:
                break
            move = int(moves[(ply * 37 + 11) % len(moves)])
            undo = np.empty(UNDO_SIZE, dtype=np.int64)
            make_move_with_accumulators(
                board,
                state,
                move,
                undo,
                self.weights.feature,
                self.weights.feature_bias,
                white,
                black,
                king_squares,
            )
            history.append((move, undo.copy()))
            self._assert_exact(board, state, white, black, king_squares)

        self.assertGreaterEqual(len(history), 40)
        for move, undo in reversed(history):
            unmake_move_with_accumulators(
                board,
                state,
                move,
                undo,
                self.weights.feature,
                self.weights.feature_bias,
                white,
                black,
                king_squares,
            )
            self._assert_exact(board, state, white, black, king_squares)

        np.testing.assert_array_equal(board, original_board)
        np.testing.assert_array_equal(state, original_state)
        np.testing.assert_array_equal(white, original_white)
        np.testing.assert_array_equal(black, original_black)
        np.testing.assert_array_equal(king_squares, original_kings)

    def test_explicit_capture(self) -> None:
        self._exercise_move(
            "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", CAPTURE
        )

    def test_explicit_en_passant(self) -> None:
        self._exercise_move(
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", EN_PASSANT
        )

    def test_explicit_quiet_and_capture_promotions(self) -> None:
        self._exercise_move("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q")
        self._exercise_move(
            "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8n", CAPTURE
        )

    def test_explicit_white_and_black_castling(self) -> None:
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        self._exercise_move(fen, "e1g1", CASTLE)
        self._exercise_move(fen.replace(" w ", " b "), "e8c8", CASTLE)

    def test_king_move_refreshes_file_mirror(self) -> None:
        self._exercise_move("7k/8/8/8/3K4/8/P7/8 w - - 0 1", "d4e4")


if __name__ == "__main__":
    unittest.main()
