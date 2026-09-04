"""Feature-schema invariants that must hold before training expensive models."""

import unittest

import chess
import numpy as np

from fastcore import position_from_fen
from nnue.fast_features import encode_fast
from nnue.features import (
    INPUT_FEATURES,
    MAX_ACTIVE_FEATURES,
    PADDING_INDEX,
    encode_board,
    encode_fen,
    perspective_indices,
)


class NNUEFeatureTests(unittest.TestCase):
    def test_four_field_fen_is_supported(self) -> None:
        encoded = encode_fen("8/8/8/4k3/8/3K4/4P3/8 w - -")
        self.assertEqual(encoded.white.shape, (MAX_ACTIVE_FEATURES,))
        self.assertEqual(encoded.turn, 1)

    def test_start_position_is_perspective_symmetric(self) -> None:
        encoded = encode_board(chess.Board())
        np.testing.assert_array_equal(encoded.white, encoded.black)
        self.assertEqual(np.count_nonzero(encoded.white != PADDING_INDEX), 30)

    def test_colour_mirror_swaps_perspectives(self) -> None:
        board = chess.Board("r3k2r/pp2qppp/2npbn2/2p5/3NP3/2N1B3/PPP2PPP/R2Q1RK1 b kq - 4 11")
        original = encode_board(board)
        mirrored = encode_board(board.mirror())
        np.testing.assert_array_equal(original.white, mirrored.black)
        np.testing.assert_array_equal(original.black, mirrored.white)
        self.assertEqual(original.turn, -mirrored.turn)

    def test_horizontal_reflection_is_normalized(self) -> None:
        board = chess.Board("4k3/8/2n5/8/3P4/8/8/2K5 w - - 0 1")
        reflected = board.transform(chess.flip_horizontal)
        np.testing.assert_array_equal(
            perspective_indices(board, chess.WHITE),
            perspective_indices(reflected, chess.WHITE),
        )
        np.testing.assert_array_equal(
            perspective_indices(board, chess.BLACK),
            perspective_indices(reflected, chess.BLACK),
        )

    def test_indices_are_in_range_or_padding(self) -> None:
        encoded = encode_board(chess.Board())
        for indices in (encoded.white, encoded.black):
            self.assertTrue(np.all(indices >= 0))
            self.assertTrue(np.all(indices <= PADDING_INDEX))
            self.assertTrue(np.all(indices[indices != PADDING_INDEX] < INPUT_FEATURES))

    def test_missing_king_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            encode_fen("8/8/8/8/8/8/4P3/4K3 w - -")

    def test_numba_encoder_matches_python_over_playout(self) -> None:
        board = chess.Board()
        for ply in range(256):
            expected = encode_board(board)
            pieces, state = position_from_fen(board.fen(en_passant="fen"))
            white, black, turn = encode_fast(pieces, state)
            np.testing.assert_array_equal(white, expected.white)
            np.testing.assert_array_equal(black, expected.black)
            self.assertEqual(turn, expected.turn)
            if board.is_game_over():
                board = chess.Board()
            else:
                moves = sorted(board.legal_moves, key=lambda move: move.uci())
                board.push(moves[(ply * 17 + 3) % len(moves)])


if __name__ == "__main__":
    unittest.main()
