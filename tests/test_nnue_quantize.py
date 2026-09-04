"""Float/integer and symmetry checks for NNUE export."""

import unittest

import chess
import numpy as np
import torch

from nnue.features import encode_board
from nnue.model import NNUEConfig, SparseNNUE
from nnue.quantize import quantize
from nnue.runtime import evaluate_quantized, evaluate_quantized_reference


class NNUEQuantizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(7)
        cls.model = SparseNNUE(NNUEConfig(feature_hidden=16)).eval()
        with torch.no_grad():
            cls.model.output.weight.mul_(300.0)
        cls.weights = quantize(cls.model)

    def _float_evaluation(self, board: chess.Board) -> float:
        encoded = encode_board(board)
        with torch.inference_mode():
            result = self.model(
                torch.from_numpy(encoded.white).unsqueeze(0),
                torch.from_numpy(encoded.black).unsqueeze(0),
                torch.tensor([encoded.turn], dtype=torch.int8),
            )
        return float(result[0])

    def test_quantized_numba_stays_close_to_float(self) -> None:
        positions = [
            chess.Board(),
            chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
            chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
        ]
        for board in positions:
            encoded = encode_board(board)
            integer = evaluate_quantized(
                encoded.white, encoded.black, int(encoded.turn), self.weights
            )
            reference = evaluate_quantized_reference(
                encoded.white, encoded.black, int(encoded.turn), self.weights
            )
            self.assertEqual(integer, reference)
            self.assertLessEqual(abs(self._float_evaluation(board) - integer), 8.0)

    def test_relative_colour_symmetry_reaches_model_output(self) -> None:
        board = chess.Board("r3k2r/pp2qppp/2npbn2/2p5/3NP3/2N1B3/PPP2PPP/R2Q1RK1 b kq - 4 11")
        self.assertAlmostEqual(
            self._float_evaluation(board),
            self._float_evaluation(board.mirror()),
            places=5,
        )

    def test_quantized_arrays_use_integer_types(self) -> None:
        self.assertEqual(self.weights.feature.dtype, np.int16)
        self.assertEqual(self.weights.feature_bias.dtype, np.int32)
        self.assertEqual(self.weights.output_weight.dtype, np.int16)


if __name__ == "__main__":
    unittest.main()
