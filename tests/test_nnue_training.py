"""Regression checks for the NNUE training target convention."""

import unittest

import numpy as np
import torch

from nnue.shards import ShardBatch
from tools.train_nnue import _tensor_batch


class NNUETrainingTests(unittest.TestCase):
    def test_lichess_targets_remain_side_to_move_relative(self) -> None:
        batch = ShardBatch(
            white=np.zeros((2, 30), dtype=np.int32),
            black=np.zeros((2, 30), dtype=np.int32),
            turn=np.asarray([1, -1], dtype=np.int8),
            target_cp=np.asarray([125.0, 250.0], dtype=np.float32),
        )

        _, _, turn, target = _tensor_batch(batch, torch.device("cpu"))

        torch.testing.assert_close(turn, torch.tensor([1, -1], dtype=torch.int8))
        torch.testing.assert_close(target, torch.tensor([125.0, 250.0]))


if __name__ == "__main__":
    unittest.main()
