import unittest
import numpy as np
import torch
from nnue.features import PADDING_INDEX
from tools.train_nnue_balanced import groups, balanced_weights, objective


class BalancedTests(unittest.TestCase):
    def test_piece_groups(self):
        w = np.full((3, 30), PADDING_INDEX)
        for row, count in enumerate((6, 14, 15)):
            w[row, :count] = 0
        np.testing.assert_array_equal(groups(w), [0, 1, 2])

    def test_bounded_weights(self):
        w = balanced_weights(np.array([1, 10, 10000]))
        self.assertTrue(np.all((w >= .25) & (w <= 4)))
        with self.assertRaises(ValueError):
            balanced_weights(np.array([0, 10, 100]))

    def test_cap_and_gradient(self):
        prediction = torch.tensor([0., 0.], requires_grad=True)
        loss = objective(prediction, torch.tensor([10000., -10000.]), torch.ones(2), 1000)
        self.assertAlmostEqual(float(loss.detach()), 9.5)
        loss.backward()
        self.assertLess(float(prediction.grad[0]), 0)
        self.assertGreater(float(prediction.grad[1]), 0)
