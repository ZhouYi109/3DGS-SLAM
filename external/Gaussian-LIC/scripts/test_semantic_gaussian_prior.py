#!/usr/bin/env python3

import unittest

import torch

from semantic_gaussian_prior_model import (
    INPUT_DIM,
    SemanticGaussianPriorHead,
    prior_loss,
)


class SemanticGaussianPriorTest(unittest.TestCase):
    def test_zero_initialized_head_preserves_baseline(self):
        model = SemanticGaussianPriorHead(hidden_dim=32, layers=3)
        output = model(torch.randn(11, INPUT_DIM))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_loss_is_finite_and_backpropagates(self):
        model = SemanticGaussianPriorHead(hidden_dim=32, layers=2)
        prediction = model(torch.randn(8, INPUT_DIM))
        target = torch.randn(8, 14)
        target[:, 6] = target[:, 6] + 1.0
        loss, groups = prior_loss(prediction, target, torch.ones(8))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(groups), {"mean", "scale", "rotation", "color", "opacity"})
        self.assertIsNotNone(model.output.weight.grad)

    def test_decoded_targets_compare_bounded_residuals(self):
        prediction = torch.zeros(4, 14)
        prediction[:, 0] = torch.tensor([-2.0, -0.5, 0.5, 2.0])
        prediction[:, 10] = prediction[:, 0]
        target = torch.zeros_like(prediction)
        target[:, 0] = torch.tanh(prediction[:, 0])
        target[:, 10] = torch.tanh(prediction[:, 10])
        target[:, 6] = 1.0
        loss, groups = prior_loss(
            prediction,
            target,
            torch.ones(4),
            decoded_residual_targets=True,
        )
        self.assertAlmostEqual(float(groups["mean"]), 0.0, places=7)
        self.assertAlmostEqual(float(groups["color"]), 0.0, places=7)
        self.assertAlmostEqual(float(loss), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
