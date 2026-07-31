#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from build_r3live_teacher_shards import (
    encode_parameter_target,
    teacher_sample_weight,
)
from semantic_gaussian_prior_model import (
    CONTEXT_INPUT_DIM,
    CONTEXT_FEATURE_NAMES,
    INPUT_DIM,
    BASE_FEATURE_NAMES,
    SEMANTIC_CONFIDENCE_INDEX,
    SemanticGaussianPriorHead,
    input_contract,
    prior_loss,
)
from train_semantic_gaussian_prior import (
    PriorShardDataset,
    configure_context_adapter_only,
    load_compatible_model_state,
    render_aware_rollout_weight,
    restore_context_adapter_base_columns,
)


class SemanticGaussianPriorTest(unittest.TestCase):
    def test_zero_initialized_head_preserves_baseline(self):
        model = SemanticGaussianPriorHead(hidden_dim=32, layers=3)
        output = model(torch.randn(11, INPUT_DIM))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_context_head_accepts_v4_contract(self):
        model = SemanticGaussianPriorHead(
            hidden_dim=32,
            layers=3,
            input_dim=CONTEXT_INPUT_DIM,
        )
        output = model(torch.randn(5, CONTEXT_INPUT_DIM))
        self.assertEqual(tuple(output.shape), (5, 14))

    def test_feature_contract_names_cover_every_input_column(self):
        base_name, base_features = input_contract(INPUT_DIM)
        context_name, context_features = input_contract(CONTEXT_INPUT_DIM)
        self.assertEqual(base_name, "base_v3")
        self.assertEqual(context_name, "context_v4")
        self.assertEqual(base_features, BASE_FEATURE_NAMES)
        self.assertEqual(
            context_features,
            BASE_FEATURE_NAMES + CONTEXT_FEATURE_NAMES,
        )
        self.assertEqual(len(base_features), INPUT_DIM)
        self.assertEqual(len(context_features), CONTEXT_INPUT_DIM)

    def test_v4_teacher_weight_uses_semantic_confidence_not_last_column(self):
        inputs = np.zeros((2, CONTEXT_INPUT_DIM), dtype=np.float32)
        inputs[:, SEMANTIC_CONFIDENCE_INDEX] = np.array(
            [0.0, 0.8],
            dtype=np.float32,
        )
        inputs[:, -1] = np.array([1.0, 0.0], dtype=np.float32)
        opacity = np.zeros((2, 1), dtype=np.float32)
        weights = teacher_sample_weight(inputs, opacity)
        np.testing.assert_allclose(weights, np.array([0.05, 0.4]))

    def test_named_feature_ablation_zeros_only_requested_column(self):
        _, feature_names = input_contract(CONTEXT_INPUT_DIM)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inputs = np.ones((2, CONTEXT_INPUT_DIM), dtype=np.float32)
            targets = np.zeros((2, 14), dtype=np.float32)
            targets[:, 6] = 1.0
            np.savez(
                root / "train.npz",
                input=inputs,
                target=targets,
                weight=np.ones(2, dtype=np.float32),
            )
            manifest = {
                "input_dim": CONTEXT_INPUT_DIM,
                "input_contract": "context_v4",
                "input_features": feature_names,
                "shards": [
                    {
                        "path": "train.npz",
                        "count": 2,
                        "split": "train",
                        "source": "r3live_teacher",
                        "sequence": "synthetic",
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            dataset = PriorShardDataset(
                manifest_path,
                "train",
                "r3live_teacher",
                ("tanh_log1p_spacing_over_scale",),
            )
            features, _, _ = dataset[0]
            self.assertEqual(float(features[-1]), 0.0)
            np.testing.assert_array_equal(
                features[:-1],
                np.ones(CONTEXT_INPUT_DIM - 1, dtype=np.float32),
            )

    def test_v3_checkpoint_expansion_starts_context_columns_at_zero(self):
        torch.manual_seed(7)
        old_model = SemanticGaussianPriorHead(
            hidden_dim=32,
            layers=3,
            input_dim=INPUT_DIM,
        )
        for parameter in old_model.parameters():
            torch.nn.init.normal_(parameter)
        new_model = SemanticGaussianPriorHead(
            hidden_dim=32,
            layers=3,
            input_dim=CONTEXT_INPUT_DIM,
        )
        load_compatible_model_state(new_model, old_model.state_dict())

        expanded_weight = new_model.state_dict()["backbone.0.weight"]
        self.assertTrue(
            torch.equal(
                expanded_weight[:, :INPUT_DIM],
                old_model.state_dict()["backbone.0.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                expanded_weight[:, INPUT_DIM:],
                torch.zeros_like(expanded_weight[:, INPUT_DIM:]),
            )
        )

        base_features = torch.randn(5, INPUT_DIM)
        context_features = torch.randn(5, CONTEXT_INPUT_DIM - INPUT_DIM)
        expanded_features = torch.cat((base_features, context_features), dim=1)
        torch.testing.assert_close(
            old_model(base_features),
            new_model(expanded_features),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_rollout_target_uses_same_bounded_decoder_contract(self):
        inputs = np.zeros((1, CONTEXT_INPUT_DIM), dtype=np.float32)
        inputs[:, 3:6] = 0.5
        base_scaling = np.zeros((1, 3), dtype=np.float32)
        base_opacity = np.zeros((1, 1), dtype=np.float32)
        target = encode_parameter_target(
            inputs,
            base_scaling,
            base_opacity,
            np.array([[2.0, -1.0, 0.0]], dtype=np.float32),
            np.array([[0.25, -0.25, 0.0]], dtype=np.float32),
            np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.75, 0.25, 0.5]], dtype=np.float32),
            np.array([[1.0]], dtype=np.float32),
            mean_offset_limit=4.0,
            target_encoding="decoded_v2",
        )
        np.testing.assert_allclose(
            target[0, 0:3],
            np.array([0.5, -0.25, 0.0], dtype=np.float32),
        )
        np.testing.assert_allclose(target[0, 3:6], [0.25, -0.25, 0.0])
        np.testing.assert_allclose(target[0, 6:10], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(target[0, 10:13], [0.999, -0.999, 0.0])
        self.assertEqual(float(target[0, 13]), 1.0)

    def test_rollout_dataset_and_render_weight(self):
        _, feature_names = input_contract(CONTEXT_INPUT_DIM)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rows = 3
            inputs = np.ones((rows, CONTEXT_INPUT_DIM), dtype=np.float32)
            targets = np.zeros((rows, 14), dtype=np.float32)
            targets[:, 6] = 1.0
            np.savez(
                root / "train.npz",
                input=inputs,
                target=targets,
                weight=np.ones(rows, dtype=np.float32),
                rollout_target=targets,
                rollout_visibility=np.array([0.0, 0.5, 1.0], dtype=np.float32),
                rollout_gradient=np.array(
                    [[0.0] * 5, [1.0] * 5, [4.0] * 5],
                    dtype=np.float32,
                ),
            )
            manifest = {
                "input_dim": CONTEXT_INPUT_DIM,
                "input_contract": "context_v4",
                "input_features": feature_names,
                "rollout_contract": {
                    "format": "render-gradient-rollout-v1",
                    "steps": 5,
                },
                "shards": [
                    {
                        "path": "train.npz",
                        "count": rows,
                        "split": "train",
                        "source": "r3live_teacher",
                        "sequence": "synthetic",
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            dataset = PriorShardDataset(
                manifest_path,
                "train",
                "r3live_teacher",
                include_rollout=True,
            )
            self.assertEqual(len(dataset[0]), 6)
            rollout_weight = render_aware_rollout_weight(
                torch.ones(rows),
                torch.tensor([0.0, 0.5, 1.0]),
                torch.tensor([[0.0] * 5, [1.0] * 5, [4.0] * 5]),
                visibility_floor=0.1,
                gradient_gain=0.1,
            )
            self.assertTrue(torch.isfinite(rollout_weight).all())
            self.assertLess(float(rollout_weight[0]), float(rollout_weight[1]))
            self.assertLess(float(rollout_weight[1]), float(rollout_weight[2]))

    def test_context_adapter_only_preserves_v3_parameters(self):
        torch.manual_seed(11)
        old_model = SemanticGaussianPriorHead(
            hidden_dim=32,
            layers=3,
            input_dim=INPUT_DIM,
        )
        for parameter in old_model.parameters():
            torch.nn.init.normal_(parameter)
        new_model = SemanticGaussianPriorHead(
            hidden_dim=32,
            layers=3,
            input_dim=CONTEXT_INPUT_DIM,
        )
        load_compatible_model_state(new_model, old_model.state_dict())
        initial_state = {
            key: value.clone() for key, value in new_model.state_dict().items()
        }
        input_weight, frozen_base_columns = configure_context_adapter_only(
            new_model
        )
        optimizer = torch.optim.AdamW(
            [input_weight],
            lr=1e-2,
            weight_decay=0.1,
        )
        loss = new_model(torch.randn(8, CONTEXT_INPUT_DIM)).square().mean()
        loss.backward()
        optimizer.step()
        restore_context_adapter_base_columns(
            input_weight,
            frozen_base_columns,
        )

        current_state = new_model.state_dict()
        self.assertTrue(
            torch.equal(
                current_state["backbone.0.weight"][:, :INPUT_DIM],
                initial_state["backbone.0.weight"][:, :INPUT_DIM],
            )
        )
        self.assertFalse(
            torch.equal(
                current_state["backbone.0.weight"][:, INPUT_DIM:],
                initial_state["backbone.0.weight"][:, INPUT_DIM:],
            )
        )
        for key, value in current_state.items():
            if key != "backbone.0.weight":
                self.assertTrue(torch.equal(value, initial_state[key]), key)

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
