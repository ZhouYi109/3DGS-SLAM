#!/usr/bin/env python3
"""Lightweight feed-forward head for Semantic Gaussian Prior."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


BASE_INPUT_DIM = 24
CONTEXT_INPUT_DIM = 38
SUPPORTED_INPUT_DIMS = (BASE_INPUT_DIM, CONTEXT_INPUT_DIM)
INPUT_DIM = BASE_INPUT_DIM
OUTPUT_DIM = 14
OBJECT_LATENT_DIM = 16
OBJECT_LATENT_START_INDEX = 7
OBJECT_LATENT_END_INDEX = OBJECT_LATENT_START_INDEX + OBJECT_LATENT_DIM
BASE_FEATURE_NAMES = (
    "xyz_tanh_x",
    "xyz_tanh_y",
    "xyz_tanh_z",
    "rgb_r",
    "rgb_g",
    "rgb_b",
    "log1p_depth",
    *(f"object_latent_{index:02d}" for index in range(OBJECT_LATENT_DIM)),
    "semantic_confidence",
)
SEMANTIC_CONFIDENCE_INDEX = BASE_FEATURE_NAMES.index("semantic_confidence")
CONTEXT_FEATURE_NAMES = (
    "pixel_u_normalized",
    "pixel_v_normalized",
    "rgb_gradient_x",
    "rgb_gradient_y",
    "rgb_gradient_magnitude",
    "relative_depth_gradient_x",
    "relative_depth_gradient_y",
    "relative_depth_gradient_magnitude",
    "appearance_reliability",
    "depth_reliability",
    "geometry_reliability",
    "pose_reliability",
    "uncovered_fraction",
    "tanh_log1p_spacing_over_scale",
)


def input_contract(input_dim: int) -> tuple[str, tuple[str, ...]]:
    if input_dim == BASE_INPUT_DIM:
        return "base_v3", BASE_FEATURE_NAMES
    if input_dim == CONTEXT_INPUT_DIM:
        return "context_v4", BASE_FEATURE_NAMES + CONTEXT_FEATURE_NAMES
    raise ValueError(f"unsupported prior input dimension: {input_dim}")


class SemanticGaussianPriorHead(nn.Module):
    """Predict bounded-decoder residuals for one Gaussian candidate."""

    def __init__(
        self,
        hidden_dim: int = 64,
        layers: int = 3,
        input_dim: int = INPUT_DIM,
    ) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("layers must be at least 2")
        if input_dim not in SUPPORTED_INPUT_DIMS:
            raise ValueError(f"unsupported prior input dimension: {input_dim}")
        self.input_dim = input_dim
        blocks = []
        in_dim = input_dim
        for _ in range(layers - 1):
            blocks.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_dim),
                ]
            )
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*blocks)
        self.output = nn.Linear(in_dim, OUTPUT_DIM)

        # A zero residual exactly reproduces Gaussian-LIC's old initializer.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Keep the literal here: TorchScript does not capture module globals.
        if features.shape[-1] != self.input_dim:
            raise RuntimeError("Semantic Gaussian Prior input dimension mismatch")
        return self.output(self.backbone(features))


def prior_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    group_weights: tuple[float, float, float, float, float] = (
        1.0,
        1.0,
        0.5,
        0.5,
        0.25,
    ),
    decoded_residual_targets: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Group-balanced distillation loss for (mu, scale, q, RGB, opacity)."""

    if prediction.shape != target.shape or prediction.shape[-1] != OUTPUT_DIM:
        raise ValueError("prediction and target must both be [N,14]")
    weight = sample_weight.reshape(-1).clamp_min(0.0)
    denominator = weight.sum().clamp_min(1e-6)

    def reduce(values: torch.Tensor) -> torch.Tensor:
        per_sample = values.reshape(values.shape[0], -1).mean(dim=1)
        return (per_sample * weight).sum() / denominator

    bounded_prediction = torch.tanh(prediction) if decoded_residual_targets else prediction
    mean = reduce(
        F.smooth_l1_loss(
            bounded_prediction[:, 0:3],
            target[:, 0:3],
            reduction="none",
        )
    )
    scale = reduce(F.smooth_l1_loss(prediction[:, 3:6], target[:, 3:6], reduction="none"))
    pred_q = prediction[:, 6:10].clone()
    pred_q[:, 0] = pred_q[:, 0] + 1.0
    pred_q = F.normalize(pred_q, dim=1)
    target_q = F.normalize(target[:, 6:10], dim=1)
    rotation = reduce(1.0 - torch.abs((pred_q * target_q).sum(dim=1)))
    color = reduce(
        F.smooth_l1_loss(
            bounded_prediction[:, 10:13],
            target[:, 10:13],
            reduction="none",
        )
    )
    opacity = reduce(F.smooth_l1_loss(prediction[:, 13:14], target[:, 13:14], reduction="none"))
    if len(group_weights) != 5 or any(value < 0.0 for value in group_weights):
        raise ValueError("group_weights must contain five non-negative values")
    total = (
        group_weights[0] * mean
        + group_weights[1] * scale
        + group_weights[2] * rotation
        + group_weights[3] * color
        + group_weights[4] * opacity
    )
    return total, {
        "mean": mean.detach(),
        "scale": scale.detach(),
        "rotation": rotation.detach(),
        "color": color.detach(),
        "opacity": opacity.detach(),
    }
