#!/usr/bin/env python3
"""Lightweight feed-forward head for Semantic Gaussian Prior."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


INPUT_DIM = 24
OUTPUT_DIM = 14
OBJECT_LATENT_DIM = 16


class SemanticGaussianPriorHead(nn.Module):
    """Predict bounded-decoder residuals for one Gaussian candidate."""

    def __init__(self, hidden_dim: int = 64, layers: int = 3) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("layers must be at least 2")
        blocks = []
        in_dim = INPUT_DIM
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
        if features.shape[-1] != 24:
            raise RuntimeError("Semantic Gaussian Prior expects 24 input values")
        return self.output(self.backbone(features))


def prior_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Group-balanced distillation loss for (mu, scale, q, RGB, opacity)."""

    if prediction.shape != target.shape or prediction.shape[-1] != OUTPUT_DIM:
        raise ValueError("prediction and target must both be [N,14]")
    weight = sample_weight.reshape(-1).clamp_min(0.0)
    denominator = weight.sum().clamp_min(1e-6)

    def reduce(values: torch.Tensor) -> torch.Tensor:
        per_sample = values.reshape(values.shape[0], -1).mean(dim=1)
        return (per_sample * weight).sum() / denominator

    mean = reduce(F.smooth_l1_loss(prediction[:, 0:3], target[:, 0:3], reduction="none"))
    scale = reduce(F.smooth_l1_loss(prediction[:, 3:6], target[:, 3:6], reduction="none"))
    pred_q = prediction[:, 6:10].clone()
    pred_q[:, 0] = pred_q[:, 0] + 1.0
    pred_q = F.normalize(pred_q, dim=1)
    target_q = F.normalize(target[:, 6:10], dim=1)
    rotation = reduce(1.0 - torch.abs((pred_q * target_q).sum(dim=1)))
    color = reduce(F.smooth_l1_loss(prediction[:, 10:13], target[:, 10:13], reduction="none"))
    opacity = reduce(F.smooth_l1_loss(prediction[:, 13:14], target[:, 13:14], reduction="none"))
    total = mean + scale + 0.5 * rotation + 0.5 * color + 0.25 * opacity
    return total, {
        "mean": mean.detach(),
        "scale": scale.detach(),
        "rotation": rotation.detach(),
        "color": color.detach(),
        "opacity": opacity.detach(),
    }
