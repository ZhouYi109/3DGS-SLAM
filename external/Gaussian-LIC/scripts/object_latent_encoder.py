#!/usr/bin/env python3
"""Non-CLIP object-state encoder used by Semantic Gaussian Prior."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class ObjectLatentEncoder:
    """Encode geometry, color, depth, and track confidence into z_object.

    A learned projection can be supplied as an ``.npz`` containing ``weight``
    with shape ``[16, latent_dim]`` and ``bias`` with shape ``[latent_dim]``.
    The identity-safe fallback keeps the pipeline executable before training.
    """

    descriptor_dim = 16

    def __init__(
        self,
        latent_dim: int = 16,
        weights_path: str = "",
        seed: int = 20260726,
    ) -> None:
        self.latent_dim = int(latent_dim)
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        self.weights_path = str(weights_path)
        self.learned_weights_loaded = False
        if self.weights_path:
            payload = np.load(Path(self.weights_path))
            self.weight = np.asarray(payload["weight"], dtype=np.float32)
            self.bias = np.asarray(payload["bias"], dtype=np.float32).reshape(-1)
            if self.weight.shape != (self.descriptor_dim, self.latent_dim):
                raise ValueError(
                    "object latent weight must have shape "
                    f"({self.descriptor_dim}, {self.latent_dim})"
                )
            if self.bias.shape != (self.latent_dim,):
                raise ValueError(
                    f"object latent bias must have shape ({self.latent_dim},)"
                )
            self.learned_weights_loaded = True
        elif self.latent_dim == self.descriptor_dim:
            self.weight = np.eye(self.descriptor_dim, dtype=np.float32)
            self.bias = np.zeros((self.latent_dim,), dtype=np.float32)
        else:
            generator = np.random.default_rng(int(seed))
            weight = generator.standard_normal(
                (self.descriptor_dim, self.latent_dim), dtype=np.float32
            )
            self.weight = weight / np.maximum(
                np.linalg.norm(weight, axis=0, keepdims=True), 1e-6
            )
            self.bias = np.zeros((self.latent_dim,), dtype=np.float32)

    @staticmethod
    def _masked_statistics(
        image_bgr: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        mask = np.asarray(mask, dtype=bool)
        colors = image_bgr[mask].astype(np.float32)[:, ::-1] / 255.0
        if colors.size:
            color_mean = colors.mean(axis=0)
            color_std = colors.std(axis=0)
        else:
            color_mean = np.zeros((3,), dtype=np.float32)
            color_std = np.zeros((3,), dtype=np.float32)
        valid_depth = depth[mask & np.isfinite(depth) & (depth > 0.0)]
        if valid_depth.size:
            depth_median = float(np.median(valid_depth))
            depth_std = float(np.std(valid_depth))
        else:
            depth_median = 0.0
            depth_std = 0.0
        area_ratio = float(mask.mean()) if mask.size else 0.0
        return color_mean, color_std, depth_median, depth_std, area_ratio

    def descriptor(
        self,
        image_bgr: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        centroid: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        confidence: float,
    ) -> np.ndarray:
        centroid = np.nan_to_num(
            np.asarray(centroid, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        extent = np.nan_to_num(
            np.asarray(bbox_max, dtype=np.float32)
            - np.asarray(bbox_min, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        extent = np.maximum(extent, 0.0)
        color_mean, color_std, depth_median, depth_std, area_ratio = (
            self._masked_statistics(image_bgr, depth, mask)
        )
        values = np.concatenate(
            [
                np.tanh(centroid / 20.0),
                np.tanh(np.log1p(extent)),
                color_mean * 2.0 - 1.0,
                np.clip(color_std * 4.0 - 1.0, -1.0, 1.0),
                np.array(
                    [
                        np.tanh(depth_median / 20.0),
                        np.tanh(np.log1p(max(0.0, depth_std))),
                        np.sqrt(max(0.0, area_ratio)) * 2.0 - 1.0,
                        np.clip(float(confidence), 0.0, 1.0) * 2.0 - 1.0,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        if values.shape != (self.descriptor_dim,):
            raise RuntimeError("object descriptor shape invariant failed")
        return values

    def encode(
        self,
        image_bgr: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        centroid: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        confidence: float,
    ) -> np.ndarray:
        descriptor = self.descriptor(
            image_bgr,
            depth,
            mask,
            centroid,
            bbox_min,
            bbox_max,
            confidence,
        )
        latent = np.tanh(descriptor @ self.weight + self.bias).astype(np.float32)
        norm = float(np.linalg.norm(latent))
        if not np.isfinite(norm) or norm < 1e-8:
            latent = np.zeros((self.latent_dim,), dtype=np.float32)
            latent[-1] = 1.0
            return latent
        return latent / norm
