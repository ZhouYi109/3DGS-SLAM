#!/usr/bin/env python3
"""Standard loader for Gaussian-LIC semantic sidecar bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SemanticBundle:
    bundle_dir: Path
    feat: np.ndarray
    mask: np.ndarray
    meta: dict

    @property
    def num_gaussians(self) -> int:
        return int(self.feat.shape[0])

    @property
    def semantic_dim(self) -> int:
        return int(self.feat.shape[1])

    @property
    def active_count(self) -> int:
        return int(self.mask.sum())


class SemanticBundleLoader:
    """Read the semantic sidecar exported by Gaussian-LIC.

    Expected files under `bundle_dir`:
    - `semantic_feat_clean.npy`
    - `semantic_mask.npy`
    - `semantic_sidecar_info.json`
    """

    def __init__(self, bundle_dir: str | Path):
        self.bundle_dir = Path(bundle_dir)

    def _require_file(self, filename: str) -> Path:
        path = self.bundle_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required semantic sidecar file not found: {path}")
        return path

    def load(self, *, mmap_mode: Optional[str] = "r") -> SemanticBundle:
        feat_path = self._require_file("semantic_feat_clean.npy")
        mask_path = self._require_file("semantic_mask.npy")
        meta_path = self._require_file("semantic_sidecar_info.json")

        feat = np.load(feat_path, mmap_mode=mmap_mode)
        mask = np.load(mask_path, mmap_mode=mmap_mode)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if feat.ndim != 2:
            raise ValueError(f"semantic_feat_clean.npy must be rank-2, got {feat.shape}")
        if mask.ndim != 1:
            raise ValueError(f"semantic_mask.npy must be rank-1, got {mask.shape}")
        if feat.shape[0] != mask.shape[0]:
            raise ValueError(
                "semantic sidecar row mismatch: "
                f"feat_rows={feat.shape[0]}, mask_rows={mask.shape[0]}"
            )

        return SemanticBundle(
            bundle_dir=self.bundle_dir,
            feat=feat,
            mask=mask.astype(bool, copy=False),
            meta=meta,
        )


def load_semantic_bundle(bundle_dir: str | Path, *, mmap_mode: Optional[str] = "r") -> SemanticBundle:
    return SemanticBundleLoader(bundle_dir).load(mmap_mode=mmap_mode)
