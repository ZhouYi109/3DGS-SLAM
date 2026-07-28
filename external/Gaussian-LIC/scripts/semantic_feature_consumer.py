#!/usr/bin/env python3
"""Materialize aligned semantic feature bundles for downstream SLAM-3DGS consumers.

This sits one step above `semantic_feature_alignment.py`:
- validate static alignment
- load dense per-Gaussian semantic features
- export a stable bundle on disk for later consumers
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np

from semantic_feature_alignment import (
    build_alignment_summary,
    load_aligned_semantic_features,
)


def write_semantic_feature_bundle(
    point_cloud_path: str | Path,
    semantic_pt_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, str]:
    """Write a lightweight semantic bundle for later SLAM-3DGS ingestion."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    loaded = load_aligned_semantic_features(point_cloud_path, semantic_pt_path, require_exact=True)
    summary = loaded["summary"]
    feat_dense = loaded["feat_dense"]
    mask_full = loaded["mask_full"]

    feat_np = feat_dense.detach().cpu().numpy()
    mask_np = mask_full.detach().cpu().numpy().astype(np.bool_)

    feat_path = output_root / "semantic_feat.npy"
    mask_path = output_root / "semantic_mask.npy"
    summary_path = output_root / "semantic_alignment_summary.json"
    bundle_meta_path = output_root / "semantic_feature_bundle.json"

    np.save(feat_path, feat_np)
    np.save(mask_path, mask_np)
    summary_path.write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    bundle_meta = {
        "point_cloud_path": str(Path(point_cloud_path)),
        "semantic_pt_path": str(Path(semantic_pt_path)),
        "semantic_feat_path": str(feat_path),
        "semantic_mask_path": str(mask_path),
        "alignment_summary_path": str(summary_path),
        "num_gaussians": int(summary.ply_vertex_count),
        "semantic_dim": int(summary.feat_dim),
        "alignment_mode": summary.alignment_mode,
        "aligned_exact": bool(summary.aligned_exact),
        "storage": {
            "semantic_feat": "dense_full_numpy",
            "semantic_mask": "dense_bool_numpy",
        },
    }
    bundle_meta_path.write_text(
        json.dumps(bundle_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "semantic_feat_path": str(feat_path),
        "semantic_mask_path": str(mask_path),
        "alignment_summary_path": str(summary_path),
        "bundle_meta_path": str(bundle_meta_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write an aligned semantic feature bundle from point_cloud.ply and semantic .pt."
    )
    parser.add_argument("--point-cloud", required=True, help="Path to Gaussian-LIC point_cloud.ply")
    parser.add_argument("--semantic-pt", required=True, help="Path to Semantic Gaussians .pt file")
    parser.add_argument("--output-dir", required=True, help="Directory for exported semantic bundle")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    summary = build_alignment_summary(args.point_cloud, args.semantic_pt)
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    outputs = write_semantic_feature_bundle(args.point_cloud, args.semantic_pt, args.output_dir)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
