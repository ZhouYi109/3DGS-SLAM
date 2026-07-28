#!/usr/bin/env python3
"""Static alignment utilities for Semantic Gaussians .pt features and Gaussian-LIC PLY outputs.

This module focuses on the lowest-risk integration step:
1. read a Gaussian-LIC `point_cloud.ply`
2. read a Semantic Gaussians `*.pt`
3. validate whether semantic features can be aligned to the exported Gaussian order
4. optionally expand sparse semantic rows to a dense per-Gaussian tensor
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SemanticPtInfo:
    feat_rows: int
    feat_dim: int
    feat_dtype: str
    mask_rows: int
    mask_true_count: int
    mask_dtype: str


@dataclass
class SemanticAlignmentSummary:
    point_cloud_path: str
    semantic_pt_path: str
    ply_vertex_count: int
    feat_rows: int
    feat_dim: int
    mask_rows: int
    mask_true_count: int
    feat_dtype: str
    mask_dtype: str
    alignment_mode: str
    aligned_exact: bool


def read_ply_vertex_count(point_cloud_path: str | Path) -> int:
    """Read only the PLY header and return the vertex count."""
    path = Path(point_cloud_path)
    if not path.is_file():
        raise FileNotFoundError(f"PLY file not found: {path}")

    vertex_count: Optional[int] = None
    with path.open("rb") as fh:
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError(f"Unexpected EOF while reading PLY header: {path}")
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break

    if vertex_count is None:
        raise ValueError(f"Could not find 'element vertex' in PLY header: {path}")
    return vertex_count


def _load_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("This module requires PyTorch to read semantic .pt files.") from exc
    return torch


def read_semantic_pt_info(semantic_pt_path: str | Path) -> SemanticPtInfo:
    """Read semantic .pt metadata without making assumptions about alignment yet."""
    torch = _load_torch()
    path = Path(semantic_pt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Semantic .pt file not found: {path}")

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Semantic .pt must contain a dict, got: {type(payload)!r}")
    if "feat" not in payload or "mask_full" not in payload:
        raise ValueError("Semantic .pt must contain 'feat' and 'mask_full'.")

    feat = payload["feat"]
    mask = payload["mask_full"]

    if feat.ndim != 2:
        raise ValueError(f"'feat' must be rank-2, got shape={tuple(feat.shape)}")
    if mask.ndim != 1:
        raise ValueError(f"'mask_full' must be rank-1, got shape={tuple(mask.shape)}")

    return SemanticPtInfo(
        feat_rows=int(feat.shape[0]),
        feat_dim=int(feat.shape[1]),
        feat_dtype=str(feat.dtype),
        mask_rows=int(mask.shape[0]),
        mask_true_count=int(mask.sum().item()),
        mask_dtype=str(mask.dtype),
    )


def build_alignment_summary(
    point_cloud_path: str | Path,
    semantic_pt_path: str | Path,
) -> SemanticAlignmentSummary:
    """Inspect the two files and determine the static alignment mode."""
    ply_vertex_count = read_ply_vertex_count(point_cloud_path)
    pt_info = read_semantic_pt_info(semantic_pt_path)

    if pt_info.mask_rows != ply_vertex_count:
        alignment_mode = "mask_length_mismatch"
        aligned_exact = False
    elif pt_info.feat_rows == ply_vertex_count:
        alignment_mode = "dense_full"
        aligned_exact = True
    elif pt_info.feat_rows == pt_info.mask_true_count:
        alignment_mode = "masked_sparse"
        aligned_exact = True
    else:
        alignment_mode = "feat_length_mismatch"
        aligned_exact = False

    return SemanticAlignmentSummary(
        point_cloud_path=str(Path(point_cloud_path)),
        semantic_pt_path=str(Path(semantic_pt_path)),
        ply_vertex_count=ply_vertex_count,
        feat_rows=pt_info.feat_rows,
        feat_dim=pt_info.feat_dim,
        mask_rows=pt_info.mask_rows,
        mask_true_count=pt_info.mask_true_count,
        feat_dtype=pt_info.feat_dtype,
        mask_dtype=pt_info.mask_dtype,
        alignment_mode=alignment_mode,
        aligned_exact=aligned_exact,
    )


def load_aligned_semantic_features(
    point_cloud_path: str | Path,
    semantic_pt_path: str | Path,
    *,
    require_exact: bool = True,
    fill_value: float = 0.0,
) -> Dict[str, Any]:
    """Return dense per-Gaussian semantic features aligned to the PLY vertex order.

    Supported cases:
    - dense_full: feat rows already equal to PLY vertex count
    - masked_sparse: feat rows equal mask_true_count; expand back to full length
    """
    torch = _load_torch()
    summary = build_alignment_summary(point_cloud_path, semantic_pt_path)
    if require_exact and not summary.aligned_exact:
        raise ValueError(
            "Semantic feature alignment failed: "
            f"mode={summary.alignment_mode}, "
            f"ply_vertex_count={summary.ply_vertex_count}, "
            f"feat_rows={summary.feat_rows}, "
            f"mask_rows={summary.mask_rows}, "
            f"mask_true_count={summary.mask_true_count}"
        )

    payload = torch.load(Path(semantic_pt_path), map_location="cpu")
    feat = payload["feat"]
    mask = payload["mask_full"].to(dtype=torch.bool)

    if summary.alignment_mode == "dense_full":
        dense_feat = feat
    elif summary.alignment_mode == "masked_sparse":
        dense_feat = torch.full(
            (summary.ply_vertex_count, summary.feat_dim),
            fill_value,
            dtype=feat.dtype,
        )
        dense_feat[mask] = feat
    else:
        raise ValueError(
            "Unsupported alignment mode for dense feature loading: "
            f"{summary.alignment_mode}"
        )

    return {
        "summary": summary,
        "feat_dense": dense_feat,
        "mask_full": mask,
    }


def write_alignment_summary(
    point_cloud_path: str | Path,
    semantic_pt_path: str | Path,
    output_json_path: str | Path,
) -> SemanticAlignmentSummary:
    summary = build_alignment_summary(point_cloud_path, semantic_pt_path)
    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize static alignment between point_cloud.ply and semantic .pt."
    )
    parser.add_argument("--point-cloud", required=True, help="Path to Gaussian-LIC point_cloud.ply")
    parser.add_argument("--semantic-pt", required=True, help="Path to Semantic Gaussians .pt file")
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional output JSON path for writing alignment summary",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    summary = build_alignment_summary(args.point_cloud, args.semantic_pt)
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    if args.out_json:
        write_alignment_summary(args.point_cloud, args.semantic_pt, args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
