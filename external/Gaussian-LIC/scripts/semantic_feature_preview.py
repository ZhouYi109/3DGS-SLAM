#!/usr/bin/env python3
"""Preview/export tool for aligned semantic Gaussian features.

Inputs:
- Gaussian-LIC `point_cloud.ply`
- semantic bundle files:
  - `semantic_feat.npy`
  - `semantic_mask.npy`

Outputs:
- colored semantic preview PLY
- summary JSON

The preview color is derived from the first three PCA components of the semantic
feature matrix. This is only for inspection and should not be treated as a
semantic label palette.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np


_PLY_TYPE_TO_DTYPE = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
}


@dataclass
class SemanticPreviewSummary:
    point_cloud_path: str
    semantic_feat_path: str
    semantic_mask_path: str
    num_vertices: int
    semantic_dim: int
    mask_true_count: int
    preview_ply_path: str
    alignment_ok: bool


def _read_ply_header(path: Path) -> Tuple[List[str], int, str]:
    header_lines: List[str] = []
    header_size = 0
    with path.open("rb") as fh:
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError(f"Unexpected EOF while reading PLY header: {path}")
            header_size += len(raw)
            line = raw.decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line == "end_header":
                break
    return header_lines, header_size, header_lines[1] if len(header_lines) > 1 else ""


def read_ply_xyz(point_cloud_path: str | Path) -> np.ndarray:
    """Read x/y/z for the vertex element from ascii or binary little-endian PLY."""
    path = Path(point_cloud_path)
    if not path.is_file():
        raise FileNotFoundError(f"PLY file not found: {path}")

    header_lines, header_size, fmt = _read_ply_header(path)
    if fmt not in {"format ascii 1.0", "format binary_little_endian 1.0"}:
        raise ValueError(f"Unsupported PLY format: {fmt}")

    vertex_count = None
    in_vertex = False
    vertex_props: List[Tuple[str, str]] = []
    for line in header_lines:
        if line.startswith("element "):
            parts = line.split()
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
                vertex_props = []
            continue
        if in_vertex and line.startswith("property "):
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Unsupported PLY property line: {line}")
            prop_type, prop_name = parts[1], parts[2]
            if prop_type not in _PLY_TYPE_TO_DTYPE:
                raise ValueError(f"Unsupported PLY property type: {prop_type}")
            vertex_props.append((prop_name, _PLY_TYPE_TO_DTYPE[prop_type]))

    if vertex_count is None or not vertex_props:
        raise ValueError(f"Could not parse vertex element from PLY: {path}")

    prop_names = [name for name, _ in vertex_props]
    if not {"x", "y", "z"}.issubset(prop_names):
        raise ValueError(f"PLY vertex properties must contain x/y/z: {path}")

    if fmt == "format ascii 1.0":
        xyz = np.empty((vertex_count, 3), dtype=np.float32)
        x_idx = prop_names.index("x")
        y_idx = prop_names.index("y")
        z_idx = prop_names.index("z")
        with path.open("r", encoding="ascii", errors="ignore") as fh:
            # Skip header
            while True:
                if fh.readline().strip() == "end_header":
                    break
            for i in range(vertex_count):
                cols = fh.readline().strip().split()
                xyz[i, 0] = float(cols[x_idx])
                xyz[i, 1] = float(cols[y_idx])
                xyz[i, 2] = float(cols[z_idx])
        return xyz

    dtype = np.dtype(vertex_props)
    with path.open("rb") as fh:
        fh.seek(header_size)
        arr = np.fromfile(fh, dtype=dtype, count=vertex_count)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32, copy=False)


def compute_preview_rgb(feat: np.ndarray) -> np.ndarray:
    """Project semantic features to 3D using PCA and map them to uint8 RGB."""
    if feat.ndim != 2:
        raise ValueError(f"Expected rank-2 feature matrix, got shape={feat.shape}")
    if feat.shape[1] < 3:
        raise ValueError(f"Expected semantic dim >= 3, got {feat.shape[1]}")

    feat32 = np.nan_to_num(feat.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    feat_centered = feat32 - feat32.mean(axis=0, keepdims=True)

    # Use a covariance eigen-decomposition on a capped sample for stability and speed.
    max_rows_for_basis = 20000
    if feat_centered.shape[0] > max_rows_for_basis:
        sample_idx = np.linspace(0, feat_centered.shape[0] - 1, max_rows_for_basis, dtype=np.int64)
        feat_sample = feat_centered[sample_idx]
    else:
        feat_sample = feat_centered

    cov = np.cov(feat_sample, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order[:3]]
    proj = feat_centered @ basis

    min_v = proj.min(axis=0, keepdims=True)
    max_v = proj.max(axis=0, keepdims=True)
    denom = np.where((max_v - min_v) < 1e-8, 1.0, (max_v - min_v))
    norm = (proj - min_v) / denom
    rgb = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return rgb


def write_preview_ply(output_path: str | Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("XYZ and RGB row counts do not match.")

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    with out.open("wb") as fh:
        fh.write(header)
        packed = np.empty(
            xyz.shape[0],
            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )
        packed["x"] = xyz[:, 0]
        packed["y"] = xyz[:, 1]
        packed["z"] = xyz[:, 2]
        packed["red"] = rgb[:, 0]
        packed["green"] = rgb[:, 1]
        packed["blue"] = rgb[:, 2]
        packed.tofile(fh)


def generate_preview(
    point_cloud_path: str | Path,
    semantic_feat_path: str | Path,
    semantic_mask_path: str | Path,
    output_dir: str | Path,
) -> SemanticPreviewSummary:
    point_cloud_path = Path(point_cloud_path)
    semantic_feat_path = Path(semantic_feat_path)
    semantic_mask_path = Path(semantic_mask_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xyz = read_ply_xyz(point_cloud_path)
    feat = np.load(semantic_feat_path)
    mask = np.load(semantic_mask_path)

    alignment_ok = xyz.shape[0] == feat.shape[0] == mask.shape[0]
    if not alignment_ok:
        raise ValueError(
            "Semantic preview input mismatch: "
            f"vertices={xyz.shape[0]}, feat_rows={feat.shape[0]}, mask_rows={mask.shape[0]}"
        )

    rgb = compute_preview_rgb(feat)
    preview_ply_path = output_dir / "semantic_preview_colored.ply"
    summary_path = output_dir / "semantic_preview_summary.json"

    write_preview_ply(preview_ply_path, xyz, rgb)

    summary = SemanticPreviewSummary(
        point_cloud_path=str(point_cloud_path),
        semantic_feat_path=str(semantic_feat_path),
        semantic_mask_path=str(semantic_mask_path),
        num_vertices=int(xyz.shape[0]),
        semantic_dim=int(feat.shape[1]),
        mask_true_count=int(mask.astype(np.bool_).sum()),
        preview_ply_path=str(preview_ply_path),
        alignment_ok=alignment_ok,
    )
    summary_path.write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a colored semantic preview PLY from semantic bundle outputs."
    )
    parser.add_argument("--point-cloud", required=True, help="Path to point_cloud.ply")
    parser.add_argument("--semantic-feat", required=True, help="Path to semantic_feat.npy")
    parser.add_argument("--semantic-mask", required=True, help="Path to semantic_mask.npy")
    parser.add_argument("--output-dir", required=True, help="Directory for preview outputs")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    summary = generate_preview(
        point_cloud_path=args.point_cloud,
        semantic_feat_path=args.semantic_feat,
        semantic_mask_path=args.semantic_mask,
        output_dir=args.output_dir,
    )
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
