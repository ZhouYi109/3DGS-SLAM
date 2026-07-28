#!/usr/bin/env python3
"""Top-K query prototype over Gaussian-LIC semantic sidecars."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np

from semantic_bundle_loader import load_semantic_bundle


@dataclass
class SemanticQueryResult:
    gaussian_indices: List[int]
    similarity_scores: List[float]
    semantic_dim: int
    topk: int
    query_norm: float


def _normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def query_topk_by_embedding(
    semantic_feat: np.ndarray,
    query_embedding: np.ndarray,
    *,
    topk: int = 10,
    active_mask: np.ndarray | None = None,
) -> SemanticQueryResult:
    if semantic_feat.ndim != 2:
        raise ValueError(f"semantic_feat must be rank-2, got {semantic_feat.shape}")
    if query_embedding.ndim != 1:
        raise ValueError(f"query_embedding must be rank-1, got {query_embedding.shape}")
    if semantic_feat.shape[1] != query_embedding.shape[0]:
        raise ValueError(
            "query dim mismatch: "
            f"semantic_dim={semantic_feat.shape[1]}, query_dim={query_embedding.shape[0]}"
        )

    feat = np.nan_to_num(np.asarray(semantic_feat, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    query = np.nan_to_num(np.asarray(query_embedding, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if active_mask is not None:
        active_mask = np.asarray(active_mask, dtype=bool)
        if active_mask.shape != (feat.shape[0],):
            raise ValueError(f"active_mask shape mismatch: expected {(feat.shape[0],)}, got {active_mask.shape}")
        valid_indices = np.flatnonzero(active_mask)
        feat = feat[valid_indices]
    else:
        valid_indices = np.arange(feat.shape[0], dtype=np.int64)

    feat = _normalize_rows(feat)
    query_norm = float(np.linalg.norm(query))
    if query_norm < 1e-8:
        raise ValueError("query embedding norm is too small")
    query = query / query_norm

    scores = feat @ query
    topk = max(1, min(int(topk), scores.shape[0]))
    top_idx_local = np.argpartition(-scores, topk - 1)[:topk]
    top_idx_local = top_idx_local[np.argsort(-scores[top_idx_local])]
    top_idx_global = valid_indices[top_idx_local]

    return SemanticQueryResult(
        gaussian_indices=[int(x) for x in top_idx_global.tolist()],
        similarity_scores=[float(scores[i]) for i in top_idx_local.tolist()],
        semantic_dim=int(semantic_feat.shape[1]),
        topk=topk,
        query_norm=query_norm,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query Top-K Gaussians from a semantic sidecar bundle.")
    parser.add_argument("--bundle-dir", required=True, help="Path to Gaussian-LIC semantic_sidecar directory")
    parser.add_argument("--query-npy", required=True, help="Path to a 1D query embedding .npy")
    parser.add_argument("--topk", type=int, default=10, help="Number of Gaussians to return")
    parser.add_argument("--ignore-mask", action="store_true", help="Do not filter by semantic_mask.npy")
    parser.add_argument("--out-json", default="", help="Optional output JSON path")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    bundle = load_semantic_bundle(args.bundle_dir, mmap_mode="r")
    query = np.load(args.query_npy)
    result = query_topk_by_embedding(
        bundle.feat,
        query,
        topk=args.topk,
        active_mask=None if args.ignore_mask else bundle.mask,
    )
    payload = asdict(result)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
