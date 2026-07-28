#!/usr/bin/env python3
"""Object-level open-vocabulary memory for Gaussian-LIC.

The core is ROS-free so association, persistence, and querying can be tested
without loading SAM2 or CLIP. Object association uses a task-specific latent;
CLIP-like query features are retained only for open-vocabulary retrieval.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def normalize_feature(feature: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(feature))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("semantic feature norm is too small")
    return feature / norm


def valid_geometry(centroid: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> bool:
    return bool(
        np.isfinite(centroid).all()
        and np.isfinite(bbox_min).all()
        and np.isfinite(bbox_max).all()
        and np.all(bbox_max >= bbox_min)
    )


def bbox_iou_3d(
    lhs_min: np.ndarray,
    lhs_max: np.ndarray,
    rhs_min: np.ndarray,
    rhs_max: np.ndarray,
) -> float:
    if not valid_geometry((lhs_min + lhs_max) * 0.5, lhs_min, lhs_max):
        return 0.0
    if not valid_geometry((rhs_min + rhs_max) * 0.5, rhs_min, rhs_max):
        return 0.0
    intersection = np.maximum(
        0.0, np.minimum(lhs_max, rhs_max) - np.maximum(lhs_min, rhs_min)
    )
    intersection_volume = float(np.prod(intersection))
    lhs_volume = float(np.prod(np.maximum(0.0, lhs_max - lhs_min)))
    rhs_volume = float(np.prod(np.maximum(0.0, rhs_max - rhs_min)))
    union = lhs_volume + rhs_volume - intersection_volume
    return intersection_volume / union if union > 1e-9 else 0.0


@dataclass
class ObjectObservation:
    feature: np.ndarray
    latent: np.ndarray
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    confidence: float
    mask_area: int
    local_id: int

    def __post_init__(self) -> None:
        self.feature = normalize_feature(self.feature)
        self.latent = normalize_feature(self.latent)
        self.centroid = np.asarray(self.centroid, dtype=np.float32).reshape(3)
        self.bbox_min = np.asarray(self.bbox_min, dtype=np.float32).reshape(3)
        self.bbox_max = np.asarray(self.bbox_max, dtype=np.float32).reshape(3)
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        self.mask_area = int(max(0, self.mask_area))
        self.local_id = int(self.local_id)


@dataclass
class ObjectNode:
    object_id: int
    feature: np.ndarray
    latent: np.ndarray
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    confidence: float
    observations: int
    first_seen: float
    last_seen: float


class ObjectMemory:
    def __init__(
        self,
        feature_similarity_gate: float = 0.72,
        latent_similarity_gate: Optional[float] = None,
        spatial_distance_gate: float = 3.0,
        feature_weight: float = 0.65,
        latent_weight: Optional[float] = None,
        spatial_weight: float = 0.25,
        bbox_weight: float = 0.10,
        spatial_sigma: float = 1.5,
        ema_min: float = 0.05,
        ema_max: float = 0.35,
        max_objects: int = 4096,
    ) -> None:
        self.latent_similarity_gate = float(
            feature_similarity_gate
            if latent_similarity_gate is None
            else latent_similarity_gate
        )
        self.spatial_distance_gate = float(spatial_distance_gate)
        self.latent_weight = float(
            feature_weight if latent_weight is None else latent_weight
        )
        self.spatial_weight = float(spatial_weight)
        self.bbox_weight = float(bbox_weight)
        self.spatial_sigma = float(max(1e-3, spatial_sigma))
        self.ema_min = float(ema_min)
        self.ema_max = float(ema_max)
        self.max_objects = int(max_objects)
        self.nodes: Dict[int, ObjectNode] = {}
        self.next_object_id = 0

    def _association_score(
        self, observation: ObjectObservation, node: ObjectNode
    ) -> Optional[float]:
        latent_score = float(np.dot(observation.latent, node.latent))
        if latent_score < self.latent_similarity_gate:
            return None

        spatial_score = 0.0
        bbox_score = 0.0
        observation_has_geometry = valid_geometry(
            observation.centroid, observation.bbox_min, observation.bbox_max
        )
        node_has_geometry = valid_geometry(node.centroid, node.bbox_min, node.bbox_max)
        if observation_has_geometry and node_has_geometry:
            distance = float(np.linalg.norm(observation.centroid - node.centroid))
            if distance > self.spatial_distance_gate:
                return None
            spatial_score = math.exp(
                -(distance * distance) / (2.0 * self.spatial_sigma * self.spatial_sigma)
            )
            bbox_score = bbox_iou_3d(
                observation.bbox_min,
                observation.bbox_max,
                node.bbox_min,
                node.bbox_max,
            )

        return (
            self.latent_weight * latent_score
            + self.spatial_weight * spatial_score
            + self.bbox_weight * bbox_score
        )

    def _new_node(self, observation: ObjectObservation, timestamp: float) -> int:
        if len(self.nodes) >= self.max_objects:
            raise RuntimeError("object memory reached max_objects")
        object_id = self.next_object_id
        self.next_object_id += 1
        self.nodes[object_id] = ObjectNode(
            object_id=object_id,
            feature=observation.feature.copy(),
            latent=observation.latent.copy(),
            centroid=observation.centroid.copy(),
            bbox_min=observation.bbox_min.copy(),
            bbox_max=observation.bbox_max.copy(),
            confidence=observation.confidence,
            observations=1,
            first_seen=float(timestamp),
            last_seen=float(timestamp),
        )
        return object_id

    def _update_node(
        self, node: ObjectNode, observation: ObjectObservation, timestamp: float
    ) -> None:
        eta = float(
            np.clip(
                observation.confidence / max(1.0, node.observations + 1.0),
                self.ema_min,
                self.ema_max,
            )
        )
        node.feature = normalize_feature(
            (1.0 - eta) * node.feature + eta * observation.feature
        )
        node.latent = normalize_feature(
            (1.0 - eta) * node.latent + eta * observation.latent
        )
        if valid_geometry(
            observation.centroid, observation.bbox_min, observation.bbox_max
        ):
            if valid_geometry(node.centroid, node.bbox_min, node.bbox_max):
                node.centroid = (1.0 - eta) * node.centroid + eta * observation.centroid
                node.bbox_min = (1.0 - eta) * node.bbox_min + eta * observation.bbox_min
                node.bbox_max = (1.0 - eta) * node.bbox_max + eta * observation.bbox_max
            else:
                node.centroid = observation.centroid.copy()
                node.bbox_min = observation.bbox_min.copy()
                node.bbox_max = observation.bbox_max.copy()
        node.confidence = float(
            np.clip((1.0 - eta) * node.confidence + eta * observation.confidence, 0.0, 1.0)
        )
        node.observations += 1
        node.last_seen = float(timestamp)

    def update(
        self, observations: Sequence[ObjectObservation], timestamp: float
    ) -> List[int]:
        assignments = [-1] * len(observations)
        candidates: List[Tuple[float, int, int]] = []
        for observation_index, observation in enumerate(observations):
            for object_id, node in self.nodes.items():
                score = self._association_score(observation, node)
                if score is not None:
                    candidates.append((score, observation_index, object_id))

        used_observations = set()
        used_objects = set()
        for _, observation_index, object_id in sorted(candidates, reverse=True):
            if observation_index in used_observations or object_id in used_objects:
                continue
            assignments[observation_index] = object_id
            used_observations.add(observation_index)
            used_objects.add(object_id)

        for observation_index, observation in enumerate(observations):
            object_id = assignments[observation_index]
            if object_id < 0:
                object_id = self._new_node(observation, timestamp)
                assignments[observation_index] = object_id
            else:
                self._update_node(self.nodes[object_id], observation, timestamp)
        return assignments

    def query(self, query_embedding: np.ndarray, topk: int = 5) -> List[Tuple[int, float]]:
        if not self.nodes:
            return []
        query = normalize_feature(query_embedding)
        scored = [
            (object_id, float(np.dot(node.feature, query)))
            for object_id, node in self.nodes.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, min(int(topk), len(scored)))]

    def snapshot(self, include_features: bool = False) -> Dict[str, object]:
        objects = []
        for object_id in sorted(self.nodes):
            node = self.nodes[object_id]
            item = {
                "object_id": object_id,
                "centroid": node.centroid.astype(float).tolist(),
                "bbox_min": node.bbox_min.astype(float).tolist(),
                "bbox_max": node.bbox_max.astype(float).tolist(),
                "confidence": node.confidence,
                "observations": node.observations,
                "first_seen": node.first_seen,
                "last_seen": node.last_seen,
            }
            if include_features:
                item["query_feature"] = node.feature.astype(float).tolist()
                item["object_latent"] = node.latent.astype(float).tolist()
            objects.append(item)
        return {
            "schema_version": 1,
            "association_embedding": "object_latent",
            "query_embedding": "open_vocabulary_feature",
            "next_object_id": self.next_object_id,
            "object_count": len(objects),
            "objects": objects,
        }

    def save(self, output_prefix: str) -> Tuple[str, str]:
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        object_ids = np.array(sorted(self.nodes), dtype=np.int32)
        if object_ids.size:
            nodes = [self.nodes[int(object_id)] for object_id in object_ids]
            features = np.stack([node.feature for node in nodes]).astype(np.float16)
            object_latents = np.stack([node.latent for node in nodes]).astype(np.float16)
            centroids = np.stack([node.centroid for node in nodes]).astype(np.float32)
            bbox_min = np.stack([node.bbox_min for node in nodes]).astype(np.float32)
            bbox_max = np.stack([node.bbox_max for node in nodes]).astype(np.float32)
            confidence = np.array([node.confidence for node in nodes], dtype=np.float32)
            observations = np.array([node.observations for node in nodes], dtype=np.int32)
            first_seen = np.array([node.first_seen for node in nodes], dtype=np.float64)
            last_seen = np.array([node.last_seen for node in nodes], dtype=np.float64)
        else:
            features = np.empty((0, 0), dtype=np.float16)
            object_latents = np.empty((0, 0), dtype=np.float16)
            centroids = np.empty((0, 3), dtype=np.float32)
            bbox_min = np.empty((0, 3), dtype=np.float32)
            bbox_max = np.empty((0, 3), dtype=np.float32)
            confidence = np.empty((0,), dtype=np.float32)
            observations = np.empty((0,), dtype=np.int32)
            first_seen = np.empty((0,), dtype=np.float64)
            last_seen = np.empty((0,), dtype=np.float64)

        npz_path = str(prefix.with_suffix(".npz"))
        json_path = str(prefix.with_suffix(".json"))
        with tempfile.NamedTemporaryFile(
            dir=str(prefix.parent), suffix=".npz", delete=False
        ) as handle:
            temp_npz = handle.name
        try:
            np.savez_compressed(
                temp_npz,
                object_ids=object_ids,
                object_latents=object_latents,
                query_features=features,
                # Keep the legacy key readable by existing query tools.
                features=features,
                centroids=centroids,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                confidence=confidence,
                observations=observations,
                first_seen=first_seen,
                last_seen=last_seen,
            )
            os.replace(temp_npz, npz_path)
        finally:
            if os.path.exists(temp_npz):
                os.unlink(temp_npz)

        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(prefix.parent),
            suffix=".json",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temp_json = handle.name
            json.dump(self.snapshot(include_features=False), handle, indent=2)
            handle.write("\n")
        try:
            os.replace(temp_json, json_path)
        finally:
            if os.path.exists(temp_json):
                os.unlink(temp_json)
        return npz_path, json_path
