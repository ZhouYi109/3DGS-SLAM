#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import numpy as np

from object_semantic_memory import ObjectMemory, ObjectObservation


def observation(
    feature,
    latent=None,
    centroid=(0.0, 0.0, 2.0),
    confidence=0.9,
    local_id=0,
):
    centroid = np.asarray(centroid, dtype=np.float32)
    return ObjectObservation(
        feature=np.asarray(feature, dtype=np.float32),
        latent=np.asarray(feature if latent is None else latent, dtype=np.float32),
        centroid=centroid,
        bbox_min=centroid - 0.25,
        bbox_max=centroid + 0.25,
        confidence=confidence,
        mask_area=1000,
        local_id=local_id,
    )


class ObjectMemoryTest(unittest.TestCase):
    def test_temporal_merge_and_query(self):
        memory = ObjectMemory(feature_similarity_gate=0.7, spatial_distance_gate=2.0)
        first = memory.update([observation([1.0, 0.0, 0.0])], 1.0)
        second = memory.update(
            [observation([0.99, 0.05, 0.0], centroid=(0.1, 0.0, 2.0))],
            2.0,
        )
        self.assertEqual(first, [0])
        self.assertEqual(second, [0])
        self.assertEqual(len(memory.nodes), 1)
        self.assertEqual(memory.nodes[0].observations, 2)
        self.assertEqual(memory.query(np.array([1.0, 0.0, 0.0]), 1)[0][0], 0)

    def test_spatial_gate_prevents_wrong_merge(self):
        memory = ObjectMemory(feature_similarity_gate=0.7, spatial_distance_gate=1.0)
        memory.update([observation([1.0, 0.0])], 1.0)
        assignment = memory.update(
            [observation([1.0, 0.0], centroid=(5.0, 0.0, 2.0))],
            2.0,
        )
        self.assertEqual(assignment, [1])
        self.assertEqual(len(memory.nodes), 2)

    def test_association_uses_object_latent_not_query_feature(self):
        memory = ObjectMemory(
            latent_similarity_gate=0.7,
            spatial_distance_gate=2.0,
        )
        memory.update(
            [observation([1.0, 0.0], latent=[0.0, 1.0])],
            1.0,
        )
        assignment = memory.update(
            [
                observation(
                    [-1.0, 0.0],
                    latent=[0.05, 0.99],
                    centroid=(0.1, 0.0, 2.0),
                )
            ],
            2.0,
        )
        self.assertEqual(assignment, [0])
        self.assertEqual(len(memory.nodes), 1)

    def test_persistence_shapes(self):
        memory = ObjectMemory()
        memory.update(
            [
                observation([1.0, 0.0, 0.0], local_id=0),
                observation([0.0, 1.0, 0.0], centroid=(2.0, 0.0, 2.0), local_id=1),
            ],
            1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            npz_path, json_path = memory.save(str(Path(directory) / "memory"))
            self.assertTrue(Path(npz_path).exists())
            self.assertTrue(Path(json_path).exists())
            # Explicitly close the archive so TemporaryDirectory cleanup also
            # works on Windows, where open files cannot be unlinked.
            with np.load(npz_path) as payload:
                self.assertEqual(payload["features"].shape, (2, 3))
                self.assertEqual(payload["query_features"].shape, (2, 3))
                self.assertEqual(payload["object_latents"].shape, (2, 3))
                self.assertEqual(payload["centroids"].shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
