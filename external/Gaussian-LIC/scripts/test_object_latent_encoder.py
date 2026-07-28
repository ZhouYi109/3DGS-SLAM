#!/usr/bin/env python3

import unittest

import numpy as np

from object_latent_encoder import ObjectLatentEncoder


class ObjectLatentEncoderTest(unittest.TestCase):
    def test_latent_is_normalized_and_not_clip_dimensional(self):
        encoder = ObjectLatentEncoder(latent_dim=16)
        image = np.full((8, 8, 3), (20, 80, 160), dtype=np.uint8)
        depth = np.full((8, 8), 2.0, dtype=np.float32)
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 1:7] = True
        latent = encoder.encode(
            image,
            depth,
            mask,
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([0.5, 1.5, 2.5], dtype=np.float32),
            np.array([1.5, 2.5, 3.5], dtype=np.float32),
            0.9,
        )
        self.assertEqual(latent.shape, (16,))
        self.assertAlmostEqual(float(np.linalg.norm(latent)), 1.0, places=5)

    def test_geometry_or_color_changes_latent(self):
        encoder = ObjectLatentEncoder(latent_dim=16)
        depth = np.ones((6, 6), dtype=np.float32)
        mask = np.ones((6, 6), dtype=bool)
        first = encoder.encode(
            np.zeros((6, 6, 3), dtype=np.uint8),
            depth,
            mask,
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            0.8,
        )
        second = encoder.encode(
            np.full((6, 6, 3), 255, dtype=np.uint8),
            depth,
            mask,
            np.array([2.0, 0.0, 0.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
            0.8,
        )
        self.assertLess(float(np.dot(first, second)), 0.99)


if __name__ == "__main__":
    unittest.main()
