import unittest

from embedding.embedding import (
    Embedding,
    PCAConfigurationSpaceEmbedding,
    RawEmbedding,
    StaticEmbedding,
    best_embedding,
)
from typing import Optional

import numpy as np


class EmbeddingTestCase(unittest.TestCase):
    simple_data = np.array(
        [
            [[0, 1], [5, 3]],
            [[0.6, 1.4], [6, 2]],
            [[1, 2], [7, 1]],
        ],
        dtype=np.float32,
    )

    def _complex_data(self) -> np.ndarray:
        # we have three vectors; we generate three vertices, each one goes
        # linearly along each vector for 30 frames starting at [10,0,-10]
        # configuration space will need 2 configurations; geometry will need 3 lines
        v1 = np.array([1, 1, 1])
        v2 = np.array([0.5, 0.5, 1])
        v3 = np.array([-1, 1, -1])
        origin = np.array([10, 0, -10])

        def p1(t) -> np.ndarray:
            return origin + t * v1

        def p2(t) -> np.ndarray:
            return origin + t * v2

        def p3(t) -> np.ndarray:
            return origin + t * v3

        data = [(p1(t), p2(t), p3(t)) for t in range(30)]

        return np.array(data)

    def _basic_embedding_tests(
        self, cls, quality: float, data: Optional[np.ndarray] = None
    ) -> bool:
        """
        Basic tests on an embedding that should work on any subclass, using
        simple_data above.

        Returns False if the data is not valid for the given embedding and
        quality bound, otherwise True.
        """
        assert issubclass(cls, Embedding)
        if data is None:
            data = self.simple_data
        if not cls.is_valid(data, quality):
            return False
        embed = cls.from_data(data, quality)
        projected = embed.project(data)
        inverted = embed.invert(projected)
        self.assertTrue(np.allclose(inverted, data, rtol=0, atol=quality))
        b = embed.tobytes()
        embed2, offset = cls.from_bytes(b)
        self.assertEqual(offset, len(b))

        b = embed2.write_projection(projected)
        projected2, offset = embed2.read_projection(b)
        self.assertEqual(offset, len(b))
        self.assertTrue(np.allclose(projected2, projected, rtol=0, atol=quality))

        inverted2 = embed2.invert(projected2)
        self.assertTrue(np.allclose(inverted2, inverted, rtol=0, atol=quality))
        self.assertTrue(np.allclose(inverted2, data, rtol=0, atol=quality))

        return True

    def test_embedding_raw(self) -> None:
        is_valid = self._basic_embedding_tests(RawEmbedding, quality=1e-6)
        self.assertTrue(is_valid)

        # Now test that actually the raw embedding is actually the identity
        # transform.  Not only close, but identical! Serialization may cause
        # rounding, is all.
        raw = RawEmbedding.from_data(self.simple_data, quality=1)
        projected = raw.project(self.simple_data)
        self.assertTrue(np.all(projected == self.simple_data))
        inverted = raw.invert(projected)
        self.assertTrue(np.all(inverted == self.simple_data))

    def test_embedding_static(self) -> None:
        # The simple data is all within 1 unit. So it's not a valid static
        # embedding at quality 1e-6 but it *is* valid at quality 1.
        is_valid = self._basic_embedding_tests(StaticEmbedding, quality=1e-6)
        self.assertFalse(is_valid)

        is_valid = self._basic_embedding_tests(StaticEmbedding, quality=1)
        self.assertTrue(is_valid)

        # It's also a valid static embedding at quality 1e-2 if I divide by 1000.
        # (Divide by 100 and check for 1e-2 hits roundoff questions.)
        is_valid = self._basic_embedding_tests(
            StaticEmbedding, quality=1e-2, data=(self.simple_data / 1000)
        )
        self.assertTrue(is_valid)

    def test_embedding_pca_configuration(self) -> None:
        is_valid = self._basic_embedding_tests(
            PCAConfigurationSpaceEmbedding, quality=1e-6
        )
        self.assertTrue(is_valid)

    def test_best_embedding(self) -> None:
        # we'll need 2 configurations + centroid makes 3 frames of data for PCA
        embed = best_embedding(self.simple_data, quality=1e-6)
        self.assertIsInstance(embed, RawEmbedding)

        # All the data is within distance 1 of the centre so we can go static, which is 1 frame.
        embed = best_embedding(self.simple_data, quality=1)
        self.assertIsInstance(embed, StaticEmbedding)

        # The "complex" data shrinks to almost nothing under PCA: just 1 frame + centroid
        # rather than 30 frames raw.
        embed = best_embedding(self._complex_data(), quality=1e-2)
        self.assertIsInstance(embed, PCAConfigurationSpaceEmbedding)
