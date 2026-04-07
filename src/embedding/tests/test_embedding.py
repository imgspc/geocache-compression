import unittest

from embedding.embedding import RawEmbedding, Embedding, StaticEmbedding
from typing import Optional

import numpy as np


class EmbeddingTestCase(unittest.TestCase):
    simple_data = np.array(
        [
            [[0, 1], [5, 3]],
            [[0.5, 1.5], [6, 2]],
            [[1, 2], [7, 1]],
        ],
        dtype=np.float32,
    )

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

        b = projected.tobytes()
        projected2, offset = embed2.read_projection(b)
        self.assertEqual(offset, len(b))

        self.assertTrue(np.allclose(projected, projected2, rtol=0, atol=quality))
        inverted2 = embed2.invert(projected2)
        self.assertTrue(np.allclose(inverted2, data, rtol=0, atol=quality))

        return True

    def test_raw(self) -> None:
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

    def test_static(self) -> None:
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
