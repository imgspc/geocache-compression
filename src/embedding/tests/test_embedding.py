import unittest

from embedding.embedding import RawEmbedding

import numpy as np


class EmbeddingTestCase(unittest.TestCase):
    def test_raw(self) -> None:
        data = np.array(
            [[[0, 1], [0.5, 1.5], [1, 2]], [[5, 3], [6, 2], [7, 1]]], dtype=np.float32
        )
        self.assertTrue(RawEmbedding.is_valid(data, quality=1))
        raw = RawEmbedding.from_data(data, quality=1)
        projected = raw.project(data)
        self.assertTrue(np.all(projected == data))
