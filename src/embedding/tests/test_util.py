import unittest
from embedding import util
import numpy as np
import math


class UtilTestCase(unittest.TestCase):
    def test_small_uint(self) -> None:
        def roundtrip(i, n):
            b = util.pack_small_uint(i)
            ii, _ = util.unpack_small_uint(b)
            self.assertEqual(i, ii)
            self.assertEqual(len(b), n)

        roundtrip(0, 1)
        roundtrip(1, 1)
        roundtrip(63, 1)
        roundtrip(64, 2)
        roundtrip(65, 2)
        roundtrip(16448, 3)
        roundtrip(1 << 32, 5)
        roundtrip(1 << 60, 9)

    def test_jagged(self) -> None:
        A = np.arange(1, 101).reshape((10, 10))
        B = util.zero_jagged(A, np.arange(10) // 2)
        self.assertEqual(B.shape[0], 10)
        self.assertEqual(B.shape[1], 4)
        nonzeros = np.count_nonzero(B, axis=1)
        for i in range(10):
            self.assertEqual(nonzeros[i], i // 2)
