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
