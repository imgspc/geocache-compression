import unittest
from embedding import clustering
import numpy as np
import math

from embedding.tests.test_data import complex_data


class ClusteringTestCase(unittest.TestCase):
    def test_issorted(self) -> None:
        unsorted = np.array([1, 5, 2, 56, 2, 25, 2])
        issorted = np.array([1, 6, 8, 9, 10, 345, 6742])
        self.assertFalse(clustering._is_sorted(unsorted))
        self.assertTrue(clustering._is_sorted(issorted))

        self.assertTrue(clustering._is_sorted(np.array([])))

    def test_covering_api(self) -> None:
        covering = clustering.Covering(np.arange(53), np.array([17]))
        self.assertEqual(2, covering.nsubsets)
        self.assertTrue(covering.is_id_permutation())
        subsets = list(covering.subsets)
        self.assertTrue(np.all(np.arange(17) == subsets[0]))
        self.assertTrue(np.all(np.arange(17, 53) == subsets[1]))
        with self.assertRaises(ValueError):
            covering.indices[3] = 7
            covering.verify()
        covering.indices[3] = 3

        b = covering.tobytes()
        covering2, offset = clustering.Covering.from_bytes(b, offset=0)
        self.assertEqual(len(b), offset)
        self.assertTrue(np.all(np.arange(53) == covering2.indices))
        self.assertEqual(2, covering.nsubsets)
        self.assertTrue(covering.is_id_permutation())
        subsets2 = list(covering2.subsets)
        self.assertTrue(np.all(np.arange(17) == subsets2[0]))
        self.assertTrue(np.all(np.arange(17, 53) == subsets2[1]))

        # TODO: the various constructors

    def test_covering_by_index(self) -> None:
        csize = 11
        covering = clustering.cluster_by_index(
            complex_data(), quality=1, cluster_size=csize
        )
        # Ensure not too many subsets. 3000 is 3 * 1000 from default settings from complex_data
        self.assertEqual(covering.nsubsets, math.ceil(3000 / csize))

        # Ensure no subset is too large.
        spare = 0
        for subset in covering.subsets:
            self.assertTrue(len(subset) <= csize)
            spare += csize - len(subset)

        # Ensure again that we don't have too many subsets.
        self.assertTrue(spare < csize)
