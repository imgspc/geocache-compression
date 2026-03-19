import unittest
from embedding import clustering
import numpy as np


class ClusteringTestCase(unittest.TestCase):
    def test_issorted(self) -> None:
        unsorted = np.array([1, 5, 2, 56, 2, 25, 2])
        issorted = np.array([1, 6, 8, 9, 10, 345, 6742])
        self.assertFalse(clustering._is_sorted(unsorted))
        self.assertTrue(clustering._is_sorted(issorted))

        self.assertTrue(clustering._is_sorted(np.array([])))

    def test_covering_api(self) -> None:
        covering = clustering.Covering(np.arange(5), np.array([]))
        self.assertEqual(1, covering.nsubsets)
        self.assertTrue(covering.is_id_permutation())
        for subset in covering.subsets:
            self.assertTrue(np.all(np.arange(5) == subset))
        with self.assertRaises(ValueError):
            covering.indices[3] = 7
            covering.verify()
            covering.indices[3] = 3

        # TODO: multiple subsets
        # TODO: the various constructors
        # TODO: serialization
