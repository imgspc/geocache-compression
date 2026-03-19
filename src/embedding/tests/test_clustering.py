import unittest
from embedding import clustering
import numpy as np


class ClusteringTestCase(unittest.TestCase):
    def test_issorted(self):
        unsorted = np.array([1, 5, 2, 56, 2, 25, 2])
        issorted = np.array([1, 6, 8, 9, 10, 345, 6742])
        self.assertFalse(clustering._is_sorted(unsorted))
        self.assertTrue(clustering._is_sorted(issorted))

        self.assertTrue(clustering._is_sorted(np.array([])))
