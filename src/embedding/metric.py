from __future__ import annotations

import numpy as np
import math
from typing import Optional

# Various metrics that take two same-shape ndarrays interpreted as
#     (nsamples, nvertices, ndim)
# and check the distance between the arrays.


def box_range(A: np.ndarray) -> np.ndarray:
    """
    Find the dimensions of a box that encloses the last dimension of the array.

    range(np.array([[1,2,3], [0,1,2]],
                   [[-1,-1,-1], [-2,-3,-2]]))
    [3,5,5]
    """
    shape = A.shape
    n = np.prod(shape[:-1])
    m = shape[-1]
    pts = A.reshape(n, m)
    lower = pts.min(axis=0)
    upper = pts.max(axis=0)
    return upper - lower


def difference(A: np.ndarray, B: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return the difference matrix, after checking:
    1. if B is None, just return A
    2. if A and B have different shape, raise a ValueError
    """
    if B is None:
        return A
    if A.shape != B.shape:
        raise ValueError(f"Array shapes {A.shape} and {B.shape} differ")
    return A - B


def point_hausdorff(A: np.ndarray, B: Optional[np.ndarray] = None) -> float:
    """
    Return the largest L2 distance between vertices that match index and timestep.

    This is the Hausdorff metric for a point cloud.
    """
    diff = difference(A, B)
    dist2 = np.vecdot(diff, diff)
    return math.sqrt(np.max(dist2))


def Linf(A: np.ndarray, B: Optional[np.ndarray] = None) -> float:
    """
    Return the largest coordinate-wise distance between two arrays.

    This is also known as L_infinity
    """
    diff = difference(A, B)
    return float(np.fabs(diff).max())


def numdifferent(A: np.ndarray, B: Optional[np.ndarray] = None) -> int:
    """
    Return the number of values that differ.
    """
    diff = difference(A, B)
    return np.count_nonzero(diff)


class Report:
    def __init__(
        self,
        predata: np.ndarray,
        postdata: np.ndarray,
        compressed_size: int,
    ):
        # Sizes in bytes.
        self.original_size = predata.itemsize * predata.size
        self.compressed_size = compressed_size
        self.original_numvalues = predata.size

        self.range = box_range(predata)

        # Compute the errors.
        errors = predata - postdata

        # Compute some metrics.
        self.hausdorff = point_hausdorff(errors)
        self.Linf = Linf(errors)

        # Verify: can we get precisely lossless results?
        corrected = postdata + errors
        self.corrected_Linf = Linf(predata, corrected)
        self.numuncorrectable = numdifferent(predata, corrected)

    def print_report(self, metersPerUnit: float):
        mpu = metersPerUnit
        compression_ratio = 1 - self.compressed_size / self.original_size
        print(
            f"{self.original_size} reduced to {self.compressed_size}: {compression_ratio:.2%} reduction"
        )
        range_string = " ".join(f"{x:.2f}" for x in self.range * mpu)
        print(f"Range: {range_string} m")
        print(f"Hausdorff (pointwise): {self.hausdorff * mpu} m")
        print(f"Linf: {self.Linf * mpu} m")
        print(f"Linf after error-correct: {self.corrected_Linf * mpu} m")
        uncorrectable_ratio = self.numuncorrectable / self.original_numvalues
        print(
            f"{self.numuncorrectable} uncorrectable entries; {uncorrectable_ratio:.2%} of entries"
        )
