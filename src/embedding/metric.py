from __future__ import annotations

import numpy as np
import math

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


# TODO: compute Hausdorff taking topology into account.
def point_hausdorff(A: np.ndarray, B: np.ndarray) -> float:
    """
    Return the largest L2 distance between vertices that match index and timestep.

    This is the Hausdorff metric for a point cloud.
    """
    if A.shape != B.shape:
        raise ValueError(f"Array shapes {A.shape} and {B.shape} differ")
    diff = A - B
    dist2 = np.vecdot(diff, diff)
    return math.sqrt(np.max(dist2))


def Linf(A: np.ndarray, B: np.ndarray) -> float:
    """
    Return the largest coordinate-wise distance between two arrays.

    This is also known as L_infinity
    """
    if A.shape != B.shape:
        raise ValueError(f"Array shapes {A.shape} and {B.shape} differ")
    diff = A - B
    return float(np.fabs(diff).max())
