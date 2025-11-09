from __future__ import annotations

import numpy as np
import struct

from typing import Iterator, Sequence, cast


#
# This file has abstractions for creating clusterings, so that we can then
# send data on to an embedding.
#
#
class Covering:
    """
    An covering of a set S is a set of subsets where all the elements in S are
    named. An exact covering names them exactly once.

    E.g. if the set S is [1,2,3,4]
    Exact covering example: [[1,2], [3,4]]
    Not a covering: [[2], [3,4]] because 1 is not in any subset.
    Not an exact covering: [[1,2,4], [3,4]] because 4 appears in two subsets.
    """

    def __init__(self, indices: np.ndarray, offsets: np.ndarray):
        """
        Use from_arrays or from_bytes instead.
        """
        self.indices = indices.astype(np.uint32)
        self.offsets = offsets.astype(np.uint32)

    @property
    def nsubsets(self) -> int:
        return len(self.offsets)

    def is_id_permutation(self) -> bool:
        """
        Return whether the covering, unsliced, is already sorted (True), or whether it
        needs to be reindexed (False).

        Recommendation: if a covering is *not* the identity, then set the first
        index to something other than 0 and we'll find it fastest.
        """
        if self.indices[0] != 0:
            return False
        if self.indices[-1] != len(self.indices) - 1:
            return False

        # is_sorted is somehow not a native operation, have to do this nonsense
        #
        # We are comparing component-wise whether the array except the last
        # element is smaller than the array except the first element, creating
        # a bool array for that, and return true if all are true.
        #
        return bool(np.all(self.indices[:-1] <= self.indices[1:]))

    @property
    def subsets(self) -> Iterator[np.ndarray]:
        n = self.nsubsets
        if n == 0:
            return
        for i in range(n - 1):
            yield self.indices[self.offsets[i] : self.offsets[i + 1]]
        # Return the last cluster if any
        last_offset = self.offsets[-1]
        yield self.indices[last_offset : len(self.indices)]

    @staticmethod
    def from_arrays(subsets: Sequence[Sequence[int]]):
        """
        Given a sequence of sequences of int, store them, and be able to serialize
        and deserialize them.
        """
        indices = np.concatenate(subsets)

        # Get lengths of everything but the last subset.
        lengths = np.array([len(subset) for subset in subsets[:-1]], dtype=int)

        # Get start offsets of each subset. The first start offset is zero,
        # then len(subset[0]), then len(subset[0] + subset[1]), etc.
        offsets = np.concatenate([[0], lengths.cumsum()], dtype=int)

        return Covering(indices, offsets)

    @staticmethod
    def from_bytes(b: bytes, offset: int = 0) -> tuple[Covering, int]:
        """
        Return a covering plus return the new offset after reading the covering
        from the bytes stream.
        """
        # TODO: compress this. It would delta-code to nothingness.

        # Read the length of the covering, then the covering.
        nindices = struct.unpack_from(">I", b, offset)[0]
        offset += 4
        indices = np.frombuffer(b, dtype=np.uint32, count=nindices, offset=offset)
        offset += nindices * indices.itemsize

        # Read the length of the offsets, then the offsets.
        noffsets = struct.unpack_from(">I", b, offset)[0]
        offset += 4
        offsets = np.frombuffer(b, dtype=np.uint32, count=noffsets, offset=offset)
        offset += noffsets * offsets.itemsize

        return (Covering(indices, offsets), offset)

    def tobytes(self) -> bytes:
        """
        Return bytes that frombytes can read and recreate into a Covering object.
        """
        # Output the two arrays, with lengths.
        return b"".join(
            (
                struct.pack(">I", len(self.indices)),
                self.indices.tobytes(),
                struct.pack(">I", len(self.offsets)),
                self.offsets.tobytes(),
            )
        )


def slice(a: np.ndarray, cover: Covering) -> Iterator[np.ndarray]:
    """
    Given an array a with shape (nsamples, nverts, ndim) and a covering of the vertices,
    return matrices with shape (nsamples, len(subset) * ndim)

    Ex 1. if the covering returns 1-element arrays [0], [1], ... then we'll return
    matrices that describe in each row the position of each vertex over time.

    Ex 2. if the covering returns 10-element arrays, then we concatenate the
    positions of those ten vertices to return a 30-column matrix of positions
    over time.
    """
    nsamples, nverts, ndim = a.shape
    for subset in cover.subsets:
        yield a.take(subset, axis=1).reshape(nsamples, len(subset) * ndim)


def unslice(slices: list[np.ndarray], cover: Covering, ndim: int) -> np.ndarray:
    """
    Given matrices with shape (nsamples, len(subset) * ndim), get back
    an array with shape (nsamples, nverts, ndim).
    """
    if len(slices) != cover.nsubsets:
        raise ValueError(f"Unmatched slicings: {len(slices)} versus {cover.nsubsets}")

    # Reshape the list of slices
    flattened = np.concatenate(slices, axis=1)
    (nsamples, ndata) = flattened.shape
    if ndata % ndim != 0:
        raise ValueError(f"Data is not {ndim}-dimensional: shape {flattened.shape}")
    nverts = ndata // ndim
    byvertex = flattened.reshape((nsamples, nverts, ndim))

    # Check if we need to reorder
    if not cover.is_id_permutation():
        raise ValueError("Reordering covers is not yet supported")

    return byvertex


def cluster_by_vertex(n: int) -> Covering:
    """
    Return a covering that just has each vertex one by one.
    """

    def array(i: int) -> Sequence[int]:
        return cast(Sequence[int], np.array([i], dtype=int))

    return Covering.from_arrays([array(i) for i in range(n)])


def cluster_monolithic(n: int) -> Covering:
    """
    Return a monolithic covering: all in one big cluster.
    """

    def array(i: int) -> Sequence[int]:
        return cast(Sequence[int], np.array(range(n), dtype=int))

    return Covering.from_arrays([array(n)])


def cluster_by_index(n: int, k: int) -> Covering:
    """
    Cluster every group of k vertices together, as if index adjacency meant
    they were close in behaviour.
    """
    # The indices are just all the vertices one after the other.
    # The offsets start at 0, and go up in steps of 10.
    indices = np.array(range(n))
    offsets = indices[0:n:k]
    return Covering(indices, offsets)
