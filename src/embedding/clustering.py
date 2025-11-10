from __future__ import annotations

import numpy as np
import struct
import math

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
    def from_indices_and_counts(indices: np.ndarray, counts: np.ndarray) -> Covering:
        """
        Given an array of indices and an array of counts per cluster,
        return the covering.

        TODO: check that it's actually a covering.
        """
        # The first offset is zero, and the offset of the end of the array
        # doesn't need to be stored.
        offsets = np.concatenate([[0], counts.cumsum()[:-1]])

        return Covering(indices, offsets)

    @staticmethod
    def from_arrays(subsets: Sequence[Sequence[int]]) -> Covering:
        """
        Given a sequence of sequences of int, store them, and be able to serialize
        and deserialize them.
        """
        indices = np.concatenate(subsets)
        lengths = np.array([len(subset) for subset in subsets], dtype=int)
        return Covering.from_indices_and_counts(indices, lengths)

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
        byvertex = byvertex[:, cover.indices, :]

    return byvertex


def cluster_by_index(data: np.ndarray, cluster_size: int = 10000) -> Covering:
    """
    Given data of shape (nsamples, nverts, ndim), return a covering of the
    vertices based just on indices.

    Cluster every group of k vertices together, as if index adjacency meant
    they were close in behaviour.
    """
    # The indices are just all the vertices one after the other.
    # The offsets start at 0, and go up in steps of 10.
    (nsamples, nverts, ndim) = data.shape
    indices = np.arange(nverts)
    offsets = indices[0:nverts:cluster_size]
    return Covering(indices, offsets)


def cluster_kmeans(data: np.ndarray, cluster_size: int = 10000) -> Covering:
    """
    Given data of shape (nsamples, nverts, ndim), return a covering of the
    vertices based on k-means for k clusters.

    k is computed as ceil(nverts / cluster_size). TODO: do this more rationally.

    Clustering is based on the distance between the motion curves relative to
    their centroids.
    """
    from sklearn.cluster import KMeans  # type: ignore

    (nsamples, nverts, ndim) = data.shape

    # Translate each curve center to the origin
    centroid = data.sum(axis=0) / nsamples
    M = data - centroid

    # Reorder to have row per vertex, each row is xyzxyzxyz... motion curves.
    byvertex = M.transpose((1, 0, 2))
    flattened = byvertex.reshape(nverts, nsamples * ndim)

    # Run k-means.
    k = int(math.ceil(nverts / cluster_size))
    kmeans = KMeans(n_clusters=k).fit(flattened)

    # Convert to a Covering
    indices = kmeans.labels_.argsort()
    _, counts = np.unique(kmeans.labels_, return_counts=True)
    return Covering.from_indices_and_counts(indices, counts)
