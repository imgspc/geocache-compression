from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import struct

from typing import Union, Optional, Iterator, Iterable, Sequence, cast

from .util import pack_small_uint, unpack_small_uint, pack_dtype, unpack_dtype


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
        self.indices = indices
        self.offsets = offsets

    @property
    def nsubsets(self) -> int:
        return len(self.offsets)

    @property
    def subsets(self) -> Iterator[np.ndarray]:
        n = self.nsubsets
        if n == 0:
            return
        for i in range(n - 1):
            yield self.indices[self.offsets[i] : self.offsets[i + 1]]
        # Return the last cluster if any
        last_offset = self.offsets[-1]
        if last_offset < len(self.indices) - 1:
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
        offset += nindices * np.uint32.itemsize

        # Read the length of the offsets, then the offsets.
        noffsets = struct.unpack_from(">I", b, offset)[0]
        offset += 4
        offsets = np.frombuffer(b, dtype=np.uint32, count=noffsets, offset=offset)
        offset += noffsets * np.uint32.itemsize

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
