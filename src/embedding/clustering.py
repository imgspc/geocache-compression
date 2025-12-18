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
        self.verify()

    def verify(self) -> None:
        # Verify that the indices are unique and cover the range.
        s = np.unique(self.indices)
        if len(s) != len(self.indices):
            raise ValueError(f"non-unique indices: {len(s)} out of {len(self.indices)}")
        if s[0] != 0:
            raise ValueError(f"least index is {s[0]}")
        if s[-1] != len(s) - 1:
            raise ValueError(f"largest index is {s[-1]} not {len(s)-1}")

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
        """
        # The first offset is zero, and the offset of the end of the array
        # doesn't need to be stored.
        offsets = np.concatenate([[0], counts.cumsum()[:-1]])

        return Covering(indices, offsets)

    @staticmethod
    def from_labels(labels: np.ndarray) -> Covering:
        """
        Given a vector where labels[i] is the number of the cluster for
        vertex i, return the covering.
        """
        indices = labels.argsort()
        _, counts = np.unique(labels, return_counts=True)
        return Covering.from_indices_and_counts(indices, counts)

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
    if cover.is_id_permutation():
        return byvertex
    else:
        # Reorder. Compute the inverse permutation and apply it.
        inverse_permutation = np.argsort(cover.indices)
        return byvertex[:, inverse_permutation, :]


def _get_vertex_curve(M: np.ndarray, i: int) -> np.ndarray:
    """
    Given an array of shape (nsamples, nverts, ndim), get back a matrix
    of shape (nsamples, ndim) representing the motion of the ith vertex.
    """
    return M[:, i, :]


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
    return Covering.from_labels(kmeans.labels_)


def cluster_pca_kmeans(data: np.ndarray, cluster_size: int = 10000) -> Covering:
    """
    Given data of shape (nsamples, nverts, ndim), return a covering of the
    vertices based on k-means for k clusters over the PCA of each motion curve.

    k is computed as ceil(nverts / cluster_size). TODO: do this more rationally.
    """
    from sklearn.cluster import KMeans  # type: ignore

    (nsamples, nverts, ndim) = data.shape

    # Translate each curve center to the origin
    centroid = data.sum(axis=0) / nsamples
    M = data - centroid

    # for each vertex, compute the PCA basis (the 3x3 orthonormal matrix)
    def get_pca_basis(curve):
        U, s, Vt = np.linalg.svd(curve, full_matrices=False)
        return Vt

    # shape: each row is a 3x3 matrix. Convert to each row is a 9-vector.
    pcas = np.array([get_pca_basis(_get_vertex_curve(M, i)) for i in range(nverts)])
    flattened = pcas.reshape(nverts, ndim * ndim)
    k = int(math.ceil(nverts / cluster_size))
    kmeans = KMeans(n_clusters=k).fit(flattened)

    # Convert to a Covering
    indices = kmeans.labels_.argsort()
    _, counts = np.unique(kmeans.labels_, return_counts=True)
    return Covering.from_indices_and_counts(indices, counts)


def _lineanglemetric(l1: np.ndarray, l2: np.ndarray) -> float:
    """
    Given two lines, each going through the origin and heading to points l1 and
    l2 respectively, compute a metric related to the angle between the lines.
    If the angle is theta, we return:
        (1 - cos(theta)) / 2
    This is 0 for parallel lines and 1 for perpendicular lines.

    Either l1 or l2 or both can be an array of lines, one line per row.

    If both l1 and l2 are single lines, we return the scalar distance.
    If either l1 or l2 is a single line while the other is an array of lines, we return a vector of distances.
    If both are arrays of lines, we return a matrix where entry i,j is the distance from l1[i] to l2[j].
    """
    dot = l1 @ l2.T
    return 1 - dot * dot


def _cluster_near_values(
    data: np.ndarray, epsilon: float, converter, metric
) -> Covering:
    """
    Given data of shape (nsamples, nverts, ndim), convert each vertex to a
    vector via the converter, then find cluster centres so that every vertex's
    vector is within distance epsilon or less to at least one cluster centre,
    according to the metric.

    This is quite slow: O(nverts * nclusters)

    converter: np.ndarray -> T
        # convert a motion curve to some type (usually np.ndarray)
    metric: T * T -> float
        # either side or both may be an array of T (as an np.ndarray)
    """
    (nsamples, nverts, ndim) = data.shape
    centroid = data.sum(axis=0) / nsamples
    M = data - centroid

    # converted is an array with a row per vertex, and values that depend on the converter.
    # the 'metric' had better be a metric over the space spanned by the converter
    converted = np.array([converter(_get_vertex_curve(M, i)) for i in range(nverts)])

    # Create clusters. Start with one centre, and keep adding centres until all
    # vertices have distance less than epsilon to at least one centre.
    centres = [converted[0]]
    best_distances = metric(converted[0], converted)

    while np.max(best_distances) > epsilon:
        index = np.argmax(best_distances)
        centres.append(converted[index])
        newdist = metric(centres[-1], converted)
        best_distances = np.minimum(best_distances, newdist)

    # Assign each vertex to a cluster.
    # all_distances[i,j] is the distance from vertex i to centre j
    # labels[i] is the index of the closest centre to vertex i
    all_distances = metric(converted, np.array(centres))
    labels = np.argmin(all_distances, axis=1)

    return Covering.from_labels(labels)


def cluster_first_axis(data: np.ndarray, cluster_size: int = 10000) -> Covering:
    """
    Given data of shape (nsamples, nverts, ndim), return a covering of the
    vertices based on having similar angles between the principal component axes
    of each motion curve in each cluster.

    TODO: give a way to choose the #degrees.
    """
    degrees = 10
    epsilon = (1 - math.cos(math.radians(degrees))) / 2

    def get_axis(curve: np.ndarray) -> np.ndarray:
        U, s, Vt = np.linalg.svd(curve, full_matrices=False)
        # Compute e1 @ Vt ... which is the first row of Vt:
        return Vt[0]

    return _cluster_near_values(data, epsilon, get_axis, _lineanglemetric)


def cluster_near_quaternions(data: np.ndarray, cluster_size: int = 10000) -> Covering:
    """
    Given data of shape (nsamples, nverts, ndim), return a covering of the
    vertices based on keeping the angle between quaternions and their cluster
    centre less than 2 degrees.

    TODO: give a way to choose the #degrees.
    """
    from scipy.spatial.transform import Rotation

    degrees = 10
    epsilon = (1 - math.cos(math.radians(degrees))) / 2

    def get_quat(curve: np.ndarray) -> np.ndarray:
        U, s, Vt = np.linalg.svd(curve, full_matrices=False)
        # If Vt isn't 3x3 then make it 3x3. The singular values are sorted,
        # so keeping the first three dimensions keeps the best ones.
        (n, m) = Vt.shape
        if (n, m) != (3, 3):
            if (n >= 3) and (m >= 3):
                # Simple case: just drop values.
                new_Vt = Vt[0:3, 0:3]
                Vt = new_Vt
            else:
                # TODO
                raise ValueError(
                    f"unimplemented: converting shape {Vt.shape} to a rotation matrix"
                )

        # Vt may have negative determinant (rotoreflection, rather than rotation).
        # If so, negating the last row will flip the sign and make a pure rotation,
        # with as little impact as possible since that affects the least
        # principal component.
        # TODO: make this make geometric sense.
        if np.linalg.det(Vt) < 0:
            Vt[-1, :] = -Vt[-1, :]
        r = Rotation.from_matrix(Vt)
        return r.as_quat()

    return _cluster_near_values(data, epsilon, get_quat, _lineanglemetric)


def cluster_static_first(
    data: np.ndarray, cluster_size: int = 10000, cluster_fn=cluster_by_index
) -> Covering:
    """
    Given data of shape (nsamples, nverts, ndim), return a covering
    with a first cluster being all the points that don't move at all,
    followed by covering the remaining vertices using some other
    cluster functions (default: by index).
    """
    (nsamples, nverts, ndim) = data.shape

    # Figure out which vertices go into the first cluster, namely those with
    # static values by vertex. (Would be nice actually to get static values *by coordinate*
    # so motion along flat ground doesn't need to encode the elevation at all).
    initial_positions = data[0, :, :]
    coord_values_are_initial = data == initial_positions
    coord_values_are_static = np.all(coord_values_are_initial, axis=0)
    vertex_values_are_static = np.any(coord_values_are_static, axis=1)

    # we want the indices where the vertices are static versus moving
    static_indices = np.nonzero(vertex_values_are_static)[0]

    # Check two extreme cases: all vertices move, or none move.
    if len(static_indices) == 0:
        return cluster_fn(data, cluster_size)
    elif len(static_indices) == nverts:
        return Covering(np.arange(nverts), np.array([0]))

    # cluster just the moving data, which will have indices refer to the view.
    moving_indices = np.nonzero(np.invert(vertex_values_are_static))[0]
    moving_data = data[:, moving_indices, :]
    moving_data_covering = cluster_fn(moving_data, cluster_size)

    # build the indices and offsets of the moving data covering,
    # in the original indexing.
    mdc_indices = moving_indices[moving_data_covering.indices]
    mdc_offsets = len(static_indices) + moving_data_covering.offsets

    # Create a covering object that starts with the static cluster and then
    # has the sub-covering, in the original indexing.
    indices = np.concatenate((static_indices, mdc_indices))
    offsets = np.concatenate(([0], mdc_offsets))
    return Covering(indices, offsets)
