from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import struct
import math
from scipy.spatial.transform import Rotation

from .util import (
    pack_small_uint,
    unpack_small_uint,
    pack_dtype,
    unpack_dtype,
    zero_jagged,
)
from .encoding import (
    ApproximatedStream,
    encode_coordinates,
    decode_coordinates,
    encode_sparse_matrix,
    decode_sparse_matrix,
    encode_tiny_ints,
    decode_tiny_ints,
)

from typing import Union, Optional
from numpy.typing import DTypeLike

#
# This file includes the generic Embeddings we've developed, in a class
# hierarchy.
#

# Type aliases to help read the intent of the code.
Domain = np.ndarray
Reduced = np.ndarray


class Embedding(ABC):
    @classmethod
    @abstractmethod
    def from_data(
        cls, data: Domain, quality: float, verbose: bool = False
    ) -> Embedding:
        """
        Compute an embedding for the given data and quality bounds.
        """
        ...

    @abstractmethod
    def project(self, data: Domain) -> Reduced:
        """
        Project the data from the domain to the reduced-dimension space.

        The data must have shape (n, m) for some n and m.

        The output will have shape (n, m'). Expect m' << m.

        Only valid to call this when you used from_data as the constructor.
        """
        ...

    @abstractmethod
    def invert(self, data: Reduced) -> Domain:
        """
        Return transformed data from the embedded space to the original space.

        Must be valid to call this if you used from_bytes as the constructor.
        """
        ...

    @classmethod
    @abstractmethod
    def is_valid(cls, data: np.ndarray, quality: float) -> bool:
        """
        Return whether this embedding can represent the given data for the given quality bound.
        """
        ...

    @abstractmethod
    def tobytes(self) -> bytes:
        """
        Return a representation of the embedding that can be restored via
        from_bytes if we know the type.

        Most callers probably want to call `embeddings.serialize`
        at the module level.
        """
        ...

    @classmethod
    @abstractmethod
    def from_bytes(cls, b: bytes, offset: int = 0) -> tuple[Embedding, int]:
        """
        Static method to create an embedding from stored bytes.

        Most callers probably want to call `embeddings.deserialize`
        `embeddings.deserialize` at the module level.
        """
        ...

    @abstractmethod
    def write_projection(self, projected: Reduced) -> bytes:
        """
        Convert the projected data to some bytes.

        Use read_projection to convert the bytes back to the domain.
        """
        ...

    @abstractmethod
    def read_projection(self, b: bytes, offset: int = 0) -> tuple[Reduced, int]:
        """
        Read some data projected according to this embedding from a bytes object,
        starting at the offset.

        Output the reduced-dimension ndarray and the new offset to start reading from.
        """
        ...


# Map of ID to embedding type; concrete classes are registered at the end of this file.
#
# ID is a uint16 -- 65k embeddings should be enough for everyone.
class _ClassMap:
    _classmap: dict[int, type] = {}
    _idmap: dict[type, int] = {}

    def register(self, ID: int, cls: type):
        if ID < 0 or ID >= 2**16:
            raise ValueError(f"ID {ID} is not a uint16")
        old_cls = self._classmap.get(ID, None)
        if old_cls and old_cls != cls:
            raise KeyError(
                f"ID {ID} already used for {old_cls}, cannot be used for {cls}"
            )
        self._classmap[ID] = cls
        self._idmap[cls] = ID

    def get_classes(self) -> list[type]:
        return list(self._idmap.keys())

    def get_class(self, ID: int) -> type:
        return self._classmap[ID]

    def get_ID(self, cls: type) -> int:
        return self._idmap[cls]

    def pack_class(self, embedding: Union[Embedding, type]) -> bytes:
        if isinstance(embedding, Embedding):
            cls = type(embedding)
        else:
            cls = embedding
        return pack_small_uint(self._idmap[cls])

    def unpack_class(self, b: bytes, offset: int = 0) -> tuple[type, int]:
        """
        Return an embedding class, and the new offset.
        """
        ID, offset = unpack_small_uint(b, offset)
        return (self.get_class(ID), offset)


_classmap = _ClassMap()


def serialize(embedding: Embedding) -> bytes:
    """
    Write an embedding into bytes in a way that we can reconstruct
    via `deserialize`

    Raises KeyError if the embedding class isn't registered.
    """
    return _classmap.pack_class(embedding) + embedding.tobytes()


def deserialize(b: bytes, offset: int = 0) -> tuple[Embedding, int]:
    """
    Convert bytes written by `serialize` back into an Embedding.

    Return the embedding and the new offset after reading it.
    """
    cls, offset = _classmap.unpack_class(b, offset)
    assert issubclass(cls, Embedding)
    return cls.from_bytes(b, offset)


class RawEmbedding(Embedding):
    """
    Ceci n'est pas un embedding.

    We store the data, rounded to the quality bound.
    """

    def __init__(
        self, nsamples: int, nverts: int, ndim: int, dtype, quality: Optional[float]
    ):
        self.nsamples = nsamples
        self.nverts = nverts
        self.ndim = ndim
        self.dtype = dtype
        self.quality = quality

    @classmethod
    def from_data(
        cls, data: Domain, quality: float, verbose: bool = False
    ) -> Embedding:
        n, m, d = data.shape
        return cls(n, m, d, data.dtype, quality)

    @classmethod
    def is_valid(cls, data: np.ndarray, quality: float) -> bool:
        return True

    def project(self, data: Domain) -> Reduced:
        return data

    def invert(self, data: Reduced) -> Domain:
        return data

    def tobytes(self) -> bytes:
        return b"".join(
            (
                pack_small_uint(self.nsamples),
                pack_small_uint(self.nverts),
                pack_small_uint(self.ndim),
                pack_dtype(self.dtype),
            )
        )

    @classmethod
    def from_bytes(cls, b: bytes, offset: int = 0) -> tuple[Embedding, int]:
        n, offset = unpack_small_uint(b, offset)
        m, offset = unpack_small_uint(b, offset)
        d, offset = unpack_small_uint(b, offset)
        t, offset = unpack_dtype(b, offset)
        return (RawEmbedding(n, m, d, t, quality=None), offset)

    def write_projection(self, projected: Reduced) -> bytes:
        if self.quality is None:
            # Lossy compression, so we must not re-export if we read
            # this data from file.
            raise ValueError("unable to re-encode data lossily")
        return encode_coordinates(projected, self.quality)

    def read_projection(self, b: bytes, offset: int = 0) -> tuple[Reduced, int]:
        shape = (self.nsamples, self.nverts, self.ndim)
        return decode_coordinates(b, offset, shape=shape, dtype=self.dtype)


class StaticEmbedding(Embedding):
    """
    Embedding vertices that aren't moving.

    We store the bounding box center, and that's what we return for all frames.

    This embedding minimizes L_inf error, not L_2 error.
    """

    def __init__(self, c: np.ndarray, nsamples: int, quality: Optional[float]):
        """
        Most users will want to use `from_data` or `from_bytes` rather than directly
        using the constructor.

        c is the centre, shape (1, nverts, ndim)
        nsamples is the number of rows in the original data
        quality is the rounding allowed; not stored in serialization
        """
        self.c = c
        self.nsamples = nsamples
        self.quality = quality

    @classmethod
    def _bbox(cls, data: Domain) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the bounding box of the data: the minimum and maximum.
        """
        minimum = np.min(data, axis=0)
        maximum = np.max(data, axis=0)
        return (minimum, maximum)

    @classmethod
    def from_data(
        cls, data: Domain, quality: float, verbose: bool = False
    ) -> Embedding:
        """
        Compute a static embedding from static data.

        We do not check validity; use is_valid if you want that.
        """
        minimum, maximum = cls._bbox(data)
        c = 0.5 * (minimum + maximum)
        nsamples, _, __ = data.shape
        return cls(c, nsamples, quality)

    @classmethod
    def is_valid(cls, data: Domain, quality: float) -> bool:
        """
        Is the data actually static?
        """
        minimum, maximum = cls._bbox(data)
        # max-min is the diameter, quality is the maximum radius
        is_in_quality = (maximum - minimum) <= 2 * quality
        return bool(np.all(is_in_quality))

    def project(self, data: Domain) -> Reduced:
        """
        Project the data to the lower-dimensional space... which is 0-dimensional.
        """
        return np.array([], dtype=self.c.dtype)

    def invert(self, data: Reduced) -> Domain:
        """
        Invert the lower-dimensional space (an empty vector) back to the original space,
        i.e. return the centre repeated every frame.
        """
        return np.tile(self.c, (self.nsamples, 1, 1))

    def tobytes(self) -> bytes:
        """
        Write the matrix size, type, and centre.
        """
        if self.quality is None:
            raise ValueError("refusing to re-encode data previous lossily compressed")
        nverts, ndim = self.c.shape

        # Write the xyz values as (potentially) compressed streams.
        # todo: reorder the vertices in a space-filling order
        return b"".join(
            (
                pack_small_uint(self.nsamples),
                pack_small_uint(nverts),
                pack_small_uint(ndim),
                pack_dtype(self.c.dtype),
                encode_coordinates(self.c, self.quality),
            )
        )

    @classmethod
    def from_bytes(cls, b: bytes, offset: int = 0) -> tuple[Embedding, int]:
        """
        Read the matrix size, type, and centre.
        """
        n, offset = unpack_small_uint(b, offset)
        m, offset = unpack_small_uint(b, offset)
        d, offset = unpack_small_uint(b, offset)
        t, offset = unpack_dtype(b, offset)
        assert issubclass(t, np.number)
        shape = (m, d)
        c, offset = decode_coordinates(b, offset, shape=shape, dtype=t)
        return (StaticEmbedding(c, n, quality=None), offset)

    def write_projection(self, projected: Reduced) -> bytes:
        """
        Return an empty bytes object, since there is no projection.
        """
        return b""

    def read_projection(self, b: bytes, offset: int = 0) -> tuple[Reduced, int]:
        """
        Read the projection to be inverted.

        There is no projection: this reads nothing and returns an empty vector.
        """
        return np.array([]), offset


class FlatCenteredPCA:
    """
    Embedding-like class based on PCA, assuming the data is centered around the
    origin and the input data has shape (n, m)

    This is *not* an Embedding because the caller needs to decide how to
    flatten the data of shape (nsamples, nvertices, ndim) into shape (n, m).
    """

    def __init__(
        self,
        n: int,
        m: int,
        V: np.ndarray,
        V_counts: np.ndarray,
        U: Optional[np.ndarray],
        U_counts: Optional[np.ndarray],
        quality: Optional[np.ndarray],
    ):
        """
        Most users will want to use `from_data` or `from_bytes` rather than directly
        using the constructor.
        """
        self.n = n
        self.m = m
        self.V = V
        self.V_counts = V_counts
        self.U = U
        self.U_counts = U_counts
        self.quality = quality

    @property
    def dtype(self) -> np.dtype:
        return self.V.dtype

    @classmethod
    def is_valid(cls, data: Domain, quality: float) -> bool:
        return True

    def project(self, data: Domain) -> Reduced:
        """
        Project the (n, m)-dimensional data to the reduced-dimension space.
        The output has shape (n, k) where k <= m depends on arguments to from_data.

        The data is assumed to be identical to what we analyzed -- and we don't
        verify that assumption.
        """
        if self.U is None:
            # It's actually possible: just invert WVt (aka transpose it) and
            # multiply. But I don't want to bother testing and debugging.
            raise ValueError("Unable to project from deserialized PCA")
        return self.U

    def invert(self, data: Reduced) -> Domain:
        return data @ self.V.T

    def tobytes(self) -> bytes:
        """
        Convert the embedding to a bytes object for serialization.

        The U array is not serialized, so calling `project` after deserializing
        will not give the identical result due to roundoff.
        """
        # quality can only be none if we have no data to encode
        assert self.quality is not None or np.max(self.V_counts) == 0
        return b"".join(
            (
                pack_small_uint(self.n),
                pack_small_uint(self.m),
                pack_dtype(self.dtype),
                encode_sparse_matrix(self.V, self.V_counts, self.quality),
            )
        )

    @classmethod
    def _epsilon(cls, A: np.ndarray, B: np.ndarray, q: float) -> np.ndarray:
        """
        Given two matrices A and B that we're mulitplying together, find a
        vector of roundoff values that enaure the maximum error is below q. We
        will round off corresponding columns of A and rows of B to the same
        value but each column/row pair gets a different epsilon.
        """
        # eps_k^2 + eps_k(maxA_k + maxB_k) - q/d == 0
        # eps = [sqrt((maxA + maxB)^2 + 4q/d) - (maxA + maxB)] / 2
        # maxA + maxB gets squared so do the calculation in 64-bit, and return to
        # lower precision later.
        maxA = np.max(np.fabs(A), axis=0).astype(np.float64)
        maxB = np.max(np.fabs(B), axis=1).astype(np.float64)
        d = len(maxA)
        maxAB = maxA + maxB

        assert len(maxA) == len(maxB)
        assert A.dtype == B.dtype
        assert d > 0

        eps = (0.5 * (np.sqrt(4 * q / d + np.square(maxAB)) - maxAB)).astype(A.dtype)
        return eps

    @classmethod
    def from_data(
        cls, M: Domain, quality: float, verbose: bool = False
    ) -> FlatCenteredPCA:
        """
        Perform PCA on the data, which must have shape (n, m) and should be
        centered about the origin to get mathematically reasonable results.

        Guarantee that the error in any single value is at most "quality";
        lower quality is better.
        """
        if len(M.shape) != 2:
            raise ValueError(f"Shape {M.shape} should be a matrix")
        n, m = M.shape
        if not n or not m:
            raise ValueError(f"Empty matrix with shape {M.shape}")
        t = M.dtype

        if quality < 0:
            raise ValueError(f"invalid quality {quality}")

        # Perform SVD:
        # Vt is the pseudo-rotation to the SVD basis (transpose of V, historical reasons)
        #       In configuration space, this is a set of configurations that get blended.
        #       In R^d, this is a rotoreflection to a new orthonormal basis.
        # W is the scaling matrix -- we only store the diagonal s.
        # U is the data rewritten in the SVD basis (it is, itself, semi-orthogonal).
        # WU can be interpreted as:
        #       In configuration space, the blend weights.
        #       In R^d, the coordinates in the new basis.
        # The svd algorithm guarantees s is returned in order largest scale first.
        #
        # We can't assume M is hermitian, so we need the full computation.
        # We definitely need U and V.
        # We don't want the full matrices, they'd be huge.
        U, s, Vt = np.linalg.svd(M, full_matrices=False)

        # Shove the s vector into the smaller of U or Vt.
        if np.prod(Vt.shape) < np.prod(U.shape):
            Vt = np.diag(s) @ Vt
        else:
            U = U * s

        # We will store U and V as sparse matrices, rounding to zero any
        # values that we don't need in order to keep the quality bound.
        # Keep U and Vt compatible in shape for now.
        C = compute_colrows_needed(M, U, Vt, quality)
        U_counts = np.max(C, axis=1)
        V_counts = np.max(C, axis=0)
        U = zero_jagged(U, U_counts)
        V = zero_jagged(Vt.T, V_counts)
        Vt = V.T
        assert U.shape[0] == n
        assert V.shape[0] == m
        assert U.shape[1] == V.shape[1]

        # Compute how much we can still round off.
        existing_roundoff = np.max(np.fabs(M - U @ Vt))
        if np.max(V_counts) == 0:
            # If we rounded U and V down to nothing, we don't have
            # a quality bound to evaluate.
            epsilon = None
        else:
            if existing_roundoff >= quality:
                existing_roundoff = 0  # TODO: we shouldn't be here, just give up
            epsilon = cls._epsilon(U, Vt, quality - existing_roundoff)

        if verbose:
            fullsize = np.prod(M.shape)
            sparsesize = np.sum(U_counts) + np.sum(V_counts)
            if sparsesize < fullsize:
                print(
                    f"PCA reduced values from {fullsize} to {sparsesize} ({np.sum(U_counts)} + {np.sum(V_counts)}) with error {existing_roundoff}"
                )
            else:
                print(
                    f"PCA unable to reduce value count: from {fullsize} to {sparsesize}"
                )

        embedding = cls(n, m, V, V_counts, U, U_counts, epsilon)
        return embedding

    @classmethod
    def from_bytes(cls, b: bytes, offset: int = 0) -> tuple[FlatCenteredPCA, int]:
        """
        Read from the bytes starting at offset, return an embedding
        and the new offset for the next read.
        """
        n, offset = unpack_small_uint(b, offset)
        m, offset = unpack_small_uint(b, offset)
        t, offset = unpack_dtype(b, offset)
        V, V_counts, offset = decode_sparse_matrix(b, offset, m, dtype=t)

        return (cls(n, m, V, V_counts, U=None, U_counts=None, quality=None), offset)

    def write_projection(self, U: np.ndarray) -> bytes:
        if self.U is None or self.U_counts is None:
            raise ValueError("can't re-encode")
        return encode_sparse_matrix(self.U, self.U_counts, self.quality)

    def read_projection(self, b: bytes, offset: int = 0) -> tuple[Reduced, int]:
        U, U_counts, offset = decode_sparse_matrix(b, offset, self.n, self.dtype)
        return U, offset


def center_data(
    data: np.ndarray, quality: float, verbose: bool = False
) -> tuple[np.ndarray, bytes]:
    """
    Find the centroids of the data.

    We compute the centroids, round them off, and return the rounded centroids
    along with the centroids encoded as bytes.
    """
    nsamples, nverts, ndim = data.shape

    # Center the data.
    c = np.sum(data, axis=0) / nsamples
    cbytes = encode_coordinates(c, quality)
    shape = (nverts, ndim)
    cprime, _ = decode_coordinates(
        cbytes,
        offset=0,
        shape=shape,
        dtype=data.dtype,
        verbose=verbose,
    )
    return (cprime, cbytes)


def compute_colrows_needed(
    M: np.ndarray, A: np.ndarray, B: np.ndarray, epsilon: float
) -> np.ndarray:
    """
    Given M = A@B, return how many columns of A and rows of B we need to keep
    to arrive at ||M' - M||_inf < epsilon.

    M is given to avoid counting roundoff already present in A @ B.

    Returns an integer matrix C of the same dimensions as M.

    C_ij is the number of columns of A / rows of B we need for entry M_ij to
    have low error.

    You can round A_ik to zero if C_ij < k for all j:
        np.max(C, axis=1)
    You can round B_kj to zero if C_ij < k for all i:
        np.max(C, axis=0)
    """
    nrows, maxk = A.shape
    _, ncols = B.shape
    if _ != maxk:
        raise ValueError("shapes {A.shape} and {B.shape} not compatible")

    nk = maxk + 1

    # Compute the error of truncating to (nrows, k) @ (k, ncols) for all k
    # errors[i,j,k] is the error in M_ij after rounding to k values in the dot
    # product.
    #
    # Leave errors[i,j,nk] = 0 so it'll always come up as OK, simplifying the code.
    errors = np.zeros((nrows, ncols, nk + 1))
    for k in range(nk):
        Ak = A[:, :k]
        Bk = B[:k, :]
        Mk = Ak @ Bk
        errors[:, :, k] = np.fabs(Mk - M)

    # Compute which error values are ok and which are too high.
    # ok[i,j,k] is true if keeping k values is good enough for M_ij.
    ok = errors <= epsilon

    # C[i,j] is the minimum number of values to keep that is good enough
    # for M_ij.
    C = np.argmax(ok, axis=2)

    # if C[i,j] exceeds nk then we have too much error in that component no
    # matter what.  TODO: we should do something about that other than silently
    # ignoring it.
    C = np.where(C > nk, nk, C)

    return C


class AbstractPCAEmbedding(Embedding):
    """
    Embedding class where we do PCA. Subclasses decide whether to do it in
    configuration space or in euclidean space.
    """

    @classmethod
    @abstractmethod
    def flatten(cls, data: np.ndarray) -> np.ndarray:
        """
        Given data of shape (nsamples, nverts, ndim) flatten it to (n, m).
        """
        ...

    @abstractmethod
    def unflatten(self, data: np.ndarray) -> np.ndarray:
        """
        Given data of shape (n, m) spread it out to shape (nsamples, nverts, ndim).
        """
        ...

    def __init__(self, pca: FlatCenteredPCA, c: np.ndarray, cbytes: bytes):
        """
        Most users will want to use `from_data` or `from_bytes` rather than directly
        using the constructor.
        """
        self.pca = pca
        self.c = c
        if len(c.shape) != 2:
            raise ValueError(
                f"centroid shape should be (nverts, ndim) rather than {c.shape}"
            )
        self.cbytes = cbytes

    @classmethod
    def from_data(
        cls, data: Domain, quality: float, verbose: bool = False
    ) -> Embedding:
        """
        Perform PCA on the data, which must have shape (nsamples, nverts, ndim).

        Guarantee that the error in any single value is at most "quality" units off.
        Lower "quality" is better. We can't guarantee we can actually achieve
        the given quality bound if it's too small.
        """
        if len(data.shape) != 3:
            raise ValueError(
                f"Shape {data.shape} should have shape (nsamples, nverts, ndim)"
            )
        nsamples, nverts, ndim = data.shape
        if not nsamples or not nverts or not ndim:
            raise ValueError(f"Empty matrix with shape {data.shape}")

        if quality <= 0:
            raise ValueError(f"invalid quality {quality}")

        # Center the data. We compute the centroid in full precision, then
        # we round-trip it through encoding and use that for calculations to
        # avoid letting roundoff accumulate.
        c, cbytes = center_data(data, quality, verbose)
        M = data - c

        # Flatten M and get its PCA
        M = cls.flatten(M)
        pca = FlatCenteredPCA.from_data(M, quality, verbose)
        return cls(pca, c, cbytes)

    @classmethod
    def is_valid(cls, data: Domain, quality: float) -> bool:
        return True

    @property
    def ndim(self) -> int:
        return self.c.shape[1]

    def project(self, data: Domain) -> Reduced:
        """
        Data is ignored, we just return the same PCA projection we started with.
        """
        if self.pca.U is None:
            # theoretically could be done but not needed right now
            raise NotImplementedError("Unable to project from deserialized PCA")
        return self.pca.U

    def invert(self, projected: Reduced) -> Domain:
        inverted_flat = self.pca.invert(projected)
        inverted = self.unflatten(inverted_flat)
        return inverted + self.c

    def tobytes(self) -> bytes:
        """
        Convert the embedding to a bytes object for serialization.
        """
        assert self.c.dtype == self.pca.dtype
        return b"".join(
            (
                self.pca.tobytes(),
                pack_small_uint(self.ndim),
                self.cbytes,
            )
        )

    @classmethod
    def from_bytes(cls, b: bytes, offset: int = 0) -> tuple[Embedding, int]:
        """
        Read from the bytes starting at offset, return an embedding
        and the new offset for the next read.
        """
        pca, offset = FlatCenteredPCA.from_bytes(b, offset)
        m = pca.m
        dtype = pca.dtype

        ndim, offset = unpack_small_uint(b, offset)
        if m % ndim != 0:
            raise ValueError(f"{m} does not divide evenly into {ndim} dimensions")
        nverts = m // ndim
        cstart = offset
        shape = (nverts, ndim)
        c, cend = decode_coordinates(b, offset, shape=shape, dtype=dtype, verbose=False)
        offset = cend

        return (cls(pca, c, b[cstart:cend]), offset)

    def write_projection(self, projected: Reduced) -> bytes:
        return self.pca.write_projection(projected)

    def read_projection(self, b: bytes, offset: int = 0) -> tuple[Reduced, int]:
        return self.pca.read_projection(b, offset)


class PCAConfigurationSpaceEmbedding(AbstractPCAEmbedding):
    """
    Embedding class where we do PCA in configuration space.

    Each row is the positions of all vertices concatenated;
    each column is a coordinate of a vertex.
    """

    def __init__(self, pca: FlatCenteredPCA, c: np.ndarray, cbytes: bytes):
        """
        Most users will want to use `from_data` or `from_bytes` rather than directly
        using the constructor.
        """
        super().__init__(pca, c, cbytes)

    @classmethod
    def flatten(cls, data: np.ndarray) -> np.ndarray:
        nsamples, nverts, ndim = data.shape
        # Vt will be serialized row by row.
        # Make sure each row is the X coords, then the Y coords, etc to
        # minimize larger diffs.
        data_by_coord = data.transpose(0, 2, 1)
        return data_by_coord.reshape((nsamples, nverts * ndim))

    def unflatten(self, data: np.ndarray) -> np.ndarray:
        nsamples, m = data.shape
        if m % self.ndim != 0:
            raise ValueError(f"{m} columns not divisible into {self.ndim} dimensions")
        nverts = m // self.ndim
        data_by_coord = data.reshape((nsamples, self.ndim, nverts))
        return data_by_coord.transpose(0, 2, 1)


class PCA3dRotationEmbedding(Embedding):
    """
    Embedding class where we do PCA in geometry space, in 3d.

    PCA produces a 3x3 rotation matrix. We convert it to an axis-angle
    representation so we store only 3-vectors each rather than 3x3 matrices.

    Not valid in other dimensions. Analogies exist in other dimensions but
    we don't have data to care about those.
    """

    def __init__(
        self,
        c: np.ndarray,
        rotation: Rotation,
        cbytes: Optional[bytes] = None,
        rbytes: Optional[bytes] = None,
        quality: Optional[float] = None,
        data: Optional[np.ndarray] = None,
    ):
        self.c = c
        self.rotation = rotation
        self.cbytes = cbytes
        self.rbytes = rbytes
        self.quality = quality
        self.data = data

    @classmethod
    def is_valid(cls, data: Domain, quality: float) -> bool:
        # SVD will give us a 3x3 rotation matrix per vertex if we have exactly
        # 3d data and at least 3 samples.
        nsamples, nverts, ndim = data.shape
        return ndim == 3 and nsamples >= 3

    @property
    def nverts(self) -> int:
        return self.c.shape[0]

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self):
        return self.c.dtype

    @classmethod
    def from_data(
        cls, data: Domain, quality: float, verbose: bool = False
    ) -> Embedding:
        """ """
        assert cls.is_valid(data, quality)
        nsamples, nverts, ndim = data.shape

        dtype = data.dtype
        c, cbytes = center_data(data, quality, verbose)
        M = data - c

        # transpose so that svd is getting a bunch of nsamples x ndim matrices,
        # and returns a U, s, and Vt matrix for each vertex.
        byvertex = M.transpose(1, 0, 2)
        U, s, Vt = np.linalg.svd(byvertex, full_matrices=False)

        # Vt is an array of 3x3 matrices, each a rotoflection. We can trivially
        # convert them into rotations by negating the last row.
        #
        # We'd need to adjust U to match, but we'll throw it away and recompute
        # it later anyway.
        reflections = np.linalg.det(Vt) < 0
        Vs_to_flip = Vt[reflections, :, :]
        negated_V_rows = -Vs_to_flip[:, -1, :]
        Vt[reflections, -1, :] = negated_V_rows

        # Store it as a set of rotations.
        rotations = Rotation.from_matrix(Vt)

        # Round-trip the rotations so we encode with what will be reconstructed.
        # We need about 20x the quality for the rotation to be sufficiently accurate
        # to avoid having too much error. Proof TBD but basically the error is about
        # 14 eps from the rotation matrix, and about 6 eps to convert from
        # axis-angle to matrix, plus we are giving quality/2 to UW,
        # so set eps = quality/40.
        rbytes = cls.tobytes_rotation(rotations, quality / 40, dtype)
        rotations, _ = cls.frombytes_rotation(rbytes, 0, nverts, dtype)

        return cls(
            c,
            rotations,
            cbytes=cbytes,
            rbytes=rbytes,
            quality=quality,
            data=data,
        )

    def project(self, data: Domain) -> Reduced:
        # From SVD:
        #     M = UWVt
        # We want to return UW. Simple:
        #     UW = MVt^-1.
        # In other words, apply the inverse rotation to the centered data.
        # Transpose to (nverts, nsamples, ndim) from (nsamples, nverts, ndim)
        nsamples, nverts, ndim = data.shape
        if ndim != 3:
            raise ValueError
        if nverts != self.nverts:
            raise ValueError

        bysample = data - self.c
        byvertex = bysample.transpose(1, 0, 2)

        # Rotation converts to matrix form internally (as of scipy 1.17.1), so
        # let's just go get it ourselves.
        # Shape is nverts, ndim, ndim
        rotation = self.rotation.as_matrix().transpose(0, 2, 1)

        # Help numpy broadcast: make the shapes be
        # byvertex - (nverts, nsamples, 1, ndim)
        # invrotation - (nverts, 1, ndim, ndim)
        # That way, we match up vertices, we apply the same rotation to each sample,
        # and applying the rotation means multiplying a row by a rotation.
        M = np.reshape(byvertex, (nverts, nsamples, 1, ndim))
        V = np.reshape(rotation, (nverts, 1, ndim, ndim))

        # multiply them to shape (nverts, nsamples, 1, ndim)
        UW = M @ V

        # drop the extra rank added for broadcasting rules
        # also the roetation got done in float64, bring it back down if needed.
        UW = np.reshape(UW, (nverts, nsamples, ndim)).astype(self.dtype)

        return UW

    def invert(self, data: Reduced) -> Domain:
        nverts, nsamples, ndim = data.shape
        if ndim != 3 or nverts != self.nverts:
            raise ValueError(
                f"expected ({self.nverts}, {nsamples}, {self.ndim}) got {data.shape}"
            )

        # From SVD:
        #     M = UWVt
        # data is UW; Vt is the rotations. We need to add dimensions to guide broadcasting,
        # and remove them later.
        UW = np.reshape(data, (nverts, nsamples, 1, ndim))
        Vt = self.rotation.as_matrix()
        Vt = np.reshape(Vt, (nverts, 1, ndim, ndim))
        M = UW @ Vt
        M = np.reshape(M, (nverts, nsamples, ndim))

        # Transpose to (nsamples, nverts, ndim) from (nverts, nsamples, ndim)
        centered = M.transpose(1, 0, 2)

        # Shift back to the original space.
        return centered + self.c

    @classmethod
    def frombytes_rotation(
        cls, b: bytes, offset: int, nverts: int, dtype: np.dtype
    ) -> tuple[Rotation, int]:
        rotvecs, offset = decode_coordinates(b, offset, (nverts, 3), dtype=dtype)
        return Rotation.from_rotvec(rotvecs), offset

    @classmethod
    def tobytes_rotation(cls, rotations, epsilon, dtype: np.dtype) -> bytes:
        rotvecs = rotations.as_rotvec().astype(dtype)
        return encode_coordinates(rotvecs, epsilon)

    @classmethod
    def from_bytes(cls, b: bytes, offset: int = 0) -> tuple[Embedding, int]:
        nverts, offset = unpack_small_uint(b, offset)
        ndim = 3
        t, offset = unpack_dtype(b, offset)
        dtype = np.dtype(t)
        c, offset = decode_coordinates(b, offset, (nverts, ndim), dtype=dtype)
        rotations, offset = cls.frombytes_rotation(b, offset, nverts, dtype)

        return (cls(c, rotations), offset)

    def tobytes(self) -> bytes:
        if self.cbytes is None or self.rbytes is None or self.quality is None:
            raise ValueError("trying to re-encode decoded data")

        return b"".join(
            [
                pack_small_uint(self.nverts),
                pack_dtype(self.dtype),
                self.cbytes,
                self.rbytes,
            ]
        )

    def read_projection(self, b: bytes, offset: int = 0) -> tuple[Reduced, int]:
        # Read the counts, which have values [0,3] so fit in 2 bits.
        nsamples, offset = unpack_small_uint(b, offset)
        counts, offset = decode_tiny_ints(b, offset, (self.nverts,), 2)

        # Read U, which was stored as sparse columns.
        U = np.zeros((self.nverts, nsamples, self.ndim))

        def dense(j: int, offset: int) -> int:
            mask = counts > j
            nonzeros = np.sum(mask.astype(int))
            if nonzeros == 0:
                return offset
            sparse, offset = decode_coordinates(
                b, offset, (nonzeros, nsamples), dtype=self.dtype
            )
            U[mask, :, j] = sparse
            return offset

        for j in range(self.ndim):
            offset = dense(j, offset)
        return U, offset

    def write_projection(self, projected: Reduced) -> bytes:
        if self.quality is None or self.data is None:
            raise ValueError
        if projected.dtype != self.dtype:
            raise ValueError
        nverts, nsamples, ndim = projected.shape
        if nverts != self.nverts or ndim != self.ndim:
            raise ValueError

        # we allow ourselves to eat up half the quality bound
        epsilon = self.quality / 2

        t = self.dtype

        # errors[i,j] is the error of keeping i values for vertex j's
        # representation in the rotated space. We can store 0, 1, 2, or 3.
        # Hack: denote keeping 4 values as having zero error.
        errors = np.zeros((5, self.nverts))
        for n in range(ndim + 1):
            U = np.zeros(projected.shape)
            U[:, :, :n] = projected[:, :, :n]
            inverted = self.invert(U)
            err_n = np.fabs(inverted - self.data)
            # err_n has shape (nsamples, nverts, ndim)
            # we want to have the max over all samples over all dimensions for
            # each vertex. So transpose to have (nverts, nsamples, ndim),
            # flatten the last two, and then max.
            error_per_vertex = err_n.transpose((1, 0, 2))
            error_per_vertex = error_per_vertex.reshape((nverts, nsamples * ndim))
            errors[n, :] = np.max(error_per_vertex, axis=1)

        # ok[i,j] means keeping i values for vertex j is sufficient
        ok = errors <= epsilon

        # counts[j] is minimum count of values to keep for vertex j.
        # if we exceed the error bound no matter what, the argmax will be 4;
        # silently just replace that with a 3 instead.
        counts = np.argmax(ok, axis=0)
        counts = np.where(counts == 4, 3, counts)

        # values are [0..3] so 2 bits is enough.
        encoded_counts = encode_tiny_ints(counts, 2)

        # Store columns of U sparsely; counts[i] > j tells us whether the
        # value of U[j,i] is nonzero. I.e. count 0 means dimensions 0..2 are zero,
        # count 2 means dimensions 0 and 1 have value but dimension 2 is zero.
        def sparse(j):
            mask = counts > j
            sparse_col = U[mask, :, j]
            coded = encode_coordinates(sparse_col, epsilon)
            return coded

        sparse_cols = [sparse(j) for j in range(self.ndim)]

        return b"".join([pack_small_uint(nsamples), encoded_counts, *sparse_cols])


def best_embedding(
    data: Domain,
    quality: float,
    verbose: bool = False,
    candidates: Optional[list[type[Embedding]]] = None,
) -> Embedding:
    nsamples, nverts, ndim = data.shape

    if candidates is None:
        candidates = _classmap.get_classes()

    embeddings = [
        candidate.from_data(data, quality, verbose)
        for candidate in candidates
        if candidate.is_valid(data, quality)
    ]
    lengths = [
        len(embed.tobytes()) + len(embed.write_projection(embed.project(data)))
        for embed in embeddings
    ]
    ordered = sorted(zip(embeddings, lengths), key=lambda el: el[1])
    best_embed, best_length = ordered[0]
    if verbose:
        best_cls = best_embed.__class__.__name__
        print(f"  stored ({nsamples} x {nverts}) as {best_length} bytes for {best_cls}")
        for other_embed, other_length in ordered[1:]:
            other_cls = other_embed.__class__.__name__
            print(f"    better than {other_length} bytes for {other_cls}")
    return best_embed


_classmap.register(1, RawEmbedding)
_classmap.register(2, StaticEmbedding)
_classmap.register(3, PCAConfigurationSpaceEmbedding)
_classmap.register(4, PCA3dRotationEmbedding)
