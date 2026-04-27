from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import struct
import math

from .util import pack_small_uint, unpack_small_uint, pack_dtype, unpack_dtype
from .encoding import ApproximatedStream, encode_coordinates, decode_coordinates

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
        k: int,
        WVt: np.ndarray,
        U: Optional[np.ndarray],
        quality: Optional[np.ndarray],
    ):
        """
        Most users will want to use `from_data` or `from_bytes` rather than directly
        using the constructor.
        """
        self.n = n
        self.m = m
        self.k = k
        self.WVt = WVt
        self.U = U
        self.quality = quality

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
        return data @ self.WVt

    def tobytes(self) -> bytes:
        """
        Convert the embedding to a bytes object for serialization.

        The U array is not serialized, so calling `project` after deserializing
        will not give the identical result due to roundoff.
        """
        assert self.quality is not None
        return b"".join(
            (
                pack_small_uint(self.n),
                pack_small_uint(self.m),
                pack_small_uint(self.k),
                pack_dtype(self.WVt.dtype),
                encode_coordinates(self.WVt.T, self.quality),
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

        # We want to drop as many dimensions as possible since each one dropped removes
        # n+m+1 values: n from U, m from Vt, and 1 from s.
        # Count how much error we sustain if we drop 'count' values.
        def error(count: int) -> float:
            if count == len(s):
                # Drop everything? Mprime is the zero matrix, so M-Mprime = M.
                return np.max(np.fabs(M))

            if count == 0:
                Uprime = U
                sprime = s
                Vtprime = Vt
            else:
                Uprime = U[:, 0:-count]
                sprime = s[0:-count]
                Vtprime = Vt[0:-count, :]
            # * to multiply by a diagonal matrix as a vector; @ for matmul
            Mprime = Uprime * sprime @ Vtprime
            return np.max(np.fabs(M - Mprime))

        # Error increases monotonically with the more dimensions we drop, so
        # binary search indices 0..len(s) to find the most we can drop while
        # keeping error < quality.
        # Binary search, adapted from bisect.bisect_right; lo ends up at the
        # smallest number of dimensions to drop that produces too much error.
        lo = 0
        hi = len(s)
        while lo < hi:
            mid = (lo + hi) // 2
            if quality < error(mid):
                hi = mid
            else:
                lo = mid + 1
        count = lo - 1
        if count < 0:
            if verbose:
                print(
                    f"PCA creates too much roundoff error even without dimensionality reduction"
                )
            count = 0
            # TODO: return None

        # Make copies to release the truncated bits.
        if count > 0:
            U = np.array(U[:, 0:-count], copy=True)
            s = np.array(s[0:-count], copy=True)
            Vt = np.array(Vt[0:-count, :], copy=True)

        # We'll store the two matrices; shove s into the smaller one, so
        # diff-coding will do a better job on the bigger matrix.
        if np.prod(Vt.shape) < np.prod(U.shape):
            Vt = np.diag(s) @ Vt
        else:
            U = U * s

        # Compute how much we can round off.
        existing_roundoff = error(count)
        if existing_roundoff >= quality:
            existing_roundoff = 0  # TODO: we shouldn't be here, just give up
        epsilon = cls._epsilon(U, Vt, quality - existing_roundoff)

        if verbose:
            if count > 0:
                print(
                    f"PCA reduced {count} dimensions from {M.shape} to {U.shape} and {Vt.shape} with error {existing_roundoff}"
                )
            else:
                print(f"PCA unable to reduce dimensions")

        embedding = cls(n, m, len(s), Vt, U, epsilon)
        return embedding

    @classmethod
    def from_bytes(cls, b: bytes, offset: int = 0) -> tuple[FlatCenteredPCA, int]:
        """
        Read from the bytes starting at offset, return an embedding
        and the new offset for the next read.
        """
        n, offset = unpack_small_uint(b, offset)
        m, offset = unpack_small_uint(b, offset)
        k, offset = unpack_small_uint(b, offset)
        t, offset = unpack_dtype(b, offset)
        # we stored WVt transposed, so the shape is (m,k) not (k,m)
        WV, offset = decode_coordinates(b, offset, dtype=t, shape=(m, k))
        WVt = WV.T

        return (cls(n, m, k, WVt, U=None, quality=None), offset)

    def write_projection(self, U: np.ndarray) -> bytes:
        if self.quality is None:
            raise ValueError("can't re-encode")
        return encode_coordinates(U, self.quality)

    def read_projection(self, b: bytes, offset: int = 0) -> tuple[Reduced, int]:
        return decode_coordinates(b, offset, (self.n, self.k), self.WVt.dtype)


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
        c = np.sum(data, axis=0) / nsamples
        cbytes = encode_coordinates(c, quality)
        shape = (nverts, ndim)
        c, _ = decode_coordinates(
            cbytes,
            offset=0,
            shape=shape,
            dtype=data.dtype,
            verbose=verbose,
        )
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
        assert self.c.dtype == self.pca.WVt.dtype
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
        t = pca.WVt.dtype
        tsize = t.itemsize

        ndim, offset = unpack_small_uint(b, offset)
        if m % ndim != 0:
            raise ValueError(f"{m} does not divide evenly into {ndim} dimensions")
        nverts = m // ndim
        cstart = offset
        shape = (nverts, ndim)
        c, cend = decode_coordinates(b, offset, shape=shape, dtype=t, verbose=False)
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


def best_embedding(
    data: Domain,
    quality: float,
    verbose: bool = False,
    candidates=[StaticEmbedding, PCAConfigurationSpaceEmbedding, RawEmbedding],
) -> Embedding:
    nsamples, nverts, ndim = data.shape

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
