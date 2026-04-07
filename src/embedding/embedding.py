from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import struct
import math

from .util import pack_small_uint, unpack_small_uint, pack_dtype, unpack_dtype

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

        In general the data must be the same as what was used to initialize
        the embedding. Some specific embeddings may relax that requirement
        """
        ...

    @abstractmethod
    def invert(self, data: Reduced) -> Domain:
        """
        Return transformed data from the embedded space to the original space.
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

    def __init__(self, nsamples: int, nverts: int, ndim: int, dtype, quality: float):
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
                struct.pack(self.dtype.char, self.quality),
            )
        )

    @classmethod
    def from_bytes(cls, b: bytes, offset: int = 0) -> tuple[Embedding, int]:
        n, offset = unpack_small_uint(b, offset)
        m, offset = unpack_small_uint(b, offset)
        d, offset = unpack_small_uint(b, offset)
        t, offset = unpack_dtype(b, offset)
        assert issubclass(t, np.number)
        fmt = np.dtype(t).char
        (q,) = struct.unpack_from(fmt, b, offset)
        offset += struct.calcsize(fmt)
        return (RawEmbedding(n, m, d, t, q), offset)

    def write_projection(self, projected: Reduced) -> bytes:
        return projected.tobytes()

    def read_projection(self, b: bytes, offset: int = 0) -> tuple[Reduced, int]:
        ndata = self.nsamples * self.nverts * self.ndim
        flatdata = np.frombuffer(b, offset=offset, dtype=self.dtype, count=ndata)
        tsize = flatdata.itemsize
        offset += tsize * ndata

        data = np.reshape(flatdata, (self.nsamples, self.nverts, self.ndim))
        return (data, offset)


class StaticEmbedding(Embedding):
    """
    Embedding vertices that aren't moving.

    We store the centroid, and that's what we return for all frames.

    If the input is actually not static, then the errors are going to be
    the distance from the actual data to the centroid. If that's a problem,
    use another embedding!
    """

    def __init__(self, c: np.ndarray, nsamples: int):
        """
        Most users will want to use `from_data` or `from_bytes` rather than directly
        using the constructor.

        c is the centroid, shape (1, nverts, ndim)
        nsamples is the number of rows in the original data
        """
        self.c = c
        self.nsamples = nsamples

    @classmethod
    def from_data(
        cls, data: Domain, quality: float, verbose: bool = False
    ) -> Embedding:
        """
        Compute a static embedding from static data.

        We do not check validity; use is_valid if you want that.
        """
        nsamples, nverts, ndim = data.shape
        c = np.sum(data, axis=0) / nsamples
        return cls(c, nsamples)

    @classmethod
    def is_valid(cls, data: Domain, quality: float) -> bool:
        """
        Is the data actually static?
        """
        nsamples, nverts, ndim = data.shape
        c = np.sum(data, axis=0) / nsamples
        M = data - c
        return np.max(np.fabs(M)) <= quality

    def project(self, data: Domain) -> Reduced:
        """
        Project the data to the lower-dimensional space... which is 0-dimensional.
        """
        return np.array([], dtype=self.c.dtype)

    def invert(self, data: Reduced) -> Domain:
        """
        Invert the lower-dimensional space (an empty vector) back to the original space,
        i.e. return the centroid repeated every frame.
        """
        return np.tile(self.c, (self.nsamples, 1, 1))

    def tobytes(self) -> bytes:
        """
        Write the matrix size, type, and centroid.
        """
        nverts, ndim = self.c.shape
        return b"".join(
            (
                pack_small_uint(self.nsamples),
                pack_small_uint(nverts),
                pack_small_uint(ndim),
                pack_dtype(self.c.dtype),
                self.c.tobytes(),
            )
        )

    @classmethod
    def from_bytes(cls, b: bytes, offset: int = 0) -> tuple[Embedding, int]:
        """
        Read the matrix size, type, and centroid.
        """
        n, offset = unpack_small_uint(b, offset)
        m, offset = unpack_small_uint(b, offset)
        d, offset = unpack_small_uint(b, offset)
        t, offset = unpack_dtype(b, offset)
        assert issubclass(t, np.number)
        c = np.frombuffer(b, offset=offset, dtype=t, count=m * d)
        c = np.reshape(c, (m, d))
        tsize = c.itemsize
        offset += tsize * m * d
        return (StaticEmbedding(c, n), offset)

    def write_projection(self, projected: Reduced) -> bytes:
        # There is no projection; write nothing.
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

    def __init__(self, n: int, m: int, k: int, WVt: np.ndarray):
        """
        Most users will want to use `from_data` or `from_bytes` rather than directly
        using the constructor.
        """
        self.n = n
        self.m = m
        self.k = k
        self.WVt = WVt
        self.U: Optional[np.ndarray] = None

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
        return b"".join(
            (
                pack_small_uint(self.n),
                pack_small_uint(self.m),
                pack_small_uint(self.k),
                pack_dtype(self.WVt.dtype),
                self.WVt.tobytes(),
            )
        )

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
        min_d = min(n, m)
        t = M.dtype

        if quality < 0:
            raise ValueError(f"invalid quality {quality}")

        # Perform SVD:
        # Vt is the pseudo-rotation to the SVD basis (transpose of V, historical reasons)
        # W is the scaling matrix -- we only store the diagonal s.
        # U is the data rewritten in the SVD basis (it is, itself, semi-orthogonal).
        # The svd algorithm guarantees s is returned in order largest scale first.
        #
        # We can't assume M is hermitian, so we need the full computation.
        # We definitely need U and V.
        # We don't want the full matrices, they'd be huge.
        U, s, Vt = np.linalg.svd(M, full_matrices=False)

        if quality == 0:
            count = min_d
            WVt = np.diag(s) @ Vt
        else:
            # Round off to U', W', Vt' so that M' = U'W'Vt' and ||M-M'||_inf < quality.
            # U and V are rotations so the rows are unit vectors.
            # W is a diagonal matrix with non-negative values, in order biggest first.
            # M = U W Vt
            # M_ij = sum_k U_ik W_kk Vt_kj
            # M_ij = sum_k U_ik s_k Vt_kj
            #
            # We will choose a vector eps of roundoff values as follows:
            # * every entry in the jth column of U rounds to the nearest multiple of eps_j
            # * every entry in the ith row of WVt rounds to the nearest multiple of eps_i
            #
            # This ensures that every term that involves a value of W has the same error bounds:
            #
            # TODO: verify interval arithmetic here:
            # M'_ij = sum_k [(U_ik +- eps_k) (s_k Vt_kj +- eps_k)]
            #       = sum_k [U_ik s_k Vt_kj +- eps_k (s_k Vt_kj + U_ik) +- eps_k^2]
            #
            # The overall error in a component of the reconstruction, M', is thus:
            # |M_ij - M'_ij| = |sum_k [U_ik s_k Vt_kj - U_ik s_k Vt_kj +- eps_k (s_k Vt_kj + U_ik) +- eps_k^2]|
            #                = |sum_k [                                +- eps_k (s_k Vt_kj + U_ik) +- eps_k^2]|
            # Given that U_i and Vt_k are unit vectors this is bounded by:
            #             <= sum_k |eps_k (s_k + 1)| + eps_k^2
            # To ensure the error is at most quality, then, it suffices to choose eps to satisfy:
            #       quality >= sum_k eps_k (s_k + 1) + eps_k^2
            # Allowing each component of eps to have equal weight (and requiring they are positive):
            #       quality / min(data.shape) == eps_k (s_k + 1) + eps_k^2
            # solve for eps_k:
            #       eps_k = [-(s_k+1) + sqrt((s_k+1)^2 + 4 quality/min(data.shape))] / 2
            # Note that eps_k is strictly positive, so we can divide by it safely.
            #
            # Once the eps vector is chosen, note that rounding s can zero out some entries entirely,
            # achieving dimensionality reduction.
            #
            # TODO: make eps be precisely a power of two so that rounding zeroes out low-order bits.

            # Given the s1^2, we need to do the math here in higher precision.
            higher: DTypeLike
            match t:
                case np.float16:
                    higher = np.float32
                case np.float32:
                    higher = np.float64
                case _:
                    higher = np.longdouble
            s1 = np.array(s, dtype=higher) + higher(1)  # type: ignore
            s1s1 = s1 * s1 + (4 * quality / min_d)
            s1s1_sqrt = np.sqrt(s1s1)
            eps = 0.5 * (s1s1_sqrt - s1)

            # Round s to see if we can achieve dimensionality reduction.
            # s' = eps * round(s / eps) ; but no need to keep s'
            count = np.count_nonzero(np.round(s / eps))

            if count == min_d:
                WVt = np.diag(s) @ Vt
            else:
                # Drop the zero-rounded values by dropping:
                # row/col from W, rows from Vt, columns from U, and values from eps
                WVt = np.diag(s[0:count]) @ Vt[0:count, :]
                U = np.array(U[:, 0:count], copy=True)
                eps = np.array(eps[0:count], copy=True)

            if verbose:
                print(
                    f"{M.shape}, projected to {WVt.shape}: {M.size} reduced to {WVt.size} + {U.size} ({(WVt.size + U.size)/M.size:.2%})"
                )

            # Now, round the matrices.
            # U we divide each row component-wise by eps, round, then multiply back out by eps
            # WVt we do that to the transpose to operate on each column component-wise
            #
            # Make sure to save as the original data size.
            U = np.array(np.round(U / eps) * eps, dtype=t)
            WVt = np.array((np.round(WVt.T / eps) * eps).T, dtype=t)

        embedding = cls(n, m, count, WVt)
        embedding.U = U
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
        assert issubclass(t, np.number)
        tsize = t(0).itemsize

        WVt_flat = np.frombuffer(b, offset=offset, dtype=t, count=k * m)
        WVt = np.reshape(WVt_flat, (k, m))
        offset += tsize * k * m

        return (cls(n, m, k, WVt), offset)

    def read_projection(self, b: bytes, offset: int = 0) -> tuple[Reduced, int]:
        t = self.WVt.dtype
        U_flat = np.frombuffer(b, offset=offset, dtype=t, count=self.k * self.n)
        U = np.reshape(U_flat, (self.n, self.k))
        offset += t.itemsize * self.k * self.n

        return (U, offset)


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

    def __init__(self, pca: FlatCenteredPCA, c: np.ndarray):
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

        if quality < 0:
            raise ValueError(f"invalid quality {quality}")

        # Center the data.
        c = np.sum(data, axis=0) / nsamples
        if quality != 0:
            c = np.round(c / quality) * quality
        M = data - c

        # Flatten M and get its PCA
        M = cls.flatten(M)
        pca = FlatCenteredPCA.from_data(M, quality, verbose)
        return cls(pca, c)

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
                self.c.tobytes(),
                pack_small_uint(self.ndim),
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

        c = np.frombuffer(b, offset=offset, dtype=t, count=m)
        offset += tsize * m

        ndim, offset = unpack_small_uint(b, offset)
        if m % ndim != 0:
            raise ValueError(f"{m} does not divide evenly into {ndim} dimensions")
        c = np.reshape(c, (m // ndim, ndim))

        return (cls(pca, c), offset)

    def write_projection(self, projected: Reduced) -> bytes:
        return projected.tobytes()

    def read_projection(self, b: bytes, offset: int = 0) -> tuple[Reduced, int]:
        return self.pca.read_projection(b, offset)


class PCAConfigurationSpaceEmbedding(AbstractPCAEmbedding):
    """
    Embedding class where we do PCA in configuration space.

    Each row is the positions of all vertices concatenated;
    each column is a coordinate of a vertex.
    """

    def __init__(self, pca: FlatCenteredPCA, c: np.ndarray):
        """
        Most users will want to use `from_data` or `from_bytes` rather than directly
        using the constructor.
        """
        super().__init__(pca, c)

    @classmethod
    def flatten(cls, data: np.ndarray) -> np.ndarray:
        nsamples, nverts, ndim = data.shape
        return np.reshape(data, (nsamples, nverts * ndim))

    def unflatten(self, data: np.ndarray) -> np.ndarray:
        n, m = data.shape
        if m % self.ndim != 0:
            raise ValueError(f"{m} columns not divisible into {self.ndim} dimensions")
        return np.reshape(data, (n, m // self.ndim, self.ndim))


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
        len(embed.tobytes()) + len(embed.project(data).tobytes())
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
