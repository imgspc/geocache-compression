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
    def read_projection(self, b: bytes, offset: int = 0) -> tuple[Reduced, int]:
        """
        Read some data projected according to this embedding from a bytes object,
        starting at the offset.

        Output the reduced-dimension ndarray and the new offset to start reading from.

        Note: to *write* the projected data, simply use data.tobytes()
        where data came from the `project` function.
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
        (ID, offset) = unpack_small_uint(b, offset)
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
    (cls, offset) = _classmap.unpack_class(b, offset)
    assert issubclass(cls, Embedding)
    return cls.from_bytes(b, offset)


class RawEmbedding(Embedding):
    """
    Ceci n'est pas un embedding.

    This is just the identity; we store the data, and do nothing with it.
    """

    __slots__ = ["nsamples", "nverts", "ndim"]

    def __init__(self, n: int, m: int, dtype):
        self.n = n
        self.m = m
        self.dtype = dtype

    def project(self, data: Domain) -> Reduced:
        return data

    def invert(self, data: Reduced) -> Domain:
        return data

    def tobytes(self) -> bytes:
        return b"".join(
            (
                pack_small_uint(self.n),
                pack_small_uint(self.m),
                pack_dtype(self.dtype),
            )
        )

    @classmethod
    def from_bytes(cls, b: bytes, offset: int = 0) -> tuple[Embedding, int]:
        (n, offset) = unpack_small_uint(b, offset)
        (m, offset) = unpack_small_uint(b, offset)
        (t, offset) = unpack_dtype(b, offset)
        assert issubclass(t, np.number)
        return (RawEmbedding(n, m, t), offset)

    def read_projection(self, b: bytes, offset: int = 0) -> tuple[Reduced, int]:
        ndata = self.n * self.m
        flatdata = np.frombuffer(b, offset=offset, dtype=self.dtype, count=ndata)
        tsize = flatdata.itemsize
        offset += tsize * ndata

        data = np.reshape(flatdata, (self.n, self.m))
        return (data, offset)


class StaticEmbedding(Embedding):
    """
    Embedding vertices that aren't moving.

    We store the centroid, and that's what we return for all frames.

    If the input is actually not static, then the errors are going to be
    the distance from the actual data to the centroid. If that's a problem,
    use another embedding!
    """

    def __init__(self, c: np.ndarray, n: int):
        """
        Most users will want to use `from_data` or `from_bytes` rather than directly
        using the constructor.

        c is the centroid, dimensions 1 x m
        n is the number of rows in the original data
        """
        self.c = c
        self.n = n

    @staticmethod
    def is_valid(data: Domain, quality: float) -> bool:
        """
        Is the data actually static?
        """
        (n, m) = data.shape
        c = np.sum(data, axis=0) / n
        M = data - c
        return np.max(np.fabs(M)) <= quality

    @staticmethod
    def from_data(
        data: Domain, quality: float, verbose: bool = False
    ) -> StaticEmbedding:
        """
        Compute a static embedding from static data.

        We do not check validity; use is_valid if you want that.
        """
        (n, m) = data.shape
        c = np.sum(data, axis=0) / n
        return StaticEmbedding(c, n)

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
        return np.tile(self.c, (self.n, 1))

    def tobytes(self) -> bytes:
        """
        Write the matrix size, type, and centroid.
        """
        return b"".join(
            (
                pack_small_uint(self.n),
                pack_small_uint(len(self.c)),
                pack_dtype(self.c.dtype),
                self.c.tobytes(),
            )
        )

    @classmethod
    def from_bytes(cls, b: bytes, offset: int = 0) -> tuple[Embedding, int]:
        """
        Read the matrix size, type, and centroid.
        """
        (n, offset) = unpack_small_uint(b, offset)
        (m, offset) = unpack_small_uint(b, offset)
        (t, offset) = unpack_dtype(b, offset)
        assert issubclass(t, np.number)
        c = np.frombuffer(b, offset=offset, dtype=t, count=m)
        tsize = c.itemsize
        offset += tsize * m
        return (StaticEmbedding(c, n), offset)

    def read_projection(self, b: bytes, offset: int = 0) -> tuple[Reduced, int]:
        """
        Read the projection to be inverted.

        There is no projection: this reads nothing and returns an empty vector.
        """
        return np.array([], dtype=self.c.dtype), offset


class CenteredPCAEmbedding(Embedding):
    """
    Embedding based on PCA, assuming the data is centered around the origin.

    Use PCAEmbedding if the data is not centered.
    """

    __slots__ = ("n", "m", "k", "WVt", "U", "WVtInv")

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

    @staticmethod
    def from_data(
        M: Domain, quality: float, verbose: bool = False
    ) -> CenteredPCAEmbedding:
        """
        Perform PCA on the data, which must have shape (n, m) and should be
        centered about the origin to get mathematically reasonable results.

        Guarantee that the error in any single value is at most "quality";
        lower quality is better.
        """
        if len(M.shape) != 2:
            raise ValueError(f"Shape {M.shape} should be a matrix")
        (n, m) = M.shape
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

            if verbose or not verbose:
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

        embedding = CenteredPCAEmbedding(n, m, count, WVt)
        embedding.U = U
        return embedding

    @classmethod
    def from_bytes(cls, b: bytes, offset: int = 0) -> tuple[Embedding, int]:
        """
        Read from the bytes starting at offset, return an embedding
        and the new offset for the next read.
        """
        (n, offset) = unpack_small_uint(b, offset)
        (m, offset) = unpack_small_uint(b, offset)
        (k, offset) = unpack_small_uint(b, offset)
        (t, offset) = unpack_dtype(b, offset)
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


class PCAEmbedding(Embedding):
    """
    Embedding based on PCA.

    The data need not be centered.
    """

    __slots__ = ("pca", "c")

    def __init__(self, pca: CenteredPCAEmbedding, c: np.ndarray):
        """
        Most users will want to use `from_data` or `from_bytes` rather than directly
        using the constructor.
        """
        self.pca = pca
        self.c = c

    def project(self, data: Domain) -> Reduced:
        """
        Data is ignored, we just return the same PCA projection we started with.
        """
        if self.pca.U is None:
            # Centre the data, then get the CenteredPCA to project.
            # Or... don't care.
            raise ValueError("Unable to project from deserialized PCA")
        return self.pca.U

    def invert(self, projected: Reduced) -> Domain:
        return self.pca.invert(projected) + self.c

    def tobytes(self) -> bytes:
        """
        Convert the embedding to a bytes object for serialization.
        """
        assert self.c.dtype == self.pca.WVt.dtype
        return b"".join(
            (
                self.pca.tobytes(),
                self.c.tobytes(),
            )
        )

    @classmethod
    def from_bytes(cls, b: bytes, offset: int = 0) -> tuple[Embedding, int]:
        """
        Read from the bytes starting at offset, return an embedding
        and the new offset for the next read.
        """
        (pca, offset) = CenteredPCAEmbedding.from_bytes(b, offset)
        assert isinstance(pca, CenteredPCAEmbedding)
        m = pca.m
        t = pca.WVt.dtype
        tsize = t.itemsize

        c = np.frombuffer(b, offset=offset, dtype=t, count=m)
        offset += tsize * m

        return (cls(pca, c), offset)

    def read_projection(self, b: bytes, offset: int = 0) -> tuple[Reduced, int]:
        return self.pca.read_projection(b, offset)

    @staticmethod
    def from_data(data: Domain, quality: float, verbose: bool = False) -> PCAEmbedding:
        """
        Perform PCA on the data, which must have shape (n, m).

        Guarantee that the error in any single value is at most "quality" units off.
        Lower "quality" is better. We can't guarantee we can actually achieve
        the given quality bound if it's too small.
        """
        if len(data.shape) != 2:
            raise ValueError(f"Shape {data.shape} should be a matrix")
        (n, m) = data.shape
        if not n or not m:
            raise ValueError(f"Empty matrix with shape {data.shape}")

        if quality < 0:
            raise ValueError(f"invalid quality {quality}")

        # Find the centroid and round it to centre the data about the origin.
        # TODO: we should likely round to a power of 2 just below alpha, so
        # that we're truncating bits.
        c = np.sum(data, axis=0) / n
        c = np.round(c / quality) * quality
        M = data - c

        # Get the PCA of the centered data and return the full embedding with the centre.
        pca = CenteredPCAEmbedding.from_data(M, quality, verbose)
        return PCAEmbedding(pca, c)


def best_embedding(data: Domain, quality: float, verbose: bool = False) -> Embedding:
    n, m = data.shape

    if StaticEmbedding.is_valid(data, quality):
        static = StaticEmbedding.from_data(data, quality, verbose)
        if verbose:
            print(f"  stored ({n} x {m}) as static {len(static.tobytes())} bytes")
        return static

    embedded = PCAEmbedding.from_data(data, quality=quality, verbose=verbose)
    raw = RawEmbedding(n, m, data.dtype)
    pcabytes = len(embedded.tobytes()) + len(embedded.project(data).tobytes())
    rawbytes = len(raw.tobytes()) + len(raw.project(data).tobytes())
    if pcabytes >= rawbytes:
        if verbose:
            print(f"  stored raw as {rawbytes} rather than {pcabytes} bytes")
        return raw
    else:
        return embedded


_classmap.register(1, RawEmbedding)
_classmap.register(2, StaticEmbedding)
_classmap.register(3, PCAEmbedding)
