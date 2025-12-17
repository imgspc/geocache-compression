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


class PCAEmbedding(Embedding):
    """
    Embedding based on PCA.

    We model the data as all fitting on a lower-dimensional subspace (or a
    full-dimensional one, but with different axes).

    E.g. Given data as a 3d position per timestep, setting k = 2 means we're modeling
    the point as moving along a plane.

    Getting to higher dimensions, it gets harder to visualize what's going on.
    """

    __slots__ = ("n", "m", "k", "c", "WVt", "U", "WVtInv")

    def __init__(self, n: int, m: int, k: int, c: np.ndarray, WVt: np.ndarray):
        """
        Most users will want to use `from_data` or `from_bytes` rather than directly
        using the constructor.
        """
        self.n = n
        self.m = m
        self.k = k
        self.c = c
        self.WVt = WVt
        self.U: Optional[np.ndarray] = None
        self.WVtInv: Optional[np.ndarray] = None

    @staticmethod
    def from_data(data: Domain, quality: float, verbose: bool = False) -> PCAEmbedding:
        """
        Perform PCA on the data, which must have shape (n, m).

        Quality should be in [0,1]
        """
        if len(data.shape) != 2:
            raise ValueError(f"Shape {data.shape} should be a matrix")
        (n, m) = data.shape
        if not n or not m:
            raise ValueError(f"Empty matrix with shape {data.shape}")

        # Translate the data to the origin.
        c = np.sum(data, axis=0) / n
        M = data - c

        # Perform SVD:
        # Vt is the orthonormal basis,
        # W is the diagonal weight matrix, stored as a vector of singular values
        # U is the data rewritten in the SVD basis.
        #
        # We can't assume M is hermitian, so we need the full computation.
        # We definitely need U and V.
        # We don't want a square U (it would be huge).
        U, s, Vt = np.linalg.svd(M, full_matrices=False)

        # Dimensionality reduction: Keep the most significant values only.
        if quality == 1:
            count = min(data.shape)
        elif quality >= 0 and quality < 1:
            # find the number of components needed to explain more than k of the variance
            # note: don't use running_sum = s.cumsum() / s.sum() because roundoff can make
            # the last cumulative sum be significantly different than the sum.
            running_sum = s.cumsum()
            running_sum = running_sum[:-1] / running_sum[-1]
            count = 1 + running_sum.searchsorted(quality)
            if verbose:
                print(f"chose {count} dimensions among {running_sum}")
        else:
            raise ValueError(
                f"invalid value {quality} ({type(quality)}) should be zero or a float in (0,1)"
            )

        # No dimensionality reduction? Store the whole thing
        if count >= min(n, m):
            WVt = np.diag(s) @ Vt
        else:
            # Drop row/col from W (by dropping values from s), drop rows from Vt, drop
            # columns from U.
            #
            # Keep copies to allow the full-dimension memory to be released.
            # WVt is already a copy due to the matmul.
            # U we need to copy explictly.
            WVt = np.diag(s[0:count]) @ Vt[0:count, :]
            U = np.array(U[:, 0:count], copy=True)

        embedding = PCAEmbedding(n, m, count, c, WVt)
        embedding.U = U
        return embedding

    def project(self, data: Domain) -> Reduced:
        """
        Project the (n, m)-dimensional data to the reduced-dimension space.
        The output has shape (n, k) where k <= m depends on arguments to from_data.

        The data is assumed to be identical to what we analyzed -- and we don't
        verify that assumption.
        """
        if self.U is not None:
            # Normally we already projected the data and got U:
            return self.U
        if self.WVtInv is None:
            # When reconstructing from bytes we don't know U so compute it:
            #    data = U W Vt
            # and we want U. Post-multiply both sides with the inverse of WVt,
            # we get:
            #   data WVt+ = U WVt WVt+ = U
            # Store WVtInv once, and then we can run project repeatedly.
            self.WVtInv = np.linalg.pinv(self.WVt)
        return data @ self.WVtInv

    def invert(self, data: Reduced) -> Domain:
        return data @ self.WVt + self.c

    def tobytes(self) -> bytes:
        """
        Convert the embedding to a bytes object for serialization.

        The U array is not serialized, so calling `project` after deserializing
        will not give the identical result due to roundoff.
        """
        assert self.c.dtype == self.WVt.dtype
        return b"".join(
            (
                pack_small_uint(self.n),
                pack_small_uint(self.m),
                pack_small_uint(self.k),
                pack_dtype(self.c.dtype),
                self.c.tobytes(),
                self.WVt.tobytes(),
            )
        )

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

        c = np.frombuffer(b, offset=offset, dtype=t, count=m)
        offset += tsize * m

        WVt_flat = np.frombuffer(b, offset=offset, dtype=t, count=k * m)
        WVt = np.reshape(WVt_flat, (k, m))
        offset += tsize * k * m

        return (cls(n, m, k, c, WVt), offset)

    def read_projection(self, b: bytes, offset: int = 0) -> tuple[Reduced, int]:
        t = self.c.dtype
        U_flat = np.frombuffer(b, offset=offset, dtype=t, count=self.k * self.n)
        U = np.reshape(U_flat, (self.n, self.k))
        offset += t.itemsize * self.k * self.n

        return (U, offset)


class RoundedPCAEmbedding(PCAEmbedding):
    @staticmethod
    def from_data(data: Domain, quality: float, verbose: bool = False) -> PCAEmbedding:
        """
        Perform PCA on the data, which must have shape (n, m).

        Guarantee that the error in any single value is at most "quality";
        lower quality is better.
        """
        if len(data.shape) != 2:
            raise ValueError(f"Shape {data.shape} should be a matrix")
        (n, m) = data.shape
        if not n or not m:
            raise ValueError(f"Empty matrix with shape {data.shape}")
        min_d = min(n, m)

        alpha = quality
        if alpha < 0:
            raise ValueError(f"invalid quality {quality}")

        # Translate the data to the origin. Round to the nearest alpha.
        c = np.sum(data, axis=0) / n
        c = np.round(c / alpha) * alpha
        M = data - c

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

        if alpha == 0:
            count = min_d
            WVt = np.diag(s) @ Vt
        else:
            # Round off to U', W', Vt' so that M' = U'W'Vt' and ||M-M'||_inf < alpha.
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
            # To ensure the error is at most alpha, then, it suffices to choose eps to satisfy:
            #       alpha >= sum_k eps_k (s_k + 1) + eps_k^2
            # Allowing each component of eps to have equal weight (and requiring they are positive):
            #       alpha / min(data.shape) == eps_k (s_k + 1) + eps_k^2
            # solve for eps_k:
            #       eps_k = [-(s_k+1) + sqrt((s_k+1)^2 + 4 alpha/min(data.shape))] / 2
            # Note that eps_k is strictly positive, so we can divide by it safely.
            #
            # Once the eps vector is chosen, note that rounding s can zero out some entries entirely,
            # achieving dimensionality reduction.
            #

            # Given the s1^2, we need to do the math here in higher precision.
            higher: DTypeLike
            match data.dtype:
                case np.float16:
                    higher = np.float32
                case np.float32:
                    higher = np.float64
                case _:
                    higher = np.longdouble
            s1 = np.array(s, dtype=higher) + higher(1)  # type: ignore
            s1s1 = s1 * s1 + (4 * alpha / min_d)
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
                    f"{n} samples, {m} coordinates, projected to {count} dims: {M.size} reduced to {WVt.size} + {U.size} ({(WVt.size + U.size)/M.size:.2%})"
                )

            # Now, round the matrices.
            # U we divide each row component-wise by eps, round, then multiply back out by eps
            # WVt we do that to the transpose to operate on each column component-wise
            #
            # Make sure to save as the original data size.
            U = np.array(np.round(U / eps) * eps, dtype=data.dtype)
            WVt = np.array((np.round(WVt.T / eps) * eps).T, dtype=data.dtype)

        embedding = PCAEmbedding(n, m, count, c, WVt)
        embedding.U = U
        return embedding


_classmap.register(1, PCAEmbedding)
