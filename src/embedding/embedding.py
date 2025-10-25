from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import struct

from typing import Union, TypeVar, SupportsAbs, Generic, Optional

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


def pack_small_uint(i: int) -> bytes:
    """
    Pack an unsigned int that is assumed to be small.
    Store the first byte. If it doesn't fit, keep trying.
    Store the first half-int. If it doesn't fit, keep trying.
    Store the first int. If it doesn't fit, keep trying.
    Store the remaining quad. If it doesn't fit, something is wrong with you.
    """
    if i < 255:
        return struct.pack(">B", i)
    else:
        i -= 255
        if i < 65535:
            return struct.pack(">BH", 255, i)
        else:
            i -= 65535
            if i < 4294967295:
                return struct.pack(">BHI", 255, 65535, i)
            else:
                i -= 4294967295
                return struct.pack(">BHIQ", 255, 65535, 4294967295, i)


def unpack_small_uint(b: bytes, offset: int = 0) -> tuple[int, int]:
    """
    Read a presumed-small uint written via pack_small_uint.
    Return the value we read, followed by the new offset.
    """
    i = struct.unpack_from(">B", b, offset)[0]
    offset += 1
    if i < 255:
        return (i, offset)
    else:
        i += struct.unpack_from(">H", b, offset)[0]
        offset += 2
        if i < 255 + 65535:
            return (i, offset)
        else:
            i += struct.unpack_from(">I", b, offset)[0]
            offset += 4
            if i < 255 + 65535 + 4294967295:
                return (i, offset)
            else:
                i += struct.unpack_from(">Q", b, offset)[0]
                offset += 8
                return (i, offset)


def calcsize_small_uint(i: int) -> int:
    if i < 255:
        return 1
    else:
        return 5


def pack_dtype(t: np.dtype) -> bytes:
    """
    Store a dtype... or rather, store one of the few recognized dtypes.

    At the moment it's the current 3 sizes of float.
    """
    match t:
        case np.float16:
            return struct.pack(">B", 16)
        case np.float32:
            return struct.pack(">B", 32)
        case np.float64:
            return struct.pack(">B", 64)
        case _:
            raise ValueError("can't serialize type {t}")


def unpack_dtype(b: bytes, offset: int = 0) -> tuple[type, int]:
    match struct.unpack_from(">B", b, offset)[0]:
        case 16:
            return (np.float16, offset + 1)
        case 32:
            return (np.float32, offset + 1)
        case 64:
            return (np.float64, offset + 1)
        case _:
            raise ValueError("value {_} doesn't map to a known numpy dtype")


def calcsize_dtype(t: type) -> int:
    return 1


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
    def from_data(data: Domain, k: Union[int, float] = 0) -> PCAEmbedding:
        """
        Perform PCA on the data, which must have shape (n, m).

        If k is a float in (0, 1) then use as many components as needed to
        explain that fraction of variance. If k is an integer, use that many
        components. If k is 0 or is at least min(n, m), use all of the components.

        We mildly assume n > m. It'll work otherwise but won't be efficient.
        """
        if len(data.shape) != 2:
            raise ValueError("Shape must be a matrix")
        (n, m) = data.shape
        if not n or not m:
            raise ValueError("Empty matrix")

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
        if k == 0:
            count = min(data.shape)
        elif isinstance(k, int) and k >= 1:
            count = k
        elif isinstance(k, float) and k > 0 and k < 1:
            # find the number of components needed to explain more than k of the variance
            running_sum = s.cumsum() / s.sum()
            print(f"{running_sum}")
            count = 1 + running_sum.searchsorted(k)
        else:
            raise ValueError(
                f"invalid value {k} ({type(k)}) should be zero, a float in (0,1) or a positive int"
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
            U = np.array(U[:, 0:count])

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

    def invert(self, data: Domain) -> Reduced:
        return data @ self.WVt + self.c

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

        return (PCAEmbedding(n, m, k, c, WVt), offset)

    def read_projection(self, b: bytes, offset: int = 0) -> tuple[Reduced, int]:
        t = self.c.dtype
        U_flat = np.frombuffer(b, offset=offset, dtype=t, count=self.k * self.n)
        U = np.reshape(U_flat, (self.k, self.n))
        offset += t.itemsize * self.k * self.n

        return (U, offset)


_classmap.register(1, PCAEmbedding)
