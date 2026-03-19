from __future__ import annotations

import numpy as np
import struct


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


def dtype_to_int(t: np.dtype) -> int:
    match t:
        case np.float16:
            return 2
        case np.float32:
            return 4
        case np.float64:
            return 8
        case _:
            raise ValueError("can't serialize type {t}")


def int_to_dtype(i: int) -> type:
    match i:
        case 2:
            return np.float16
        case 4:
            return np.float32
        case 8:
            return np.float64
        case _:
            raise ValueError("can't deserialize type with id {i}")


def pack_dtype(t: np.dtype) -> bytes:
    """
    Store a dtype... or rather, store one of the few recognized dtypes.

    At the moment it's the current 3 sizes of float.
    """
    return struct.pack("B", dtype_to_int(t))


def unpack_dtype(b: bytes, offset: int = 0) -> tuple[type, int]:
    i = struct.unpack_from("B", b, offset)[0]
    return (int_to_dtype(i), offset + 1)
