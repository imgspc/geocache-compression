from __future__ import annotations

import bisect
import numpy as np
import struct
import concurrent.futures

from typing import Optional


# For testing, use the LinearExecutor to make operation be single-threaded
# and in sequential order. For production, switch to ThreadPoolExecutor.
class LinearExecutor(concurrent.futures.Executor):
    def __init__(self):
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def submit(self, f, *args, **kwargs) -> concurrent.futures.Future:  # type: ignore
        return self.pool.submit(f, *args, **kwargs)

    def map(self, fn, *iterables, timeout=None, chunksize=1, buffersize=None):
        for args in zip(*iterables):
            yield self.submit(fn, *args).result()


def make_thread_pool():
    # return concurrent.futures.ThreadPoolExecutor()
    return LinearExecutor()


class SmallIntPackFmt:
    """
    We use part of the first byte (least-significant byte) to encode the
    number of additional bytes. If the least bits are
        00 = 1 byte, total 6 bits, [0,64)
        01 = 2 bytes, total 14 bits, [64, 16448)
        10 = 3 bytes, total 22 bits, [16448, 4210752)
        0011 => 4 bytes, total 28 bits [4210752, 272646208)
        0111 => 5 bytes, total 36 bits [272646208, 68992122944)
        1011 => 6 bytes, total 44 bits [68992122944, 17661178167360)
        01111 => 7 bytes, total 51 bits [17661178167360, 2269460991852608)
        011111 => 8 bytes, total 58 bits [2269460991852608, 290499837143564352)
        111111 => ignore header byte, read 8-byte unsigned after, total of 9 bytes

    This depends on writing little-endian so the least significant byte goes
    first.
    """

    def __init__(self, fmt, shift, mask, nbytes, base):
        self.fmt = fmt
        self.shift = shift
        self.mask = mask
        self.nbytes = nbytes
        self.base = base

    def matches(self, byte) -> bool:
        mask = (1 << self.shift) - 1
        return (byte & mask) == self.mask

    def maxvalue(self) -> int:
        nbits = self.nbytes * 8 - self.shift
        return self.base + (1 << nbits)


_pack_fmts = (
    SmallIntPackFmt("<B", 2, 0b00, 1, 0),
    SmallIntPackFmt("<H", 2, 0b01, 2, 64),
    SmallIntPackFmt("<I", 2, 0b10, 3, 16448),
    SmallIntPackFmt("<Q", 4, 0b0011, 4, 4210752),
    SmallIntPackFmt("<Q", 4, 0b0111, 5, 272646208),
    SmallIntPackFmt("<Q", 4, 0b1011, 6, 68992122944),
    SmallIntPackFmt("<Q", 5, 0b01111, 7, 68992122944),
    SmallIntPackFmt("<Q", 6, 0b011111, 8, 2269460991852608),
    SmallIntPackFmt("<Q", 6, 0b111111, 9, 290499837143564352),
)
# The first entry *must* be correct. The others, we fix up here.
for i in range(1, len(_pack_fmts)):
    _pack_fmts[i].base = _pack_fmts[i - 1].maxvalue()


def pack_small_uint(i: int) -> bytes:
    """
    Pack an unsigned int that is assumed to be small.
    """
    index = bisect.bisect_right(_pack_fmts, i, key=lambda f: f.maxvalue())
    foo = _pack_fmts[index]
    if foo.nbytes == 9:
        return bytes([foo.mask]) + struct.pack(foo.fmt, i)
    else:
        b = struct.pack(foo.fmt, ((i - foo.base) << foo.shift) | foo.mask)
        assert not np.any(b[foo.nbytes :])
        return b[: foo.nbytes]


def unpack_small_uint(b: bytes, offset: int = 0) -> tuple[int, int]:
    """
    Read a presumed-small uint written via pack_small_uint.
    Return the value we read, followed by the new offset.
    """
    header = b[offset]
    foo = next((pf for pf in _pack_fmts if pf.matches(header)))
    if foo.nbytes == 9:
        (i,) = struct.unpack_from(foo.fmt, b, offset + 1)
    else:
        # We truncate high-order zeroes so we can't just read the format
        # from the buffer. We need to read out the payload and extend it
        # as needed. There must be a better way than this hackery:
        payload = np.zeros(struct.calcsize(foo.fmt), dtype=np.uint8)
        values = np.frombuffer(b, offset=offset, count=foo.nbytes, dtype=np.uint8)
        payload[: foo.nbytes] = values
        (mangled,) = struct.unpack_from(foo.fmt, payload)
        i = foo.base + (mangled >> foo.shift)
    return i, offset + foo.nbytes


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


def intrange_to_width(a: np.ndarray) -> tuple[int, bool]:
    """
    Return 1, 2 or 3 depending on whether the range of integer values fit
    in int8, int16, or int32 (or their unsigned counterparts).

    Return true if we need the sign, false if unsigned.
    """
    lo = np.min(a)
    hi = np.max(a)
    signed = lo < 0
    if signed:
        if lo >= -128 and hi < 127:
            return 1, signed
        if lo >= -32768 and hi < 32767:
            return 2, signed
        else:
            return 3, signed
    else:
        if hi < 255:
            return 1, signed
        if hi < 65535:
            return 2, signed
        else:
            return 3, signed


def intwidth_to_type(length: int, sign: bool) -> Optional[type]:
    if length == 0:
        return None
    if length < 0 or length > 3:
        raise ValueError(f"Invalid int width {length}")
    if sign:
        t: tuple[type, type, type] = (np.int8, np.int16, np.int32)
    else:
        t = (np.uint8, np.uint16, np.uint32)
    return t[length - 1]


def float_to_hex(value: float) -> str:
    # TODO: handle float64 vs float32 (we're doing 32 here)
    fbits = struct.pack("f", value)
    (unsigned,) = struct.unpack("I", fbits)
    return "0x" + hex(unsigned)
