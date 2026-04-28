from __future__ import annotations

import math
import numpy as np
import struct
import subprocess
import tempfile
import concurrent.futures

from abc import abstractmethod
from typing import Optional, Union
from numpy.typing import DTypeLike
from pathlib import Path
from .util import (
    pack_small_uint,
    unpack_small_uint,
    pack_dtype,
    unpack_dtype,
    float_to_hex,
)

neats_dir = Path.home() / "projects/geocache-compression/install/bin"
neats_compress = neats_dir / "neats_lossy_compress"
neats_decompress = neats_dir / "neats_lossy_decompress"

# Set up a thread pool for encoding.
_executor = concurrent.futures.ThreadPoolExecutor()


def temp_filename():
    """
    Return a context manager object with a name field.
    The file it denotes will be auto-deleted at end of context.
    """
    return tempfile.NamedTemporaryFile(delete_on_close=False)


class Codec:
    """
    Basic capability to read/write a vector of floats in a specific encoding.
    """

    @classmethod
    @abstractmethod
    def tobytes(
        cls, data: np.ndarray, quality: float, verbose: bool
    ) -> Optional[bytes]: ...

    @classmethod
    @abstractmethod
    def from_bytes(
        cls, payload: bytes, dtype: Union[type, DTypeLike], verbose: bool
    ) -> np.ndarray: ...


class RawCodec(Codec):
    @classmethod
    def tobytes(
        cls, data: np.ndarray, quality: float, verbose: bool
    ) -> Optional[bytes]:
        return data.tobytes()

    @classmethod
    def from_bytes(
        cls, payload: bytes, dtype: Union[type, DTypeLike], verbose: bool
    ) -> np.ndarray:
        return np.frombuffer(payload, dtype=dtype)


class NeatsCodec(Codec):
    @classmethod
    def tobytes(
        cls, data: np.ndarray, quality: float, verbose: bool
    ) -> Optional[bytes]:
        if data.dtype != np.float32:
            # TODO: handle float64 and float16
            return None
        if len(data) == 0:
            return None
        with temp_filename() as binfile:
            with temp_filename() as neatsfile:
                data.tofile(binfile)

                binfile.close()
                neatsfile.close()

                args = [
                    neats_compress,
                    binfile.name,
                    "-o",
                    neatsfile.name,
                    "-q",
                    float_to_hex(quality),
                ]
                if verbose:
                    args.append("-v")
                subprocess.run(args)

                with open(neatsfile.name, "rb") as f:
                    return f.read()

    @classmethod
    def from_bytes(
        cls, payload: bytes, dtype: Union[type, DTypeLike], verbose: bool
    ) -> np.ndarray:
        with temp_filename() as neatsfile:
            with temp_filename() as binfile:
                neatsfile.write(payload)
                neatsfile.close()
                binfile.close()

                args = [neats_decompress, neatsfile.name, "-o", binfile.name]
                if verbose:
                    subprocess.run(args)
                else:
                    subprocess.run(args, stdout=subprocess.DEVNULL)
                return np.fromfile(binfile.name, dtype=dtype)


class IntifyCodec(Codec):
    @classmethod
    def tobytes(
        cls, data: np.ndarray, quality: float, verbose: bool
    ) -> Optional[bytes]:
        """
        We store the data as fixed-point integer multiples of the quality bound,
        difference-coded, using the smallest int that will fit all the
        differences.
        """
        if len(data) == 0:
            return None
        # boost to 64 bits for intermediate calculations; we squeeze it later
        # so it doesn't cost us anything.
        rounded = np.round(data.astype(np.float64) / quality).astype(np.int64)
        diffcoded = np.ediff1d(rounded, to_begin=rounded[0])

        # min_scalar_type is tricky to use with signed values; rewrite the
        # differences with a bias so it's all unsigned values.
        base = np.min(diffcoded)
        positive = diffcoded - base
        largest = np.max(positive)
        t = np.min_scalar_type(largest)

        diff_bytes = positive.astype(t).tobytes()
        tchar = t.char[0].encode()
        return struct.pack("<qfc", base, quality, tchar) + diff_bytes

    @classmethod
    def from_bytes(
        cls, payload: bytes, dtype: Union[type, DTypeLike], verbose: bool
    ) -> np.ndarray:
        base, quality, tchar = struct.unpack_from("<qfc", payload)
        offset = struct.calcsize("<qfc")
        diff_values = np.frombuffer(payload, offset=offset, dtype=np.dtype(tchar))
        diffcoded = np.array(diff_values, dtype=np.int64) + base
        rounded = np.cumsum(diffcoded)
        return (rounded * np.float64(quality)).astype(dtype)


_codecs = [RawCodec, IntifyCodec]


class ApproximatedStream:
    """
    Represents a stream of data that will be encoded lossily to precision epsilon.

    Chooses the best codec.
    """

    def __init__(
        self,
        stream: np.ndarray,
        epsilon: Optional[float],
    ):
        """
        Create an ApproximatedStream for the data, with the given error bound epsilon.

        If epsilon is None then the stream can't be converted to bytes.
        """
        if len(stream.shape) > 1:
            raise ValueError(f"Shape must be flat, not {stream.shape}")
        self.stream = stream
        self.epsilon = epsilon

    def tobytes_dataonly(self, count: Optional[int], verbose=False) -> bytes:
        """
        Return the bytes just of the data stream.

        count is len(self.stream) but should be provided only if the
        decoder will be able to know it without decoding this stream (i.e. if
        the data is known from other data). Otherwise, provide None.
        """
        if self.epsilon is None:
            raise ValueError("unable to re-compress a stream")

        encodings = [
            codec.tobytes(self.stream, self.epsilon, verbose) for codec in _codecs
        ]
        valid = [
            (index, encoding)
            for (index, encoding) in enumerate(encodings)
            if encoding is not None
        ]
        # The raw encoding is always valid.
        assert len(valid) >= 1

        def key(ce: tuple[int, bytes]) -> int:
            return len(ce[1])

        best_index, data = min(valid, key=key)

        # Encoding:
        # codec index (1 byte)
        # payload length in bytes, written as a small_uint -- unless this is known
        #       (namely: count is known and the codec is raw)
        # encoded stream

        header = struct.pack("B", best_index)
        if count is None or _codecs[best_index] is not RawCodec:
            header += pack_small_uint(len(data))
        if verbose:
            print(
                f"wrote {len(self.stream)} values in {len(header)} + {len(data)} = {len(header) + len(data)} bytes"
            )
        return header + data

    class Header:
        def __init__(
            self, offset: int, codec_index: int, headerlen: int, payloadlen: int
        ):
            self.offset = offset
            self.codec_index = codec_index
            self.headerlen = headerlen
            self.payloadlen = payloadlen

        def next_offset(self) -> int:
            return self.offset + self.headerlen + self.payloadlen

    @staticmethod
    def read_dataonly_header(
        dtype: Union[DTypeLike, type],
        count: Optional[int],
        b: bytes,
        offset: int,
        verbose: bool,
    ) -> ApproximatedStream.Header:
        """
        Return the header (codec, header length, payload length) but don't decode payload yet.

        Useful to schedule parallel decompression.

        See from_bytes_dataonly_with_header.
        """
        start_offset = offset
        (codec_index,) = struct.unpack_from("B", b, offset=offset)
        offset += 1
        if count is not None and _codecs[codec_index] == RawCodec:
            length = count * np.dtype(dtype).itemsize
        else:
            length, offset = unpack_small_uint(b, offset)
        headerlen = offset - start_offset

        if verbose:
            storage = _codecs[codec_index].__name__
            print(
                f"reading {storage} {length} bytes + header {headerlen} = {length + headerlen}"
            )
        return ApproximatedStream.Header(start_offset, codec_index, headerlen, length)

    @staticmethod
    def from_bytes_dataonly_with_header(
        dtype: Union[DTypeLike, type],
        count: Optional[int],
        b: bytes,
        h: ApproximatedStream.Header,
        verbose: bool,
    ) -> tuple[ApproximatedStream, int]:
        """
        Having read the header for this stream, decompress the actual data.

        See from_bytes_dataonly.
        """
        offset = h.offset
        offset += h.headerlen
        payload = b[offset : offset + h.payloadlen]
        offset += h.payloadlen

        codec = _codecs[h.codec_index]
        uncompressed = codec.from_bytes(payload, dtype, verbose)
        astream = ApproximatedStream(uncompressed, epsilon=None)
        return (astream, offset)

    @classmethod
    def from_bytes_dataonly(
        cls,
        dtype: Union[DTypeLike, type],
        count: Optional[int],
        b: bytes,
        offset: int,
        verbose=False,
    ) -> tuple[ApproximatedStream, int]:
        """
        Read the data from the bytes.

        count is the number of values we expect to be decoding, if known
        from exogenous data.  The corresponding to_bytes_dataonly call must
        have the same count value (i.e. the same number, or both are None).
        """
        header = cls.read_dataonly_header(dtype, count, b, offset, verbose)
        return cls.from_bytes_dataonly_with_header(dtype, count, b, header, verbose)

    def tobytes(self, verbose=False) -> bytes:
        return b"".join(
            [
                pack_dtype(self.stream.dtype),
                self.tobytes_dataonly(count=None, verbose=verbose),
            ]
        )

    @staticmethod
    def from_bytes(
        b: bytes, offset: int = 0, verbose=False
    ) -> tuple[ApproximatedStream, int]:
        dt, offset = unpack_dtype(b, offset)
        assert issubclass(dt, np.number)
        return ApproximatedStream.from_bytes_dataonly(
            dtype=dt, count=None, b=b, offset=offset, verbose=verbose
        )


def encode_coordinates(
    data: np.ndarray, quality: Union[float, np.ndarray], verbose=False
) -> bytes:
    """
    Given a vector, encode it as a stream. Quality must be a single value.

    Given an array of shape (..., ndim) encode ndim streams with error bound
    quality[d] and return the whole, concatenated.

    Quality can be one value used for every coordinate, or a vector of length
    ndim denoting a separate quality per coordinate.
    """
    ndim = data.shape[-1]

    if not isinstance(quality, np.ndarray):
        quality = np.full(ndim, quality)

    if len(data.shape) == 1:
        return ApproximatedStream(data, quality[0]).tobytes_dataonly(len(data))

    # flatten the greater dimensions
    count = int(np.prod(data.shape[:-1]))
    flattened = data.reshape(count, ndim)

    def write_coordinate(d: int) -> bytes:
        column = flattened[:, d]
        stream = ApproximatedStream(column, quality[d])
        return stream.tobytes_dataonly(count=count, verbose=verbose)

    bytestreams = _executor.map(write_coordinate, range(ndim))
    b = b"".join(bytestreams)
    if verbose:
        print(f"output of {ndim} streams is {len(b)} bytes")
    return b


def decode_coordinates(
    b: bytes,
    offset: int,
    shape: Union[np.ndarray, tuple[int, ...], list[int]],
    dtype: Union[DTypeLike, type],
    verbose=False,
) -> tuple[np.ndarray, int]:
    """
    Return an array of the given shape from the given byte
    stream, plus the offset to the next byte to read.

    The bytes must have been output from encode_coordinates.
    """
    if len(shape) == 1:
        ndim = 1
        count = shape[0]
    else:
        ndim = shape[-1]
        count = np.prod(shape[:-1])

    headers: list[ApproximatedStream.Header] = []
    for d in range(ndim):
        if verbose:
            print(
                f"read coord {d} starting at {offset} / {len(b)} ({len(b) - offset} remaining)"
            )
        header = ApproximatedStream.read_dataonly_header(
            dtype=dtype, count=count, b=b, offset=offset, verbose=verbose
        )
        headers.append(header)
        offset = header.next_offset()

    # Decompress in parallel.
    def decompress(d: int) -> ApproximatedStream:
        stream, _ = ApproximatedStream.from_bytes_dataonly_with_header(
            dtype, count, b, headers[d], verbose=verbose
        )
        return stream

    streams = list(_executor.map(decompress, range(ndim)))

    if len(shape) == 1:
        data = streams[0].stream
    else:
        # if we're returning an array we need to make coordinates be columns,
        # then restore the original shape.
        row_coords = np.array([stream.stream for stream in streams])
        data = row_coords.T.reshape(shape)
    return data, headers[-1].next_offset()


def tiny_int_dtype(width: int) -> type:
    if width <= 0 or width > 64:
        raise ValueError("width {width} out of bounds (0,64]")
    match width:
        case 64:
            return np.uint64
        case 32:
            return np.uint32
        case 16:
            return np.uint16
        case 8:
            return np.uint8
        case 4:
            return np.uint8
        case 2:
            return np.uint8
        case 1:
            return np.bool
    return np.uint64


def packmultibits(values: np.ndarray, width: int) -> np.ndarray:
    """
    Packs the elements of a width-bit-valued unsigned int array into bits in an
    array of a native uint type.

    Use unpackmultibits to round-trip the values.
    """
    if width < 0 or width > 64:
        raise ValueError("width {width} out of bounds [0,64]")

    # Flatten the array; the caller will provide the shape when decoding.
    values = values.flatten()

    maxvalue = (1 << width) - 1
    if np.max(values) > maxvalue:
        raise ValueError("max value {np.max(values)} exceeds {width} bits")

    if width == 0:
        return np.array([])
    if width == 1:
        return np.packbits(values)

    t = tiny_int_dtype(width)
    dtype = np.dtype(t)
    values = values.astype(dtype)

    # If we're in a native type, just changing the type was enough.
    wordsize = 8 * dtype.itemsize
    if wordsize == width:
        return values

    # Otherwise we break the values up into values_per_word cols; then shift
    # each col so that when we OR the rows together, we have packed
    # values_per_word values together into one 8-bit or 64-bit word.
    values_per_word = wordsize // width

    # reshape, padding so the length divides evenly
    extravalues = len(values) % values_per_word
    if extravalues != 0:
        numpad = values_per_word - extravalues
        padded = np.pad(values, (0, numpad))
    else:
        padded = values
    nwords = len(padded) // values_per_word
    reshaped = padded.reshape((nwords, values_per_word))

    # shift each col so the bits don't overlap
    shifts = width * np.arange(values_per_word, dtype=dtype)
    shifted = reshaped << shifts

    # OR the rows (which is the same as summing since the bits don't overlap)
    # For some reason, numpy upcasts here, so we need to clobber it back down.
    orred = shifted.sum(axis=1).astype(dtype)
    return orred


def unpackmultibits(words: np.ndarray, width: int, count: int) -> np.ndarray:
    """
    Perform almost the inverse of packbits: take packed bits and convert back
    to an flat array of unsigned values of the narrowest possible type.

    The original shape and type are lost.
    """
    if width == 0:
        return np.zeros(count, dtype=np.uint8)

    if width == 1:
        return np.unpackbits(words, count=count)

    dtype = words.dtype
    assert dtype == np.dtype(tiny_int_dtype(width))
    wordsize = 8 * dtype.itemsize
    if wordsize == width:
        # Native size; no packing needed.
        return words.astype(dtype)

    # Break the words up into columns of values of width bits.
    # Then flatten the columns to get the padded output.
    # Finally, trim off the pad values.
    values_per_word = wordsize // width

    # Make a column from each value so we can shift and mask them.
    columns = np.tile(words, (values_per_word, 1)).T

    # Shift each column.
    shifts = width * np.arange(values_per_word, dtype=dtype)
    shifted = columns >> shifts

    # Mask all the values to truncate high bits.
    maxvalue = (1 << width) - 1
    masked = shifted & maxvalue

    # Flatten the columns.
    padded = masked.flatten()

    # Trim the padding if any.
    if len(padded) == count:
        return padded
    else:
        return np.array(padded[:count], copy=True)


def encode_tiny_ints(values: np.ndarray, width: int) -> bytes:
    """
    Encode an array of unsigned ints that each fit in some small number of bits.
    """
    return packmultibits(values, width).tobytes()


def decode_tiny_ints(
    b: bytes, offset: int, shape: tuple[int, ...], width: int
) -> tuple[np.ndarray, int]:
    """
    Decode tiny ints encoded by encode_tiny_ints.

    Returns an array of the appropriate shape, and the new offset. The original
    dtype is lost, we output the narrowest dtype that fits the width.
    """
    if width < 0 or width > 64:
        raise ValueError("width {width} out of bounds [0,64]")
    if width == 0:
        return np.zeros(shape, dtype=np.uint8), offset

    t = tiny_int_dtype(width)
    dtype = np.dtype(t)
    wordsize = 8 * dtype.itemsize
    values_per_word = wordsize // width

    count = int(np.prod(shape))
    padded_count = ((count - 1) // values_per_word) + 1

    packed = np.frombuffer(b[offset : offset + padded_count], dtype=dtype)
    offset += padded_count * packed.dtype.itemsize
    unpacked = unpackmultibits(packed, width, count)
    return unpacked.reshape(shape), offset
