from __future__ import annotations

import math
import numpy as np
import struct
import subprocess
import tempfile

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
    uint_width,
    make_thread_pool,
)

neats_dir = Path.home() / "projects/geocache-compression/install/bin"
neats_compress = neats_dir / "neats_lossy_compress"
neats_decompress = neats_dir / "neats_lossy_decompress"

# Set up a thread pool for encoding.
_executor = make_thread_pool()


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
        count = int(np.prod(data.shape))
        rounded = np.round(data.astype(np.float64) / quality).astype(np.int64)
        diffcoded = np.ediff1d(rounded, to_begin=rounded[0])

        # min_scalar_type is tricky to use with signed values; rewrite the
        # differences with a bias so it's all unsigned values.
        base = np.min(diffcoded)
        positive = diffcoded - base
        largest = np.max(positive)
        if largest == 0:
            width = 0
        else:
            width = math.floor(math.log2(largest)) + 1
            assert largest < (1 << width)
        diff_bytes = encode_tiny_ints(positive, width)
        return (
            struct.pack("<qfB", base, quality, width)
            + pack_small_uint(count)
            + diff_bytes
        )

    @classmethod
    def from_bytes(
        cls, payload: bytes, dtype: Union[type, DTypeLike], verbose: bool
    ) -> np.ndarray:
        base, quality, width = struct.unpack_from("<qfB", payload)
        offset = struct.calcsize("<qfB")
        count, offset = unpack_small_uint(payload, offset)
        diff_values, offset = decode_tiny_ints(
            payload, offset=offset, width=width, shape=(count,)
        )
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
        if count is not None and count <= 0:
            raise ValueError(f"count {count}")

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

        def streamlength(ce: tuple[int, bytes]) -> int:
            return len(ce[1])

        best_index, data = min(valid, key=streamlength)

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
        if count is not None and count <= 0:
            raise ValueError(f"count {count}")
        start_offset = offset
        (codec_index,) = struct.unpack_from("B", b, offset=offset)
        offset += 1
        if codec_index >= len(_codecs):
            raise ValueError(f"read codec {codec_index} in stream with count {count}")
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
        return cls.from_bytes_dataonly_with_header(dtype, b, header, verbose)

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
    shape = data.shape
    if len(shape) == 1:
        ndim = 1
        count = shape[0]
    else:
        ndim = shape[-1]
        count = np.prod(shape[:-1])

    if ndim == 0 or count == 0:
        return b""

    if not isinstance(quality, np.ndarray):
        quality = np.full(ndim, quality)

    if ndim == 1:
        return ApproximatedStream(data, quality[0]).tobytes_dataonly(len(data))

    # flatten the greater dimensions
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

    # 0-length storage is actually pretty common
    if count == 0 or ndim == 0:
        return np.array([], dtype=dtype), offset

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
            dtype, b, headers[d], verbose=verbose
        )
        return stream

    streams = list(_executor.map(decompress, range(ndim)))

    if ndim == 1:
        data = streams[0].stream
    else:
        # if we're returning an array we need to make coordinates be columns,
        # then restore the original shape.
        row_coords = np.array([stream.stream for stream in streams])
        data = row_coords.T.reshape(shape)
    return data, headers[-1].next_offset()


def encode_sparse_matrix(
    data: np.ndarray,
    counts: np.ndarray,
    quality: Union[None, float, np.ndarray],
    verbose=False,
) -> bytes:
    """
    Given a sparse matrix provided as:
        a matrix of nrows x ncols
        a vector of nrows counts (the number of columns to keep for each row)
    Encode the sparse matrix.

    Columns that are entirely zero are not encoded:
        decode_sparse_matrix(encode_sparse_matrix(...))
    may return fewer columns than passed in.

    Quality may be None only if np.max(counts) == 0.
    """
    nrows, _ = data.shape
    ncols = np.max(counts)

    if ncols == 0:
        return struct.pack("B", 0)

    if not isinstance(quality, np.ndarray):
        quality = np.full(ncols, quality)

    # Write the values column by column, sparsely.
    def write_column(col: int) -> bytes:
        column = data[col < counts, col]
        stream = ApproximatedStream(column, quality[col])
        return stream.tobytes_dataonly(count=len(column), verbose=verbose)

    bytestreams = _executor.map(write_column, range(ncols))

    # Write the counts.
    width = uint_width(counts)
    cbytes = encode_tiny_ints(counts, width)

    b = b"".join([struct.pack("B", width), cbytes, *bytestreams])
    if verbose:
        print(
            f"stored {np.prod(data.shape)} values sparsely as {np.sum(counts)} in {len(b)} bytes"
        )
    return b


def decode_sparse_matrix(
    b: bytes,
    offset: int,
    nrows: int,
    dtype: Union[DTypeLike, type],
    verbose=False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Decode a sparse matrix previously encoded by encode_sparse_matrix.

    The number of columns returned is the minimum number needed to encode
    the non-zero data.

    Returns:
        * the matrix in the given shape (must be a matrix)
        * the row counts
        * the offset after reading the data
    """
    # First read the counts.
    (width,) = struct.unpack_from("B", b, offset=offset)
    offset += 1

    if width == 0:
        return (
            np.zeros((nrows, 0), dtype=dtype),
            np.zeros((nrows,), dtype=np.uint8),
            offset,
        )

    row_counts, offset = decode_tiny_ints(b, offset, (nrows,), width)
    ncols = np.max(row_counts)

    # Build up the mask of which values are actually stored.
    mask = np.tile(np.arange(ncols), (1, nrows)) < row_counts[:, np.newaxis]

    # Count how many values each *column* has (counts is by row).
    col_counts = np.sum(mask, axis=0)

    # Now read the headers for the columns.
    headers: list[ApproximatedStream.Header] = []
    for col in range(ncols):
        if verbose:
            print(
                f"read column {col} starting at {offset} / {len(b)} ({len(b) - offset} remaining)"
            )
        header = ApproximatedStream.read_dataonly_header(
            dtype=dtype, count=col_counts[col], b=b, offset=offset, verbose=verbose
        )
        headers.append(header)
        offset = header.next_offset()

    # Decode in parallel.
    def decode(col: int) -> np.ndarray:
        stream, _ = ApproximatedStream.from_bytes_dataonly_with_header(
            dtype, b, headers[col], verbose=verbose
        )
        return stream.stream

    columns = list(_executor.map(decode, range(ncols)))

    # Desparsify
    data = np.zeros((nrows, ncols), dtype=dtype)
    for col in range(ncols):
        data[mask[:, col], col] = columns[col]

    # We're done!
    return data, row_counts, offset


def tiny_int_dtype(width: int) -> np.dtype:
    if width < 1 or width > 64:
        raise ValueError(f"width {width} out of bounds [2,64]")

    def waste_per_64bits(dtype):
        wordsize = dtype.itemsize * 8
        if wordsize < width:
            return 64
        nwords = 64 // wordsize
        waste_per_word = wordsize % width
        return waste_per_word * nwords

    dtypes = map(np.dtype, (np.uint8, np.uint16, np.uint32, np.uint64))
    return min(dtypes, key=waste_per_64bits)


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

    dtype = tiny_int_dtype(width)
    values = values.astype(dtype)

    # If we're in a native type, just changing the type was enough.
    wordsize = 8 * dtype.itemsize
    if wordsize == width:
        return values

    # Otherwise we break the values up into values_per_word cols; then shift
    # each col so that when we OR the rows together, we have packed
    # values_per_word values together into each word without overlap.
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
    # Prevent numpy from upcasting to uint64 here; we know there's no carry.
    orred = shifted.sum(axis=1, dtype=dtype)
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
    padded_len = padded_count * dtype.itemsize
    if offset + padded_len > len(b):
        raise ValueError(
            "buffer of length {len(b)} overrun by {padded_count}-byte array starting at {offset}"
        )

    packed = np.frombuffer(b[offset : offset + padded_len], dtype=dtype)
    offset += padded_len
    unpacked = unpackmultibits(packed, width, count)
    return unpacked.reshape(shape), offset
