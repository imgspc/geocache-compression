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

    @abstractmethod
    def tobytes(
        self, data: np.ndarray, quality: float, verbose: bool
    ) -> Optional[bytes]: ...

    @classmethod
    @abstractmethod
    def from_bytes(
        cls, payload: bytes, dtype: Union[type, DTypeLike], verbose: bool
    ) -> np.ndarray: ...


class RawCodec(Codec):
    def tobytes(
        self, data: np.ndarray, quality: float, verbose: bool
    ) -> Optional[bytes]:
        return data.tobytes()

    @classmethod
    def from_bytes(
        cls, payload: bytes, dtype: Union[type, DTypeLike], verbose: bool
    ) -> np.ndarray:
        return np.frombuffer(payload, dtype=dtype)


class NeatsCodec(Codec):
    def tobytes(
        self, data: np.ndarray, quality: float, verbose: bool
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

        compressed = NeatsCodec().tobytes(self.stream, self.epsilon, verbose)
        raw = RawCodec().tobytes(self.stream, self.epsilon, verbose)
        assert raw is not None

        compressed_header = 4
        raw_header = 4 if count is None else 1

        # Encoding:
        # if we're storing compressed, length in bytes as positive big-endian signed int
        # if we're storing raw:
        #       if count is known, header is 0xff
        #       if count is not known, header is 4-byte signed int, corresponds to -length in bytes
        if compressed is None:
            use_raw = True
        elif len(raw) + raw_header <= len(compressed) + compressed_header:
            use_raw = True
        else:
            use_raw = False

        if use_raw:
            data = raw
            if count is not None:
                header = b"\xff"
                # length is unrestricted
            else:
                header = struct.pack(">i", -len(raw))

                # check the length is in bounds (we can negate it)
                if len(raw) >= 0xA0000000:
                    raise NotImplementedError(
                        "length {len(data)} too long; TODO: support 64-bit length"
                    )
        else:
            assert compressed is not None
            data = compressed
            header = struct.pack(">i", len(compressed))

            # check the length is in bounds (positive as int32)
            if len(compressed) >= 0xA0000000:
                raise NotImplementedError(
                    "length {len(data)} too long; TODO: support 64-bit length"
                )
        if verbose:
            storage = "raw" if use_raw else "compressed"
            print(
                f"wrote {storage} {len(data)} bytes + header {len(header)} = {len(data) + len(header)}"
            )
        return header + data

    class Header:
        def __init__(self, offset, is_raw, headerlen, payloadlen):
            self.offset = offset
            self.is_raw = is_raw
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
        Return:
        * payloadlen

        Useful to parallelize decompression.

        See from_bytes_dataonly_with_header.
        """
        if count is None:
            (length,) = struct.unpack_from(">i", b, offset)
            headerlen = 4

            if length <= 0:
                # negative or zero means raw (because -0 is still zero).
                length = -length
                is_raw = True
            else:
                is_raw = False
        else:
            if b[offset] == 0xFF:
                is_raw = True
                length = count * np.dtype(dtype).itemsize
                headerlen = 1
            else:
                is_raw = False
                (length,) = struct.unpack_from(">I", b, offset)
                headerlen = 4

        if verbose:
            storage = "raw" if is_raw else "compressed"
            print(
                f"reading {storage} {length} bytes + header {headerlen} = {length + headerlen}"
            )
        return ApproximatedStream.Header(offset, is_raw, headerlen, length)

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

        codec = RawCodec if h.is_raw else NeatsCodec
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
