from __future__ import annotations

import math
import numpy as np
import struct
import subprocess
import tempfile
import concurrent.futures

from typing import Optional
from pathlib import Path
from .util import pack_small_uint, unpack_small_uint, pack_dtype, unpack_dtype

neats_dir = Path.home() / "projects/geocache-compression/install/bin"
neats_compress = neats_dir / "neats_lossy_compress"
neats_decompress = neats_dir / "neats_lossy_decompress"


def temp_filename():
    """
    Return a context manager object with a name field.
    The file it denotes will be auto-deleted at end of context.
    """
    return tempfile.NamedTemporaryFile(delete_on_close=False)


class ApproximatedStream:
    """
    Represents a stream of data that will be encoded lossily to precision epsilon.
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

    def _quality_to_hex(self) -> str:
        if self.epsilon is None:
            raise ValueError("unable to compress a stream without an error bound")
        if self.stream.dtype != np.float32:
            raise NotImplementedError("compression only available for float32")
        fbits = struct.pack("f", self.epsilon)
        (unsigned,) = struct.unpack("I", fbits)
        return "0x" + hex(unsigned)

    def _tobytes_compressed(self, verbose: bool) -> Optional[bytes]:
        """
        Compress the stream and return its bytes.
        """
        if self.stream.dtype != np.float32:
            return None
        if len(self.stream) == 0:
            return None
        with temp_filename() as binfile:
            with temp_filename() as neatsfile:
                self.stream.tofile(binfile)
                binfile.close()
                neatsfile.close()

                args = [
                    neats_compress,
                    binfile.name,
                    "-o",
                    neatsfile.name,
                    "-q",
                    self._quality_to_hex(),
                ]
                if verbose:
                    args.append("-v")
                subprocess.run(args)

                with open(neatsfile.name, "rb") as f:
                    return f.read()

    def _tobytes_raw(self, verbose: bool) -> bytes:
        return self.stream.tobytes()

    def tobytes_dataonly(self, count: Optional[int], verbose=False) -> bytes:
        """
        Return the bytes just of the data stream.

        count is len(self.stream) but should be provided only if the
        decoder will be able to know it without decoding this stream (i.e. if
        the data is known from other data). Otherwise, provide None.
        """
        if self.epsilon is None:
            raise ValueError("unable to re-compress a stream")

        compressed = self._tobytes_compressed(verbose)
        raw = self._tobytes_raw(verbose)

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

    @staticmethod
    def from_bytes_dataonly(
        dtype: type, count: Optional[int], b: bytes, offset: int, verbose=False
    ) -> tuple[ApproximatedStream, int]:
        """
        Read the data from the bytes.

        count is the number of values we expect to be decoding, if known
        from exogenous data.  The corresponding to_bytes_dataonly call must
        have the same count value (i.e. the same number, or both are None).
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
        offset += headerlen
        payload = b[offset : offset + length]
        offset += length

        if is_raw:
            uncompressed: np.ndarray = np.frombuffer(payload, dtype=dtype)
        else:
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
                    uncompressed = np.fromfile(binfile.name, dtype=dtype)

        astream = ApproximatedStream(uncompressed, epsilon=None)
        return (astream, offset)

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


def encode_coordinates(data: np.ndarray, quality: float, verbose=False) -> bytes:
    """
    Given an array of shape (nsamples, nverts, ndim) or (nverts, ndim), the latter of
    which is reinterpreted as (1, nverts, ndim).

    Output ndim streams each of which is the samples of each vertex,
    concatenated.
    """
    if len(data.shape) == 2:
        data = data[np.newaxis, :]
    nsamples, nverts, ndim = data.shape
    count = nsamples * nverts

    reordered = data.transpose(2, 0, 1)
    by_dimension = reordered.reshape(ndim, count)

    def write_coordinate(d: int) -> bytes:
        column = by_dimension[d, :]
        stream = ApproximatedStream(column, quality)
        return stream.tobytes_dataonly(count=count, verbose=verbose)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=ndim)
    bytestreams = executor.map(write_coordinate, range(ndim))
    b = b"".join(bytestreams)
    if verbose:
        print(f"output of {ndim} streams is {len(b)} bytes")
    return b


def decode_coordinates(
    b: bytes,
    offset: int,
    nsamples: int,
    nverts: int,
    ndim: int,
    dtype: type,
    verbose=False,
) -> tuple[np.ndarray, int]:
    """
    Return an array of shape (nsamples, nverts, ndim) from the given byte
    stream, plus the offset to the next byte to read.

    The bytes must have been output from encode_coordinates.
    """
    count = nsamples * nverts

    streams = []
    for d in range(ndim):
        if verbose:
            print(
                f"read coord {d} starting at {offset} / {len(b)} ({len(b) - offset} remaining)"
            )
        stream, offset = ApproximatedStream.from_bytes_dataonly(
            dtype=dtype, count=count, b=b, offset=offset, verbose=verbose
        )
        streams.append(stream)

    by_dimension = np.array([stream.stream for stream in streams])
    reordered = by_dimension.reshape((ndim, nsamples, nverts))
    data = reordered.transpose(1, 2, 0)
    return data, offset
