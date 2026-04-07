from __future__ import annotations

import math
import numpy as np
import struct
import subprocess
import tempfile

from typing import Optional
from pathlib import Path
from .util import pack_small_uint, unpack_small_uint, pack_dtype, unpack_dtype

neats_dir = Path.home() / "projects/geocache-compression/build/neats"
neats_compress = neats_dir / "neats_compress"
neats_decompress = neats_dir / "neats_decompress"


def temp_filename():
    """
    Return a context manager object with a name field.
    The file it denotes will be auto-deleted at end of context.
    """
    return tempfile.NamedTemporaryFile(delete_on_close=False)


def epsilon_to_decimals(epsilon: float) -> int:
    # This function is far from any critical path; should be possible to do it
    # much faster if necessary.
    n = 0
    if epsilon <= 0 or math.isnan(epsilon):
        raise ValueError(f"{epsilon} must be strictly positive")
    while epsilon < 1:
        n += 1
        epsilon *= 10
    return n


class ApproximatedStream:
    """
    Represents a stream of data that will be encoded lossily to precision epsilon
    (or, if decimals is set, to a fixed number of decimal places)
    """

    def __init__(
        self, epsilon: float, stream: np.ndarray, decimals: Optional[int] = None
    ):
        """
        Create an ApproximatedStream for the data, with the given error bound epsilon.

        Epsilon is ignored if `decimals` is provided.
        """
        if len(stream.shape) > 1:
            raise ValueError(f"Shape must be flat, not {stream.shape}")
        if decimals is None and (epsilon <= 0 or not math.isfinite(epsilon)):
            raise ValueError(f"epsilon must be finite and positive {epsilon}")
        self.stream = stream

        if decimals is not None:
            self.decimals = decimals
        else:
            self.decimals = epsilon_to_decimals(epsilon)

    def tobytes_dataonly(self) -> bytes:
        """
        Return the bytes just of the data stream.

        Presumably the caller knows what the data type and precision are.

        The first word is the length in bytes of the compressed data stream.
        """
        # TODO: get neats to accept a stream in from stdin and output to stdout rather than
        # requiring temp files.
        with temp_filename() as binfile:
            with temp_filename() as neatsfile:
                self.stream.tofile(binfile)
                binfile.close()
                neatsfile.close()

                subprocess.run(
                    [
                        neats_compress,
                        binfile.name,
                        "-o",
                        neatsfile.name,
                        "-d",
                        str(self.decimals),
                        "-s",
                    ]
                )
                with open(neatsfile.name, "rb") as f:
                    b = f.read()
                length = struct.pack("<I", len(b))
                return length + b

    @staticmethod
    def from_bytes_dataonly(
        dtype: type, decimals: int, b: bytes, offset: int
    ) -> tuple[ApproximatedStream, int]:
        """
        Read the data from the bytes, using a known header.
        """
        length = struct.unpack_from("<I", b, offset)[0]
        offset += 4
        neats_data = b[offset : offset + length]
        offset += length
        with temp_filename() as neatsfile:
            with temp_filename() as binfile:
                neatsfile.write(neats_data)
                neatsfile.close()
                binfile.close()

                subprocess.run([neats_decompress, neatsfile.name, "-o", binfile.name])
                uncompressed: np.ndarray = np.fromfile(binfile.name, dtype=dtype)

        astream = ApproximatedStream(0, uncompressed, decimals)
        return (astream, offset)

    def tobytes(self) -> bytes:
        return b"".join(
            [
                pack_dtype(self.stream.dtype),
                pack_small_uint(self.decimals),
                self.tobytes_dataonly(),
            ]
        )

    @staticmethod
    def from_bytes(b: bytes, offset: int) -> tuple[ApproximatedStream, int]:
        dt, offset = unpack_dtype(b, offset)
        assert issubclass(dt, np.number)
        decimals, offset = unpack_small_uint(b, offset)
        return ApproximatedStream.from_bytes_dataonly(dt, decimals, b, offset)
