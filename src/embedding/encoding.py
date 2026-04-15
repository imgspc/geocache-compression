from __future__ import annotations

import math
import numpy as np
import struct
import subprocess
import tempfile

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
        # not checked: that we're in 32 bits
        fbits: bytes = struct.pack("f", self.epsilon)
        (unsigned,) = struct.unpack("I", fbits)
        return "0x" + hex(unsigned)

    def tobytes_dataonly(self, verbose=False) -> bytes:
        """
        Return the bytes just of the data stream.

        Presumably the caller knows what the data type and precision are.

        The first word is the length in bytes of the compressed data stream.
        """
        if self.epsilon is None:
            raise ValueError("unable to compress a stream without an error bound")

        if self.stream.dtype != np.float32:
            # TODO: handle float16 and float64.
            b = self.stream.tobytes()
            length = struct.pack("<I", len(b))
            return length + b

        # TODO: get neats to accept a stream in from stdin and output to stdout rather than
        # requiring temp files.
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
                    b = f.read()
                length = struct.pack("<I", len(b))
                return length + b

    @staticmethod
    def from_bytes_dataonly(
        dtype: type, b: bytes, offset: int, verbose=False
    ) -> tuple[ApproximatedStream, int]:
        """
        Read the data from the bytes, using a known header.
        """
        (length,) = struct.unpack_from("<I", b, offset)
        offset += 4
        payload = b[offset : offset + length]
        offset += length

        if dtype != np.float32:
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
                self.tobytes_dataonly(verbose),
            ]
        )

    @staticmethod
    def from_bytes(
        b: bytes, offset: int = 0, verbose=False
    ) -> tuple[ApproximatedStream, int]:
        dt, offset = unpack_dtype(b, offset)
        assert issubclass(dt, np.number)
        return ApproximatedStream.from_bytes_dataonly(
            dt, b, offset=offset, verbose=verbose
        )
