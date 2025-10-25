from __future__ import annotations

import argparse
import json
import numpy as np
import os
import struct
import sys

from typing import Any, Optional, Iterable, Union
from numpy.typing import DTypeLike
from embedding import embedding

# Inputs:
#       .json -- provides nsamples, type, size, extent
#       .bin -- an ndarray of shape (nsamples, size, extent) and the given type; must match an entry in the json file
#
# Output:
#       .embed-header.bin
#       .embed.bin
#
# The embed-header is a list of Embedding objects, serialized. One object per vertex.
#
# The embed file is a list of reduced-dimension ndarrays, one per vertex. The overall shape
# is a ragged array of (size, nsamples, k) where k may vary per vertex.
#
# See reconstruct.py to lossily reverse from the embedded data to the original.
#


class Header:
    def __init__(
        self,
        floatsize: int,
        extent: int,
        size: int,
        nsamples: int,
        k: int = 0,
    ):
        self.floatsize = floatsize
        self.extent = extent
        self.k = k
        self.size = size
        self.nsamples = nsamples

    def verify_shape(self, f: np.ndarray) -> None:
        """
        Verify that the array's shape matches the header values for size,
        extent, and samples.
        """
        shape = f.shape
        expected = (self.nsamples, self.size, self.extent)
        if shape != expected:
            raise ValueError(
                f"expected shape {expected}, actual binfile is shape {shape}"
            )

    # pack little-endian since most machines are little-endian
    # pack type, extent, k as bytes, followed by quads for size and nsamples
    packformat = "<BBBQQ"

    def pack(self) -> bytes:
        return struct.pack(
            self.packformat,
            self.floatsize,
            self.extent,
            self.k,
            self.size,
            self.nsamples,
        )

    @classmethod
    def unpack(cls, b: bytes) -> Header:
        (f, e, k, s, n) = struct.unpack(cls.packformat, b)
        return cls(floatsize=f, extent=e, size=s, nsamples=n, k=k)

    def __str__(self) -> str:
        return f"{self.extent} float{self.floatsize}_t per point, {self.size} points, {self.nsamples} samples"


# Return a bool array where different values are listed as 'true'.
# Note: negative and positive zero are equal.
vector_are_ne = np.vectorize(lambda a, b: a != b)

# Choose a or b depending on c (a is for True, b is for False).
vector_mux = np.vectorize(lambda a, b, c: a if c else b)


class SVD:
    def __init__(
        self,
        c: np.ndarray,
        WVt: np.ndarray,
        U: np.ndarray,
        epsilon: np.ndarray,
        clobbers: np.ndarray,
    ):
        """
        U WVt + c should reconstruct the original data as well
        as is possible given dimensionality reduction and roundoff.

        U WVt + c + epsilon should reconstruct the original data exactly.

        There is a possibility that no value for epsilon can work. This can be true iff
        (U WVt)[i,j] is very large compared to column[i,j]. In that case,
        clobbers[i,j] will be True and epsilon[i,j] will be column[i,j].
        """
        self.c = c
        self.WVt = WVt
        self.U = U
        self.epsilon = epsilon
        self.clobbers = clobbers
        assert clobbers.dtype == np.bool

    @staticmethod
    def _finish_compute(column: np.ndarray, c, WVt, U) -> SVD:
        """
        The SVD math having been performed, compute the epsilon and clobbers
        arrays, and package it all into an SVD object.
        """
        # Compute the prediction, which will be approximate due to roundoff and
        # dimensionality reduction.
        predicted = U @ WVt + c

        # Compute the error in the prediction.
        epsilon = column - predicted

        # Check that we can reconstruct precisely (bitwise)
        reconstructed = predicted + epsilon

        # The 'clobbers' array is all the values that didn't get reconstructed.
        clobbers = vector_are_ne(column, reconstructed)

        # If 'clobbers' is true then we use the actual data, and if it's false
        # we use the error term that manages to actually reconstruct the data.
        epsilon_mux = vector_mux(column, epsilon, clobbers)
        return SVD(c, WVt, U, epsilon_mux, clobbers)

    @staticmethod
    def compute(column: np.ndarray, k: int = 0) -> SVD:
        """
        Perform SVD, optionally reducing to k dimensions.
        """
        n = len(column)
        if not n:
            raise ValueError("Can't run SVD on empty matrix")

        # Translate the data to the origin.
        c = sum(column) / n
        M = column - c

        # Perform SVD:
        # Vt is the orthonormal basis,
        # W is the diagonal weight matrix, stored as a vector of singular values
        # U is the data rewritten in the SVD basis.
        #
        # We can't assume M is hermitian, so we need the full computation.
        # We definitely need U and V.
        # We don't want a square U, we only need it to be nsamples x n.
        # TODO: handle case of nsamples < n
        U, s, Vt = np.linalg.svd(M, full_matrices=False)

        # No dimensionality reduction? Return right away.
        if k == 0 or k == n:
            WVt = np.diag(s) @ Vt
            return SVD._finish_compute(column, c, WVt, U)

        # Dimensionality reduction: Keep the most significant k values only.
        # Drop row/col from W (by dropping values from s), drop rows from Vt, drop
        # columns from U.
        #
        # Return copies to allow the full-dimension memory to be released.
        # WVt_hat is already a copy due to the dot product.
        # U_hat we need to copy explictly.
        WVt_hat = np.diag(s[0:k]) @ Vt[0:k, :]
        U_hat = np.array(U[:, 0:k])
        return SVD._finish_compute(column, c, WVt_hat, U_hat)

    @staticmethod
    def header_size(extent: int, k: int, floatsize: int) -> int:
        """
        Return the number of bytes needed for the SVD header given the extent, k,
        and float size (in bits).
        """
        if k == 0:
            k = extent
        return (floatsize // 8) * (extent + extent * k)

    def pack_header(self) -> bytes:
        """
        Return the bytes of the SVD header we currently store.
        """
        return self.c.tobytes() + self.WVt.tobytes()


def read_file(binfile: str, header: Header) -> np.ndarray:
    """
    Read the file into an row-major array structured with nsamples rows and size columns,
    each column being an extent-tuple of floatsize values.

    Use [:,i,:] to get just the samples of vertex i.
    """
    # Build the datatype
    match header.floatsize:
        case 16:
            nptype: DTypeLike = np.float16
        case 32:
            nptype = np.float32
        case 64:
            nptype = np.float64
        case _:
            raise ValueError(
                f"can't handle type float{header.floatsize}_t parsing {binfile}"
            )

    # datatype is that each sample row is a matrix of size rows by extent columns
    dtype = np.dtype((nptype, (header.size, header.extent)))

    # return the entire file, parsed into one array with shape (nsamples, size, extent)
    return np.fromfile(binfile, dtype=dtype)


def svd_file(f: np.ndarray, k: int = 0) -> Iterable[SVD]:
    """
    Given a parsed file as a nsamples x size x extent array, run SVD
    piecewise on each nsamples x extent slice.
    """
    (nsamples, size, extent) = f.shape

    return (SVD.compute(f[:, i, :], k) for i in range(size))


def parse_json(jsonfile: str, binfile: str) -> Header:
    """
    Given a json file describing the ABC file, and a binary file, report the
    details needed to form the header and parse the binary file.

    Raises KeyError if binfile isn't associated with a property in the ABC file.
    """
    binbasename = os.path.basename(binfile)
    with open(jsonfile) as f:
        # Look up the components, find the one where 'bin' matches binfile by basename.
        # Report that. Ignore the dirname.
        parsed_json = json.load(f)
        for component in parsed_json["components"]:
            if component["bin"] == binbasename:
                float_type = component["type"]
                if isinstance(float_type, int):
                    floatsize = float_type
                elif float_type == "float16_t":
                    floatsize = 16
                elif float_type == "float32_t":
                    floatsize = 32
                elif float_type == "float64_t":
                    floatsize = 64
                else:
                    raise ValueError(f"unhandled type {float_type} for {binbasename}")
                return Header(
                    floatsize=floatsize,
                    extent=component["extent"],
                    size=component["size"],
                    nsamples=component["samples"],
                )
    # If we are here then we didn't find the file.
    raise KeyError(f"no property has 'bin' matching {binbasename}")


def convert(jsonfile: str, binfile: str) -> str:
    """
    Convert the input file (with metadata in the json file) to embed files.
    """
    header = parse_json(jsonfile, binfile)
    f = read_file(binfile, header)
    print(f"read {binfile} as {header}")
    header.verify_shape(f)

    # TODO: parallelize the computation.
    embedded = [
        embedding.PCAEmbedding.from_data(f[:, i, :], 0.95) for i in range(header.size)
    ]

    basename = os.path.splitext(binfile)[0]
    headersbin = basename + ".embed-header.bin"
    projectedbin = basename + ".embed.bin"
    with open(headersbin, "wb") as headerfile:
        with open(projectedbin, "wb") as projectedfile:
            for i in range(header.size):
                headerfile.write(embedded[i].tobytes())
                projected = embedded[i].project(f[:, i, :])
                projectedfile.write(projected.tobytes())

    return projectedbin


def main(jsonfile: str, binfile: str):
    outname = convert(jsonfile, binfile)
    print(f"output written to {outname}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
