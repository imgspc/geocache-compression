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
from embedding import clustering

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

    def __str__(self) -> str:
        return f"{self.extent} float{self.floatsize}_t per point, {self.size} points, {self.nsamples} samples"


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

    # Form the clusters
    cover = clustering.cluster_monolithic(header.size)
    print(f"found {cover.nsubsets} clusters")

    # TODO: parallelize the computation.
    clusters = list(clustering.slice(f, cover))
    embedded = [embedding.PCAEmbedding.from_data(cluster, 0.95) for cluster in clusters]

    basename = os.path.splitext(binfile)[0]
    headersbin = basename + ".embed-header.bin"
    projectedbin = basename + ".embed.bin"
    with open(headersbin, "wb") as headerfile:
        with open(projectedbin, "wb") as projectedfile:
            for i, cluster in enumerate(clusters):
                headerfile.write(embedded[i].tobytes())
                projected = embedded[i].project(cluster)
                projectedfile.write(projected.tobytes())

    return projectedbin


def main(jsonfile: str, binfile: str):
    outname = convert(jsonfile, binfile)
    print(f"output written to {outname}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
