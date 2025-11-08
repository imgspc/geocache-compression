from __future__ import annotations

import argparse
import json
import numpy as np
import os
import struct
import sys

from typing import Any, Optional, Iterable, Union
from numpy.typing import DTypeLike
from embedding import clustering, embedding, metric
from embedding.io import Header, read_binfile, parse_json

# Inputs:
#       .json -- provides nsamples, type, size, extent
#       .bin -- an ndarray of shape (nsamples, size, extent) and the given type; must match an entry in the json file
#       .embed-header.bin (unnamed, inferred)
#       .embed.bin (unnamed, inferred)
#
# Output:
#       .approx.bin -- an ndarray same shape as .bin with the embedded data projected back to the original domain
#       stdout -- some statistics about errors


def verify(jsonfile: str, binfile: str) -> None:
    """
    Convert the input file (with metadata in the json file) and embed files to approx files.
    """
    basename = os.path.splitext(binfile)[0]
    package = parse_json(jsonfile)
    header = package.get_header(binfile)
    predata = read_binfile(header)
    print(f"read {basename} as {header}")
    header.verify_shape(predata)

    # generate input filenames (todo: centralize)
    headersbin = basename + ".embed-header.bin"
    projectedbin = basename + ".embed.bin"
    clusterbin = basename + ".embed-clusters.bin"

    # generate output filenames
    invertbin = basename + ".embed-inverted.bin"
    errorsbin = basename + ".embed-errors.bin"

    # Read the clusters
    with open(clusterbin, "rb") as f:
        (cover, _) = clustering.Covering.from_bytes(f.read())
    print(f"found {cover.nsubsets} clusters")

    with open(headersbin, "rb") as headerfile:
        headeroff = 0
        headerbytes = headerfile.read()
    with open(projectedbin, "rb") as projectedfile:
        projectedoff = 0
        projectedbytes = projectedfile.read()

    # Read the embedded data and invert it back to the original domain.
    slices: list[np.ndarray] = []
    for _ in range(cover.nsubsets):
        (embed, headeroff) = embedding.deserialize(headerbytes, headeroff)
        (projected, projectedoff) = embed.read_projection(projectedbytes, projectedoff)
        slices.append(embed.invert(projected))

    # The data is now sliced up, unslice it.
    postdata = clustering.unslice(slices, cover, header.extent)

    # Now compute the errors.
    errors = predata - postdata

    # Output the inverted data.
    with open(invertbin, "wb") as invertfile:
        invertfile.write(postdata.tobytes())

    # Calculate the errors and write them out.
    with open(errorsbin, "wb") as errorsfile:
        errorsfile.write(errors.tobytes())

    # Compute some metrics.
    box = metric.box_range(predata)
    print(f"Original range: {box}")

    distance = metric.point_hausdorff(predata, postdata)
    print(f"Hausdorff (pointwise): {distance}")
    distance = metric.Linf(predata, postdata)
    print(f"Linf: {distance}")

    # Verify: can we get precisely lossless results?
    distance = metric.Linf(predata, postdata + errors)
    print(f"Linf after error-correct: {distance}")

    print(f"Data reconstructed in {invertbin}")


def main(jsonfile: str, binfile: str):
    verify(jsonfile, binfile)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
