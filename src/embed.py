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
from embedding.io import Header, read_file, parse_json

# Inputs:
#       .json -- provides nsamples, type, size, extent
#       .bin -- an ndarray of shape (nsamples, size, extent) and the given type; must match an entry in the json file
#
# Output:
#       .embed-header.bin
#       .embed.bin
#       .embed-clusters.bin
#
# The embed-header is a list of Embedding objects, serialized. One object per vertex.
#
# The embed file is a list of reduced-dimension ndarrays, one per vertex. The overall shape
# is a ragged array of (size, nsamples, k) where k may vary per vertex.
#
# See reconstruct.py to lossily reverse from the embedded data to the original.
#


def convert(jsonfile: str, binfile: str) -> str:
    """
    Convert the input file (with metadata in the json file) to embed files.
    """
    header = parse_json(jsonfile, binfile)
    f = read_file(binfile, header)
    print(f"read {binfile} as {header}")
    header.verify_shape(f)

    # Form the clusters
    cover = clustering.cluster_by_index(header.size, 10)
    print(f"found {cover.nsubsets} clusters")

    # TODO: parallelize the computation.
    clusters = list(clustering.slice(f, cover))
    embedded = [
        embedding.PCAEmbedding.from_data(cluster, 0.995) for cluster in clusters
    ]

    basename = os.path.splitext(binfile)[0]
    headersbin = basename + ".embed-header.bin"
    projectedbin = basename + ".embed.bin"
    clusterbin = basename + ".embed-clusters.bin"
    with open(headersbin, "wb") as headerfile:
        with open(projectedbin, "wb") as projectedfile:
            for i, cluster in enumerate(clusters):
                headerfile.write(embedding.serialize(embedded[i]))
                projected = embedded[i].project(cluster)
                projectedfile.write(projected.tobytes())
    with open(clusterbin, "wb") as clusterfile:
        clusterfile.write(cover.tobytes())

    return projectedbin


def main(jsonfile: str, binfile: str):
    outname = convert(jsonfile, binfile)
    print(f"output written to {outname}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
