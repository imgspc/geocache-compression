from __future__ import annotations

import argparse
import json
import numpy as np
import os
import struct
import sys
import re

from typing import Any, Optional, Iterable, Union
from numpy.typing import DTypeLike
from embedding import embedding, clustering, metric

# Inputs:
#       .json -- provides nsamples, type, size, extent for all properties
#       .bin -- provides ndarray (nsamples, size, extent) and dtype=type; also, tells us which property to look for
#


class Header:
    def __init__(
        self,
        objpath: str,
        floatsize: int,
        extent: int,
        size: int,
        nsamples: int,
        binpath: str = "",
    ):
        self.floatsize = floatsize
        self.extent = extent
        self.size = size
        self.nsamples = nsamples
        self.path = objpath
        if binpath:
            self.binpath = binpath
        else:
            self.binpath = re.sub(r"[-./ ()]", "-", objpath).strip("-")

    def numbytes(self) -> int:
        return (self.floatsize // 8) * self.size * self.extent * self.nsamples

    def matches(self, name: str) -> bool:
        """
        Does the name match this header?

        It can match the object path, or it can match the binary filename (only
        the basename needs to match).
        """
        if name == self.path:
            return True
        if name == self.binpath:
            return True
        if os.path.basename(name) == os.path.basename(self.binpath):
            return True
        return False

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

    def tojson(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "type": self.floatsize,
            "samples": self.nsamples,
            "size": self.size,
            "extent": self.extent,
            "bin": self.binpath,
        }

    @staticmethod
    def fromjson(component: dict[str, Any]) -> Header:
        float_type = component["type"]
        if isinstance(float_type, int):
            floatsize: int = float_type
        else:
            match float_type:
                case "float16" | "float16_t":
                    floatsize = 16
                case "float32" | "float32_t":
                    floatsize = 32
                case "float64" | "float64_t":
                    floatsize = 64
                case _:
                    component_name = component["path"]
                    raise ValueError(
                        f"unhandled type {float_type} for {component_name}"
                    )
        return Header(
            objpath=str(component["path"]),
            floatsize=floatsize,
            extent=int(component["extent"]),
            size=int(component["size"]),
            nsamples=int(component["samples"]),
            binpath=str(component["bin"]),
        )

    def __str__(self) -> str:
        return f"{self.path} -- {self.extent} float{self.floatsize}_t per point, {self.size} points, {self.nsamples} samples"


class Package:
    def __init__(self, inputfile: str, headers: list[Header]):
        self.inputfile = inputfile
        self.headers = headers

    def get_header(self, key: str) -> Header:
        for header in self.headers:
            if header.matches(key):
                return header
        raise KeyError(f"{key} not found in package for {self.inputfile}")

    def tojson(self) -> dict[str, Any]:
        return {
            "path": self.inputfile,
            "components": [header.tojson() for header in self.headers],
        }


def read_binfile(header: Header) -> np.ndarray:
    """
    Read the file into an row-major array structured with nsamples rows and size columns,
    each column being an extent-tuple of floatsize values.
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
                f"can't handle type float{header.floatsize} parsing {header.binpath}"
            )

    # datatype is that each sample row is a matrix of size rows by extent columns
    dtype = np.dtype((nptype, (header.size, header.extent)))

    # return the entire file, parsed into one array with shape (nsamples, size, extent)
    return np.fromfile(header.binpath, dtype=dtype)


def parse_json(jsonstr: str) -> Package:
    """
    Parse the json string and convert it to a Package (basically, a list of Headers).

    See parse_json_file if you have a file path.
    """
    parsed_json = json.loads(jsonstr)

    headers = [Header.fromjson(component) for component in parsed_json["components"]]
    filename = ""
    if "abc" in parsed_json:
        filename = parsed_json["abc"]
    elif "usd" in parsed_json:
        filename = parsed_json["usd"]
    return Package(filename, headers)


def parse_json_file(jsonfile: str) -> Package:
    with open(jsonfile) as f:
        data = f.read()
    return parse_json(data)


def separate_usd(usdfile: str, outdir: str, verbose: bool = False) -> Package:
    """
    Separate out a USD file and return the resulting package.

    Also writes the relevant .bin and .json files.

    Requires `pip install usd-core`
    """
    # Import here so we can use the rest without USD being installed.
    # USD doesn't yet have official type stubs as of usd-core 25.11
    from pxr import Usd, UsdGeom, Sdf  # type: ignore

    stage = Usd.Stage.Open(usdfile)

    headers: list[Header] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        attr = mesh.GetPointsAttr()
        if attr.GetVariability() != Sdf.VariabilityVarying:
            continue
        if attr.GetNumTimeSamples() == 0:
            continue
        times = attr.GetTimeSamples()
        data = np.array([attr.Get(t) for t in times])
        if verbose:
            print(f"{prim.GetPath()} {data.shape} {data.dtype}")
        filename = re.sub(r"[-./ ()]", "-", str(prim.GetPath())).strip("-")
        filepath = f"{outdir}/{filename}.bin"
        with open(filepath, "wb") as f:
            f.write(data.tobytes())
        headers.append(
            Header.fromjson(
                {
                    "path": prim.GetPath(),
                    "type": data.dtype,
                    "samples": data.shape[0],
                    "size": data.shape[1],
                    "extent": data.shape[2],
                    "bin": filepath,
                }
            )
        )
    package = Package(usdfile, headers)
    filepath = f"{outdir}/{os.path.basename(usdfile)}.json"
    with open(filepath, "w") as f:
        json.dump(package.tojson(), f)
    return package


def create_embedding(
    header: Header, quality: float = 0.999, clustersize: int = 10000, verbose=False
) -> tuple[str, str, str]:
    """
    Create an embedding for the specific property.

    Return the names of all the binary files created.
    """
    if verbose:
        print(f"reducing dimension of {header}")

    # Read the data
    data = read_binfile(header)

    if verbose:
        print(f"  read {data.size * data.itemsize} bytes")

    # Form the clusters
    cover = clustering.cluster_by_index(header.size, clustersize)
    if verbose:
        print(f"  created {cover.nsubsets} clusters")

    # TODO: parallelize the computation.
    clusters = list(clustering.slice(data, cover))
    embeddings = [
        embedding.PCAEmbedding.from_data(cluster, quality, verbose=verbose)
        for cluster in clusters
    ]

    basename = os.path.splitext(header.binpath)[0]
    headersbin = basename + ".embed-header.bin"
    projectedbin = basename + ".embed.bin"
    clusterbin = basename + ".embed-clusters.bin"

    with open(headersbin, "wb") as headerfile:
        with open(projectedbin, "wb") as projectedfile:
            for cluster, embed in zip(clusters, embeddings):
                headerfile.write(embedding.serialize(embed))
                projected = embed.project(cluster)
                projectedfile.write(projected.tobytes())
            projectedfilesize = projectedfile.tell()
        headerfilesize = headerfile.tell()
    with open(clusterbin, "wb") as clusterfile:
        clusterfile.write(cover.tobytes())
        clusterfilesize = clusterfile.tell()
    if verbose:
        print(f"  wrote {clusterfilesize + headerfilesize + projectedfilesize} bytes")

    return (headersbin, projectedbin, clusterbin)


def read_embedding(
    header: Header, files: tuple[str, str, str], verbose: bool = False
) -> np.ndarray:
    """
    Read an embedding and output the reconstructed data.
    """
    (headersbin, projectedbin, clusterbin) = files

    # Read the clusters
    with open(clusterbin, "rb") as f:
        (cover, _) = clustering.Covering.from_bytes(f.read())

    # Read the headers and projections, start offsets at zero for the future loop.
    with open(headersbin, "rb") as f:
        headeroff = 0
        headerbytes = f.read()
    with open(projectedbin, "rb") as f:
        projectedoff = 0
        projectedbytes = f.read()

    # Read the embedded data and invert it back to the original domain.
    slices: list[np.ndarray] = []
    for _ in range(cover.nsubsets):
        (embed, headeroff) = embedding.deserialize(headerbytes, headeroff)
        (projected, projectedoff) = embed.read_projection(projectedbytes, projectedoff)
        slices.append(embed.invert(projected))

    # The data is now sliced up, unslice it.
    return clustering.unslice(slices, cover, header.extent)
