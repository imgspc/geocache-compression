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
from embedding import embedding
from embedding import clustering

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
        if os.path.basename(name) == os.path.basename(name):
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

    def __str__(self) -> str:
        return f"{self.extent} float{self.floatsize}_t per point, {self.size} points, {self.nsamples} samples"


class Package:
    def __init__(self, inputfile: str, headers: list[Header]):
        self.inputfile = inputfile
        self.headers = headers

    def get_header(self, key: str) -> Header:
        for header in self.headers:
            if header.matches(key):
                return header
        raise KeyError(f"{key} not found in package for {self.inputfile}")


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

    def make_header(component) -> Header:
        float_type = component["type"]
        if isinstance(float_type, int):
            floatsize = float_type
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
            objpath=component["path"],
            floatsize=floatsize,
            extent=component["extent"],
            size=component["size"],
            nsamples=component["samples"],
            binpath=component["bin"],
        )

    headers = [make_header(component) for component in parsed_json["components"]]
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


def separate_usd(usdfile: str, outdir: str) -> Package:
    """
    Separate out a USD file and return the resulting package.

    Also writes the relevant .bin and .json files.

    Requires `pip install usd-core`
    """
    # Import here so we can use the rest without USD being installed.
    # USD doesn't yet have official type stubs as of usd-core 25.11
    from pxr import Usd, UsdGeom, Sdf  # type: ignore

    stage = Usd.Stage.Open(usdfile)

    components = []
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
        print(f"{prim.GetPath()} {data.shape} {data.dtype}")
        filename = re.sub(r"[-./ ()]", "-", str(prim.GetPath())).strip("-")
        filepath = f"{outdir}/{filename}.bin"
        print(filepath)
        with open(filepath, "wb") as f:
            f.write(data.tobytes())
        components.append(
            {
                "path": str(prim.GetPath()),
                "type": str(data.dtype),
                "samples": int(data.shape[0]),
                "size": int(data.shape[1]),
                "extent": int(data.shape[2]),
                "bin": filepath,
            }
        )
    jsondata = {
        "usd": usdfile,
        "components": components,
    }
    filepath = f"{outdir}/{os.path.basename(usdfile)}.json"
    jsonstr = json.dumps(jsondata)
    with open(filepath, "w") as f:
        f.write(jsonstr)
    return parse_json(jsonstr)
