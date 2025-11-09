# Geocache Compression

This is a research project aiming to explore how to compress animated geocache
files, e.g. Alembic files or USD vertex caches.

The overall idea is:
* Each coordinate is a time series, so use time series compression. E.g. Guerra et all 2025: https://arxiv.org/pdf/2412.16266
* Figure out how to move to high dimension (hundreds of thousands) but short length (tens to hundreds of samples), rather than 1-d and long timeseries.
* In particular, figure out how to reduce dimension, e.g. with clustering and dimensionality reduction techniques.
* Use the Academy Software Foundation's DPEL and the alembic octopus for sample data

The target is SIGGRAPH 2026 or maybe eurographics.

## USD

Initial tests are with the [ALab](https://github.com/DigitalProductionExampleLibrary/ALab) project. Note: the script to install everything doesn't seem to work on Windows. Works fine on macos and likely on linux.

Open the [Jupyter notebook](https://github.com/imgspc/geocache-compression/blob/main/src/usd-separate.ipynb) and follow the directions in the first cell.

### Running for USD

After following the instructions in the Jupyter notebook, you'll be in Jupyter. Hit Shift-enter to execute each of the cells and see what happens.

Then play around.

Restart the kernel when you change the code in src/embedding.

## Alembic

Initial tests are with the ancient `Alembic_Octopus_Example.tgz` from the [google code downloads page](https://code.google.com/archive/p/alembic/downloads).

First, install [Alembic 1.8.8](https://github.com/alembic/alembic) (or probably most any other version). Note: you must install it with HDF5 support, and you must install the ILM math program (which gives float16 support). On macos, I was able to `brew install hdf5 imath` and let the cmake files find it all.

Then to build geocache-compression:
```
export ALEMBIC_ROOT=$HOME/projects/alembic-install
mkdir build
mkdir install
cd build
cmake ../src -DCMAKE_INSTALL_PREFIX=`pwd`/../install
make install
```

### Running Alembic

Run this from the build directory of this (so you can clean it out easily).
```
cd build
../install/bin/abc-parse ~/Downloads/Alembic_Octopus_Example/alembic_octopus.abc > octopus.json
python ../install/bin/abc-separate.py --verbose octopus.json
python ../install/bin/abc-combine.py --verbose octopus.json out.abc
../install/bin/abc-compare --verbose ~/Downloads/Alembic_Octopus_Example/alembic_octopus.abc out.abc
```

That parses an .abc file, separates out all the properties into their own .bin files, combines them, and compares the results.

## Developing

We use clang-format and black for formatting; and mypy for type checking the Python code.
```
pip install mypy black
sh pre-submit.sh
```

Pre-submit reformats all the C++ and Python code it can find, and typechecks Python.

## Testing

Good idea for later.

## Links

* Alembic discussion group: https://groups.google.com/g/alembic-discussion
* Alembic google code downloads page (for the octopus): https://code.google.com/archive/p/alembic/downloads
* DPEL: https://dpel.aswf.io/
