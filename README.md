# Geocache Compression

This is a research project aiming to explore how to compress animated geocache
files, e.g. Alembic files.

The overall idea is:
* Each coordinate is a time series, so use time series compression. E.g. Guerra et all 2025: https://arxiv.org/pdf/2412.16266
** Need to experiment with how to deal with positions versus normals, trying to cluster rather than seeing each coordinate of each vertex as independent, etc.
* Use the Academy Software Foundation's DPEL for sample data (and also the octopus).

The target is SIGGRAPH 2026.

## Building

First, install [Alembic 1.8.8](https://github.com/alembic/alembic) (or probably most any other version). I installed
it in `~/projects/alembic-install` but it doesn't matter, you just have to know
where you put it.

```
export ALEMBIC_ROOT=$HOME/projects/alembic-install
mkdir build
mkdir install
cd build
cmake ../src -DCMAKE_INSTALL_PREFIX=`pwd`/../install
make install
```

## Testing

Good idea for later.

## Running

Download `Alembic_Octopus_Example.tgz` from the [google code downloads page](https://code.google.com/archive/p/alembic/downloads) into your downloads folder and unzip it in place. To use it, you need to have built Alembic with HDF5 support (which is not the default), since the example is ancient.

Full loop. Run this from the build directory (so you can clean it out easily).
```
cd build
../install/bin/abc-parse ~/Downloads/Alembic_Octopus_Example/alembic_octopus.abc > octopus.json
python ../install/bin/abc-separate.py --verbose octopus.json
python ../install/bin/abc-combine.py --verbose octopus.json out.abc
../install/bin/abc-compare --verbose ~/Downloads/Alembic_Octopus_Example/alembic_octopus.abc out.abc
```

That parses an .abc file, separates out all the properties into their own .bin files, combines them, and compares the results.

## Links

* Alembic discussion group: https://groups.google.com/g/alembic-discussion
* Alembic google code downloads page (for the octopus): https://code.google.com/archive/p/alembic/downloads
* DPEL: https://dpel.aswf.io/
