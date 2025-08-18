# Geocache Compression

This is a research project aiming to explore how to compress animated geocache
files, e.g. Alembic files.

The overall idea is:
* Each coordinate is a time series, so use time series compression. E.g. Guerra et all 2025: https://arxiv.org/pdf/2412.16266
** Need to experiment with how to deal with positions versus normals, trying to cluster rather than seeing each coordinate of each vertex as independent, etc.
* Use the Academy Software Foundation's DPEL for sample data (and also the octopus).

The target is SIGGRAPH 2026.
