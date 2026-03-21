#! /bin/sh
set -e

cd `dirname "$0"`

# Reformat C++
find src -name \*.cpp -print0 | xargs -0 clang-format -i --verbose

# Strip output cells from jupyter notebooks
find src -name .ipynb_checkpoints -prune -o -name \*.ipynb -print0 | xargs -0 python -m nbstripout

# Reformat python and jupyter notebooks
python -m black src

# Type-check python (but not jupyter)
python -m mypy src

# Type-check jupyter
find src -name .ipynb_checkpoints -prune -o -name \*.ipynb -print0 | xargs -0 python -m nbqa mypy

if [ x"$1" == x ] ; then
    cd src
    python -m unittest discover -v -s embedding/tests
fi
