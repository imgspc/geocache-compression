#! /bin/sh

find src -name \*.cpp | xargs clang-format -i --verbose

python -m black src

python -m mypy src
