#! /bin/sh

cd `dirname "$0"`

find src -name \*.cpp | xargs clang-format -i --verbose

python -m black src

python -m mypy src
