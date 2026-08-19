#!/bin/bash
set -e

# pass --valgrind to check for memory errors and leaks
RUNNER=""
if [ "$1" = "--valgrind" ]; then
    RUNNER="valgrind --leak-check=full --error-exitcode=1"
fi

mkdir -p build
cd build
cmake ..
make VERBOSE=1
$RUNNER ./pure_cpp
