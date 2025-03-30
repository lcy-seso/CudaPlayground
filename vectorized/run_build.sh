#!/bin/bash

set -e

if [ ! -d _build ]; then
    mkdir _build || {
        echo "Failed to create _build directory"
        exit 1
    }
fi

cd _build

if [ -f vec_test ]; then
    rm vec_test
fi

# Clean previous build artifacts
if [ -d CMakeFiles ]; then
    rm -rf CMakeFiles
fi

if [ -f CMakeCache.txt ]; then
    rm -f CMakeCache.txt
fi

cmake ..

make -j32

if [ -f vec_test ]; then
    echo "build success"
    ./vec_test >../run.log 2>&1

    RUN_STATUS=$?
    cat ../run.log
    if [ $RUN_STATUS -ne 0 ]; then
        echo "Program execution failed"
        exit 1
    fi
else
    echo "build failed"
    exit 1
fi

cd ../
