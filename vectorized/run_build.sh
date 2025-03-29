#!/bin/bash

set -e # Exit on error

# Create build directory if it doesn't exist
if [ ! -d _build ]; then
    mkdir _build || {
        echo "Failed to create _build directory"
        exit 1
    }
fi

# Clean previous build artifacts
if [ -d _build/CMakeFiles ]; then
    rm -rf _build/CMakeFiles
fi

if [ -f _build/CMakeCache.txt ]; then
    rm -f _build/CMakeCache.txt
fi

cd _build

cmake ..

make

cd ../

# Run the executable and capture output
./_build/vec_test >run.log 2>&1
RUN_STATUS=$?
cat run.log
if [ $RUN_STATUS -ne 0 ]; then
    echo "Program execution failed"
    exit 1
fi
