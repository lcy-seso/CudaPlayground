#!/bin/bash
set -o pipefail

if [ ! -d "build" ]; then
  mkdir build
fi

cd build || exit 1

if [ -d CMakeCache.txt ]; then
  rm CMakeCache.txt
fi

if [ -d CMakeFiles ]; then
  rm -rf CMakeFiles
fi

# Store compiler paths before using them
C_COMPILER="$(command -v gcc)"
CXX_COMPILER="$(command -v g++)"

# Preserve exit code with pipefail
cmake -DCMAKE_C_COMPILER="${C_COMPILER}" \
  -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
  -DCMAKE_BUILD_TYPE=Release \
  ../ 2>&1 | tee ../build.log

CMAKE_EXIT=$?
if [ $CMAKE_EXIT -ne 0 ]; then
  echo "CMake configuration failed with exit code $CMAKE_EXIT"
  exit $CMAKE_EXIT
fi

# Get the number of processors first to avoid masking return value
NUM_PROCS=$(nproc)
if [ $? -ne 0 ]; then
  echo "Failed to determine number of processors with nproc"
  NUM_PROCS=1
fi

# Preserve exit code with variable
make -j "$NUM_PROCS" 2>&1 | tee -a ../build.log
MAKE_EXIT=$?
if [ $MAKE_EXIT -ne 0 ]; then
  echo "Make build failed with exit code $MAKE_EXIT"
  exit $MAKE_EXIT
fi
