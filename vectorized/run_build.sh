#!/bin/bash

if [ ! -d "build" ]; then
  mkdir build
fi

cd build

if [ -d CMakeCache.txt ]; then
  rm CMakeCache.txt
fi

if [ -d CMakeFiles ]; then
  rm -rf CMakeFiles
fi

cmake -DCMAKE_C_COMPILER=`which gcc` \
   -DCMAKE_CXX_COMPILER=`which g++` \
   -DCMAKE_BUILD_TYPE=Release \
   ../ 2>&1 | tee ../build.log

make -j `nproc` 2>&1 | tee -a ../build.log
