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
   ../

make -j `nproc` 2>&1 | tee build.log
