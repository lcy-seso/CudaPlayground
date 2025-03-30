#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

inline void __cudaCheck(const cudaError err, const char* file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s(%d): CUDA error: %s.\n", file, line,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
#define CudaCheck(call) __cudaCheck(call, __FILE__, __LINE__)
