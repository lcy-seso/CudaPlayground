#pragma once

#include "vec.cuh"

__global__ void vec_add4(const __bfloat164* a, const __bfloat164* b,
                         __bfloat164* c, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

__global__ void vec_add8(const __bfloat168* a, const __bfloat168* b,
                         __bfloat168* c, int n) {
  int i = 8 * (threadIdx.x + blockDim.x * blockIdx.x);
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

/// helper functions
template <typename DType>
__global__ void init_data(DType* a, DType* b, DType* c, __bfloat16 v1,
                          __bfloat16 v2, __bfloat16 v3, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    a[i] = v1;
    b[i] = v2;
    c[i] = v3;
  }
}

__global__ void debug_print_bfloat164(const __bfloat164* a, int n) {
  for (int i = 0; i < n; ++i) {
    print_bfloat164(a[i]);
  }
}

__global__ void debug_print_bfloat168(const __bfloat168* a, int n) {
  for (int i = 0; i < n; ++i) {
    print_bfloat168(a[i]);
  }
}
