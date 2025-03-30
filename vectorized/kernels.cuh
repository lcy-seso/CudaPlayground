#pragma once

#include "vec.cuh"

template <typename DType>
__global__ void vec_add(const DType* a, const DType* b, DType* c, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
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

__global__ void debug_print_bfloat16(const __bfloat16* a, int n) {
  for (int i = 0; i < n; ++i) {
    printf("%.2f\n", __bfloat162float(a[i]));
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
