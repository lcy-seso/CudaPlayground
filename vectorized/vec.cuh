#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

typedef __nv_bfloat16 __bfloat16;
typedef __nv_bfloat162 __bfloat162;

struct __align__(8) __bfloat4 {  // packed 64-bit data
  __bfloat162 x, y;

  // Constructor
  __forceinline__ __host__ __device__ __bfloat4() : x(), y() {}

  // Constructor from 4 bfloat16 values
  __forceinline__ __host__ __device__ __bfloat4(__bfloat162 x_, __bfloat162 y_)
      : x(x_), y(y_) {}

  // Add constructor that takes 4 individual bfloat16 values
  __forceinline__ __host__ __device__ __bfloat4(__bfloat16 a, __bfloat16 b,
                                                __bfloat16 c, __bfloat16 d)
      : x(make_bfloat162(a, b)), y(make_bfloat162(c, d)) {}

  // Constructor from single bfloat16 value (broadcasts to all elements)
  __forceinline__ __host__ __device__ __bfloat4(__nv_bfloat162 v)
      : x(v), y(v) {}

  // Assignment operator as member function
  __forceinline__ __host__ __device__ __bfloat4& operator+=(
      const __bfloat4& b) {
    x = __hadd2(x, b.x);
    y = __hadd2(y, b.y);
    return *this;
  }
};

// Binary operators as non-member functions
__forceinline__ __host__ __device__ __bfloat4 operator+(const __bfloat4& a,
                                                        const __bfloat4& b) {
  __bfloat4 res;
  res.x = __hadd2(a.x, b.x);
  res.y = __hadd2(a.y, b.y);
  return res;
}

__forceinline__ __host__ __device__ __bfloat4 operator*(const __bfloat4& a,
                                                        const __bfloat4& b) {
  return __bfloat4(__hmul2(a.x, b.x), __hmul2(a.y, b.y));
}

__global__ void vecAdd(const __bfloat4* a, const __bfloat4* b, __bfloat4* c,
                       int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}
