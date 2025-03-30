#include "cuda_timer.hpp"
#include "kernels.cuh"
#include "vec.cuh"

#include <stdio.h>

template <const int kN, const unsigned int kThreads>
float test_vec_add1(int warmup = 10, int repeat = 500) {
  using DType = __bfloat16;

  DType *d_a, *d_b, *d_c;
  cudaError_t err;

  err = cudaMalloc(&d_a, kN * sizeof(DType));
  if (err != cudaSuccess) {
    printf("cudaMalloc failed for d_a: %s\n", cudaGetErrorString(err));
    return -1;
  }

  err = cudaMalloc(&d_b, kN * sizeof(DType));
  if (err != cudaSuccess) {
    printf("cudaMalloc failed for d_b: %s\n", cudaGetErrorString(err));
    return -1;
  }

  err = cudaMalloc(&d_c, kN * sizeof(DType));
  if (err != cudaSuccess) {
    printf("cudaMalloc failed for d_c: %s\n", cudaGetErrorString(err));
    return -1;
  }

  unsigned int blocks = kN / kThreads;
  dim3 threads{kThreads, 1, 1};
  dim3 grid{blocks, 1, 1};

  init_data<<<grid, threads>>>(d_a, d_b, d_c, 1.5, 2.3, 0.0, kN);
  cudaDeviceSynchronize();

  for (int i = 0; i < warmup; ++i)
    vec_add<DType><<<grid, threads>>>(d_a, d_b, d_c, kN);
  cudaDeviceSynchronize();

  CudaTimer timer;
  timer.start();
  for (int i = 0; i < repeat; ++i) {
    vec_add<DType><<<grid, threads>>>(d_a, d_b, d_c, kN);
  }
  cudaDeviceSynchronize();
  float time = timer.stop() / repeat;

  {
#ifdef DEBUG
    dim3 threads{1, 1, 1};
    dim3 grid{1, 1, 1};
    debug_print_bfloat16<<<grid, threads>>>(d_c, kN);
    cudaDeviceSynchronize();
#endif
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return time;
}

template <const int kN, const unsigned int kThreads>
float test_vec_add4(int warmup = 10, int repeat = 500) {
  using DType = __bfloat164;

  DType *d_a, *d_b, *d_c;
  cudaError_t err;

  int numel = kN / 4;
  err = cudaMalloc(&d_a, numel * sizeof(DType));
  if (err != cudaSuccess) {
    printf("cudaMalloc failed for d_a: %s\n", cudaGetErrorString(err));
    return -1;
  }

  err = cudaMalloc(&d_b, numel * sizeof(DType));
  if (err != cudaSuccess) {
    printf("cudaMalloc failed for d_b: %s\n", cudaGetErrorString(err));
    return -1;
  }

  err = cudaMalloc(&d_c, numel * sizeof(DType));
  if (err != cudaSuccess) {
    printf("cudaMalloc failed for d_c: %s\n", cudaGetErrorString(err));
    return -1;
  }

  unsigned int blocks = kN / (4 * kThreads);
  dim3 threads{kThreads, 1, 1};
  dim3 grid{blocks, 1, 1};

  init_data<<<grid, threads>>>(d_a, d_b, d_c, 1.5, 2.3, 0.0, numel);
  cudaDeviceSynchronize();

  for (int i = 0; i < warmup; ++i)
    vec_add<DType><<<grid, threads>>>(d_a, d_b, d_c, numel);
  cudaDeviceSynchronize();

  CudaTimer timer;
  timer.start();
  for (int i = 0; i < repeat; ++i) {
    vec_add<DType><<<grid, threads>>>(d_a, d_b, d_c, numel);
  }
  cudaDeviceSynchronize();
  float time = timer.stop() / repeat;

  {
#ifdef DEBUG
    dim3 threads{1, 1, 1};
    dim3 grid{1, 1, 1};
    debug_print_bfloat164<<<grid, threads>>>(d_c, numel);
    cudaDeviceSynchronize();
#endif
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return time;
}

template <const int kN, const unsigned int kThreads>
float test_vec_add8(int warmup = 10, int repeat = 500) {
  using DType = __bfloat168;

  DType *d_a, *d_b, *d_c;
  cudaError_t err;

  int numel = kN / 8;  // Each __bfloat168 contains 8 elements
  err = cudaMalloc(&d_a, numel * sizeof(DType));
  if (err != cudaSuccess) {
    printf("cudaMalloc failed for d_a: %s\n", cudaGetErrorString(err));
    return -1;
  }

  err = cudaMalloc(&d_b, numel * sizeof(DType));
  if (err != cudaSuccess) {
    printf("cudaMalloc failed for d_b: %s\n", cudaGetErrorString(err));
    return -1;
  }

  err = cudaMalloc(&d_c, numel * sizeof(DType));
  if (err != cudaSuccess) {
    printf("cudaMalloc failed for d_c: %s\n", cudaGetErrorString(err));
    return -1;
  }

  dim3 threads{kThreads, 1, 1};
  int blocks = kN / (8 * kThreads);
  dim3 grid{(unsigned int)blocks, 1, 1};

  init_data<<<grid, threads>>>(d_a, d_b, d_c, 1.5f, 2.3f, 0.0f, numel);
  cudaDeviceSynchronize();

  for (int i = 0; i < warmup; ++i)
    vec_add<DType><<<grid, threads>>>(d_a, d_b, d_c, numel);
  cudaDeviceSynchronize();

  CudaTimer timer;
  timer.start();
  for (int i = 0; i < repeat; ++i) {
    vec_add<DType><<<grid, threads>>>(d_a, d_b, d_c, numel);
  }
  cudaDeviceSynchronize();
  float time = timer.stop() / repeat;

  {
#ifdef DEBUG
    dim3 threads{1, 1, 1};
    dim3 grid{1, 1, 1};
    debug_print_bfloat168<<<grid, threads>>>(d_c, numel);
    cudaDeviceSynchronize();
#endif
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return time;
}

int main() {
  float time1 = test_vec_add1<10240, 256 /* threads */>();
  float time2 = test_vec_add4<10240, 256 /* threads */>();
  float time3 = test_vec_add8<10240, 256 /* threads */>();
  printf("time1: %f, time2: %f, time3: %f\n", time1, time2, time3);
  return 0;
}
