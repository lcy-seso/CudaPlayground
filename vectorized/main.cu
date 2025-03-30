#include "kernels.cuh"
#include "vec.cuh"

#include <stdio.h>

template <const int kN, const unsigned int kThreads>
int test_vec_add4() {
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

  vec_add4<<<grid, threads>>>(d_a, d_b, d_c, numel);
  cudaDeviceSynchronize();

  {  // debug print
    dim3 threads{1, 1, 1};
    dim3 grid{1, 1, 1};
    debug_print_bfloat164<<<grid, threads>>>(d_c, numel);
    cudaDeviceSynchronize();
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}

template <const int kN, const unsigned int kThreads>
int test_vec_add8() {
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

  vec_add8<<<grid, threads>>>(d_a, d_b, d_c, numel);
  cudaDeviceSynchronize();

  {  // debug print
    dim3 threads{1, 1, 1};
    dim3 grid{1, 1, 1};
    printf("d_c:\n");
    debug_print_bfloat168<<<grid, threads>>>(d_c, numel);
    cudaDeviceSynchronize();
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}

int main() {
  test_vec_add4<1024, 128 /* threads */>();
  // test_vec_add8<12800, 128 /* threads */>();
  return 0;
}
