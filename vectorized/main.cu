#include "kernels.cuh"
#include "vec.cuh"

#include <stdio.h>

int test_vec_add4() {
  using DType = __bfloat164;
  const int kN = 128;  // each thread process a single bfloat164

  // Allocate device memory
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

  dim3 threads{128, 1, 1};
  dim3 grid{1, 1, 1};
  init_data<<<grid, threads>>>(d_a, d_b, d_c, 1.5, 2.3, 0.0, kN);

  dim3 threads2{1, 1, 1};
  dim3 grid2{1, 1, 1};

#if defined(DEBUG)
  printf("d_a:\n");
  debug_print_bfloat164<<<grid2, threads2>>>(d_a, kN);
  cudaDeviceSynchronize();

  printf("d_b:\n");
  debug_print_bfloat164<<<grid2, threads2>>>(d_b, kN);
  cudaDeviceSynchronize();
#endif

  vec_add4<<<grid, threads>>>(d_a, d_b, d_c, kN);
  cudaDeviceSynchronize();

  printf("results:\n");
  debug_print_bfloat164<<<grid2, threads2>>>(d_c, kN);
  cudaDeviceSynchronize();

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}

int test_vec_add8() {
  using DType = __bfloat168;
  const int kN = 128;  // each thread process a single bfloat168

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

  dim3 threads{128, 1, 1};
  dim3 grid{1, 1, 1};

  init_data<<<grid, threads>>>(d_a, d_b, d_c, 1.5f, 2.3f, 0.0f, kN);

  dim3 threads2{1, 1, 1};
  dim3 grid2{1, 1, 1};

#if defined(DEBUG)
  printf("d_a:\n");
  debug_print_bfloat164<<<grid2, threads2>>>(d_a, kN);
  cudaDeviceSynchronize();

  printf("d_b:\n");
  debug_print_bfloat164<<<grid2, threads2>>>(d_b, kN);
  cudaDeviceSynchronize();
#endif

  vec_add8<<<grid, threads>>>(d_a, d_b, d_c, kN);
  cudaDeviceSynchronize();

  printf("d_c:\n");
  debug_print_bfloat168<<<grid2, threads2>>>(d_c, kN);
  cudaDeviceSynchronize();

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}

int main() {
  test_vec_add4();
  test_vec_add8();
  return 0;
}
