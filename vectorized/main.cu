
#include "vec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main() {
  using DType = __bfloat16;  // bfloat16 is not defined in the host
  const int kN = 256;

  // Host vectors for float4 (to match __bfloat4 structure)
  thrust::host_vector<float4> h_a(kN);
  thrust::host_vector<float4> h_b(kN);

  // Initialize with some values
  for (int i = 0; i < kN; ++i) {
    h_a[i] = make_float4(i, i + 1, i + 2, i + 3);
    h_b[i] = make_float4(2 * i, 2 * (i + 1), 2 * (i + 2), 2 * (i + 3));
  }

  // Device vectors for __bfloat4
  thrust::device_vector<__bfloat4> d_a(kN);
  thrust::device_vector<__bfloat4> d_b(kN);
  thrust::device_vector<__bfloat4> d_c(kN);

  // Custom kernel to convert and copy data
  auto convert_kernel = [=] __device__(const float4& f) {
    return __bfloat4(__float2bfloat16(f.x), __float2bfloat16(f.y),
                     __float2bfloat16(f.z), __float2bfloat16(f.w));
  };

  // Convert and copy data
  thrust::transform(h_a.begin(), h_a.end(), d_a.begin(), convert_kernel);
  thrust::transform(h_b.begin(), h_b.end(), d_b.begin(), convert_kernel);

  dim3 threads{32, 1, 1};
  dim3 grid{1, 1, 1};

  vecAdd<<<grid, threads>>>(thrust::raw_pointer_cast(d_a.data()),
                            thrust::raw_pointer_cast(d_b.data()),
                            thrust::raw_pointer_cast(d_c.data()), kN);

  return 0;
}
