#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#define CUDA_OK(call)                                                          \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e));          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float bf16(float x) {
  unsigned short bits;
  asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(x));
  return __uint_as_float(static_cast<unsigned>(bits) << 16);
}

// One lane per independent input. Small and deliberately not timed.
__global__ void measure(const float *gates, float *traces, float *pairs,
                        int n) {
  int task = blockIdx.x * blockDim.x + threadIdx.x;
  if (task >= n * 3)
    return;
  int c = 16 << (task % 3), input = task / 3;
  float s[64], d[64], u[64];
  float acc = 0;
  for (int i = 0; i < c; ++i) {
    acc = __fadd_rn(acc, __fmul_rn(gates[input * 64 + i], 0x1.715476p+0f));
    s[i] = acc;
    float de = ex2(acc), inv = ex2(-acc);
    d[i] = bf16(de);
    u[i] = bf16(inv);
    int offset = (task * 64 + i) * 5;
    traces[offset] = acc;
    traces[offset + 1] = de;
    traces[offset + 2] = inv;
    traces[offset + 3] = d[i];
    traces[offset + 4] = u[i];
  }
  for (int i = 0; i < c; ++i)
    for (int j = 0; j <= i; ++j) {
      int offset = (task * 4096 + i * 64 + j) * 2;
      pairs[offset] = bf16(ex2(__fsub_rn(s[i], s[j])));
      pairs[offset + 1] = __fmul_rn(d[i], u[j]);
    }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: cuda_range inputs.txt output_directory\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  int n = 0;
  in >> n;
  if (n <= 0 || n > 1000)
    return 2;
  std::vector<float> gates(n * 64);
  for (auto &x : gates) {
    if (!(in >> x))
      return 2;
  }
  cudaDeviceProp prop;
  CUDA_OK(cudaGetDeviceProperties(&prop, 0));
  if (prop.major != 10 || prop.minor != 3) {
    std::cerr << "Expected B300 sm_103\n";
    return 3;
  }
  int driver, runtime;
  CUDA_OK(cudaDriverGetVersion(&driver));
  CUDA_OK(cudaRuntimeGetVersion(&runtime));
  std::cout << "device=" << prop.name << " cc=" << prop.major << '.'
            << prop.minor << " SMs=" << prop.multiProcessorCount
            << " driver=" << driver << " runtime=" << runtime << '\n';
  std::vector<float> traces(n * 3 * 64 * 5), pairs(n * 3 * 4096 * 2);
  float *g, *t, *p;
  CUDA_OK(cudaMalloc(&g, gates.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&t, traces.size() * sizeof(float)));
  CUDA_OK(cudaMalloc(&p, pairs.size() * sizeof(float)));
  CUDA_OK(cudaMemcpy(g, gates.data(), gates.size() * sizeof(float),
                     cudaMemcpyHostToDevice));
  measure<<<(n * 3 + 31) / 32, 32>>>(g, t, p, n);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaMemcpy(traces.data(), t, traces.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(pairs.data(), p, pairs.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
  std::ofstream ts(std::string(argv[2]) + "/gpu_traces.csv"),
      ps(std::string(argv[2]) + "/gpu_pairs.csv");
  if (!ts || !ps)
    return 2;
  ts << "input,chunk,token,prefix_log2,exp_fp32,inverse_fp32,decay,inverse\n"
     << std::setprecision(9);
  ps << "input,chunk,i,j,direct,factorized\n" << std::setprecision(9);
  for (int task = 0; task < n * 3; ++task) {
    int c = 16 << (task % 3);
    for (int i = 0; i < c; ++i) {
      ts << task / 3 << ',' << c << ',' << i + 1;
      for (int k = 0; k < 5; ++k)
        ts << ',' << traces[(task * 64 + i) * 5 + k];
      ts << '\n';
      for (int j = 0; j <= i; ++j) {
        int off = (task * 4096 + i * 64 + j) * 2;
        ps << task / 3 << ',' << c << ',' << i + 1 << ',' << j + 1 << ','
           << pairs[off] << ',' << pairs[off + 1] << '\n';
      }
    }
  }
  CUDA_OK(cudaFree(g));
  CUDA_OK(cudaFree(t));
  CUDA_OK(cudaFree(p));
}
