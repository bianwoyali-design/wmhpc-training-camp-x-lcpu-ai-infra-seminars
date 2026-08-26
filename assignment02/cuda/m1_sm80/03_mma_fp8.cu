#include "../common.h"
#include <cstdlib>
#include <cuda_fp8.h>
#include <random>

__global__ void mma(const __nv_fp8_e4m3 *A, const __nv_fp8_e4m3 *B, float *D) {
  const int laneid = threadIdx.x & 31;
  const int groupID = laneid >> 2;
  const int threadID_in_group = laneid & 0x3;

  unsigned a[4];

#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int row = groupID + (8 * (r & 0x1));
    const int col = (threadID_in_group * 4) + (16 * (r >> 1));
    a[r] = *reinterpret_cast<const uint32_t *>(&A[row * 32 + col]);
  }

  unsigned b[2];

#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int row = (threadID_in_group * 4) + (16 * r);
    const int col = groupID;
    b[r] = *reinterpret_cast<const uint32_t *>(&B[col * 32 + row]);
  }

  float c[4] = {0.f, 0.f, 0.f, 0.f}, d[4];
  asm volatile(
      R"(mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};
)"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));

  D[groupID * 8 + threadID_in_group * 2] = d[0];
  D[groupID * 8 + threadID_in_group * 2 + 1] = d[1];
  D[(groupID + 8) * 8 + threadID_in_group * 2] = d[2];
  D[(groupID + 8) * 8 + threadID_in_group * 2 + 1] = d[3];
}

int main(int argc, char *argv[]) {
  unsigned seed =
      argc > 1 ? static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10)) : 1u;

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 15);
  __nv_fp8_e4m3 hA[16 * 32], hB[8 * 32];
  float fA[16 * 32], fB[8 * 32], ref[16 * 8] = {};
  for (int i = 0; i < 16 * 32; ++i) {
    auto v = __nv_fp8_e4m3((float)(dist(rng) - 8));
    hA[i] = v;
    fA[i] = float(v);
  }
  for (int i = 0; i < 8 * 32; ++i) {
    auto v = __nv_fp8_e4m3((float)(dist(rng) - 8));
    hB[i] = v;
    fB[i] = float(v);
  }
  for (int r = 0; r < 16; ++r) {
    for (int n = 0; n < 8; ++n) {
      for (int k = 0; k < 32; ++k) {
        ref[r * 8 + n] += fA[r * 32 + k] * fB[n * 32 + k];
      }
    }
  }

  __nv_fp8_e4m3 *dA, *dB;
  float *dD;
  CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
  CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
  CUDA_CHECK(cudaMalloc(&dD, 16 * 8 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
  mma<<<1, 32>>>(dA, dB, dD);
  CUDA_CHECK_KERNEL();
  float got[16 * 8];
  CUDA_CHECK(cudaMemcpy(got, dD, sizeof(got), cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int i = 0; i < 16 * 8; ++i) {
    bad += got[i] != ref[i];
  }
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dD);

  printf("%s\n", !bad ? "PASS" : "MISMATCH");
  return 0;
}