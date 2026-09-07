// 问题 5.3(c):ceiling 探针。
//
// 目标:测出"5.3(b) 这种访存形状的上限带宽"。方法是写一个和你的
// quant kernel 访存完全同形(读同样的 bf16、写同样位置的 8 byte 数据
// 与 1 byte SF)、但不做任何数学的 kernel——读进来的位 xor 一下直接
// 写出去即可。它的耗时就是这个访存模式在这块卡上的地板。
//
// 跑完把三个数放在一起:探针 GB/s、你的 quant kernel GB/s(03b 的
// 输出)、两者比值。报告里回答:你的 kernel 离自己的上限还有多远,
// 差距是访存还是计算(ncu 的 SM% / DRAM% 可以佐证)。
//
// 在下面实现探针 kernel 和 launch;main 不需要修改。
#include "../common.h"
#include "nvfp4_common.h"
#include <random>
#include <vector>

template <int BLOCK>
__global__ void probe_kernel(const __nv_bfloat16 *__restrict__ in,
                             uint8_t *__restrict__ dataOut,
                             uint8_t *__restrict__ sfOut, int M, int K) {
  const int groups_per_row = K / NVFP4_GROUP;
  const int num_groups = M * groups_per_row;

  const int group_ID = blockIdx.x * BLOCK + threadIdx.x;
  if (group_ID >= num_groups) {
    return;
  }

  const int row = group_ID / groups_per_row;
  const int k_group = group_ID % groups_per_row;

  const auto *src =
      reinterpret_cast<const uint64_t *>(in + row * K + k_group * NVFP4_GROUP);

  const uint64_t x0 = src[0];
  const uint64_t x1 = src[1];
  const uint64_t x2 = src[2];
  const uint64_t x3 = src[3];

  const uint64_t x = x0 ^ x1 ^ x2 ^ x3;

  *reinterpret_cast<uint8_t *>(dataOut + row * (K / 2) + k_group * 8) = x;

  const int ktiles = nvfp4_num_ktiles(K);
  sfOut[sf_swizzled_offset(row, k_group, ktiles)] = static_cast<uint8_t>(x);
}

static void launch_probe(const __nv_bfloat16 *in, uint8_t *dataOut,
                         uint8_t *sfOut, int M, int K, int sms) {
  constexpr int BLOCK = 256;

  const int groups_per_row = K / NVFP4_GROUP;
  const int num_groups = M * groups_per_row;
  const int grid = (num_groups + BLOCK - 1) / BLOCK;

  probe_kernel<BLOCK><<<grid, BLOCK>>>(in, dataOut, sfOut, M, K);
}

int main() {
  int sms;
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  for (const auto &shape :
       {std::pair{4096, 7168}, {16384, 4096}, {16384, 8192}}) {
    int M = shape.first;
    int K = shape.second;
    size_t n = (size_t)M * K;
    __nv_bfloat16 *dx;
    uint8_t *dd, *dsf;
    CUDA_CHECK(cudaMalloc(&dx, n * 2));
    CUDA_CHECK(cudaMalloc(&dd, n / 2));
    CUDA_CHECK(cudaMalloc(&dsf, nvfp4_sf_bytes(M, K)));
    CUDA_CHECK(cudaMemset(dx, 0x3c, n * 2));
    float ms = time_avg_ms([&] { launch_probe(dx, dd, dsf, M, K, sms); }, 50);
    CUDA_CHECK_KERNEL();
    double bytes = n * 2.0 + n * 0.5 + n / 16.0;
    printf("M=%-6d K=%-5d  probe %8.2f us  %6.0f GB/s\n", M, K, ms * 1e3,
           effective_gbps(bytes, ms));
    cudaFree(dx);
    cudaFree(dd);
    cudaFree(dsf);
  }
  return 0;
}
