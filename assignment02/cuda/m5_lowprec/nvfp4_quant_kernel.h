// 问题 5.3(b):实现 NVFP4 quant kernel。
//
// 输入 bf16 矩阵 [M, K](K 是 16 的倍数),输出:
//   dataOut:e2m1 打包数据,每行 K/2 byte,低 nibble 放偶数下标元素
//   sfOut:  e4m3 SF,swizzled 布局(偏移用 nvfp4_common.h 的
//           sf_swizzled_offset;整个 SF 张量已在调用前清零)
//
// 每组的计算顺序(判测按同一顺序生成真值,逐 byte 严格相等):
//   amax = 组内 16 个值的绝对值最大
//   sf8  = __nv_fp8_e4m3(amax / 6.0f)
//   sf   = float(sf8)
//   inv  = sf != 0 ? 1.0f / sf : 0.0f
//   nibble[i] = encode(v[i] * inv)
//
// 设备侧的 e2m1 转换直接用 cuda_fp4.h 的 __nv_fp4x2_e2m1(float2 的 .x
// 进低 nibble),它在 sm_100 家族上是单条硬件指令;你在 5.3(a) 写的
// 编码器语义与它一致,host 参考用的就是它。
//
// 组织建议:16 元素 = 32 byte,一个线程恰好负责一个组,天然免掉组内
// 线程协作;quant 没有行间依赖,grid 怎么铺完全自由。
#pragma once
#include "nvfp4_common.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

template <int BLOCK>
__global__ void nvfp4_quant_kernel(const __nv_bfloat16 *__restrict__ in,
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

  const int k0 = k_group * NVFP4_GROUP;
  const int in_base = row * K + k0;

  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < NVFP4_GROUP; ++i) {
    const float v = __bfloat162float(in[in_base + i]);
    amax = std::fmax(amax, std::fabs(v));
  }

  const __nv_fp8_e4m3 sf8(amax / 6.0f);
  const float sf = static_cast<float>(sf8);
  const float inv = sf != 0 ? 1.0f / sf : 0.0f;

  const int ktiles = nvfp4_num_ktiles(K);
  sfOut[sf_swizzled_offset(row, k_group, ktiles)] = sf8.__x;

  const int out_base = row * (K / 2) + k_group * (NVFP4_GROUP / 2);

#pragma unroll
  for (int i = 0; i < NVFP4_GROUP; i += 2) {
    const float x = __bfloat162float(in[in_base + i]) * inv;
    const float y = __bfloat162float(in[in_base + i + 1]) * inv;
    const __nv_fp4x2_e2m1 q(make_float2(x, y));
    dataOut[out_base + i / 2] = q.__x;
  }
}

// 判测和 5.4 会按这个签名调用;grid 大小你自己定,写在这里。
inline void launch_nvfp4_quant(const __nv_bfloat16 *in, uint8_t *dataOut,
                               uint8_t *sfOut, int M, int K, int sms) {
  constexpr int BLOCK = 256;

  const int groups_per_row = K / NVFP4_GROUP;
  const int num_groups = M * groups_per_row;
  const int grid = (num_groups + BLOCK - 1) / BLOCK;

  nvfp4_quant_kernel<BLOCK><<<grid, BLOCK>>>(in, dataOut, sfOut, M, K);
}
