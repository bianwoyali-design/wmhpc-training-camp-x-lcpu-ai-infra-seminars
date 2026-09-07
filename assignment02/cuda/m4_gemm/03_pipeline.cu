// 问题 4.3(FROM-SCRATCH,模块压轴):多级缓冲流水。
//
// 从你自己的 02_tma.cu 出发,把单缓冲扩成 STAGES 级循环缓冲:TMA 往
// 前预取后续 K 段,mma 消费当前段,装载与计算重叠。STAGES 是编译参数:
//   STAGES=4 make -B run/m4_gemm/03_pipeline
// (-B 不能省:只改 -D 不改文件,make 会认为无需重编。)
//
// 明确不要求:warp specialization、persistent kernel、epilogue 融合。
// 不设达成率门槛,评分看实验与归因质量。
//
// 两个已知事实,直接告知:
//   1. smem 用量 = STAGES*(BM+BN)*BK*2,STAGES>=3 起超过 48KB 静态
//      上限,必须动态 smem + cudaFuncSetAttribute(main 已配好)。
//   2. 一条真实的流水线 hazard(我们开发答案时踩到的,写出来让你避开):
//      "机会式预取"(try_wait 非阻塞,空了就发)不能替代"强制发射"。
//      若本轮要消费的那段 TMA 在早先检查时 stage 未空而被跳过,后面
//      wait full 等的就是一条从未发出的拷贝——死锁。症状签名很典型:
//      1024^3 侥幸全过,4096^3 必挂(13 万次机会必中一次)。正确结构:
//      本轮要消费的 TMA 用阻塞等 empty 保证发出,机会式 try_wait 只
//      用于更深的预取。另外 empty mbarrier 必须每 stage 一个:单个
//      mbar 的 parity 区分不了相隔 2 轮的完成,STAGES>=2 必然歧义。
//
// 交付:
//   - 梯子表第三行(4096^3,默认 STAGES=3)
//   - stages 扫描表:S ∈ {2,3,4,6},在两个形状上各扫一遍——4096^3 与
//     M=256 N=4096 K=16384(小 grid、长 K)。两张表的 S 敏感度不一样,
//     解释差异来自什么(提示方向:每 SM 常驻 block 数怎么随 smem 用量
//     变、块间并发本身能隐藏多少延迟)。./sweep_stages.sh 会跑全表
//   - 流水时空图:任选一个 S,画出稳态下 TMA/mma 在各 stage 上的重叠
//   - handout 4.3 的三问:瓶颈移动;梯子表逐级归因(含 assignment01
//     的 naive matmul 同口径对照);smem 与 TMEM 谁先顶住扩 stage/tile
//
// 运行:make run/m4_gemm/03_pipeline;./bin/m4_gemm/03_pipeline M N K
#include "../common.h"
#include <cstdint>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <random>
#include <vector>

#ifndef STAGES
#define STAGES 3
#endif

#define CUDA_DRIVER_CHECK(call)                                                \
  do {                                                                         \
    CUresult err_ = (call);                                                    \
    if (err_ != CUDA_SUCCESS) {                                                \
      const char *name_ = nullptr;                                             \
      const char *message_ = nullptr;                                          \
      cuGetErrorName(err_, &name_);                                            \
      cuGetErrorString(err_, &message_);                                       \
      fprintf(stderr, "CUDA Driver error %s at %s:%d: %s\n",                   \
              name_ ? name_ : "unknown", __FILE__, __LINE__,                   \
              message_ ? message_ : "unknown");                                \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

constexpr int BM = 128, BN = 64, BK = 64;
constexpr int NSTAGE = STAGES;

constexpr int A_BYTES = BK * BM * sizeof(__nv_bfloat16),
              B_BYTES = BK * BN * sizeof(__nv_bfloat16);
constexpr int STAGE_BYTES = A_BYTES + B_BYTES;

__device__ inline uint64_t make_desc_sm100(uint32_t saddr, uint32_t lbo,
                                           uint32_t sbo, uint32_t layout) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)layout << 61;
  return d;
}

__device__ inline void mbar_wait(uint32_t mbar, uint32_t phase) {
  uint32_t done = 0;
  while (!done)
    asm volatile("{\n.reg .pred p;\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.b32 %0, 1, 0, p;\n}"
                 : "=r"(done)
                 : "r"(mbar), "r"(phase));
}

// 非阻塞版:成功返回 true。机会式深预取用它。
__device__ inline bool mbar_try(uint32_t mbar, uint32_t phase) {
  uint32_t done;
  asm volatile("{\n.reg .pred p;\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
               "selp.b32 %0, 1, 0, p;\n}"
               : "=r"(done)
               : "r"(mbar), "r"(phase));
  return done;
}

template <int S>
__device__ inline void
issue_tma_tile(int tile_iterator, uint8_t *smem, uint64_t *mbar_full, int tileM,
               int tileN, const CUtensorMap *tmapA, const CUtensorMap *tmapB) {
  const int stage = tile_iterator % S;
  const uint32_t mbar_full_u32 =
      static_cast<uint32_t>(__cvta_generic_to_shared(&mbar_full[stage]));
  uint8_t *sA = smem + stage * STAGE_BYTES;
  uint8_t *sB = sA + A_BYTES;
  const uint32_t a_base = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
  const uint32_t b_base = static_cast<uint32_t>(__cvta_generic_to_shared(sB));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   mbar_full_u32),
               "r"(STAGE_BYTES)
               : "memory");
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
      "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(a_base),
      "l"(tmapA), "r"(tile_iterator * BK), "r"(tileM), "r"(mbar_full_u32)
      : "memory");
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile."
      "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(b_base),
      "l"(tmapB), "r"(tile_iterator * BK), "r"(tileN), "r"(mbar_full_u32)
      : "memory");
}

__global__ void gemm_pipeline(const __nv_bfloat16 *gA, const __nv_bfloat16 *gB,
                              float *gD, int M, int N, int K,
                              const __grid_constant__ CUtensorMap tmapA,
                              const __grid_constant__ CUtensorMap tmapB) {
  extern __shared__ uint8_t smem_raw[];
  uint8_t *smem = (uint8_t *)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  // TODO:把你 4.2 的 kernel 扩成 NSTAGE 级流水。参考结构:
  // (1) smem 划成 NSTAGE 段,stage s 的 A/B 起点自己排;mbarrier 每
  //     stage 两个:full[s](TMA 到达)、empty[s](mma 消费完成)
  __shared__ uint32_t s_taddr;
  __shared__ __align__(8) uint64_t mbar_full[NSTAGE], mbar_empty[NSTAGE];
  if (!warp_id) {
    if (!lane) {
      for (int s = 0; s < NSTAGE; ++s) {
        const uint32_t mbar_full_u32 = static_cast<uint32_t>(
                           __cvta_generic_to_shared(&mbar_full[s])),
                       mbar_empty_u32 = static_cast<uint32_t>(
                           __cvta_generic_to_shared(&mbar_empty[s]));
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], %2;\n"
            "mbarrier.init.shared::cta.b64 [%1], %2;\n" ::"r"(mbar_full_u32),
            "r"(mbar_empty_u32), "r"(1)
            : "memory");
      }
      asm volatile("fence.mbarrier_init.release.cluster;\n");
    }
    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&s_taddr));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
            "r"(dst),
        "r"(64));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();
  const int tileM = blockIdx.x * BM;
  const int tileN = blockIdx.y * BN;
  const uint32_t taddr = s_taddr;
  const uint32_t idesc =
      (1u << 4) | (1u << 7) | (1u << 10) | (8u << 17) | (8u << 24);
  uint32_t elected;
  asm volatile("{\n.reg .pred P;\nelect.sync _|P, 0xffffffff;\nselp.b32 %0, "
               "1, 0, P;\n}"
               : "=r"(elected));
  bool issuer = (!warp_id && elected);
  // (2) 预热:先发 min(NSTAGE, iters) 轮 TMA(发第 it 轮 = 对 stage
  //     it%NSTAGE 做 arrive.expect_tx + 两条 cp.async.bulk.tensor)
  const int iters = K / BK;
  const int warmup = iters < NSTAGE ? iters : NSTAGE;
  if (issuer) {
    for (int p = 0; p < warmup; ++p) {
      issue_tma_tile<NSTAGE>(p, smem, mbar_full, tileM, tileN, &tmapA, &tmapB);
    }
  }
  int next_issue = warmup;
  // (3) 主循环 it:
  //     - 强制发射:若第 it 轮 TMA 还没发,阻塞等 empty[it%NSTAGE]
  //       后补发(见文件头 hazard;empty 的 parity 按该 stage 被复用
  //       的轮次算,第一次复用等的是上一轮使用的完成)
  //     - 机会式深预取:try_wait 下一个待发 stage 的 empty,成功就
  //       继续发,失败立刻停,不许阻塞
  //     - 等 full[it%NSTAGE](parity = (it/NSTAGE)&1)→ tcgen05.fence
  //       → mma(与 4.2 相同,累加位口径不变)→ commit 到
  //       empty[it%NSTAGE]
  for (int it = 0; it < iters; ++it) {
    const int stage = it % NSTAGE;
    const int reuse = it / NSTAGE;
    const uint32_t mbar_full_u32 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&mbar_full[stage]));
    const uint32_t mbar_empty_u32 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&mbar_empty[stage]));
    if (issuer) {
      if (next_issue == it) {
        const int empty_phase = (reuse - 1) & 1;
        mbar_wait(mbar_empty_u32, empty_phase);
        issue_tma_tile<NSTAGE>(it, smem, mbar_full, tileM, tileN, &tmapA,
                               &tmapB);
        ++next_issue;
      }
      while (next_issue < iters) {
        const int cand_stage = next_issue % NSTAGE;
        const int cand_reuse = next_issue / NSTAGE;
        const int empty_phase = (cand_reuse - 1) & 1;
        const uint32_t cand_mbar_empty_u32 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&mbar_empty[cand_stage]));
        if (mbar_try(cand_mbar_empty_u32, empty_phase)) {
          issue_tma_tile<NSTAGE>(next_issue, smem, mbar_full, tileM, tileN,
                                 &tmapA, &tmapB);
        } else {
          break;
        }
        ++next_issue;
      }
    }
    const int full_phase = reuse & 1;
    mbar_wait(mbar_full_u32, full_phase);
    if (issuer) {
      uint8_t *sA = smem + stage * STAGE_BYTES;
      uint8_t *sB = sA + A_BYTES;
      const uint32_t a_base =
          static_cast<uint32_t>(__cvta_generic_to_shared(sA));
      const uint32_t b_base =
          static_cast<uint32_t>(__cvta_generic_to_shared(sB));
      asm volatile("tcgen05.fence::after_thread_sync;");
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const uint64_t adesc = make_desc_sm100(a_base + (kk << 5), 0, 1024, 2);
        const uint64_t bdesc = make_desc_sm100(b_base + (kk << 5), 0, 1024, 2);
        uint32_t accumulate = (it || kk);
        asm volatile(
            "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::
                "r"(taddr),
            "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accumulate));
      }
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one"
                   ".shared::cluster.b64 [%0];" ::"r"(mbar_empty_u32)
                   : "memory");
    }
  }
  // (4) drain:等最后一轮 mma 的 empty 到达,再进 epilogue
  const uint32_t mbar_empty_u32 = static_cast<uint32_t>(
      __cvta_generic_to_shared(&mbar_empty[(iters - 1) % NSTAGE]));
  mbar_wait(mbar_empty_u32, ((iters - 1) / NSTAGE) & 1);
  asm volatile("tcgen05.fence::after_thread_sync;");
  const int row = (warp_id << 5) + lane;
  for (int col = 0; col < 64; col += 8) {
    const uint32_t addr = taddr + ((warp_id << 5) << 16) + col;
    float r[8];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=f"(r[0]), "=f"(r[1]), "=f"(r[2]), "=f"(r[3]), "=f"(r[4]),
                   "=f"(r[5]), "=f"(r[6]), "=f"(r[7])
                 : "r"(addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;");
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      gD[(tileM + row) * N + (tileN + col + i)] = r[i];
    }
  }
  __syncthreads();
  if (!warp_id) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
        "r"(64));
  }
}

int main(int argc, char **argv) {
  int M = argc > 3 ? atoi(argv[1]) : 4096;
  int N = argc > 3 ? atoi(argv[2]) : 4096;
  int K = argc > 3 ? atoi(argv[3]) : 4096;
  if (M % BM || N % BN || K % BK) {
    printf("形状需按 %dx%dx%d 对齐\n", BM, BN, BK);
    return 1;
  }
  size_t nA = (size_t)M * K, nB = (size_t)N * K, nD = (size_t)M * N;
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(-3, 3);
  std::vector<__nv_bfloat16> hA(nA), hB(nB);
  for (auto &v : hA)
    v = __float2bfloat16((float)dist(rng));
  for (auto &v : hB)
    v = __float2bfloat16((float)dist(rng));
  __nv_bfloat16 *dA, *dB;
  float *dD, *dRef;
  CUDA_CHECK(cudaMalloc(&dA, nA * 2));
  CUDA_CHECK(cudaMalloc(&dB, nB * 2));
  CUDA_CHECK(cudaMalloc(&dD, nD * 4));
  CUDA_CHECK(cudaMalloc(&dRef, nD * 4));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dD, 0xFF, nD * 4));

  CUtensorMap tmapA = {}, tmapB = {};
  cuuint64_t globalDimA[2] = {static_cast<cuuint64_t>(K),
                              static_cast<cuuint64_t>(M)},
             globalDimB[2] = {static_cast<cuuint64_t>(K),
                              static_cast<cuuint64_t>(N)};
  cuuint64_t globalStrides[1] = {static_cast<cuuint64_t>(K) *
                                 sizeof(__nv_bfloat16)};
  cuuint32_t boxDimA[2] = {static_cast<cuuint32_t>(BK),
                           static_cast<cuuint32_t>(BM)},
             boxDimB[2] = {static_cast<cuuint32_t>(BK),
                           static_cast<cuuint32_t>(BN)};
  cuuint32_t elementStrides[2] = {1, 1};
  CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(
      &tmapA, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dA, globalDimA,
      globalStrides, boxDimA, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(
      &tmapB, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dB, globalDimB,
      globalStrides, boxDimB, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  dim3 grid(M / BM, N / BN);
  // NSTAGE=3 时 72KB+对齐余量,超 48KB 静态上限,动态 smem 必须。
  size_t smemBytes = (size_t)NSTAGE * (BM + BN) * BK * 2 + 1024;
  CUDA_CHECK(cudaFuncSetAttribute(gemm_pipeline,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  (int)smemBytes));
  auto launch = [&] {
    gemm_pipeline<<<grid, 128, smemBytes>>>(dA, dB, dD, M, N, K, tmapA, tmapB);
  };
  launch();
  CUDA_CHECK_KERNEL();

  cublasHandle_t h;
  cublasCreate(&h);
  float alpha = 1.f, beta = 0.f;
  cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16BF, K,
               dA, CUDA_R_16BF, K, &beta, dRef, CUDA_R_32F, N,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(nD), ref(nD);
  CUDA_CHECK(cudaMemcpy(got.data(), dD, nD * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ref.data(), dRef, nD * 4, cudaMemcpyDeviceToHost));
  long bad = 0;
  for (size_t i = 0; i < nD; i++)
    bad += got[i] != ref[i];

  int iters = (size_t)M * N >= (size_t)4096 * 4096 ? 20 : 100;
  float ms = time_avg_ms(launch, iters);
  double tflops = 2.0 * M * N * K / (ms * 1e9);
  float cub_ms = time_avg_ms(
      [&] {
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB,
                     CUDA_R_16BF, K, dA, CUDA_R_16BF, K, &beta, dRef,
                     CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
      },
      iters);
  double cub_tflops = 2.0 * M * N * K / (cub_ms * 1e9);
  printf("[4.3 pipeline S=%d] M=%d N=%d K=%d  %s(bad=%ld)  %.2f ms  %.1f "
         "TFLOPS  (cuBLAS %.1f, 达成率 %.0f%%)\n",
         NSTAGE, M, N, K, bad ? "FAIL" : "PASS", bad, ms, tflops, cub_tflops,
         100.0 * tflops / cub_tflops);
  cublasDestroy(h);
  return bad != 0;
}
