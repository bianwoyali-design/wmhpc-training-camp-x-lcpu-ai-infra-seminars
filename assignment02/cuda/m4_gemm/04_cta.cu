#include "../common.h"
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>

// CUDA 13.0 ships an older CCCL whose generated tcgen05 guards predate the
// sm_100f/sm_103a targets, even though ptxas supports these instructions for
// both targets.  Newer CCCL releases use __CUDA_HAS_ARCH_{FAMILY_,}SPECIFIC;
// bridge that detection locally until the toolkit copy is updated.
#if !defined(__CUDA_ARCH_FEAT_SM100_ALL) &&                                  \
    (__CUDA_HAS_ARCH_FAMILY_SPECIFIC(100) || __CUDA_HAS_ARCH_SPECIFIC(103))
#define CTA_CCCL_TCGEN05_TARGET_WORKAROUND 1
#define __CUDA_ARCH_FEAT_SM100_ALL 1
#endif
#include <cuda/ptx>
#if defined(CTA_CCCL_TCGEN05_TARGET_WORKAROUND)
#undef __CUDA_ARCH_FEAT_SM100_ALL
#undef CTA_CCCL_TCGEN05_TARGET_WORKAROUND
#endif
#include <random>
#include <vector>
namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

#ifndef STAGES
#define STAGES 4
#endif

#ifndef CTA_N
#define CTA_N 512
#endif

#ifndef TMEM_LD_COLUMNS
#define TMEM_LD_COLUMNS 16
#endif

#ifndef CTA_CLUSTER_SCHED
#define CTA_CLUSTER_SCHED 0
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

// A cta_group::2 operation produces 256 rows.  Each CTA owns 128 rows and
// half of each B panel.  Reusing one A tile across two N=256 MMA panels
// amortizes its TMA traffic and the cross-CTA handoff over a 256x512 output
// tile.  CTA_N remains a compile-time tuning knob for narrower experiments.
constexpr int BM = 128, BN = CTA_N, BK = 64;
constexpr int NSTAGE = STAGES;
constexpr int MMA_N = BN < 256 ? BN : 256;
constexpr int NPANELS = BN / MMA_N;
constexpr int LD_COLS = TMEM_LD_COLUMNS;
static_assert(BN % MMA_N == 0 && MMA_N % 32 == 0 && BN <= 512,
              "CTA_N must tile into at most two 32-column-aligned MMA panels");
static_assert(BN % LD_COLS == 0,
              "the TMEM load width must divide the CTA N tile");

constexpr int A_BYTES = BK * BM * sizeof(__nv_bfloat16),
              B_PANEL_BYTES = BK * (MMA_N / 2) * sizeof(__nv_bfloat16),
              B_BYTES = NPANELS * B_PANEL_BYTES;
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

__device__ inline void mbar_wait(uint64_t *mbar, uint32_t phase) {
  while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cluster,
                                        mbar, phase)) {
  }
}

__device__ inline void mbar_wait_local(uint64_t *mbar, uint32_t phase) {
  while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta,
                                        mbar, phase)) {
  }
}

__device__ inline bool mbar_try(uint64_t *mbar, uint32_t phase) {
  return ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cluster,
                                       mbar, phase);
}

template <int S>
__device__ inline void
issue_tma_tile(int tile_iterator, int rank, uint8_t *smem, uint64_t *mbar_full,
               int tileM, int tileN, const CUtensorMap *tmapA,
               const CUtensorMap *tmapB) {
  const int stage = tile_iterator % S;
  uint8_t *sA = smem + stage * STAGE_BYTES;
  uint8_t *sB = sA + A_BYTES;
  uint64_t *full = &mbar_full[stage];
  ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta,
                                 ptx::space_shared, full, STAGE_BYTES);
  const int32_t a_coords[2] = {tile_iterator * BK, tileM};
  ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, sA, tmapA,
                            a_coords, full);
#pragma unroll
  for (int panel = 0; panel < NPANELS; ++panel) {
    const int32_t b_coords[2] = {tile_iterator * BK,
                                 tileN + panel * MMA_N + rank * (MMA_N / 2)};
    ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global,
                              sB + panel * B_PANEL_BYTES, tmapB, b_coords,
                              full);
  }
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
  auto cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  __shared__ uint32_t s_taddr;
  __shared__ uint32_t stage_ready[NSTAGE];
  __shared__ __align__(8) uint64_t mbar_full[NSTAGE], mbar_empty[NSTAGE];
  if (!warp_id) {
    if (!lane) {
      for (int s = 0; s < NSTAGE; ++s) {
        ptx::mbarrier_init(&mbar_full[s], 1);
        ptx::mbarrier_init(&mbar_empty[s], 1);
        stage_ready[s] = 0;
      }
      ptx::fence_mbarrier_init(ptx::sem_release, ptx::scope_cluster);
    }
    ptx::tcgen05_alloc(ptx::cta_group_2, &s_taddr, BN);
    ptx::tcgen05_relinquish_alloc_permit(ptx::cta_group_2);
  }
  // One cluster barrier publishes the initialized DSMEM handshake words and
  // the symmetric TMEM allocation.  The hot K loop below uses a one-thread
  // DSMEM handoff instead of synchronizing all 256 threads every stage.
  cluster.sync();
  const int tileM = blockIdx.x * BM;
  const int tileN = blockIdx.y * BN;
  const uint32_t taddr = s_taddr;
  const uint32_t idesc =
      (1u << 4) | (1u << 7) | (1u << 10) | ((MMA_N / 8) << 17) | (16u << 24);
  // Warp 0 is converged here, so lane 0 is the deterministic elected thread.
  const bool elected_warp0 = (warp_id == 0 && lane == 0);
  const bool tma_issuer = elected_warp0;
  const bool mma_issuer = elected_warp0 && rank == 0;
  volatile uint32_t *rank1_ready = cluster.map_shared_rank(stage_ready, 1);
  const int iters = K / BK;
  const int warmup = iters < NSTAGE ? iters : NSTAGE;
  if (tma_issuer) {
    for (int p = 0; p < warmup; ++p) {
      issue_tma_tile<NSTAGE>(p, rank, smem, mbar_full, tileM, tileN, &tmapA,
                             &tmapB);
    }
  }
  int next_issue = warmup;
  for (int it = 0; it < iters; ++it) {
    const int stage = it % NSTAGE;
    const int reuse = it / NSTAGE;
    uint64_t *full = &mbar_full[stage];
    uint64_t *empty = &mbar_empty[stage];
    if (tma_issuer) {
      if (next_issue == it) {
        const int empty_phase = (reuse - 1) & 1;
        mbar_wait(empty, empty_phase);
        issue_tma_tile<NSTAGE>(it, rank, smem, mbar_full, tileM, tileN, &tmapA,
                               &tmapB);
        ++next_issue;
      }
      while (next_issue < iters) {
        const int cand_stage = next_issue % NSTAGE;
        const int cand_reuse = next_issue / NSTAGE;
        const int empty_phase = (cand_reuse - 1) & 1;
        if (mbar_try(&mbar_empty[cand_stage], empty_phase)) {
          issue_tma_tile<NSTAGE>(next_issue, rank, smem, mbar_full, tileM,
                                 tileN, &tmapA, &tmapB);
        } else {
          break;
        }
        ++next_issue;
      }
    }
    const int full_phase = reuse & 1;
    // One thread per CTA observes its local TMA completion.  Rank 1 publishes
    // a monotonically increasing ticket through DSMEM; rank 0 consumes it
    // before issuing the two-CTA MMA.  This replaces a full cluster barrier.
    if (tma_issuer) {
      mbar_wait_local(full, full_phase);
      if (rank == 1) {
        ptx::fence(ptx::sem_release, ptx::scope_cluster);
        *reinterpret_cast<volatile uint32_t *>(&stage_ready[stage]) =
            static_cast<uint32_t>(it + 1);
      }
    }
    if (mma_issuer) {
      while (rank1_ready[stage] < static_cast<uint32_t>(it + 1)) {
      }
      ptx::fence(ptx::sem_acquire, ptx::scope_cluster);
    }
    if (mma_issuer) {
      uint8_t *sA = smem + stage * STAGE_BYTES;
      uint8_t *sB = sA + A_BYTES;
      const uint32_t a_base =
          static_cast<uint32_t>(__cvta_generic_to_shared(sA));
      const uint32_t b_base =
          static_cast<uint32_t>(__cvta_generic_to_shared(sB));
      ptx::tcgen05_fence_after_thread_sync();
#pragma unroll
      for (int panel = 0; panel < NPANELS; ++panel) {
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const uint64_t adesc =
              make_desc_sm100(a_base + (kk << 5), 0, 1024, 2);
          const uint64_t bdesc = make_desc_sm100(
              b_base + panel * B_PANEL_BYTES + (kk << 5), 0, 1024, 2);
          const bool accumulate = (it || kk);
          ptx::tcgen05_mma(ptx::kind_f16, ptx::cta_group_2,
                           taddr + panel * MMA_N, adesc, bdesc, idesc,
                           accumulate);
        }
      }
      ptx::tcgen05_commit_multicast(ptx::cta_group_2, empty,
                                    static_cast<uint16_t>(0x3));
    }
  }
  uint64_t *last_empty = &mbar_empty[(iters - 1) % NSTAGE];
  if (tma_issuer) {
    mbar_wait(last_empty, ((iters - 1) / NSTAGE) & 1);
  }
  __syncthreads();
  ptx::tcgen05_fence_after_thread_sync();
  const int row = (warp_id << 5) + lane;
  for (int col = 0; col < BN; col += LD_COLS) {
    const uint32_t addr = taddr + ((warp_id << 5) << 16) + col;
    uint32_t r[LD_COLS];
    ptx::tcgen05_ld_32x32b(r, addr);
    ptx::tcgen05_wait_ld();
#pragma unroll
    for (int i = 0; i < LD_COLS; ++i) {
      gD[(tileM + row) * N + (tileN + col + i)] = __uint_as_float(r[i]);
    }
  }
  cluster.sync();
  if (!warp_id) {
    ptx::tcgen05_dealloc(ptx::cta_group_2, taddr, BN);
  }
}

int main(int argc, char **argv) {
  int M = argc > 3 ? atoi(argv[1]) : 4096;
  int N = argc > 3 ? atoi(argv[2]) : 4096;
  int K = argc > 3 ? atoi(argv[3]) : 4096;
  if (M <= 0 || N <= 0 || K <= 0 || M % (2 * BM) || N % BN || K % BK) {
    printf("形状需按 %dx%dx%d 对齐\n", 2 * BM, BN, BK);
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
                           static_cast<cuuint32_t>(MMA_N / 2)};
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
  // Shared storage is staged dynamically; the extra KiB aligns the first
  // tensor-map destination to the descriptor's 1024-byte requirement.
  size_t smemBytes = (size_t)NSTAGE * STAGE_BYTES + 1024;
  CUDA_CHECK(cudaFuncSetAttribute(gemm_pipeline,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  (int)smemBytes));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = smemBytes;
  cudaLaunchAttribute attrs[2] = {};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {2, 1, 1};
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  if constexpr (CTA_CLUSTER_SCHED != 0) {
    attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attrs[1].val.clusterSchedulingPolicyPreference =
        static_cast<cudaClusterSchedulingPolicy>(CTA_CLUSTER_SCHED);
    cfg.numAttrs = 2;
  }
  auto launch = [&] {
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, gemm_pipeline, dA, dB, dD, M, N, K,
                                  tmapA, tmapB));
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
  printf("[4.4 CTA group::2 tile=%dx%d S=%d LD=%d] M=%d N=%d K=%d  "
         "%s(bad=%ld)  %.2f ms  %.1f TFLOPS  (cuBLAS %.1f, 达成率 %.0f%%)\n",
         2 * BM, BN, NSTAGE, LD_COLS, M, N, K, bad ? "FAIL" : "PASS", bad,
         ms, tflops, cub_tflops, 100.0 * tflops / cub_tflops);
  cublasDestroy(h);
  return bad != 0;
}
