// Same BF16 inputs and FP32 outputs for warp MMA and tcgen05.
// B is supplied as row-major [K,N]; layout conversion is inside both kernels.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
using bf16 = __nv_bfloat16;
using namespace nvcuda;

// K-major shared-memory layout: 8-row atoms, 32B/128B XOR swizzle.
template <int Rows, int K> __device__ int offset(int row, int k) {
  constexpr int S = K >= 64 ? 64 : 16;
  return (k / S) * Rows * S + (row / 8) * 8 * S + (row % 8) * S +
         ((k % S / 8) ^ ((row / (64 / S)) & (S / 8 - 1))) * 8 + k % 8;
}
template <int Rows, int K>
__device__ uint64_t descriptor(const bf16 *p, int k) {
  constexpr int S = K >= 64 ? 64 : 16;
  uint32_t addr = uint32_t(__cvta_generic_to_shared(p)) +
                  (k / S) * Rows * S * 2 + (k % S) * 2;
  return uint64_t((addr >> 4) & 0x3fff) | (uint64_t(8 * S * 2 / 16) << 32) |
         (uint64_t(1) << 46) | (uint64_t(S == 64 ? 2 : 6) << 61);
}
__device__ void wait_barrier(uint32_t b) {
  uint32_t done = 0;
  while (!done)
    asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 "
                 "p,[%1],0; selp.b32 %0,1,0,p;}"
                 : "=r"(done)
                 : "r"(b)
                 : "memory");
}

template <int M, int N, int K, int W, bool Transpose = false, bool Pad = false>
__global__ void warp_mma(const bf16 *a, const bf16 *b, float *out,
                         int repeats) {
  constexpr int RM = Transpose ? N : M, RN = Transpose ? M : N;
  constexpr int PM = Pad && RM < 64 ? 64 : RM;
  constexpr int AK = K + 8, BN = RN + 8;
  __shared__ __align__(32) bf16 sa[PM * AK], sb[K * BN];
  int tid = threadIdx.x, warp = tid / 32;
  a += blockIdx.x * M * K;
  b += blockIdx.x * K * N;
  out += blockIdx.x * M * N;
  for (int i = tid; i < (PM - RM) * K; i += 32 * W)
    sa[(RM + i / K) * AK + i % K] = __float2bfloat16(0);
  // Read contiguous global elements; +8 shared leading dimensions reduce bank
  // conflicts.
  for (int i = tid; i < M * K; i += 32 * W) {
    if constexpr (Transpose)
      sb[(i % K) * BN + i / K] = a[i];
    else
      sa[(i / K) * AK + i % K] = a[i];
  }
  for (int i = tid; i < K * N; i += 32 * W) {
    if constexpr (Transpose)
      sa[(i % N) * AK + i / N] = b[i];
    else
      sb[(i / N) * BN + i % N] = b[i];
  }
  __syncthreads();
  for (int tile = warp; tile < (PM / 16) * (RN / 16); tile += W) {
    int row = (tile / (RN / 16)) * 16, col = (tile % (RN / 16)) * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    for (int rep = 0; rep < repeats; ++rep) {
      for (int k = 0; k < K; k += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> bf;
        wmma::load_matrix_sync(af, sa + row * AK + k, AK);
        wmma::load_matrix_sync(bf, sb + k * BN + col, BN);
        wmma::mma_sync(acc, af, bf, acc);
      }
    }
    // Common output interface, including transposition cost when requested.
    if (row < RM) {
      if constexpr (Transpose)
        wmma::store_matrix_sync(out + col * N + row, acc, N,
                                wmma::mem_col_major);
      else
        wmma::store_matrix_sync(out + row * N + col, acc, N,
                                wmma::mem_row_major);
    }
  }
}

template <int M, int N, int K, bool Transpose = false>
__global__ void tensor_mma(const bf16 *a, const bf16 *b, float *out,
                           int repeats) {
  constexpr int RM = Transpose ? N : M, RN = Transpose ? M : N;
  constexpr int PM = RM < 64 ? 64 : RM, Columns = RN < 32 ? 32 : RN;
  __shared__ __align__(1024) bf16 sa[PM * K];
  __shared__ __align__(1024) bf16 sb[RN * K];
  __shared__ float output_tile[Transpose ? 1 : 4 * 32 * 9];
  __shared__ uint32_t tmem;
  __shared__ __align__(8) uint64_t barrier;
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  uint32_t bar = uint32_t(__cvta_generic_to_shared(&barrier));
  if (warp == 0) {
    if (lane == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bar)
                   : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    uint32_t dst = uint32_t(__cvta_generic_to_shared(&tmem));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;" ::
            "r"(dst),
        "r"(Columns)
        : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
  }
  a += blockIdx.x * M * K;
  b += blockIdx.x * K * N;
  out += blockIdx.x * M * N;
  for (int i = tid; i < (PM - RM) * K; i += 128)
    sa[offset<PM, K>(RM + i / K, i % K)] = __float2bfloat16(0);
  for (int i = tid; i < M * K; i += 128) {
    if constexpr (Transpose)
      sb[offset<RN, K>(i / K, i % K)] = a[i];
    else
      sa[offset<PM, K>(i / K, i % K)] = a[i];
  }
  for (int i = tid; i < K * N; i += 128) {
    if constexpr (Transpose)
      sa[offset<PM, K>(i % N, i / N)] = b[i];
    else
      sb[offset<RN, K>(i % N, i / N)] = b[i];
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  uint32_t addr = tmem;
  constexpr uint32_t idesc =
      (1u << 4) | (1u << 7) | (1u << 10) | ((RN / 8) << 17) | ((PM / 16) << 24);
  if (tid == 0) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    for (int rep = 0; rep < repeats; ++rep) {
      for (int k = 0; k < K; k += 16) {
        uint64_t ad = descriptor<PM, K>(sa, k), bd = descriptor<RN, K>(sb, k);
        int accumulate = rep != 0 || k != 0;
        asm volatile(
            "{.reg .pred p; setp.ne.b32 p,%4,0; "
            "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;}" ::"r"(addr),
            "l"(ad), "l"(bd), "r"(idesc), "r"(accumulate)
            : "memory");
      }
    }
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
                 "cluster.b64 [%0];" ::"r"(bar)
                 : "memory");
  }
  wait_barrier(bar);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  // M=64 occupies lanes 0..15 of each 32-lane TMEM subpartition.
  int row = PM == 64 ? warp * 16 + lane : warp * 32 + lane;
  for (int col = 0; col < RN; col += 8) {
    uint32_t src = addr + ((warp * 32) << 16) + col;
    float x[8];
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];"
        : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3]), "=f"(x[4]),
          "=f"(x[5]), "=f"(x[6]), "=f"(x[7])
        : "r"(src)
        : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    if constexpr (Transpose) {
      if ((PM != 64 || lane < 16) && row < RM) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          out[(col + j) * N + row] = x[j];
      }
    } else {
// Reorder each warp's 32x8 output tile to coalesce global stores.
#pragma unroll
      for (int j = 0; j < 8; ++j)
        output_tile[warp * 32 * 9 + lane * 9 + j] = x[j];
      __syncwarp();
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        int r = (j * 32 + lane) / 8, c = lane % 8;
        int logical_row = (PM == 64 ? warp * 16 : warp * 32) + r;
        if ((PM != 64 || r < 16) && logical_row < RM)
          out[logical_row * N + col + c] =
              output_tile[warp * 32 * 9 + r * 9 + c];
      }
      __syncwarp();
    }
  }
  __syncthreads();
  if (warp == 0)
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;" ::"r"(addr),
        "r"(Columns)
        : "memory");
  if (tid == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}

template <int M, int N, int K>
int dispatch(const bf16 *a, const bf16 *b, float *o, int batch, int method,
             int repeats, cudaStream_t stream, int *resources) {
#define CHOOSE(ID, FUNC, THREADS, ...)                                         \
  if (method == ID) {                                                          \
    if (resources) {                                                           \
      cudaFuncAttributes f;                                                    \
      cudaError_t e = cudaFuncGetAttributes(&f, FUNC<__VA_ARGS__>);            \
      if (e)                                                                   \
        return int(e);                                                         \
      int blocks;                                                              \
      e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(                       \
          &blocks, FUNC<__VA_ARGS__>, THREADS, 0);                             \
      if (e)                                                                   \
        return int(e);                                                         \
      resources[0] = f.numRegs;                                                \
      resources[1] = f.sharedSizeBytes;                                        \
      resources[2] = f.localSizeBytes;                                         \
      resources[3] = blocks;                                                   \
    } else                                                                     \
      FUNC<__VA_ARGS__><<<batch, THREADS, 0, stream>>>(a, b, o, repeats);      \
    return int(cudaGetLastError());                                            \
  }
  CHOOSE(0, warp_mma, 32, M, N, K, 1)
  CHOOSE(1, warp_mma, 128, M, N, K, 4)
  CHOOSE(2, warp_mma, 256, M, N, K, 8)
  CHOOSE(3, warp_mma, 128, M, N, K, 4, false, true)
  CHOOSE(4, warp_mma, 128, M, N, K, 4, true, false)
  CHOOSE(5, tensor_mma, 128, M, N, K, false)
  CHOOSE(6, tensor_mma, 128, M, N, K, true)
#undef CHOOSE
  return int(cudaErrorInvalidValue);
}
extern "C" int launch(const void *a, const void *b, float *o, int batch,
                      int shape, int method, int repeats, void *stream,
                      int *resources) {
  const bf16 *aa = static_cast<const bf16 *>(a),
             *bb = static_cast<const bf16 *>(b);
  auto s = static_cast<cudaStream_t>(stream);
  if (shape == 0)
    return dispatch<16, 16, 128>(aa, bb, o, batch, method, repeats, s,
                                 resources);
  if (shape == 1)
    return dispatch<16, 128, 128>(aa, bb, o, batch, method, repeats, s,
                                  resources);
  if (shape == 2)
    return dispatch<16, 128, 16>(aa, bb, o, batch, method, repeats, s,
                                 resources);
  if (shape == 3)
    return dispatch<128, 128, 16>(aa, bb, o, batch, method, repeats, s,
                                  resources);
  if (shape == 4)
    return dispatch<16, 16, 16>(aa, bb, o, batch, method, repeats, s,
                                resources);
  return int(cudaErrorInvalidValue);
}
