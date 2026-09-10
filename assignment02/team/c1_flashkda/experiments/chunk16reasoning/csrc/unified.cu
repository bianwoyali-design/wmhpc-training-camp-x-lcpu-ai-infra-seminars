#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

// W warps per matrix, distributing 16x16 output tiles across warps.
// Each 16x16 tile is accumulated in FP32,
// rounded to FP16 at the GEMM boundary, then X addition is rounded to FP16.
template <int C, int W, bool Add>
__device__ void product(const half *a, const half *b, const half *old,
                        half *dst, float *tile) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  tile += warp * 256;
  for (int tix = warp; tix < (C / 16) * (C / 16); tix += W) {
    int i = (tix / (C / 16)) * 16, j = (tix % (C / 16)) * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    for (int k = 0; k < C; k += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
      wmma::load_matrix_sync(af, a + i * C + k, C);
      wmma::load_matrix_sync(bf, b + k * C + j, C);
      wmma::mma_sync(acc, af, bf, acc);
    }
    wmma::store_matrix_sync(tile, acc, 16, wmma::mem_row_major);
    __syncwarp();
    for (int t = lane; t < 256; t += 32) {
      int ix = (i + t / 16) * C + j + t % 16;
      half val = __float2half_rn(tile[t]);
      if constexpr (Add)
        val = __hadd(old[ix], val);
      dst[ix] = val;
    }
    __syncwarp();
  }
}

template <int W> __device__ __forceinline__ void sync_matrix() {
  if constexpr (W == 1)
    __syncwarp();
  else
    __syncthreads();
}

template <int C, int W = 1>
__global__ void unified(const float *in, float *out) {
  __shared__ __align__(32) half p0[C * C], p1[C * C], x0[C * C], x1[C * C];
  __shared__ __align__(32) float tile[256 * W];
  int base = blockIdx.x * C * C;
  for (int t = threadIdx.x; t < C * C; t += 32 * W) {
    half v = __float2half_rn(in[base + t]);
    p0[t] = v;
    x0[t] = __float2half_rn((t / C == t % C ? 1.f : 0.f) - __half2float(v));
  }
  sync_matrix<W>();
  half *p = p0, *pn = p1, *x = x0, *xn = x1;
  for (int power = 2; power < C; power *= 2) {
    product<C, W, false>(p, p, nullptr, pn, tile);
    sync_matrix<W>();
    product<C, W, true>(x, pn, x, xn, tile);
    sync_matrix<W>();
    half *tmp = p;
    p = pn;
    pn = tmp;
    tmp = x;
    x = xn;
    xn = tmp;
  }
  for (int t = threadIdx.x; t < C * C; t += 32 * W)
    out[base + t] = __bfloat162float(__float2bfloat16_rn(__half2float(x[t])));
}

extern "C" int launch(const float *in, float *out, int batch, int c,
                      void *stream) {
  auto s = static_cast<cudaStream_t>(stream);
  if (c == 16)
    unified<16><<<batch, 32, 0, s>>>(in, out);
  else if (c == 32)
    unified<32><<<batch, 32, 0, s>>>(in, out);
  else if (c == 64)
    unified<64><<<batch, 32, 0, s>>>(in, out);
  else
    return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
template <int C, int W = 1> int attrs(int *out) {
  cudaFuncAttributes a;
  auto e = cudaFuncGetAttributes(&a, unified<C, W>);
  if (e)
    return int(e);
  int blocks;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, unified<C, W>,
                                                    32 * W, 0);
  if (e)
    return int(e);
  out[0] = a.numRegs;
  out[1] = a.sharedSizeBytes;
  out[2] = a.localSizeBytes;
  out[3] = blocks;
  return 0;
}
extern "C" int resources(int c, int *out) {
  if (c == 16)
    return attrs<16>(out);
  if (c == 32)
    return attrs<32>(out);
  if (c == 64)
    return attrs<64>(out);
  return int(cudaErrorInvalidValue);
}

template <int C>
int dispatch(const float *in, float *out, int batch, int warps,
             cudaStream_t s) {
  if (warps == 1)
    unified<C, 1><<<batch, 32, 0, s>>>(in, out);
  else if (warps == 4)
    unified<C, 4><<<batch, 128, 0, s>>>(in, out);
  else if (warps == 8)
    unified<C, 8><<<batch, 256, 0, s>>>(in, out);
  else
    return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
extern "C" int launch_config(const float *in, float *out, int batch, int c,
                             int warps, void *stream) {
  auto s = static_cast<cudaStream_t>(stream);
  if (c == 16)
    return dispatch<16>(in, out, batch, warps, s);
  if (c == 32)
    return dispatch<32>(in, out, batch, warps, s);
  if (c == 64)
    return dispatch<64>(in, out, batch, warps, s);
  return int(cudaErrorInvalidValue);
}
template <int C> int resource_config(int warps, int *out) {
  if (warps == 1)
    return attrs<C, 1>(out);
  if (warps == 4)
    return attrs<C, 4>(out);
  if (warps == 8)
    return attrs<C, 8>(out);
  return int(cudaErrorInvalidValue);
}
extern "C" int resources_config(int c, int warps, int *out) {
  if (c == 16)
    return resource_config<16>(warps, out);
  if (c == 32)
    return resource_config<32>(warps, out);
  if (c == 64)
    return resource_config<64>(warps, out);
  return int(cudaErrorInvalidValue);
}
