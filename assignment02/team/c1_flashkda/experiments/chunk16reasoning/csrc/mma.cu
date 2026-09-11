// Actual KDA matrix shapes, isolated BF16 x BF16 -> FP32 WMMA.
// Independent matrices: does NOT model chunk-to-chunk state dependencies.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;
template <int M, int N, int K, int W>
__global__ void shape_mma(const __nv_bfloat16 *a, const __nv_bfloat16 *b,
                          float *out) {
  __shared__ __align__(32) __nv_bfloat16 as[M * K], bs[K * N];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int i = threadIdx.x; i < M * K; i += 32 * W)
    as[i] = a[blockIdx.x * M * K + i];
  for (int i = threadIdx.x; i < K * N; i += 32 * W)
    bs[i] = b[blockIdx.x * K * N + i];
  __syncthreads();
  for (int tile = warp; tile < (M / 16) * (N / 16); tile += W) {
    int row = (tile / (N / 16)) * 16, col = (tile % (N / 16)) * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    for (int k = 0; k < K; k += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>
          af;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major>
          bf;
      wmma::load_matrix_sync(af, as + row * K + k, K);
      wmma::load_matrix_sync(bf, bs + k * N + col, N);
      wmma::mma_sync(acc, af, bf, acc);
    }
    wmma::store_matrix_sync(out + blockIdx.x * M * N + row * N + col, acc, N,
                            wmma::mem_row_major);
  }
}
template <int M, int N, int K>
int dispatch_mma(const __nv_bfloat16 *a, const __nv_bfloat16 *b, float *o,
                 int batch, int w, cudaStream_t s) {
  if (w == 1)
    shape_mma<M, N, K, 1><<<batch, 32, 0, s>>>(a, b, o);
  else if (w == 4)
    shape_mma<M, N, K, 4><<<batch, 128, 0, s>>>(a, b, o);
  else if (w == 8)
    shape_mma<M, N, K, 8><<<batch, 256, 0, s>>>(a, b, o);
  else
    return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
template <int C>
int stage_mma(const __nv_bfloat16 *a, const __nv_bfloat16 *b, float *o, int n,
              int stage, int w, cudaStream_t s) {
  if (stage == 0)
    return dispatch_mma<C, C, 128>(a, b, o, n, w, s); // Gram / Mqk
  if (stage == 1)
    return dispatch_mma<C, 128, 128>(a, b, o, n, w, s); // K @ state
  if (stage == 2)
    return dispatch_mma<C, 128, C>(a, b, o, n, w, s); // INV @ U / Mqk @ U
  if (stage == 3)
    return dispatch_mma<128, 128, C>(a, b, o, n, w, s); // state update
  return int(cudaErrorInvalidValue);
}
extern "C" int launch_mma(const void *a, const void *b, float *o, int n, int c,
                          int stage, int w, void *stream) {
  auto aa = static_cast<const __nv_bfloat16 *>(a),
       bb = static_cast<const __nv_bfloat16 *>(b);
  auto s = static_cast<cudaStream_t>(stream);
  if (c == 16)
    return stage_mma<16>(aa, bb, o, n, stage, w, s);
  if (c == 32)
    return stage_mma<32>(aa, bb, o, n, stage, w, s);
  if (c == 64)
    return stage_mma<64>(aa, bb, o, n, stage, w, s);
  return int(cudaErrorInvalidValue);
}
