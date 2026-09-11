#include "utils.cuh"

using LM =
    decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
                           make_shape(Int<16>{}, Int<16>{}), LayoutLeft{}));
__global__ void official(const float *input, float *output) {
  using F = cutlass::half_t;
  using B = cutlass::bfloat16_t;
  __shared__ __align__(128) F l[cosize_v<LM>];
  __shared__ __align__(128) F inv[cosize_v<LM>];
  __shared__ __align__(128) B out[cosize_v<LM>];
  auto lt = make_tensor(make_smem_ptr(l), LM{});
  auto it = make_tensor(make_smem_ptr(inv), LM{});
  auto ot = make_tensor(make_smem_ptr(out), LM{});
  for (int k = threadIdx.x; k < 256; k += 32) {
    int i = k / 16, j = k % 16;
    F v = F(input[blockIdx.x * 256 + k]);
    lt(i, j) = v;
    it(i, j) = i == j ? F(1) : F(-float(v));
  }
  __syncwarp();
  neumann_inv_fused_1warp(lt, it, ot, threadIdx.x);
  __syncwarp();
  for (int k = threadIdx.x; k < 256; k += 32)
    output[blockIdx.x * 256 + k] = float(ot(k / 16, k % 16));
}
extern "C" int launch_official(const float *input, float *output, int batch,
                               void *stream) {
  official<<<batch, 32, 0, static_cast<cudaStream_t>(stream)>>>(input, output);
  return int(cudaGetLastError());
}
