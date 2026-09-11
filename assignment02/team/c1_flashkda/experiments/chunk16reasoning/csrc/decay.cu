// Range/rescaling microbenchmark: one scalar gate channel, same CxC output.
// This is not an implementation of the full KDA prepare or recurrent state.
#include <cuda_runtime.h>
__device__ __forceinline__ float exp_bf16(float x) {
  float y;
  unsigned short bits;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(y));
  return __uint_as_float(unsigned(bits) << 16);
}
// 0=raw factorization, 1=one center, 2=16-token tiled split rescaling,
// 3=direct causal exp(s_i-s_j) reference implementation.
template <int C, int Mode> __global__ void decay(const float *g, float *out) {
  __shared__ float s[C], d[C], u[C];
  int lane = threadIdx.x;
  if (lane == 0) {
    float acc = 0;
    for (int i = 0; i < C; ++i) {
      acc = __fadd_rn(acc, __fmul_rn(g[blockIdx.x * C + i], 0x1.715476p+0f));
      s[i] = acc;
    }
  }
  __syncwarp();
  float *dst = out + blockIdx.x * C * C;
  for (int p = lane; p < C * C; p += 32)
    dst[p] = 0;
  __syncwarp();
  if constexpr (Mode == 0 || Mode == 1) {
    float center = Mode == 0 ? 0.f : 0.5f * (s[0] + s[C - 1]);
    for (int i = lane; i < C; i += 32) {
      d[i] = exp_bf16(s[i] - center);
      u[i] = exp_bf16(center - s[i]);
    }
    __syncwarp();
    for (int p = lane; p < C * C; p += 32) {
      int i = p / C, j = p % C;
      if (j <= i)
        dst[p] = __fmul_rn(d[i], u[j]);
    }
  } else if constexpr (Mode == 2) {
    for (int bi = 0; bi < C; bi += 16)
      for (int bj = 0; bj <= bi; bj += 16) {
        float ci = 0.5f * (s[bi] + s[bi + 15]),
              cj = 0.5f * (s[bj] + s[bj + 15]);
        float delta = 0.5f * (ci - cj);
        if (lane < 16)
          d[lane] = exp_bf16((s[bi + lane] - ci) + delta);
        else
          u[lane - 16] = exp_bf16((cj - s[bj + lane - 16]) + delta);
        __syncwarp();
        for (int p = lane; p < 256; p += 32) {
          int i = bi + p / 16, j = bj + p % 16;
          if (j <= i)
            dst[i * C + j] = __fmul_rn(d[p / 16], u[p % 16]);
        }
        __syncwarp();
      }
  } else {
    for (int p = lane; p < C * C; p += 32) {
      int i = p / C, j = p % C;
      if (j <= i)
        dst[p] = exp_bf16(s[i] - s[j]);
    }
  }
}
template <int C>
int dispatch_decay(const float *g, float *out, int n, int mode,
                   cudaStream_t s) {
  if (mode == 0)
    decay<C, 0><<<n, 32, 0, s>>>(g, out);
  else if (mode == 1)
    decay<C, 1><<<n, 32, 0, s>>>(g, out);
  else if (mode == 2)
    decay<C, 2><<<n, 32, 0, s>>>(g, out);
  else if (mode == 3)
    decay<C, 3><<<n, 32, 0, s>>>(g, out);
  else
    return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
extern "C" int launch_decay(const float *g, float *out, int n, int c, int mode,
                            void *stream) {
  auto s = static_cast<cudaStream_t>(stream);
  if (c == 16)
    return dispatch_decay<16>(g, out, n, mode, s);
  if (c == 32)
    return dispatch_decay<32>(g, out, n, mode, s);
  if (c == 64)
    return dispatch_decay<64>(g, out, n, mode, s);
  return int(cudaErrorInvalidValue);
}
