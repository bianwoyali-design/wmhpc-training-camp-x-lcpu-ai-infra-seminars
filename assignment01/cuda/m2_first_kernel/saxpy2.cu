#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s at %s:%d : %s\n", cudaGetErrorName(err_), \
              __FILE__, __LINE__, cudaGetErrorString(err_));                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUDA_KERNEL_CHECK()                                                    \
  do {                                                                         \
    CUDA_CHECK(cudaGetLastError());                                            \
  } while (0)

struct GpuTimer {
  cudaEvent_t start_, stop_;

  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }

  ~GpuTimer() {
    CUDA_CHECK(cudaEventDestroy(start_));
    CUDA_CHECK(cudaEventDestroy(stop_));
  }

  void start() { CUDA_CHECK(cudaEventRecord(start_)); }

  float stop_ms() {
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }
};

static inline void fill_x(float *x, int n) {
  for (int i = 0; i < n; ++i) {
    x[i] = ((i % 2048) - 1024) * 0.5f;
  }
}

static inline void fill_y(float *y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = (i % 1024) - 512;
  }
}

__global__ void saxpy(const int n, const float a, const float *__restrict__ x,
                      float *__restrict__ y) {
#pragma unroll 4
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    y[i] = a * x[i] + y[i];
  }
}

int main(int argc, char *argv[]) {
  int n = std::atoi(argv[1]);

  CUDA_CHECK(cudaFree(0));

  if (n == 0) {
    printf("SUM=0\n");
    return 0;
  }

  size_t bytes = (size_t)n * sizeof(float);

  float *h_x, *h_y, *h_z;
  CUDA_CHECK(cudaMallocHost(&h_x, bytes));
  CUDA_CHECK(cudaMallocHost(&h_y, bytes));
  CUDA_CHECK(cudaMallocHost(&h_z, bytes));

  fill_x(h_x, n);
  fill_y(h_y, n);

  float *d_x, *d_y;
  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));

  CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

  int threads = 256;

  constexpr int devId = 0;
  int numSMs;
  CUDA_CHECK(
      cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId));

  GpuTimer timer;
  timer.start();
  saxpy<<<32 * numSMs, threads>>>(n, 2.0f, d_x, d_y);
  float ms = timer.stop_ms();
  CUDA_KERNEL_CHECK();

  CUDA_CHECK(cudaMemcpy(h_z, d_y, bytes, cudaMemcpyDeviceToHost));

  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += (double)h_z[i];
  }

  cudaFree(d_x);
  cudaFree(d_y);

  CUDA_CHECK(cudaFreeHost(h_x));
  CUDA_CHECK(cudaFreeHost(h_y));
  CUDA_CHECK(cudaFreeHost(h_z));

  printf("SUM=%.0f, n=%d, kernel_ms=%0.3f\n", sum, n, ms);
  return 0;
}
