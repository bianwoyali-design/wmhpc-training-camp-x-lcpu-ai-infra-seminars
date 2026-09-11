// Inspect captured official kernels and enable individual stages without
// editing FlashKDA.
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
extern "C" int node_count(void *graph, int *count) {
  size_t n = 0;
  auto e = cudaGraphGetNodes((cudaGraph_t)graph, nullptr, &n);
  *count = int(n);
  return int(e);
}
extern "C" int describe(void *graph, int index, void **node, char *name,
                        int capacity, int *info) {
  size_t n = 0;
  auto e = cudaGraphGetNodes((cudaGraph_t)graph, nullptr, &n);
  if (e)
    return int(e);
  std::vector<cudaGraphNode_t> nodes(n);
  e = cudaGraphGetNodes((cudaGraph_t)graph, nodes.data(), &n);
  if (e)
    return int(e);
  if (index < 0 || index >= n)
    return int(cudaErrorInvalidValue);
  *node = nodes[index];
  cudaGraphNodeType type;
  e = cudaGraphNodeGetType(nodes[index], &type);
  if (e)
    return int(e);
  info[0] = int(type);
  name[0] = 0;
  if (type != cudaGraphNodeTypeKernel)
    return 0;
  // Graph capture stores a driver CUfunction, not necessarily a runtime host
  // stub.
  CUDA_KERNEL_NODE_PARAMS p{};
  CUresult result = cuGraphKernelNodeGetParams((CUgraphNode)nodes[index], &p);
  if (result)
    return int(result);
  const char *func_name = nullptr;
  result = cuFuncGetName(&func_name, p.func);
  if (result)
    return int(result);
  std::strncpy(name, func_name, capacity - 1);
  name[capacity - 1] = 0;
  info[1] = p.gridDimX;
  info[2] = p.gridDimY;
  info[3] = p.gridDimZ;
  info[4] = p.blockDimX;
  info[5] = p.blockDimY;
  info[6] = p.blockDimZ;
  info[7] = p.sharedMemBytes;
  const CUfunction_attribute attrs[] = {CU_FUNC_ATTRIBUTE_NUM_REGS,
                                        CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                        CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES};
  for (int i = 0; i < 3; ++i) {
    result = cuFuncGetAttribute(&info[8 + i], attrs[i], p.func);
    if (result)
      return int(result);
  }
  result = cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &info[11], p.func, p.blockDimX * p.blockDimY * p.blockDimZ,
      p.sharedMemBytes);
  return int(result);
}
extern "C" int enable(void *executable, void *node, int on) {
  return int(cudaGraphNodeSetEnabled((cudaGraphExec_t)executable,
                                     (cudaGraphNode_t)node, on));
}
