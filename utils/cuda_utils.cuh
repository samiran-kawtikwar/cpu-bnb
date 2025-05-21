#pragma once

#include <cuda.h>
#include "logger.cuh"

#define CUDA_RUNTIME(ans)                 \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false)
{

  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

    /*if (abort) */ exit(1);
  }
}

#define execKernel(kernel, gridSize, BSize, deviceId, verbose, ...)                       \
  {                                                                                       \
    dim3 grid(gridSize);                                                                  \
    dim3 block(BSize);                                                                    \
                                                                                          \
    CUDA_RUNTIME(cudaSetDevice(deviceId));                                                \
    if (verbose)                                                                          \
      Log(info, "Launching %s with nblocks: %u, blockDim: %u", #kernel, gridSize, BSize); \
    kernel<<<grid, block>>>(__VA_ARGS__);                                                 \
    CUDA_RUNTIME(cudaGetLastError());                                                     \
    CUDA_RUNTIME(cudaDeviceSynchronize());                                                \
  }

#define execKernelStream(kernel, gridSize, BSize, __stream, verbose, ...) \
  {                                                                       \
    /* 1) Prep */                                                         \
    dim3 grid(gridSize);                                                  \
    dim3 block(BSize);                                                    \
    if (verbose)                                                          \
      Log(info, "Launching %s on stream %p with nblocks:%u, blockDim:%u", \
          #kernel, (void *)__stream, gridSize, BSize);                    \
    kernel<<<grid, block, 0, __stream>>>(__VA_ARGS__);                    \
    CUDA_RUNTIME(cudaGetLastError());                                     \
    CUDA_RUNTIME(cudaStreamSynchronize(__stream));                        \
  }

inline cudaStream_t &thread_stream()
{
  static thread_local cudaStream_t s = []
  {
    cudaStream_t tmp;
    cudaStreamCreate(&tmp);
    return tmp;
  }();
  return s;
}