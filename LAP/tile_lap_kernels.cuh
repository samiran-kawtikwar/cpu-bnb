#pragma once
#include "block_lap_kernels.cuh"
#include "../utils/cuda_utils.cuh"
#include "../defs.cuh"

namespace cg = cooperative_groups;

template <typename data = float>
__global__ void THA(TILED_HANDLE<data> th)
{
  __shared__ PARTITION_HANDLE<data> ph;
  cg::thread_block block = cg::this_thread_block();
  TILE tile = cg::tiled_partition<TileSize>(block);

  set_handles(tile, ph, th);
  sync(tile);
  PHA(tile, ph);
  sync(tile);
  get_objective(tile, ph);
  return;
}

template <typename data = float>
__device__ void THA_device(TILED_HANDLE<data> &th)
{
  __shared__ PARTITION_HANDLE<data> ph;
  cg::thread_block block = cg::this_thread_block();
  TILE tile = cg::tiled_partition<TileSize>(block);

  set_handles(tile, ph, th);
  sync(tile);
  PHA(tile, ph);
  sync(tile);
  get_objective(tile, ph);
  return;
}