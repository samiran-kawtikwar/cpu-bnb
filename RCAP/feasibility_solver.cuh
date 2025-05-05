#pragma once

#include "../defs.cuh"
#include "../utils/logger.cuh"
#include "../utils/cuda_utils.cuh"
#include "../LAP/Hung_Tlap.cuh"
#include "rcap_kernels.cuh"

struct feasibility_space
{
  int *row_fa, *col_fa;
  float *lap_costs_fa;
  TLAP<float> T;

  __host__ void allocate(uint N, uint K, uint devID = 0)
  {
    // allocate space for lap_costs_fa
    CUDA_RUNTIME(cudaMallocManaged((void **)&row_fa, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_fa, N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&lap_costs_fa, N * N * K * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(lap_costs_fa, 0, N * N * K * sizeof(float)));

    T = TLAP<float>(K, N, devID);
  };

  __host__ void clear()
  {
    CUDA_RUNTIME(cudaFree(row_fa));
    CUDA_RUNTIME(cudaFree(col_fa));
    CUDA_RUNTIME(cudaFree(lap_costs_fa));
    T.clear();
  }

  static void allocate_all(feasibility_space *space, uint N, uint K, uint nworkers, uint devID = 0)
  {
    for (uint i = 0; i < nworkers; i++)
      space[i].allocate(N, K, devID);
  }
  static void free_all(feasibility_space *space, uint nworkers)
  {
    for (uint i = 0; i < nworkers; i++)
      space[i].clear();
  }
};

// called by a block
__device__ __forceinline__ void feas_check(const problem_info *pinfo, TILE tile,
                                           node *a, const int *row_fa, int *col_fa,
                                           float *lap_costs, bool &feasible,
                                           PARTITION_HANDLE<float> &ph)
{
  const uint psize = pinfo->psize, ncommmodities = pinfo->ncommodities;
  const uint local_id = tile.thread_rank();

  if (local_id == 0)
  {
    feasible = true;
    ph.cost = lap_costs;
  }
  sync(tile);

  // set col_fa using row_fa
  for (uint i = local_id; i < psize; i += TileSize)
  {
    col_fa[i] = -1;
  }
  sync(tile);

  for (uint i = local_id; i < psize; i += TileSize)
  {
    if (row_fa[i] > -1)
      col_fa[row_fa[i]] = i;
  }
  sync(tile);

  for (uint k = 0; k < ncommmodities; k++)
  {
    // copy weights to lap_costs for further operations
    for (uint i = local_id; i < psize * psize; i += TileSize)
    {
      lap_costs[i] = float(pinfo->weights[k * psize * psize + i]);
    }
    sync(tile);

    PHA_fa<float>(tile, ph, row_fa, col_fa, 1);
    sync(tile);
    get_objective(tile, ph);
    if (tile.thread_rank() == 0)
    {
      float used_budget = ph.objective[0];
      if (used_budget > float(pinfo->budgets[k]))
        feasible = false;
      // printf("%u \t commodity %u: used budget %f, avl budget %f\n", blockIdx.x, k, used_budget, float(pinfo->budgets[k]));
    }
    sync(tile);
    if (!feasible)
      break;
  }
  sync(tile);
}

// kernel configuration: psize
__global__ void g_feas_check(const problem_info *pinfo, node *a,
                             feasibility_space *space,
                             bool *feasible)
{
  __shared__ PARTITION_HANDLE<float> ph;
  cg::thread_block block = cg::this_thread_block();
  TILE tile = cg::tiled_partition<TileSize>(block);
  set_handles(tile, ph, space->T.th);
  const int *row_fa = space[blockIdx.x].row_fa;

  feas_check(pinfo, tile,
             &a[blockIdx.x], row_fa, space[blockIdx.x].col_fa,
             space[blockIdx.x].lap_costs_fa, feasible[blockIdx.x], ph);
}

void feas_check_gpu(const problem_info *pinfo, const uint nchild, std::vector<node> &nodes, bool *feasible,
                    node *d_children, feasibility_space *d_feas_space, const uint dev_ = 0)
{
  const uint N = pinfo->psize;

  // copy children nodes to gpu
  CUDA_RUNTIME(cudaMemcpy(d_children, nodes.data(), sizeof(node) * nchild, cudaMemcpyHostToDevice));
  // copy fixed assignments to gpu
  for (uint ch_id = 0; ch_id < nchild; ch_id++)
  {
    int *d_row_fa = d_feas_space[ch_id].row_fa;
    int *row_fa = nodes[ch_id].value->fixed_assignments;
    CUDA_RUNTIME(cudaMemset(d_row_fa, -1, N * sizeof(int)));
    for (uint i = 0; i < N; i++)
      d_row_fa[i] = row_fa[i];
  }
  CUDA_RUNTIME(cudaDeviceSynchronize());

  execKernel(g_feas_check, nchild, BlockSize, dev_, false,
             pinfo, d_children,
             d_feas_space, feasible);
  CUDA_RUNTIME(cudaDeviceSynchronize());
  return;
}
