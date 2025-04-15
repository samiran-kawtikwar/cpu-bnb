#pragma once
#include "config.h"
#include "problem_info.h"
#include "../utils/logger.cuh"
#include "../defs.cuh"
#include <cmath>
#include "../LAP-cpu/hungarian-algorithm-cpp/Hungarian.h"
#include "../LAP/Hung_Tlap.cuh"

void sanity_check(double *costs, int N)
{
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      if (costs[i * N + j] < 0)
      {
        Log(critical, "Negative cost found at (%d, %d): %f\n", i, j, costs[i * N + j]);
        printHostMatrix(costs, N, N, "costs");
        assert(false);
      }
    }
  }
}

template <typename cost_type = uint>
__global__ void populate_costs(const uint N, const int *fa, const int *la,
                               const cost_type *distances, const cost_type *flows, TILED_HANDLE<cost_type> th)
{
  // Retrieve problem size and pointers to distance and flow arrays.
  cost_type *costs = &th.slack[blockIdx.x * N * N];
  cost_type *cpy = &th.cost[blockIdx.x * N * N];
  const uint i = blockIdx.x / N, k = blockIdx.x % N;
  // Calculate a unique linear thread ID from the 2D thread indices.
  const uint tid = threadIdx.x;
  const uint total_elements = N * N;

  // Each thread processes multiple cost elements.
  for (uint idx = tid; idx < total_elements; idx += blockDim.x)
  {
    // Map the linear index to 2D indices.
    const uint j = idx / N;
    const uint l = idx % N;

    if (fa[j] > -1)
    {
      // Fixed assignment mapping: only one location is valid.
      if (l == static_cast<uint>(fa[j]))
        costs[idx] = flows[i * N + j] * distances[k * N + l];
      else
        costs[idx] = MAX_DATA;
    }
    else
    {
      // When facility j is unassigned.
      if (la[l] == -1)
        costs[idx] = flows[i * N + j] * distances[k * N + l];
      else
        costs[idx] = MAX_DATA;

      // Override cost values when additional restrictions apply.
      if ((j == i && k != l) || (l == k && i != j))
        costs[idx] = MAX_DATA;
    }
  }
  __syncthreads();

  // only need to solve these problems
  if ((fa[i] == -1 && la[k] == -1) || (fa[i] > -1 && la[k] == i))
  {
    // Copy the computed costs
    for (uint idx = tid; idx <= total_elements; idx += blockDim.x)
      cpy[idx] = costs[idx];
    __syncthreads();
    THA_device<cost_type>(th);
  }
}

template <typename cost_type = uint>
void populate_costs(const problem_info info, const uint i, const uint k, const int *fa, const int *la, cost_type *costs)
{
  const uint N = info.N;
  cost_type *distances = info.distances, *flows = info.flows;

  std::fill(costs, costs + N * N, -1.0);
  // iterate over all facilities
  for (uint j = 0; j < N; j++)
  {
    // Fixed assignment mapping
    if (fa[j] > -1)
    {
      for (uint l = 0; l < N; l++)
      {
        if (l == fa[j])
          costs[j * N + l] = flows[i * N + j] * distances[k * N + l];
        else
          costs[j * N + l] = MAX_DATA;
      }
      continue;
    }
    for (uint l = 0; l < N; l++)
    {
      // if location l is unoccupied
      if (la[l] == -1)
        costs[j * N + l] = flows[i * N + j] * distances[k * N + l];
      else // l is occupied but j is unoccupied
        costs[j * N + l] = MAX_DATA;
      if ((j == i && k != l) || (l == k && i != j))
        costs[j * N + l] = MAX_DATA;
    }
  }
}

template <typename cost_type = uint>
cost_type update_bounds_GL(const problem_info &pinfo, node &node, TLAP<cost_type> &tlap)
{
  static Timer t;
  const uint N = pinfo.N;
  cost_type *dist = pinfo.distances;
  cost_type *flows = pinfo.flows;
  int *fa = node.value->fixed_assignments; // facilities assigned
  int *la = new int[N];                    // locations assigned
  int *temp_ass = new int[N];
  std::fill(la, la + N, -1);
  // Update problem according to fixed assignments
  for (uint i = 0; i < N; i++)
  {
    if (fa[i] > -1)
      la[fa[i]] = i;
  }
  // printHostArray(la, N, "la");
  double *z = new double[N * N];
  HungarianAlgorithm HungAlgo;
  TILED_HANDLE<cost_type> th = tlap.th;
  int *dfa, *dla;
  CUDA_RUNTIME(cudaMalloc((void **)&dfa, N * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc((void **)&dla, N * sizeof(int)));
  CUDA_RUNTIME(cudaMemcpy(dfa, fa, N * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_RUNTIME(cudaMemcpy(dla, la, N * sizeof(int), cudaMemcpyHostToDevice));

  execKernel(populate_costs<cost_type>, N * N, BlockSize, 0, false, N, dfa, dla, dist, flows, th);
  // execKernel(THA<cost_type>, N * N, BlockSize, 0, false, tlap.th);
  // printDeviceArray<cost_type>(th.objective, N * N, "Objectives");
  for (uint i = 0; i < N; i++)
  {
    for (uint k = 0; k < N; k++)
    {
      if ((fa[i] == -1 && la[k] == -1) || (fa[i] > -1 && la[k] == i)) // the facility and locations are unassigned
        z[N * i + k] = double(th.objective[N * i + k]);
      else
        z[N * i + k] = DBL_MAX;
    }
  }

  // printHostMatrix<double>(z, N, N, "Z costs:");
  double GL_bound = 0.0;
  HungAlgo.assignmentoptimal(temp_ass, &GL_bound, z, int(N), int(N));
  delete[] z;
  delete[] temp_ass;
  delete[] la;
  CUDA_RUNTIME(cudaFree(dfa));
  CUDA_RUNTIME(cudaFree(dla));
  return cost_type(GL_bound);
}