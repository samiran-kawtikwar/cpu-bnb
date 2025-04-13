#pragma once
#include "config.h"
#include "problem_info.h"
#include "../utils/logger.cuh"
#include "../defs.cuh"
#include <cmath>
#include "../LAP-cpu/hungarian-algorithm-cpp/Hungarian.h"

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
void populate_costs(const problem_info *info, const uint i, const uint k, const int *fa, const int *la, double *costs)
{
  const uint N = info->N;
  cost_type *distances = info->distances, *flows = info->flows;

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
          costs[j * N + l] = (double)flows[i * N + j] * distances[k * N + l];
        else
          costs[j * N + l] = DBL_MAX;
      }
      continue;
    }
    for (uint l = 0; l < N; l++)
    {
      // if location l is unoccupied
      if (la[l] == -1)
        costs[j * N + l] = (double)flows[i * N + j] * distances[k * N + l];
      else // l is occupied but j is unoccupied
        costs[j * N + l] = DBL_MAX;
      if ((j == i && k != l) || (l == k && i != j))
        costs[j * N + l] = DBL_MAX;
    }
  }
}

template <typename cost_type = uint>
cost_type update_bounds_GL(const problem_info *pinfo, node &node, cost_type UB)
{
  static Timer t;
  const uint N = pinfo->N;
  int *fa = node.value->fixed_assignments; // facilities assigned
  int *la = new int[N];                    // locations assigned
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

  double *temp_costs = new double[N * N];
  int *temp_ass = new int[N];
  for (uint i = 0; i < N; i++)
  {
    for (uint k = 0; k < N; k++)
    {
      if ((fa[i] == -1 && la[k] == -1) || (fa[i] > -1 && la[k] == i)) // the facility and locations are unassigned
      {                                                               // z_{i,k} = LAP solution if i is assigned to k
        populate_costs<cost_type>(pinfo, i, k, fa, la, temp_costs);
        // printHostMatrix(temp_costs, N, N, "temp_costs");
        // sanity_check(temp_costs, N);
        // Log(info, "Passed check for i: %u, k: %u", i, k);
        // if (fa[0] <= 0 && fa[1] == -1 && fa[2] == -1 && fa[3] == 1)
        // {
        //   std::string str = "c_" + std::to_string(i) + std::to_string(k);
        //   printHostMatrix(temp_costs, N, N, str.c_str());
        // }
        // Solve LAP
        HungAlgo.assignmentoptimal(temp_ass, &z[N * i + k], temp_costs, int(N), int(N));
      }
      else
        z[N * i + k] = DBL_MAX;
    }
  }
  delete[] temp_costs;

  // printHostMatrix(z, N, N, "z costs");
  double GL_bound = 0.0;
  HungAlgo.assignmentoptimal(temp_ass, &GL_bound, z, int(N), int(N));
  delete[] z;
  delete[] temp_ass;
  delete[] la;
  return cost_type(GL_bound);
}