#pragma once
#include "config.h"
#include "problem_info.h"
#include "../utils/logger.cuh"
#include "../defs.cuh"
#include <cmath>
#include "../LAP-cpu/hungarian-algorithm-cpp/Hungarian.h"

cost_type update_bounds_GL(const problem_info *pinfo, node &node, cost_type UB)
{
  const uint N = pinfo->N;
  uint *distances = pinfo->distances, *flows = pinfo->flows;
  int *fa = node.value->fixed_assignments; // facilities assigned
  int *la = new int[N];                    // locations assigned
  std::fill(la, la + N, -1);
  // Update problem according to fixed assignments
  for (uint i = 0; i < N; i++)
  {
    if (fa[i] > -1)
      la[fa[i]] = i;
  }

  double *z = new double[N * N];
  HungarianAlgorithm HungAlgo;

  double *temp_costs = new double[N * N];
  int *temp_ass = new int[N];
  for (uint i = 0; i < N; i++)
  {
    for (uint k = 0; k < N; k++)
    {
      // z_{i,k} = LAP solution if i is assigned to k
      std::fill(temp_costs, temp_costs + N * N, -1.0);
      // iterate over all facilities
      for (uint j = 0; j < N; j++)
      {
        // Fixed assignment mapping
        if (fa[j] > -1)
        {
          for (uint l = 0; l < N; l++)
          {
            if (l == fa[j])
              temp_costs[j * N + l] = (double)flows[i * N + j] * distances[k * N + l];
            else
              temp_costs[j * N + l] = DBL_MAX;
          }
          continue;
        }
        for (uint l = 0; l < N; l++)
        {
          // if location l is unoccupied
          if (la[l] == -1)
            temp_costs[j * N + l] = (double)flows[i * N + j] * distances[k * N + l];

          if ((j == i && k != l) || (l == k && i != j))
            temp_costs[j * N + l] = DBL_MAX;
        }
      }
      printHostMatrix(temp_costs, N, N, "temp_costs");

      // Solve LAP
      HungAlgo.assignmentoptimal(temp_ass, &z[N * i + k], temp_costs, int(N), int(N));
    }
  }
  delete[] temp_costs;

  printHostMatrix(z, N, N, "z");
  double GL_bound = 0.0;
  HungAlgo.assignmentoptimal(temp_ass, &GL_bound, z, int(N), int(N));
  delete[] z;
  delete[] temp_ass;
  delete[] la;
  return cost_type(GL_bound);
}