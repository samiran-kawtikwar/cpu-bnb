#pragma once
#include "config.h"
#include "problem_info.h"
#include "../utils/logger.cuh"
#include "../defs.cuh"
#include <cmath>
#include <omp.h>
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
cost_type update_bounds_GL(const problem_info *pinfo, node &node)
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

#pragma omp parallel for
  for (uint id = 0; id < N * N; id++)
  {
    uint i = id / N;
    uint k = id % N;
    HungarianAlgorithm HungAlgo;
    double *temp_costs = new double[N * N];
    int *temp_ass = new int[N];
    if ((fa[i] == -1 && la[k] == -1) || (fa[i] > -1 && la[k] == i)) // the facility and locations are unassigned
    {
      // z_{i,k} = LAP solution if i is assigned to k
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
    delete[] temp_costs;
    delete[] temp_ass;
  }

  // printHostMatrix(z, N, N, "z costs");
  double GL_bound = 0.0;
  int *temp_ass = new int[N];
  HungarianAlgorithm HungAlgo;
  HungAlgo.assignmentoptimal(temp_ass, &GL_bound, z, int(N), int(N));
  delete[] z;
  delete[] temp_ass;
  delete[] la;
  return cost_type(GL_bound);
}

template <typename cost_type = uint>
void update_bounds_poly_GL(const problem_info *pinfo, const uint level, std::vector<node> &children)
{
  const uint N = pinfo->N;
  for (uint child_id = 0; child_id < N - level; child_id++)
  {
    int *fa = children[child_id].value->fixed_assignments; // facilities assigned
    int *la = new int[N];                                  // locations assigned
    std::fill(la, la + N, -1);
    // Update problem according to fixed assignments
    for (uint i = 0; i < N; i++)
    {
      if (fa[i] > -1)
        la[fa[i]] = i;
    }
    // printHostArray(la, N, "la");
    double *z = new double[N * N];

#pragma omp parallel for
    for (uint id = 0; id < N * N; id++)
    {
      uint i = id / N;
      uint k = id % N;
      HungarianAlgorithm HungAlgo;
      double *temp_costs = new double[N * N];
      int *temp_ass = new int[N];
      if ((fa[i] == -1 && la[k] == -1) || (fa[i] > -1 && la[k] == i)) // the facility and locations are unassigned
      {
        // z_{i,k} = LAP solution if i is assigned to k
        populate_costs<cost_type>(pinfo, i, k, fa, la, temp_costs);
        // Solve LAP
        HungAlgo.assignmentoptimal(temp_ass, &z[N * i + k], temp_costs, int(N), int(N));
      }
      else
        z[N * i + k] = DBL_MAX;
      delete[] temp_costs;
      delete[] temp_ass;
    }

    // printHostMatrix(z, N, N, "z costs");
    double GL_bound = 0.0;
    int *temp_ass = new int[N];
    HungarianAlgorithm HungAlgo;
    HungAlgo.assignmentoptimal(temp_ass, &GL_bound, z, int(N), int(N));
    children[child_id].key = cost_type(GL_bound);
    children[child_id].value->LB = children[child_id].key;
    delete[] temp_ass, la, z;
  }
  return;
}

template <typename cost_type = uint>
void update_bounds_poly_GL_parallel(const problem_info *pinfo,
                                    const uint level,
                                    std::vector<node> &children)
{
  const uint N = pinfo->N;
  const uint nCh = N - level;

  // --- precompute each child's la[] and a big z[] block ---
  std::vector<std::vector<int>> las(nCh, std::vector<int>(N, -1));
  std::vector<double> Z(nCh * N * N, DBL_MAX);

// 1) build la for each child
#pragma omp for schedule(static)
  for (uint c = 0; c < nCh; ++c)
  {
    auto &la = las[c];
    int *fa = children[c].value->fixed_assignments;
    std::fill(la.begin(), la.end(), -1);
    for (uint i = 0; i < N; ++i)
      if (fa[i] >= 0)
        la[fa[i]] = int(i);
    // Z is already initialized to DBL_MAX
  }
  Log(info, "populated la");
// 2) fill z[c,i,k] in one big 3-D loop
#pragma omp for collapse(3) schedule(dynamic)
  for (uint c = 0; c < nCh; ++c)
  {
    for (uint i = 0; i < N; ++i)
    {
      for (uint k = 0; k < N; ++k)
      {
        int *fa = children[c].value->fixed_assignments;
        auto &la = las[c];
        bool ok = (fa[i] == -1 && la[k] == -1) || (fa[i] >= 0 && la[k] == int(i));
        if (!ok)
          continue;

        std::vector<double> tmp_costs(N * N);
        std::vector<int> tmp_ass(N);

        populate_costs<cost_type>(
            pinfo, i, k, fa, la.data(), tmp_costs.data());

        double &zck = Z[c * N * N + i * N + k];
        HungarianAlgorithm hung;
        hung.assignmentoptimal(
            tmp_ass.data(), &zck,
            tmp_costs.data(), int(N), int(N));
      }
    }
  }
  Log(info, "found z matrices");
// 3) final LAP on each child's Z-block
#pragma omp for schedule(dynamic)
  for (uint c = 0; c < nCh; ++c)
  {
    double glb = 0.0;
    std::vector<int> final_ass(N);
    HungarianAlgorithm hung;
    hung.assignmentoptimal(
        final_ass.data(),
        &glb,
        &Z[c * N * N],
        int(N),
        int(N));
    children[c].key = cost_type(glb);
    children[c].value->LB = children[c].key;
  }
}
