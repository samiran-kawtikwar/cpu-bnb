#include "../utils/logger.cuh"
#include "../defs.cuh"
#include "../LAP-cpu/hungarian-algorithm-cpp/Hungarian.h"
#include <cmath>
#include <omp.h>
#include <thread>

double round_to(double val, int places)
{
  float factor = pow(10, places);
  return round(val * factor) / factor;
}

bool feas_check_naive(const problem_info *pinfo, node &node)
{
  const uint N = pinfo->psize, K = pinfo->ncommodities;

  uint *weights = pinfo->weights;
  uint *budgets = pinfo->budgets;
  int *row_fa = node.value->fixed_assignments;

  for (int k = 0; k < K; k++)
  {
    double sum = 0;
    for (int i = 0; i < N; i++)
    {
      if (row_fa[i] != -1)
        sum += weights[k * N * N + i * N + row_fa[i]];
    }
    if (sum > budgets[k])
      return false;
  }
  return true;
}

bool feas_check(const problem_info *pinfo, node &node)
{
  bool feasible = true;
  const uint N = pinfo->psize, K = pinfo->ncommodities;
  uint *weights = pinfo->weights;
  uint *budgets = pinfo->budgets;
  int *row_fa = node.value->fixed_assignments;
  int *row_assignments = new int[N];
  double *lap_costs_fa = new double[N * N];
  double used_budget = 0;
  for (uint k = 0; k < K; k++)
  {
    // Update lap_costs_fa according to fixed assignments
    for (uint i = 0; i < N; i++)
    {
      for (uint j = 0; j < N; j++)
      {
        lap_costs_fa[i * N + j] = double(weights[k * N * N + i * N + j]);
      }
    }
    for (uint i = 0; i < N; i++)
    {
      for (uint j = 0; j < N; j++)
      {
        if (row_fa[i] > -1 && j != row_fa[i])
        {
          lap_costs_fa[i * N + j] = double(MAX_DATA);
        }
      }
    }
    HungarianAlgorithm HungAlgo;
    HungAlgo.assignmentoptimal(row_assignments, &used_budget, lap_costs_fa, int(N), int(N));
    if (used_budget > budgets[k])
    {
      feasible = false;
      break;
    }
  }
  delete[] row_assignments;
  delete[] lap_costs_fa;
  // Log(debug, "Status of node: %s", feasible ? "Feasible" : "Infeasible");
  return feasible;
}

void feas_check_parallel(const problem_info *pinfo, std::vector<node> &nodes, std::vector<bool> feasible)
{
  uint level = nodes[0].value->level - 1;
  const uint N = pinfo->psize, K = pinfo->ncommodities;
  uint *weights = pinfo->weights;
  uint *budgets = pinfo->budgets;

  std::vector<bool> local_feasible(N * K, true);
#pragma omp parallel num_threads(min(N *K, (uint)thread::hardware_concurrency() - 3))
  {
    int *row_assignments = new int[N];
    double *lap_costs_fa = new double[N * N];
#pragma omp for collapse(2) schedule(dynamic)
    for (uint ch_id = 0; ch_id < N - level; ch_id++)
    {
      for (uint k = 0; k < K; k++)
      {
        if (!feasible[ch_id])
          continue;
        int *row_fa = nodes[ch_id].value->fixed_assignments;
        double used_budget = 0;
        // Update lap_costs_fa according to fixed assignments
        for (uint i = 0; i < N; i++)
        {
          for (uint j = 0; j < N; j++)
          {
            lap_costs_fa[i * N + j] = double(weights[k * N * N + i * N + j]);
          }
        }
        for (uint i = 0; i < N; i++)
        {
          for (uint j = 0; j < N; j++)
          {
            if (row_fa[i] > -1 && j != row_fa[i])
            {
              lap_costs_fa[i * N + j] = double(MAX_DATA);
            }
          }
        }
        HungarianAlgorithm HungAlgo;
        HungAlgo.assignmentoptimal(row_assignments, &used_budget, lap_costs_fa, int(N), int(N));
        if (used_budget > budgets[k])
        {
          local_feasible[ch_id * K + k] = false;
        }
      }
    }
    delete[] row_assignments;
    delete[] lap_costs_fa;
  }
  // Log(debug, "Status of node: %s", feasible ? "Feasible" : "Infeasible");
  for (uint ch_id = 0; ch_id < N - level; ch_id++)
  {
    bool is_feasible = true;
    for (uint k = 0; k < K; k++)
    {
      if (!local_feasible[ch_id * K + k])
        is_feasible = false;
    }
    feasible[ch_id] = is_feasible;
  }
}

void update_bounds(const problem_info *pinfo, node &node)
{
  cost_type *h_costs = pinfo->costs;
  const uint psize = pinfo->psize;
  node.value->LB = 0;
  for (uint i = 0; i < psize; i++)
  {
    if (node.value->fixed_assignments[i] != -1)
    {
      node.value->LB += h_costs[i * psize + node.value->fixed_assignments[i]];
    }
  }
  node.key = node.value->LB;
}

uint update_bounds_subgrad(const problem_info *pinfo, node &node, cost_type UB)
{
  const uint N = pinfo->psize, K = pinfo->ncommodities;
  uint *original_costs = pinfo->costs;
  uint *weights = pinfo->weights;
  uint *budgets = pinfo->budgets;
  int *row_fa = node.value->fixed_assignments;
  // Update UB according to the fixed assignments
  for (uint i = 0; i < N; i++)
  {
    if (row_fa[i] > -1)
      UB += original_costs[i * N + row_fa[i]];
  }

  double *LB = new double[MAX_ITER], max_LB;
  double *mult = new double[K];
  int *X = new int[N * N];
  int *row_assignments = new int[N];
  double *g = new double[K];
  float lrate = 2;

  std::fill(LB, LB + MAX_ITER, 0);
  std::fill(mult, mult + K, 0);
  // set max_LB as the max of LB list
  max_LB = *std::max_element(LB, LB + MAX_ITER);

  double *lap_costs = new double[N * N];
  double *lap_costs_fa = new double[N * N];
  for (int t = 0; t < MAX_ITER; t++)
  {
    std::fill(lap_costs, lap_costs + N * N, 0);
    std::fill(g, g + K, 0);
    std::fill(X, X + N * N, 0);

    for (uint i = 0; i < N; i++)
    {
      for (uint j = 0; j < N; j++)
      {
        lap_costs[i * N + j] = double(original_costs[i * N + j]);

        double sum = 0;
        for (uint k = 0; k < K; k++)
        {
          sum += mult[k] * double(weights[k * N * N + i * N + j]);
        }
        lap_costs[i * N + j] += sum;
      }
    }
    memcpy(lap_costs_fa, lap_costs, N * N * sizeof(double));
    // Update lap_costs_fa according to fixed assignments
    for (uint i = 0; i < N; i++)
    {
      for (uint j = 0; j < N; j++)
      {
        if (row_fa[i] > -1 && j != row_fa[i])
        {
          lap_costs_fa[i * N + j] = double(MAX_DATA);
        }
      }
    }
    HungarianAlgorithm HungAlgo;
    HungAlgo.assignmentoptimal(row_assignments, &LB[t], lap_costs_fa, int(N), int(N));

    for (size_t k = 0; k < K; k++)
      LB[t] -= mult[k] * budgets[k];

    // get X from row_assignments
    for (uint i = 0; i < N; i++)
    {
      X[row_assignments[i] * N + i] = 1;
    }

    // Find the difference between the sum of the costs and the budgets
    for (size_t k = 0; k < K; k++)
    {
      for (size_t i = 0; i < N; i++)
      {
        for (size_t j = 0; j < N; j++)
        {
          g[k] += double(X[i * N + j] * weights[k * N * N + i * N + j]);
        }
      }
      g[k] -= double(budgets[k]);
    }
    double denom = 0, feas = 0;
    for (size_t k = 0; k < K; k++)
    {
      denom += g[k] * g[k];
      feas += max(float(0), g[k]);
    }
    if (feas < least_count)
    {
      // Log(debug, "Found feasible solution");
      // Update UB if needed

      goto ret;
    }

    // update multipliers according to subgradient rule
    for (size_t k = 0; k < K; k++)
    {
      mult[k] += max(double(0), lrate * (g[k] * (UB - LB[t])) / denom);
    }

    // Log(info, "Iteration %d, LB: %.3f, UB: %u, lrate: %.3f, Infeasibility: %.3f", t, LB[t], UB, lrate, feas);
    if ((t > 0 && t < 5 && LB[t] < LB[t - 1]) || LB[t] < 0)
    {
      // Log(debug, "Initial Step size too large, restart with smaller step size");
      lrate /= 2;
      t = 0;
      std::fill(mult, mult + K, 0);
    }

    if ((t + 1) % 5 == 0 && LB[t] <= LB[t - 4])
      lrate /= 2;
    if (lrate < 0.005)
      break;
  }
ret:
  // for (uint t = 0; t < MAX_ITER; t++)
  // {
  //   if (LB[t] > 0)
  //     printf("%u, ", uint(ceil(round_to(LB[t], 2))));
  //   else
  //     break;
  // }
  // printf("\n");
  max_LB = max(ceil(round_to(*std::max_element(LB, LB + MAX_ITER), 2)), double(0));
  // for (uint i = 0; i < N; i++)
  // {
  //   printf("%d ", row_fa[i]);
  // }
  // printf("\n");
  // Log(debug, "LB: %u, UB: %u\n", uint(max_LB), UB);

  // Log(info, "Subgrad Solver Gap: %.3lf%%", (UB - max_LB) * 100 / UB);
  delete[] mult;
  delete[] X;
  delete[] g;
  delete[] LB;
  delete[] lap_costs;
  delete[] lap_costs_fa;
  delete[] row_assignments;

  return uint(max_LB);
}

void update_bounds_subgrad_parallel(const problem_info *pinfo, std::vector<node> &nodes,
                                    std::vector<bool> feasible, cost_type UB)
{
  uint level = nodes[0].value->level - 1;
  const uint N = pinfo->psize, K = pinfo->ncommodities;
  uint *original_costs = pinfo->costs;
  uint *weights = pinfo->weights;
  uint *budgets = pinfo->budgets;

#pragma omp parallel for // num_threads(min(N - level, (uint)thread::hardware_concurrency() - 3))
  for (uint ch_id = 0; ch_id < N - level; ch_id++)
  {
    uint my_UB = UB;
    if (feasible[ch_id])
    {
      int *row_fa = nodes[ch_id].value->fixed_assignments;
      // Update UB according to the fixed assignments
      for (uint i = 0; i < N; i++)
      {
        if (row_fa[i] > -1)
          my_UB += original_costs[i * N + row_fa[i]];
      }

      double *LB = new double[MAX_ITER], max_LB;
      double *mult = new double[K];
      int *X = new int[N * N];
      int *row_assignments = new int[N];
      double *g = new double[K];
      float lrate = 2;

      std::fill(LB, LB + MAX_ITER, 0);
      std::fill(mult, mult + K, 0);
      // set max_LB as the max of LB list
      max_LB = *std::max_element(LB, LB + MAX_ITER);

      double *lap_costs = new double[N * N];
      double *lap_costs_fa = new double[N * N];
      for (int t = 0; t < MAX_ITER; t++)
      {
        std::fill(lap_costs, lap_costs + N * N, 0);
        std::fill(g, g + K, 0);
        std::fill(X, X + N * N, 0);

        for (uint i = 0; i < N; i++)
        {
          for (uint j = 0; j < N; j++)
          {
            lap_costs[i * N + j] = double(original_costs[i * N + j]);

            double sum = 0;
            for (uint k = 0; k < K; k++)
            {
              sum += mult[k] * double(weights[k * N * N + i * N + j]);
            }
            lap_costs[i * N + j] += sum;
          }
        }
        memcpy(lap_costs_fa, lap_costs, N * N * sizeof(double));
        // Update lap_costs_fa according to fixed assignments
        for (uint i = 0; i < N; i++)
        {
          for (uint j = 0; j < N; j++)
          {
            if (row_fa[i] > -1 && j != row_fa[i])
            {
              lap_costs_fa[i * N + j] = double(MAX_DATA);
            }
          }
        }
        HungarianAlgorithm HungAlgo;
        HungAlgo.assignmentoptimal(row_assignments, &LB[t], lap_costs_fa, int(N), int(N));

        for (size_t k = 0; k < K; k++)
          LB[t] -= mult[k] * budgets[k];

        // get X from row_assignments
        for (uint i = 0; i < N; i++)
        {
          X[row_assignments[i] * N + i] = 1;
        }

        // Find the difference between the sum of the costs and the budgets
        for (size_t k = 0; k < K; k++)
        {
          for (size_t i = 0; i < N; i++)
          {
            for (size_t j = 0; j < N; j++)
            {
              g[k] += double(X[i * N + j] * weights[k * N * N + i * N + j]);
            }
          }
          g[k] -= double(budgets[k]);
        }
        double denom = 0, feas = 0;
        for (size_t k = 0; k < K; k++)
        {
          denom += g[k] * g[k];
          feas += max(float(0), g[k]);
        }
        if (feas < least_count)
        {
          // Log(debug, "Found feasible solution");
          // Update UB if needed

          goto ret;
        }

        // update multipliers according to subgradient rule
        for (size_t k = 0; k < K; k++)
        {
          mult[k] += max(double(0), lrate * (g[k] * (my_UB - LB[t])) / denom);
        }

        // Log(info, "Iteration %d, LB: %.3f, UB: %u, lrate: %.3f, Infeasibility: %.3f", t, LB[t], UB, lrate, feas);
        if ((t > 0 && t < 5 && LB[t] < LB[t - 1]) || LB[t] < 0)
        {
          // Log(debug, "Initial Step size too large, restart with smaller step size");
          lrate /= 2;
          t = 0;
          std::fill(mult, mult + K, 0);
        }

        if ((t + 1) % 5 == 0 && LB[t] <= LB[t - 4])
          lrate /= 2;
        if (lrate < 0.005)
          break;
      }
    ret:

      max_LB = max(ceil(round_to(*std::max_element(LB, LB + MAX_ITER), 2)), double(0));

      delete[] lap_costs;
      delete[] lap_costs_fa;
      delete[] LB;
      delete[] mult;
      delete[] X;
      delete[] row_assignments;
      delete[] g;
      nodes[ch_id].value->LB = uint(max_LB);
      nodes[ch_id].key = uint(max_LB);
    }
  }
}