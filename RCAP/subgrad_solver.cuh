#pragma once

#include "subgrad_utils.cuh"

struct subgrad_space
{
  float *mult, *g, *lap_costs, *LB, *real_obj, *max_LB;
  int *X;
  int *col_fixed_assignments;

  __host__ void allocate(uint N, uint K, uint nworkers = 0, uint devID = 0)
  {
    nworkers = (nworkers == 0) ? N : nworkers;
    // allocate space for mult, g, lap_costs, LB, LB_old, X, and th
    CUDA_RUNTIME(cudaMalloc((void **)&mult, nworkers * K * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&g, nworkers * K * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&lap_costs, nworkers * N * N * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&X, nworkers * N * N * sizeof(int)));
    CUDA_RUNTIME(cudaMalloc((void **)&LB, nworkers * MAX_ITER * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&max_LB, nworkers * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&real_obj, nworkers * sizeof(float)));
    CUDA_RUNTIME(cudaMalloc((void **)&col_fixed_assignments, nworkers * N * sizeof(int)));

    CUDA_RUNTIME(cudaMemset(mult, 0, nworkers * K * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(g, 0, nworkers * K * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(lap_costs, 0, nworkers * N * N * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(X, 0, nworkers * N * N * sizeof(int)));
    CUDA_RUNTIME(cudaMemset(LB, 0, nworkers * MAX_ITER * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(max_LB, 0, nworkers * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(real_obj, 0, nworkers * sizeof(float)));
    CUDA_RUNTIME(cudaMemset(col_fixed_assignments, 0, nworkers * N * sizeof(int)));

    T = TLAP<float>(nworkers, N, devID);
    // T.allocate(nworkers, N, devID);
  };
  __host__ void clear()
  {
    CUDA_RUNTIME(cudaFree(mult));
    CUDA_RUNTIME(cudaFree(g));
    CUDA_RUNTIME(cudaFree(lap_costs));
    CUDA_RUNTIME(cudaFree(LB));
    CUDA_RUNTIME(cudaFree(real_obj));
    CUDA_RUNTIME(cudaFree(max_LB));
    CUDA_RUNTIME(cudaFree(X));
    CUDA_RUNTIME(cudaFree(col_fixed_assignments));
    T.th.clear();
  }
};

template <typename cost_type, typename weight_type>
weight_type subgrad_solver(const cost_type *original_costs, cost_type upper, const weight_type *weights, weight_type *budgets, uint N, uint K, uint dev_ = 0)
{
  Log(info, "Starting subgrad solver");
  // For root node
  float *LB = new float[MAX_ITER], UB = float(upper), max_LB;
  float *mult = new float[K];
  int *X = new int[N * N];
  float *g = new float[K];
  std::fill(mult, mult + K, 0);
  std::fill(X, X + N * N, 0);
  std::fill(LB, LB + MAX_ITER, 0);
  float *lap_costs;
  CUDA_RUNTIME(cudaMallocManaged((void **)&lap_costs, N * N * sizeof(float)));
  float lrate = 2;
  for (size_t t = 0; t < MAX_ITER; t++)
  {
    std::fill(lap_costs, lap_costs + N * N, 0);
    std::fill(g, g + K, 0);

    for (size_t i = 0; i < N; i++)
    {
      for (size_t j = 0; j < N; j++)
      {
        lap_costs[i * N + j] = original_costs[i * N + j];
        float sum = 0;
        for (size_t k = 0; k < K; k++)
        {
          sum += mult[k] * weights[k * N * N + i * N + j];
        }
        lap_costs[i * N + j] += sum;
      }
    }
    float neg = 0;
    for (size_t k = 0; k < K; k++)
    {
      neg += mult[k] * budgets[k];
    }

    LAP<float> lap = LAP<float>(lap_costs, N, dev_);
    LB[t] = lap.full_solve() - neg;
    // lap.print_solution();
    lap.get_X(X);

    // Find the difference between the sum of the costs and the budgets
    for (size_t k = 0; k < K; k++)
    {
      for (size_t i = 0; i < N; i++)
      {
        for (size_t j = 0; j < N; j++)
        {
          g[k] += float(X[i * N + j] * weights[k * N * N + i * N + j]);
        }
      }
      g[k] -= float(budgets[k]);
    }

    float denom = 0, feas = 0;
    for (size_t k = 0; k < K; k++)
    {
      denom += g[k] * g[k];
      feas += max(float(0), g[k]);
    }
    if (feas < least_count)
    {
      Log(debug, "Found feasible solution");
      // Update UB if needed

      goto ret;
    }
    Log(info, "It %d, LB: %.3f, UB: %.3f, lrate: %.3f, Infeasibility: %.3f", t, LB[t], UB, lrate, feas);
    // Update multipliers according to subgradient rule
    for (size_t k = 0; k < K; k++)
    {
      mult[k] += max(float(0), lrate * (g[k] * (UB - LB[t])) / denom);
    }
    // print lap_costs
    // for (size_t i = 0; i < N; i++)
    // {
    //   for (size_t j = 0; j < N; j++)
    //   {
    //     printf("%.2f ", lap_costs[i * N + j]);
    //   }
    //   printf("\n");
    // }

    // print mult and g
    // for (size_t i = 0; i < K; i++)
    //   printf("%.2f ", mult[i]);
    // printf("\n");
    // for (size_t i = 0; i < K; i++)
    //   printf("%.2f ", g[i]);
    // printf("\n");
    // printf("denom: %.2f\n", denom);
    if ((t > 0 && t < 5 && LB[t] < LB[t - 1]) || LB[t] < 0)
    {
      Log(debug, "Initial Step size too large, restart with smaller step size");
      lrate /= 2;
      t = 0;
      std::fill(mult, mult + K, 0);
    }

    if ((t + 1) % 5 == 0 && LB[t] <= LB[t - 4])
      lrate /= 2;
    if (lrate < 0.005)
      break;
  }
  // max_LB = max(LB)
ret:
  max_LB = ceil(*std::max_element(LB, LB + MAX_ITER));
  Log(info, "Subgrad Solver Gap: %.3f%%", (UB - max_LB) * 100 / UB);
  delete[] mult;
  delete[] X;
  delete[] g;
  delete[] LB;
  CUDA_RUNTIME(cudaFree(lap_costs));
  // Print Gap
  return uint(max_LB);
}
