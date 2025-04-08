#pragma once
#include <random>
#include <omp.h>
#include <iostream>
#include <cstring>
#include "config.h"
#include "problem_info.h"
#include "../utils/timer.h"
#include "../utils/logger.cuh"
#include "sfmt/sfmt.h"

using namespace std;

int *createProbGenData(Config config, const unsigned long seed)
{
  Log(debug, "Generating cycle data");
  uint N = config.user_nnodes;
  uint K = config.user_nframes;

  // Create problem gen data
  int *cycle = new int[N * K];
  CRandomSFMT randomGenerator(seed);
  for (int i = 0; i < K; i++)
  {
    for (int j = 0; j < N; j++)
    {
      cycle[i * N + j] = j;
    }

    for (int j = 0; j < N; j++)
    {
      int j1 = randomGenerator.IRandomX(j, N - 1);
      int temp = cycle[i * N + j1];
      cycle[i * N + j1] = cycle[i * N + j];
      cycle[i * N + j] = temp;
    }
  }

  // *********print cycle array *********
  // for (uint i = 0; i < N; i++)
  // {
  //   for (uint j = 0; j < K; j++)
  //   {
  //     cout << cycle[i * N + j] << " ";
  //   }
  //   cout << endl;
  // }
  Log(debug, "Generated cycle data");
  return cycle;
}

int *createDim1(Config config)
{
  uint K = config.user_nframes;
  int SP = K * (K - 1) / 2;

  int *dim1 = new int[SP];
  memset(dim1, 0, SP * sizeof(int));
  uint loc = 0;
  for (int i = 0; i < K; i++)
  {
    for (int j = i + 1; j < K; j++)
    {
      dim1[loc] = i;
      loc++;
    }
  }
  // *********print dim1 array *********
  // printf(("loc: %u\n"), loc);
  // cout << "dim1: ";
  // for (uint i = 0; i < SP; i++)
  // {
  //   cout << dim1[i] << " ";
  // }
  // cout << endl;
  return dim1;
}

int *createDim2(Config config)
{
  uint K = config.user_nframes;
  int SP = K * (K - 1) / 2;

  int *dim2 = new int[SP];
  memset(dim2, 0, SP * sizeof(int));
  uint loc = 0;
  for (int i = 0; i < K; i++)
  {
    for (int j = i + 1; j < K; j++)
    {
      dim2[loc] = j;
      loc++;
    }
  }
  // *********print dim1 array *********
  // printf(("loc: %u\n"), loc);
  // cout << "dim2: ";
  // for (uint i = 0; i < SP; i++)
  // {
  //   cout << dim2[i] << " ";
  // }
  // cout << endl;
  return dim2;
}

template <typename T = double>
T *generateNormalSubProblem(Config config, problem_info<T> *info, unsigned long seed)
{
  Log(debug, "Generating cost matrix with gaussian noise");
  info->dim1 = createDim1(config);
  info->dim2 = createDim2(config);
  Log(debug, "dim1 and dim2 created");
  uint N = config.user_nnodes;

  int *dim1 = info->dim1;
  int *dim2 = info->dim2;
  int SP = info->SP;
  int *cycle = info->cycle;
  double sigma = config.sigma;
  uint64_t cost_matrix_size = uint64_t(N) * N * SP;
  double *cost_matrix = new double[cost_matrix_size];
  //	double m = 0;
  Log(debug, "Generating cost matrix");

  for (int i = 0; i < SP; i++) // SP = K * (K - 1) / 2
  {
    CRandomSFMT randomGenerator(seed + i);
    for (int j = 0; j < N; j++)
    {
      for (int k = 0; k < N; k++)
      {
        int iSP = dim1[i], jSP = dim2[i];
        double val = randomGenerator.Normal(std::abs(cycle[iSP * N + j] - cycle[jSP * N + k]) - 1, sigma);
        uint64_t index = uint64_t(i * N * N) + (j * N) + k;
        assert(index < uint64_t(cost_matrix_size));
        cost_matrix[index] = val;
      }
    }
  }

  Log(debug, "Cost matrix generated");
  //************print cost_matrix**********
  // for (int i = 0; i < SP; i++)
  // {
  //   for (int j = 0; j < N; j++)
  //   {
  //     for (int k = 0; k < N; k++)
  //     {
  //       int iSP = dim1[i], jSP = dim2[i];
  //       uint64_t index = uint64_t(i * N * N) + (j * N) + k;
  //       cout << cost_matrix[index] << " ";
  //     }
  //     cout << endl;
  //   }
  //   cout << endl
  //        << endl;
  // }
  // cout << endl;
  return cost_matrix;
}

template <typename T = double>
problem_info<T> *generate_problem(Config config, int seed = 45345)
{
  Log(debug, "Generating problem with seed %d", seed);
  problem_info<T> *info = new problem_info(config.user_nnodes, config.user_nframes);

  // Generate cycle data
  info->cycle = createProbGenData(config, seed);

  // Generate cost matrix
  info->cost_matrix = generateNormalSubProblem(config, info, seed);

  return info;
}

template <typename T = double>
void print(problem_info<T> *info, bool cycle = true, bool cost_matrix = true)
{
  uint N = info->N;
  uint K = info->K;
  uint SP = info->SP;
  if (cycle)
  {
    Log(debug, "Cycle: ");
    for (size_t i = 0; i < K; i++)
    {
      for (size_t j = 0; j < N; j++)
      {
        printf("%u, ", info->cycle[i * N + j]);
      }
      printf("\n");
    }
  }
  if (cost_matrix)
  {
    Log(debug, "Cost Matrix: ");
    for (size_t i = 0; i < SP; i++)
    {
      for (size_t j = 0; j < N; j++)
      {
        for (size_t k = 0; k < N; k++)
        {
          uint64_t index = uint64_t(i * N * N) + (j * N) + k;
          printf("%.3f, ", float(info->cost_matrix[index]));
        }
        printf("\n");
      }
      printf("\n\n");
    }
    printf("\n");
  }
}