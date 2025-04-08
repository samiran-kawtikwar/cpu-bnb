#pragma once
#include "config.h"

template <typename T = double>
struct problem_info
{
  uint N, K, SP;

  int *cycle;
  T *cost_matrix;
  int *dim1, *dim2;
  problem_info(uint nodes, uint frames)
  {
    N = nodes;
    K = frames;
    SP = K * (K - 1) / 2;
  }
  ~problem_info()
  {
    delete[] cycle;
    delete[] cost_matrix;
    delete[] dim1;
    delete[] dim2;
  }
};