#pragma once
#include "config.h"

struct problem_info
{
  uint N;

  cost_type *distances;
  cost_type *flows;
  uint opt_objective;

  problem_info(uint user_n)
  {
    N = user_n;
  }
  ~problem_info()
  {
    delete[] distances;
    delete[] flows;
  }
};