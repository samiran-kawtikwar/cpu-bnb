#pragma once
#include "config.h"

struct problem_info
{
  uint psize, ncommodities;
  cost_type *costs;     // cost of assigning
  weight_type *weights; // weight of each commodity
  weight_type *budgets; // capacity of each commodity

  ~problem_info()
  {
    delete[] costs;
    delete[] weights;
    delete[] budgets;
  }
};