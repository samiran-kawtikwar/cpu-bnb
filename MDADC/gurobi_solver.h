#pragma once

typedef unsigned int uint;
#include "problem_info.h"
#include "../utils/logger.cuh"

template <typename cost_type>
cost_type solve_with_gurobi(problem_info<cost_type> *pinfo);