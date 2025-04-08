// Solve RCAP with Gurobi

#include <gurobi_c++.h>
#include <sstream>
#include "stdio.h"
#include "gurobi_solver.h"
#include <cassert>

uint64_t reverse_index(uint p, uint q, uint K)
{
  uint64_t index = K * (K - 1) / 2 - (K - p) * (K - p - 1) / 2 + (q - 1 - p);
  if (index >= K * (K - 1) / 2)
  {
    Log(error, "Error: index out of bounds");
    Log(info, "p: %u, q: %u, K: %u", p, q, K);
    Log(info, "index: %lu", index);
    Log(info, "Max: %lu", K * (K - 1) / 2);
    exit(-1);
  }
  return index;
}

template <typename cost_type = double>
cost_type solve_with_gurobi(problem_info<cost_type> *pinfo)
{
  const uint N = pinfo->N;
  const uint K = pinfo->K;
  const uint SP = pinfo->SP;
  const uint N2 = N * N;
  cost_type *costs = pinfo->cost_matrix;
  int *dim1 = pinfo->dim1;
  int *dim2 = pinfo->dim2;
  Log(debug, "Starting to construct Gurobi model");
  try
  {
    GRBEnv env = GRBEnv();
    env.set(GRB_IntParam_OutputFlag, 0);
    GRBModel model = GRBModel(env);
    Log(debug, "Defined Gurobi model");
    // Create variables
    uint64_t num_vars = uint64_t(N2) * SP;
    Log(debug, "Defining %lu variables", num_vars);
    GRBVar *x = new GRBVar[num_vars];
    Log(debug, "Variable defined");
    for (uint t = 0; t < SP; t++)
    {
      int p = dim1[t], q = dim2[t];
      for (uint i = 0; i < N; i++)
      {
        for (uint j = 0; j < N; j++)
        {
          std::stringstream s;
          s << "X_" << i << "_" << j << "_" << p << "_" << q;
          uint64_t index = uint64_t(t * N2) + (i * N) + j;
          x[index] = model.addVar(0, 1, costs[index], GRB_BINARY, s.str());
        }
      }
    }
    model.update();
    Log(debug, "Variables added to model");
    Log(debug, "Adding assignment constraints");
    // Add assignment constraints
    for (uint t = 0; t < SP; t++)
    {
      int p = dim1[t], q = dim2[t];
      for (uint i = 0; i < N; i++)
      {
        GRBLinExpr expr1 = 0, expr2 = 0;
        size_t index1 = size_t(t * N2) + (i * N);
        size_t index2 = size_t(t * N2) + i;
        for (uint j = 0; j < N; j++)
        {
          expr1 += x[index1 + j];     // row sum
          expr2 += x[index2 + j * N]; // column sum
        }
        model.addConstr(expr1 == 1);
        model.addConstr(expr2 == 1);
      }
    }
    model.update();
    Log(debug, "Assignment constraints added");
    Log(debug, "Adding continuity constraints");

    // Add continuity constraints
    for (uint p = 0; p < K; p++)
    {
      for (uint q = p + 1; q < K; q++)
      {
        for (uint r = q + 1; r < K; r++)
        {
          for (uint i = 0; i < N; i++)
          {
            for (uint j = 0; j < N; j++)
            {
              for (uint k = 0; k < N; k++)
              {
                // clang-format off
                model.addConstr(
                    x[reverse_index(p, q, K) * N2 + i * N + j] 
                    + x[reverse_index(q, r, K) * N2 + j * N + k] 
                    - x[reverse_index(p, r, K) * N2 + i * N + k] <= 1);
                model.addConstr(
                  x[reverse_index(q, r, K) * N2 + j * N + k] 
                  + x[reverse_index(p, r, K) * N2 + i * N + k] 
                  - x[reverse_index(p, q, K) * N2 + i * N + j] <= 1);
                model.addConstr(
                  x[reverse_index(p, r, K) * N2 + i * N + k] 
                  + x[reverse_index(p, q, K) * N2 + i * N + j] 
                  - x[reverse_index(q, r, K) * N2 + j * N + k] <= 1);
                // clang-format on
              }
            }
          }
        }
      }
    }
    model.update();
    Log(debug, "Continuity constraints added");
    model.optimize();
    Log(debug, "Model optimized");
    // Get objective value
    double UB = model.get(GRB_DoubleAttr_ObjVal);
    Log(debug, "UB: %f", UB);
    // model.write("scratch/model.lp");
    // print solution values
    // for (uint j = 0; j < N; j++)
    // {
    //   for (uint i = 0; i < N; i++)
    //   {
    //     if (x[i * N + j].get(GRB_DoubleAttr_X) > 0.5)
    //     {
    //       Log(debug, "x[%u][%u] = %d", i, j, int(x[i * N + j].get(GRB_DoubleAttr_X)));
    //     }
    //   }
    // }
    delete[] x;
    return UB;
  }
  catch (GRBException e)
  {
    Log(error, "Error code = %d", e.getErrorCode());
    std::cout << e.getErrorCode() << "\n\n"
              << e.getMessage() << std::endl;
    exit(-1);
  }
  catch (...)
  {
    Log(error, "Exception during optimization");
  }
  return cost_type(0);
}

// Explicit instantiations
template uint solve_with_gurobi<uint>(problem_info<uint> *);
template float solve_with_gurobi<float>(problem_info<float> *);
template double solve_with_gurobi<double>(problem_info<double> *);