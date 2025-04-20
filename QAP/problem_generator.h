#pragma once
#include <random>
#include <fstream>
#include "config.h"
#include "problem_info.h"
#include "gurobi_solver.h"
#include "../utils/timer.h"
#include "../utils/logger.cuh"
#include "../defs.cuh"

using namespace std;

template <typename cost_type = uint>
problem_info *generate_problem(Config &config, const int seed = 45345)
{
  size_t user_n = config.user_n;
  double frac = 10;

  problem_info *info = new problem_info(user_n);
  cost_type *distances, *flows;

  if (config.problemType == generated)
  {
    info->N = user_n;
    distances = new cost_type[user_n * user_n];
    flows = new cost_type[user_n * user_n];
    if (user_n > 50)
    {
      Log(critical, "Problem size too large, Implementation not ready yet. Use problem size <= 50");
      exit(-1);
    }
    // Generate random distances and flows
    default_random_engine generator(seed);
    generator.discard(1);
    uniform_int_distribution<cost_type> distribution(0, frac * user_n - 1);
    for (size_t i = 0; i < user_n; i++)
    {
      for (size_t j = 0; j < user_n; j++)
      {
        if (i == j)
        {
          distances[user_n * i + j] = 0;
          flows[user_n * i + j] = 0;
        }
        else
        {
          distances[user_n * i + j] = (cost_type)distribution(generator);
          flows[user_n * i + j] = (cost_type)distribution(generator);
        }
      }
    }

    // solve the problem to get the optimal objective
    info->opt_objective = solve_with_gurobi(distances, flows, user_n);
  }
  else if (config.problemType == qaplib)
  {
    // Read distances and flows from file
    string fileAddress = string("./QAP/instances/") + config.inputfile;
    ifstream infile(fileAddress.c_str());
    if (!infile.is_open())
    {
      Log(error, "Could not open input file: %s", config.inputfile);
      exit(1);
    }
    while (infile.is_open() && infile.good())
    {
      infile >> info->N;
      uint n = info->N;
      config.user_n = n;

      distances = new cost_type[n * n];
      flows = new cost_type[n * n];
      infile >> info->opt_objective;
      int counter = 0;
      cost_type *ptr1 = flows;
      cost_type *ptr2 = distances;
      while (!infile.eof())
      {
        if (counter < n * n)
          infile >> *(ptr1++);

        else
          infile >> *(ptr2++);

        counter++;
      }

      if (counter < 2 * n * n)
      {
        std::cerr << "Error: input size mismatch: " << fileAddress << std::endl;
        exit(1);
      }

      infile.close();
    }
    infile.close();
  }
  else
  {
    Log(error, "Invalid problem type");
    exit(1);
  }
  // Copy distances and flows to info
  info->distances = distances;
  info->flows = flows;
  if (info->N > 50)
  {
    Log(critical, "Problem size too large, Implementation not ready yet. Use problem size <= 50");
    exit(-1);
  }
  return info;
}

template <typename cost_type = uint>
void print(problem_info *pinfo, bool print_distances = true, bool print_flows = true)
{
  uint N = pinfo->N;
  Log(info, "Optimal objective: %u", pinfo->opt_objective);
  if (print_distances)
  {
    Log(debug, "Distances: ");
    for (size_t i = 0; i < N; i++)
    {
      for (size_t j = 0; j < N; j++)
      {
        printf("%u, ", pinfo->distances[i * N + j]);
      }
      printf("\n");
    }
  }
  if (print_flows)
  {
    Log(debug, "Flows: ");
    for (size_t i = 0; i < N; i++)
    {
      for (size_t j = 0; j < N; j++)
      {
        printf("%u, ", pinfo->flows[i * N + j]);
      }
      printf("\n");
    }
  }
}