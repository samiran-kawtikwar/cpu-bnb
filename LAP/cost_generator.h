#pragma once
#include <random>
#include <omp.h>
#include <thread>
#include <fstream>
#include "config.h"
#include "../utils/timer.h"
#include "../utils/logger.cuh"
#include "../defs.cuh"

using namespace std;

template <typename T>
T *generate_cost(Config config, const int seed = 45345)
{
  size_t user_n = config.user_n;
  size_t nrows = user_n;
  size_t ncols = user_n;
  double frac = config.frac;
  double range = frac * user_n;

  T *cost = new T[user_n * user_n];
  memset(cost, 0, user_n * user_n * sizeof(T));

  // use all available CPU threads for generating cost
  uint nthreads = min(user_n, (size_t)thread::hardware_concurrency() - 3); // remove 3 threads for OS and other tasks
  uint rows_per_thread = ceil((nrows * 1.0) / nthreads);
#pragma omp parallel for num_threads(nthreads)
  for (uint tid = 0; tid < nthreads; tid++)
  {
    uint first_row = tid * rows_per_thread;
    uint last_row = min(first_row + rows_per_thread, (uint)nrows);
    for (size_t r = first_row; r < last_row; r++)
    {
      default_random_engine generator(seed + r);
      generator.discard(1);
      uniform_int_distribution<T> distribution(0, range - 1);
      for (size_t c = 0; c < ncols; c++)
      {
        if (c < user_n && r < user_n)
        {
          double gen = distribution(generator);
          cost[user_n * r + c] = (T)gen;
        }
        else
        {
          if (c == r)
            cost[user_n * c + r] = 0;
          else
            cost[user_n * c + r] = UINT32_MAX;
        }
      }
    }
  }

  // ********* print cost array *********
  // for (uint i = 0; i < user_n; i++)
  // {
  //   for (uint j = 0; j < user_n; j++)
  //   {
  //     cout << cost[i * ncols + j] << " ";
  //   }
  //   cout << endl;
  // }

  // ********* write cost array to csv file *********
  // ofstream out("matrix_test.csv");
  // for (uint i = 0; i < user_n; i++)
  // {
  //   for (uint j = 0; j < user_n; j++)
  //   {
  //     out << cost[i * ncols + j] << ", ";
  //   }
  //   out << '\n';
  // }

  // ********* get frequency of all numbers *********
  // uint *freq = new uint[(uint)ceil(user_n * frac)];
  // memset(freq, 0, user_n * frac * sizeof(uint));
  // for (uint i = 0; i < user_n; i++)
  // {
  //   for (uint j = 0; j < user_n; j++)
  //   {
  //     freq[cost[i * ncols + j]]++;
  //   }
  // }
  // ofstream out("freq_test.csv");
  // for (size_t i = 0; i < user_n * frac; i++)
  // {
  //   out << freq[i] << ",\n";
  // }
  // delete[] freq;
  return cost;
}