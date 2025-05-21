#pragma once
#include <random>
#include <fstream>
#include <stdexcept>
#include <string>

#include "config.h"
#include "problem_info.h"
#include "../utils/timer.h"
#include "../utils/logger.cuh"
#include "../defs.cuh"

using namespace std;

// Download a URL to local_path (using wget), then open it in an ifstream.
// Throws std::runtime_error on failure.
std::ifstream download_and_open(const std::string &raw_url,
                                const std::string &local_path)
{
  // Build the wget command: -q for quiet, -O to set output filename
  std::string cmd = "wget -q -O " + local_path + " " + raw_url;
  int rc = std::system(cmd.c_str());
  if (rc != 0)
  {
    throw std::runtime_error("Download failed (exit code " +
                             std::to_string(rc) + ")");
  }

  std::ifstream in(local_path, std::ios::binary);
  if (!in.is_open())
  {
    throw std::runtime_error("Failed to open file \"" + local_path + "\"");
  }

  // Delete the directory entry immediately.
  // On POSIX, `in` stays valid; on Windows this will error.
  if (std::remove(local_path.c_str()) != 0)
    std::perror(("Warning: could not delete temp file " + local_path).c_str());

  return in;
}

// Convenience overload: just pass the instance name (e.g. "chr12a.dat")
// and it will format the raw‑GitHub URL for your repo.
std::ifstream download_instance(const std::string &instance_name)
{
  const std::string base =
      "https://raw.githubusercontent.com/"
      "samiran-kawtikwar/QAPLIB-instances/"
      "main/instances/";
  std::string url = base + instance_name;
  return download_and_open(url, instance_name);
}

template <typename cost_type = uint>
problem_info *generate_problem(Config &config, const int seed = 45345)
{
  size_t user_n = config.user_n;
  problem_info *info = new problem_info(user_n);
  cost_type *distances, *flows;

  if (config.problemType == generated)
  {
    Log(critical, "Generated problem tyoe not supported on this branch");
  }
  else if (config.problemType == qaplib)
  {
    // Read distances and flows from file
    ifstream infile = download_instance(config.inputfile);
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
        std::cerr << "Error: input size mismatch: " << config.inputfile << std::endl;
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
    Log(warn, "Distances: ");
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
    Log(warn, "Flows: ");
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