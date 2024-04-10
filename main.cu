#include <stdio.h>
#include <cmath>
#include <vector>
#include "utils/logger.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/timer.h"
#include "defs.cuh"
#include "LAP/config.h"
#include "LAP/cost_generator.h"
#include "LAP/device_utils.cuh"
#include "LAP/Hung_lap.cuh"
#include "LAP/lap_kernels.cuh"

#include <queue>

int main(int argc, char **argv)
{
  Log(info, "Starting program");
  Config config = parseArgs(argc, argv);
  printConfig(config);
  int dev_ = config.deviceId;
  uint psize = config.user_n;
  if (psize > 100)
  {
    Log(critical, "Problem size too large, Implementation not ready yet. Use problem size <= 100");
    exit(-1);
  }
  CUDA_RUNTIME(cudaDeviceReset());
  CUDA_RUNTIME(cudaSetDevice(dev_));

  cost_type *h_costs = generate_cost<cost_type>(config, config.seed);

  // print h_costs
  // for (size_t i = 0; i < psize; i++)
  // {
  //   for (size_t j = 0; j < psize; j++)
  //   {
  //     printf("%u, ", h_costs[i * psize + j]);
  //   }
  //   printf("\n");
  // }
  cost_type *d_costs;

  CUDA_RUNTIME(cudaMalloc((void **)&d_costs, psize * psize * sizeof(cost_type)));
  CUDA_RUNTIME(cudaMemcpy(d_costs, h_costs, psize * psize * sizeof(cost_type), cudaMemcpyHostToDevice));

  LAP<cost_type> *lap = new LAP<cost_type>(h_costs, psize, dev_);
  lap->solve();
  const cost_type UB = lap->objective;

  Log(info, "LAP solved succesfully, objective %u\n", (uint)UB);
  lap->print_solution();
  delete lap;

  Log(debug, "Solving LAP with Branching");
  Timer t = Timer();

  // Define a heap from the standard queue package
  std::priority_queue<node, std::vector<node>, std::greater<node>> heap;
  bnb_stats stats;
  stats.nodes_explored = 0;
  stats.nodes_pruned = 0;
  stats.max_heap_size = 0;

  node_info *root_info = new node_info(psize);
  root_info->LB = 0;
  root_info->level = 0;
  node root = node(0, root_info);
  heap.push(root);
  stats.max_heap_size = max(stats.max_heap_size, (uint)heap.size());
  // start branch and bound
  bool optimal = false;
  node opt_node = node(0, new node_info(psize));
  // uint iter = 0;
  do
  {
    // Log(debug, "Starting iteration# %u", iter++);
    // get the best node from the heap
    node best_node = node(0, new node_info(psize));
    best_node.copy(heap.top(), psize);
    heap.pop();
    stats.nodes_explored++;
    // Log(info, "best node key %u", (uint)best_node.key);

    // Update bound of the best node
    best_node.value->LB = 0;
    for (uint i = 0; i < psize; i++)
    {
      if (best_node.value->fixed_assignments[i] != -1)
      {
        best_node.value->LB += h_costs[i * psize + best_node.value->fixed_assignments[i]];
      }
    }
    best_node.key = best_node.value->LB;
    uint level = best_node.value->level;
    if (best_node.key < UB)
    {

      // Branch on the best node to create (psize - level) new children nodes
      for (uint i = 0; i < psize - level; i++)
      {
        // Create a new child node
        node_info *child_info = new node_info(psize);
        child_info->LB = best_node.value->LB;
        child_info->level = level + 1;
        for (uint j = 0; j < psize; j++)
        {
          child_info->fixed_assignments[j] = best_node.value->fixed_assignments[j];
        }

        // Update fixed assignments of the child by updating the ith unassigned assignment to level
        uint counter = 0;
        for (uint index = 0; index < psize; index++)
        {
          if (counter == i && child_info->fixed_assignments[index] == -1)
          {
            // Log(debug, "Code reached here\n");
            child_info->fixed_assignments[index] = level;

            break;
          }
          if (child_info->fixed_assignments[index] == -1)
            counter++;
        }

        node child = node(best_node.key, child_info);
        heap.push(child);
      }
      stats.max_heap_size = max(stats.max_heap_size, (uint)heap.size());
    }
    else if (best_node.key == UB && best_node.value->level == psize)
    {
      optimal = true;
      Log(critical, "Optimality Reached");
      opt_node.copy(best_node, psize);
      break;
    }
    else
    {
      // Prune the node
      stats.nodes_pruned++;
    }
  } while (!optimal || !heap.empty());

  if (optimal)
  {
    Log(critical, "Optimal solution found with objective %u", (uint)opt_node.key);
  }
  else
  {
    Log(critical, "Optimal solution not found");
  }
  Log(info, "Max heap size during execution: %lu", stats.max_heap_size);
  Log(info, "Nodes explored: %u, Pruned: %u", stats.nodes_explored, stats.nodes_pruned);

  delete[] h_costs;
  CUDA_RUNTIME(cudaFree(d_costs));
  Log(info, "Exiting program");
  Log(info, "Total time taken: %f sec", t.elapsed());
}