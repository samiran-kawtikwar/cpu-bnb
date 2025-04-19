#include <stdio.h>
#include <cmath>
#include <vector>
#include "utils/logger.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/timer.h"
#include "defs.cuh"
#include "LAP/device_utils.cuh"
#include "LAP/Hung_Tlap.cuh"

#include "QAP/config.h"
#include "QAP/problem_generator.h"
#include "QAP/gurobi_solver.h"
#include "QAP/qap_functions.cuh"

#include <queue>

static inline void atomicIncr(uint &counter)
{
#pragma omp atomic
  ++counter;
}

int main(int argc, char **argv)
{
  Log(info, "Starting program");
  Config config = parseArgs(argc, argv);

  problem_info *h_pinfo = generate_problem(config, config.seed);
  print(h_pinfo, false, false);
  printConfig(config);
  size_t psize = h_pinfo->N;
  cost_type UB = h_pinfo->opt_objective;
  CUDA_RUNTIME(cudaSetDevice(config.deviceId));
  Log(info, "Solving QAP with Branching");
  Timer t = Timer();
  // Define a heap from the standard priority queue package
  std::priority_queue<node, std::vector<node>, std::greater<node>> heap;
  bnb_stats stats = bnb_stats();

  // start branch and bound
  std::atomic<bool> optimal{false};
  node opt_node = node(0, new node_info(psize));
  problem_info d_pinfo = problem_info(psize);
  CUDA_RUNTIME(cudaMalloc((void **)&d_pinfo.distances, psize * psize * sizeof(cost_type)));
  CUDA_RUNTIME(cudaMalloc((void **)&d_pinfo.flows, psize * psize * sizeof(cost_type)));
  CUDA_RUNTIME(cudaMemcpy(d_pinfo.distances, h_pinfo->distances, psize * psize * sizeof(cost_type), cudaMemcpyHostToDevice));
  CUDA_RUNTIME(cudaMemcpy(d_pinfo.flows, h_pinfo->flows, psize * psize * sizeof(cost_type), cudaMemcpyHostToDevice));
  TLAP<cost_type> *tlaps = static_cast<TLAP<cost_type> *>(std::malloc(psize * sizeof(TLAP<cost_type>)));

  // define psize GL_Handles
  GL_handle **handles = new GL_handle *[psize];
  for (uint i = 0; i < psize; ++i)
    handles[i] = new GL_handle(psize, config.deviceId);

  for (int i = 0; i < psize; ++i)
    new (&tlaps[i]) TLAP<cost_type>(psize * psize, psize, config.deviceId);

  node_info *root_info = new node_info(psize);
  root_info->LB = 0;
  root_info->level = 0;
  node root = node(0, root_info);
  root.key = update_bounds_GL(d_pinfo, root, tlaps[0], handles[0]);
  root_info->LB = root.key;
  if (root.key <= UB)
  {
    heap.push(root);
    stats.nodes_explored++;
  }
  else
    stats.nodes_pruned_incumbent++;

  uint counter_1000 = 0;
  std::vector<bool> isValid(psize, false);
  std::vector<node> children(psize);

  while (!optimal.load(std::memory_order_relaxed) && !heap.empty())
  {
    // Log(debug, "Starting iteration# %u", iter++);
    // get the best node from the heap
    node best_node = node(0, new node_info(psize));
    best_node.copyFrom(heap.top(), psize);
    delete heap.top().value;
    heap.pop();
    uint level = best_node.value->level;

// Log(info, "popped node from heap with LB %u, level %u", (uint)best_node.key, level);
#pragma omp parallel for num_threads(psize - level)
    for (uint i = 0; i < psize - level; i++)
    {
      // 1) Early‑exit if some thread already found the true leaf
      if (optimal.load(std::memory_order_acquire))
        continue;

      // Branch on the best node to create (psize - level) new children nodes
      // Create a new child node
      node_info *child_info = new node_info(psize);
      best_node.value->deepcopy(child_info, psize);
      child_info->level = level + 1;
      // Log<colon>(info, "Level %u, child %u", level, i);
      // printHostArray(child_info->fixed_assignments, psize, "Input fa");
      // Update fixed assignments of the child by updating the ith unassigned assignment to level
      uint counter = 0;
      for (uint index = 0; index < psize; index++)
      {
        if (counter == i && child_info->fixed_assignments[index] == -1)
        {
          // Log(debug, "Code reached here\n");
          child_info->fixed_assignments[index] = level; // fixes the assignment at counter
          break;
        }
        if (child_info->fixed_assignments[index] == -1)
          counter++;
      }
      // printHostArray(child_info->fixed_assignments, psize, "Output fa");
      // Update bounds of the child node
      node child = node(best_node.key, child_info);
      child_info->LB = update_bounds_GL(d_pinfo, child, tlaps[i], handles[i]);
      child.key = child_info->LB;
      if (child.key <= UB && child_info->level == psize)
      {
        // Log(debug, "Code reached here\n");
        Log(debug, "Optimality reached at line %u", __LINE__);
        bool was_set = optimal.exchange(true, std::memory_order_acq_rel);
        if (!was_set)
          child.copyTo(opt_node, psize);
        continue;
      }
      else if (child.key <= UB)
      {
        children[i] = child;
        children[i].value = child_info;
        isValid[i] = true;
        atomicIncr(stats.nodes_explored);
      }
      else
      {
        delete child_info;
        atomicIncr(stats.nodes_pruned_incumbent);
      }
    }
    for (uint i = 0; i < psize - level; i++)
    {
      if (isValid[i])
        heap.push(children[i]);
    }
    stats.max_heap_size = max(stats.max_heap_size, (uint)heap.size());
    delete best_node.value;
    uint total_nodes = stats.nodes_explored + stats.nodes_pruned_incumbent;
    if (total_nodes > 1000 * counter_1000)
    {
      Log(debug, "Total nodes explored: %u, exp: %u, pru: %u", total_nodes, stats.nodes_explored,
          stats.nodes_pruned_incumbent);
      Log(debug, "Heap size: %u", (uint)heap.size());
      counter_1000++;
    }
    // reset the isValid array
    isValid.assign(psize, false);
  }

  if (optimal.load(std::memory_order_relaxed))
    Log(critical, "Optimal solution found with objective %u", (uint)opt_node.key);
  else
    Log(critical, "Optimal solution not found: infeasible problem or wrong UB");

  Log(info, "Max heap size during execution: %lu", stats.max_heap_size);
  Log(info, "Nodes Explored: %u, Incumbant: %u, Infeasible: %u", stats.nodes_explored,
      stats.nodes_pruned_incumbent, stats.nodes_pruned_infeasible);

  Log(info, "Exiting program");
  Log(info, "Total time taken: %f sec", t.elapsed());

  delete h_pinfo;
  while (!heap.empty())
  {
    delete heap.top().value;
    heap.pop();
  }
  delete opt_node.value;
  free(tlaps);
  for (int i = 0; i < psize; ++i)
    delete handles[i];
  delete[] handles;

  return 0;
}