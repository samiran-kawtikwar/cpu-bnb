#include <stdio.h>
#include <cmath>
#include <vector>
#include "utils/logger.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/timer.h"
#include "defs.cuh"
#include "LAP/device_utils.cuh"
#include "LAP/Hung_lap.cuh"
#include "LAP/lap_kernels.cuh"
#include <omp.h>
#include "QAP/config.h"
#include "QAP/problem_generator.h"
#include "QAP/gurobi_solver.h"
#include "QAP/qap_functions.cuh"

#include <queue>

int main(int argc, char **argv)
{
  Log(info, "Starting program");
  Config config = parseArgs(argc, argv);

  problem_info *h_problem_info = generate_problem(config, config.seed);
  print(h_problem_info, true, true);
  printConfig(config);
  size_t psize = h_problem_info->N;
  cost_type UB = h_problem_info->opt_objective;

  Log(info, "Solving QAP with Branching");
  Timer t = Timer();
  // Define a heap from the standard priority queue package
  std::priority_queue<node, std::vector<node>, std::greater<node>> heap;
  bnb_stats stats = bnb_stats();

  node_info *root_info = new node_info(psize);
  root_info->LB = 0;
  root_info->level = 0;
  node root = node(0, root_info);
  root.key = update_bounds_GL(h_problem_info, root);
  root.value->LB = root.key;
  heap.push(root);
  stats.max_heap_size = max(stats.max_heap_size, (uint)heap.size());
  // start branch and bound
  bool optimal = false;

  if (!(root.key <= UB))
    return -1; // infeasible problem
  node opt_node = node(0, new node_info(psize));
  // Log(debug, " Subgrad on root node");
  // Log(debug, "Root node key %u", (uint)root.key);
  std::vector<node> children(psize);

  while (!optimal || !heap.empty())
  {
    static uint iter = 0;
    node best_node = node(0, new node_info(psize));
    best_node.copyFrom(heap.top(), psize);
    delete heap.top().value;
    heap.pop();
    uint level = best_node.value->level;
    Log(debug, "Starting iteration %u from node with bound %u, level %u", ++iter, best_node.key, best_node.value->level);

    for (uint i = 0; i < psize - level; i++)
    {
      // Create a new child node
      node_info *child_info = new node_info(psize);
      best_node.value->deepcopy(child_info, psize);
      child_info->level = level + 1;
      // Update fixed assignments of the child by updating the ith unassigned assignment to level
      uint counter = 0;
      for (uint index = 0; index < psize; index++)
      {
        if (counter == i && child_info->fixed_assignments[index] == -1)
        {
          child_info->fixed_assignments[index] = level;
          break;
        }
        if (child_info->fixed_assignments[index] == -1)
          counter++;
      }
      children[i].value = child_info;
    }
#pragma omp parallel
    {
#pragma omp single
      update_bounds_poly_GL_parallel(h_problem_info, level, children);
    }
    for (uint i = 0; i < psize - level; i++)
    {
      if (children[i].key <= UB && children[i].value->level == psize)
      {
        optimal = true;
        children[i].copyTo(opt_node, psize);
        break;
      }
      else if (children[i].key <= UB)
      {
        heap.push(children[i]);
        stats.nodes_explored++;
        stats.max_heap_size = max(stats.max_heap_size, (uint)heap.size());
      }
      else
      {
        delete children[i].value;
        stats.nodes_pruned_incumbent++;
      }
    }
  }

  if (optimal)
    Log(critical, "Optimal solution found with objective %u", (uint)opt_node.key);
  else
    Log(critical, "Optimal solution not found");

  Log(info, "Max heap size during execution: %lu", stats.max_heap_size);
  Log(info, "Nodes Explored: %u, Incumbant: %u, Infeasible: %u", stats.nodes_explored, stats.nodes_pruned_incumbent, stats.nodes_pruned_infeasible);

  Log(info, "Exiting program");
  Log(info, "Total time taken: %f sec", t.elapsed());

  delete h_problem_info;
  while (!heap.empty())
  {
    delete heap.top().value;
    heap.pop();
  }
  delete opt_node.value;

  return 0;
}