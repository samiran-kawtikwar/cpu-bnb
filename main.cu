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

#include "QAP/config.h"
#include "QAP/problem_generator.h"
#include "QAP/gurobi_solver.h"
// #include "RCAP/rcap_functions.cuh"

#include <queue>

int main(int argc, char **argv)
{
  Log(info, "Starting program");
  Config config = parseArgs(argc, argv);

  problem_info *h_problem_info = generate_problem(config, config.seed);
  print(h_problem_info, true, true);
  printConfig(config);

  if (config.user_n > 50)
  {
    Log(critical, "Problem size too large, Implementation not ready yet. Use problem size <= 50");
    exit(-1);
  }

#if 0

  Timer t = Timer();

  // Solve RCAP
  cost_type UB = solve_with_gurobi<cost_type, weight_type>(h_problem_info->costs, h_problem_info->weights, h_problem_info->budgets, psize, ncommodities);
  Log(info, "RCAP solved with GUROBI: objective %u\n", (uint)UB);

  Log(info, "Time taken by Gurobi: %f sec", t.elapsed());

  Log(debug, "Solving RCAP with Branching");
  t.reset();

  // Define a heap from the standard priority queue package
  std::priority_queue<node, std::vector<node>, std::greater<node>> heap;
  bnb_stats stats = bnb_stats();

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

  // Log(debug, " Subgrad on root node");
  // root.key = update_bounds_subgrad(h_problem_info, root, UB);
  // exit(-1);
  do
  {
    // Log(debug, "Starting iteration# %u", iter++);
    // get the best node from the heap
    node best_node = node(0, new node_info(psize));
    best_node.copy(heap.top(), psize);
    delete heap.top().value;
    heap.pop();
    // Log(info, "best node key %u", (uint)best_node.key);
    // bool feasible = feas_check_naive(h_problem_info, best_node);
    bool feasible = feas_check(h_problem_info, best_node);
    if (feasible)
    {
      // Update bound of the best node
      // update_bounds(h_problem_info, best_node);
      best_node.key = update_bounds_subgrad(h_problem_info, best_node, UB);

      uint level = best_node.value->level;
      if (best_node.key <= UB && best_node.value->level == psize)
      {
        optimal = true;
        Log(critical, "Optimality Reached");
        opt_node.copy(best_node, psize);
        delete best_node.value;
        break;
      }
      else if (best_node.key <= UB)
      {
        stats.nodes_explored++;
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
      else
      {
        // Prune the node
        stats.nodes_pruned_incumbent++;
      }
    }
    else
    {
      if (heap.size() <= 0)
      {
        Log(critical, "Heap underflow");
        exit(-1);
      }
      // Prune the node
      stats.nodes_pruned_infeasible++;
    }
    delete best_node.value;
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
#endif
  return 0;
}