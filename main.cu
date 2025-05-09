#include <stdio.h>
#include <cmath>
#include <vector>
#include "utils/logger.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/timer.h"
#include "defs.cuh"
#include "LAP/device_utils.cuh"
#include "LAP/Hung_Tlap.cuh"

#include "RCAP/config.h"
#include "RCAP/cost_generator.h"
#include "RCAP/gurobi_solver.h"
#include "RCAP/rcap_functions.cuh"
#include "RCAP/feasibility_solver.cuh"
#include "RCAP/subgrad_solver.cuh"

#include <queue>

int main(int argc, char **argv)
{
  Log(info, "Starting program");
  Config config = parseArgs(argc, argv);
  printConfig(config);
  int dev_ = config.deviceId;
  uint psize = config.user_n;
  uint ncommodities = config.user_ncommodities;

  if (psize > 100)
  {
    Log(critical, "Problem size too large, Implementation not ready yet. Use problem size <= 100");
    exit(-1);
  }

  problem_info *h_problem_info = generate_problem<cost_type>(config, config.seed);
  // print(h_problem_info, true, true, false);

  Timer t = Timer();
  cost_type UB = solve_with_gurobi<cost_type, weight_type>(h_problem_info->costs, h_problem_info->weights, h_problem_info->budgets, psize, ncommodities);
  Log(info, "RCAP solved with GUROBI: objective %u\n", (uint)UB);
  Log(info, "Time taken by Gurobi: %f sec", t.elapsed());

  Log(debug, "Solving RCAP with Branching");
  t.reset();

  // Define a heap from the standard priority queue package
  std::priority_queue<node, std::vector<node>, std::greater<node>> heap;
  bnb_stats stats = bnb_stats();

  Log(debug, "Creating scratch space for workers");
  worker_info *d_worker_space;
  CUDA_RUNTIME(cudaMallocManaged((void **)&d_worker_space, sizeof(worker_info) * psize));
  worker_info::allocate_all(d_worker_space, psize, psize); // max psize workers

  Log(debug, "Creating space for subgrad solver");
  subgrad_space *d_subgrad_space; // managed by each subworker
  CUDA_RUNTIME(cudaMallocManaged((void **)&d_subgrad_space, psize * sizeof(subgrad_space)));
  subgrad_space::allocate_all(d_subgrad_space, psize, psize, ncommodities, dev_);

  Log(debug, "Creating space for feasibility check");
  feasibility_space *d_feas_space;
  CUDA_RUNTIME(cudaMallocManaged((void **)&d_feas_space, psize * sizeof(feasibility_space)));
  feasibility_space::allocate_all(d_feas_space, psize, ncommodities, psize, dev_);

  node_info *root_info = new node_info(psize);
  root_info->LB = 0;
  root_info->level = 0;
  node root = node(0, root_info);
  root.key = update_bounds_subgrad(h_problem_info, root, UB);
  if (root.key <= UB)
    heap.push(root);
  else
  {
    Log(critical, "Infeasible solution");
    return -1;
  }

  stats.max_heap_size = max(stats.max_heap_size, (uint)heap.size());
  // start branch and bound
  bool optimal = false;
  node opt_node = node(0, new node_info(psize));
  // uint iter = 0;

  std::vector<node> children(psize);
  // std::vector<bool> feasible(psize, false);
  bool *feasible;
  CUDA_RUNTIME(cudaMallocManaged((void **)&feasible, sizeof(bool) * psize));
  CUDA_RUNTIME(cudaMemset(feasible, 0, sizeof(bool) * psize));

  node *d_children;
  CUDA_RUNTIME(cudaMalloc((void **)&d_children, psize * sizeof(node)));

  while (!optimal && !heap.empty())
  {
    // Log(debug, "Starting iteration# %u", iter++);
    // get the best node from the heap
    node best_node = node(0, new node_info(psize));
    best_node.copy(heap.top(), psize);
    delete heap.top().value;
    heap.pop();
    uint level = best_node.value->level;
    // Log(info, "best node key %u", (uint)best_node.key);
    // Branch on the best node to create (psize - level) new children nodes
    for (uint i = 0; i < psize - level; i++)
    {
      // Create a new child node
      node_info *child_info = new node_info(psize);
      child_info->LB = best_node.value->LB;
      child_info->level = level + 1;
      for (uint j = 0; j < psize; j++)
        child_info->fixed_assignments[j] = best_node.value->fixed_assignments[j];

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
      children[i].value = child_info;
      feasible[i] = true;
    }
    // feas_check_parallel(h_problem_info, children, feasible);
    feas_check_gpu(h_problem_info, psize - level, children, feasible, d_children, d_feas_space, dev_);
    // copy row_fa from feasibility space to subgrad space
    for (uint i = 0; i < psize - level; i++)
      CUDA_RUNTIME(cudaMemcpy(d_subgrad_space[i].row_fa, d_feas_space[i].row_fa, sizeof(int) * psize, cudaMemcpyDeviceToDevice));
    update_bounds_subgrad_gpu(h_problem_info, psize - level,
                              children, d_children,
                              d_subgrad_space, feasible, UB, dev_);

    for (uint i = 0; i < psize - level; i++)
    {
      if (feasible[i])
      {
        // assert(children[i].value->LB == children[i].key);
        if (children[i].key <= UB && children[i].value->level == psize)
        {
          optimal = true;
          Log(critical, "Optimality Reached");
          opt_node.copy(children[i], psize);
          delete children[i].value;
          break;
        }
        else if (children[i].key <= UB)
        {
          stats.nodes_explored++;
          heap.push(children[i]);
          // Don't delete
        }
        else
        {
          // Prune the node
          stats.nodes_pruned_incumbent++;
          delete children[i].value;
        }
      }
      else
      {
        // printf("Child %u is infeasible\n", i);
        delete children[i].value;
        stats.nodes_pruned_infeasible++;
      }
    }
    stats.max_heap_size = max(stats.max_heap_size, (uint)heap.size());
    delete best_node.value;
  }

  if (optimal)
  {
    Log(critical, "Optimal solution found with objective %u", (uint)opt_node.key);
  }
  else
  {
    if (heap.size() <= 0)
    {
      Log(critical, "Heap underflow, infeasible problem");
    }
    // Prune the node
    Log(critical, "Optimal solution not found");
    exit(-1);
  }
  Log(info, "Max heap size during execution: %lu", stats.max_heap_size);
  Log(info, "Nodes Explored: %u, Incumbant: %u, Infeasible: %u", stats.nodes_explored, stats.nodes_pruned_incumbent, stats.nodes_pruned_infeasible);

  Log(info, "Exiting program");
  Log(info, "Total time taken: %f sec", t.elapsed());

  CUDA_RUNTIME(cudaFree(feasible));

  worker_info::free_all(d_worker_space, psize);
  CUDA_RUNTIME(cudaFree(d_worker_space));
  subgrad_space::free_all(d_subgrad_space, psize);
  CUDA_RUNTIME(cudaFree(d_subgrad_space));
  feasibility_space::free_all(d_feas_space, psize);
  CUDA_RUNTIME(cudaFree(d_feas_space));
  CUDA_RUNTIME(cudaFree(d_children));

  CUDA_RUNTIME(cudaFree(h_problem_info));
  while (!heap.empty())
  {
    delete heap.top().value;
    heap.pop();
  }
  delete opt_node.value;
}
