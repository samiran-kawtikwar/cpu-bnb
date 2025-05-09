#pragma once

#define __DEBUG__
// #define MAX_HEAP_SIZE 1000000
#define MAX_TOKENS 100
#define MAX_ITER 100
// #define MAX_DATA 0xffffffff
#define least_count 1e-6

typedef unsigned long long int uint64;
typedef unsigned int uint;
typedef uint cost_type;
typedef uint weight_type;

#define BlockSize 64U
#define TileSize 64U
#define TilesPerBlock (BlockSize / TileSize)
#define TILE cg::thread_block_tile<TileSize>

struct problem_info
{
  uint psize, ncommodities;
  cost_type *costs;     // cost of assigning
  weight_type *weights; // weight of each commodity
  weight_type *budgets; // capacity of each commodity

  ~problem_info()
  {
    CUDA_RUNTIME(cudaFree(costs));
    CUDA_RUNTIME(cudaFree(weights));
    CUDA_RUNTIME(cudaFree(budgets));
  }
};

struct node_info
{
  int *fixed_assignments; // To be changed later using appropriate partitions
  float LB;
  uint level;
  uint id; // For mapping with memory queue; DON'T UPDATE
  __host__ node_info(uint psize)
  {
    fixed_assignments = (int *)malloc(psize * sizeof(int));
    std::fill(fixed_assignments, fixed_assignments + psize, -1);
  };
  ~node_info()
  {
    free(fixed_assignments);
  };
};

struct node
{
  float key;
  node_info *value;
  __host__ __device__ node() {};
  __host__ __device__ node(float a, node_info *b) : key(a), value(b) {};
  // Define a comparator called greater for the node
  __host__ __device__ bool operator>(const node &other) const
  {
    return key > other.key;
  }
  // ~node() { delete[] value; };
  void copy(node other, uint psize)
  {
    key = other.key;
    value->LB = other.value->LB;
    value->level = other.value->level;
    value->id = other.value->id;
    memcpy(value->fixed_assignments, other.value->fixed_assignments, sizeof(int) * psize);
  }
  template <typename... Args>
  void print(const char *message, Args... args)
  {
    printf(message, args...);
    Log<nun>(info, "Key: %f, level: %u\t", key, value->level);
  }
  template <typename... Args>
  void print(const uint psize, const char *message, Args... args)
  {
    printf(message, args...);
    Log<nun>(info, "Key: %f, level: %u\t", key, value->level);
    // print fixed assignments
    printf("FA: ");
    for (uint i = 0; i < psize; i++)
      printf("%d ", value->fixed_assignments[i]);
    printf("\n");
  }
};

struct worker_info
{
  node nodes[MAX_TOKENS];
  float LB[MAX_TOKENS];
  uint level[MAX_TOKENS];
  bool feasible[MAX_TOKENS];
  int *fixed_assignments; // To temporarily store fixed assignments

  static void allocate_all(worker_info *d_worker_space, size_t nworkers, size_t psize)
  {
    for (size_t i = 0; i < nworkers; i++)
    {
      d_worker_space[i].allocate(psize);
    }
  }

  // Static function to free memory for an array of work_info instances
  static void free_all(worker_info *d_worker_space, size_t nworkers)
  {
    for (size_t i = 0; i < nworkers; i++)
    {
      d_worker_space[i].free();
    }
  }
  // Function to allocate memory for this instance
  void allocate(size_t psize)
  {
    CUDA_RUNTIME(cudaMalloc((void **)&fixed_assignments, psize * psize * sizeof(int)));
    CUDA_RUNTIME(cudaMemset(fixed_assignments, 0, psize * psize * sizeof(int)));
  }
  // Function to free allocated memory for this instance
  void free()
  {
    CUDA_RUNTIME(cudaFree(fixed_assignments));
  }
};

struct bnb_stats
{
  uint max_heap_size;
  uint nodes_explored;
  uint nodes_pruned_incumbent;
  uint nodes_pruned_infeasible;

  // Define a constructor for the bnb_stats
  __host__ bnb_stats()
  {
    max_heap_size = 0;
    nodes_explored = 0;
    nodes_pruned_incumbent = 0;
    nodes_pruned_infeasible = 0;
  }
};