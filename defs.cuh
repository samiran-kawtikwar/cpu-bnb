#pragma once

#define __DEBUG__
// #define MAX_HEAP_SIZE 1000000
#define MAX_TOKENS 100
#define MAX_ITER 100
#define least_count 1e-6
#define MAX_DATA 0xffffffff
#define DBL_MAX 1e10

typedef unsigned long long int uint64;
typedef unsigned int uint;

#define BlockSize 64U
#define TileSize 64U
#define TilesPerBlock (BlockSize / TileSize)
#define TILE cg::thread_block_tile<TileSize>

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
  __host__ void deepcopy(node_info *other, const uint psize)
  {
    other->LB = LB;
    other->level = level;
    other->id = id;
    memcpy(other->fixed_assignments, fixed_assignments, sizeof(int) * psize);
  }
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
  void copyTo(node &other, uint psize) // copies node and node_info
  {
    other.key = key;                     // copies to other->key
    value->deepcopy(other.value, psize); // copies to other->value
  }
  void copyFrom(const node &other, uint psize) // copies node and node_info
  {
    key = other.key;                     // copies from other->key
    other.value->deepcopy(value, psize); // copies from other->value
  }
  __host__ void print(const uint psize, std::string message = "NULL")
  {
    Log<colon>(debug, message.c_str());
    printHostArray(value->fixed_assignments, psize);
    printf("LB: %.0f\t level: %u\n", key, value->level);
  };
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