#pragma once

#define __DEBUG__
// #define MAX_HEAP_SIZE 1000000
#define MAX_TOKENS 100
#define MAX_DATA 0xffffffff
#define eps 1e-6

typedef unsigned long long int uint64;
typedef unsigned int uint;
typedef uint cost_type;

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
    delete[] fixed_assignments;
  };
};

struct node
{
  float key;
  node_info *value;
  __host__ __device__ node(){};
  __host__ __device__ node(float a, node_info *b) : key(a), value(b){};
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
};

struct bnb_stats
{
  uint max_heap_size;
  uint nodes_explored;
  uint nodes_pruned;
};