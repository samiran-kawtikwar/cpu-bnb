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
  int fixed_assignments[100]; // To be changed later using appropriate partitions
  float LB;
  uint level;
  uint id; // For mapping with memory queue; DON'T UPDATE
  __host__ node_info() { std::fill(fixed_assignments, fixed_assignments + 100, -1); };
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
};

struct bnb_stats
{
  uint max_heap_size;
  uint nodes_explored;
  uint nodes_pruned;
};