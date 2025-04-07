#pragma once
#include <stdio.h>
#include <string>
#include <unistd.h>

typedef unsigned int uint;
typedef double cost_type;

struct Config
{
  uint user_nnodes, user_nframes;
  double sigma;
  int deviceId;
  int seed;
};

static void usage()
{
  fprintf(stderr,
          "\nUsage: [options]\n"
          "\n-n <Number of targets>"
          "\n-k number of frames"
          "\n-f sigma"
          "\n-d <deviceId"
          "\n-s <seed-value>"
          "\n");
}

static void printConfig(Config config)
{
  printf("  nodes: %u\n", config.user_nnodes);
  printf("  frames: %u\n", config.user_nframes);
  printf("  sigma: %f\n", config.sigma);
  printf("  Device: %u\n", config.deviceId);
  printf("  seed value: %d\n", config.seed);
}

static Config parseArgs(int argc, char **argv)
{
  Config config;
  config.user_nnodes = 10;
  config.user_nframes = 10;
  config.sigma = 0.1;
  config.deviceId = 0;
  config.seed = 45345;

  int opt;
  while ((opt = getopt(argc, argv, "n:f:d:s:h:k:")) >= 0)
  {
    switch (opt)
    {
    case 'n':
      config.user_nnodes = atoi(optarg);
      break;
    case 'f':
      config.sigma = std::stod(optarg);
      break;
    case 'd':
      config.deviceId = atoi(optarg);
      break;
    case 's':
      config.seed = atoi(optarg);
      break;
    case 'k':
      config.user_nframes = atoi(optarg);
      break;
    case 'h':
      usage();
      exit(0);
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return config;
}