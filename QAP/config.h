#pragma onece
#include <stdio.h>
#include <string>
#include <unistd.h>

typedef unsigned int uint;

struct Config
{
  uint user_n;
  int deviceId;
  char *inputfile;
  int seed;
};

static void usage()
{
  fprintf(stderr,
          "\nUsage: [options]\n"
          "\n-n <size of the problem>"
          "\n-d <deviceId"
          "\n-s <seed-value>"
          "\n-i <input filename>"
          "\n");
}

static void printConfig(Config config)
{
  printf("  size: %u\n", config.user_n);
  printf("  Device: %u\n", config.deviceId);
  if (config.inputfile)
    printf("  input file: %s\n", config.inputfile);
  else
    printf("  seed value: %d\n", config.seed);
}

static Config parseArgs(int argc, char **argv)
{
  Config config;
  config.user_n = 10;
  config.deviceId = 0;
  config.inputfile = nullptr;
  config.seed = 45345;

  int opt;
  while ((opt = getopt(argc, argv, "n:d:s:i:h:")) >= 0)
  {
    switch (opt)
    {
    case 'n':
      config.user_n = atoi(optarg);
      break;
    case 'd':
      config.deviceId = atoi(optarg);
      break;
    case 's':
      config.seed = atoi(optarg);
      break;
    case 'i':
      config.inputfile = optarg;
      break;
    case 'h':
      usage();
      exit(0);
    default:
      usage();
      exit(1);
    }
  }
  return config;
}