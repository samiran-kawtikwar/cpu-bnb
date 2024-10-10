#!/bin/sh
cd ~/RCAP-Project/cpu-bnb/
module load cuda

source ~/.bashrc

export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

./build/main.exe -n 5 -k 5 -f 10 -d 0