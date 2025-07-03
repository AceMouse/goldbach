#!/bin/bash

python gen_binpow.py > check_composite.cu
nvcc -rdc=true lockstep_parallel_miller_rabin.cu -o check -Wno-deprecated-gpu-targets
