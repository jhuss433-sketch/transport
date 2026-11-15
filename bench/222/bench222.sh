#!/bin/bash
# SNPE 2.22 benchmark, run from the board container (snapdragon-snapdragon-1)
export SNPE_ROOT=/workspace/shared/sdk/2.22.6.240515
export PATH=$SNPE_ROOT/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$SNPE_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export PYTHONPATH=$SNPE_ROOT/lib/python
python3 $SNPE_ROOT/benchmarks/SNPE/snpe_bench.py "$@"
