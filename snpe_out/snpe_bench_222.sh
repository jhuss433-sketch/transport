#!/bin/bash
# Run SNPE 2.22's snpe_bench.py with SNPE's common_utils ahead of QNN's
PYTHONPATH=$SNPE_ROOT/lib/python:$PYTHONPATH \
  python3 $SNPE_ROOT/benchmarks/SNPE/snpe_bench.py "$@"
