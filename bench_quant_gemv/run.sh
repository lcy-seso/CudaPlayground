#!/bin/bash
set -o pipefail

# Run the benchmark command and capture its output
python3 bench_quant_gemv.py \
    --batch_size 16 \
    --in_features 4096 \
    --out_features 4096 \
    --num_centroids 8192 \
    --num_res_centroids 256 \
    --num_codebooks 1 \
    --vector_length 8 \
    --dtype bfloat16 \
    --num_warmup 10 \
    --num_iters 100 2>&1 | tee bench_quant_gemv.log

# Check the exit status of the python command
# (pipefail ensures we get python's status, not tee's)
PYTHON_EXIT=$?
if [ $PYTHON_EXIT -ne 0 ]; then
    echo "Benchmark failed with exit code $PYTHON_EXIT"
    exit $PYTHON_EXIT
fi
