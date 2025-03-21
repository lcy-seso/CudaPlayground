#!/bin/bash
set -o pipefail

echo "Running quantized GEMV benchmarks..."

# Create benchmark directory if it doesn't exist
mkdir -p benchmark

# Run the benchmarks
python -m pytest bench_quant_gemv.py -v \
    --benchmark-only \
    --benchmark-json benchmark/results.json

# Check the exit status
status=$?
if [ $status -ne 0 ]; then
    echo "Benchmark failed with exit code $status"
    exit $status
fi

echo "Benchmarks completed successfully!"
echo "Results saved to benchmark/results.json"
