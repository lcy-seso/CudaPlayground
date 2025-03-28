#!/bin/bash

# Get absolute paths
PYTHON_PATH=$(command -v python)
NCU_PATH=$(command -v ncu)

sudo $NCU_PATH \
    --set full \
    -o profile -f \
    $PYTHON_PATH bench_quant_gemv.py
