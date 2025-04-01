#!/bin/bash

# Get absolute paths
PYTHON_PATH=$(command -v python)
NCU_PATH=$(command -v ncu)

sudo $NCU_PATH \
    --kernel-name ke_quant_gemv_v2 \
    --set full \
    -o gemv -f \
    $PYTHON_PATH bench/for_profile.py
