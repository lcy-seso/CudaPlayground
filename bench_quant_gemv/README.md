## Quantized GEMV Benchmarks

The project includes performance benchmarks for quantized GEMV operations.

### Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

### Running Benchmarks

The easiest way to run benchmarks is using the provided script:

```bash
# Make the script executable
chmod +x bench_quant_gemv/run.sh

# Run benchmarks
./bench_quant_gemv/run.sh
```

Or you can run pytest commands directly:

```bash
# Run all tests including benchmarks
python -m pytest bench_quant_gemv.py -v

# Run only benchmarks
python -m pytest bench_quant_gemv.py -v --benchmark-only

# Generate a JSON report
python -m pytest bench_quant_gemv.py -v --benchmark-json benchmark/results.json
```

### Test Environment

- **GPU**: NVIDIA Tesla A100 / H100
- **CUDA Version**: 12.6
