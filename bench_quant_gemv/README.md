## Benchmarks

### Quantized GEMV Benchmarks

The project includes performance benchmarks for quantized GEMV operations.

#### Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

#### Running Benchmarks

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

The benchmark tests different combinations of:

- Batch sizes: 1, 8, 15
- Input feature dimensions: 1024, 2048, 4096
- Output feature dimensions: 1024, 4096
