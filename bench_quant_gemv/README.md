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

- **GPU**: NVIDIA Tesla A100
- **CUDA Version**: 12.6
- **Based on VPTQ's commit**: `91d3cbe514dd55cb0c283278f2bf85ba5d96bc83`

## Performance breakdown

Hyper-parameters:

- batch_size = 1

- in_feature = 10240

- out_feature = 81920

- num_centroid = 8192 = 2^13

- num_res_centroid = 256 = 2^8

- seq_len = 1

- vec_len = 8

| No. | Step                          | Accumulated Time (ms) | Elapsed Time (ms) |
| :-: | :---------------------------- | :-------------------: | :---------------- |
|  1  | Load codebook                 |        0.4655         | 0.4655            |
|  2  | Load tiled inputs             |        1.4945         | 1.0290            |
|  3  | Decode and compute over tiles |        1.5358         | 0.0413            |
|  4  | Accumulate between tiles      |        2.6208         | 1.0850            |
|  5  | Store results                 |        2.6631         | 0.0423            |
|     | Total                         |                       | 2.6631            |
