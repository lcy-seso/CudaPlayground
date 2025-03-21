"""Benchmark the quantized GEMV operation."""

import pytest
import torch
import vptq

torch.manual_seed(1234)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
device = torch.device("cuda")


def benchmark_quant_gemv(
    data: tuple,
    vector_length: int,
    num_codebooks: int,
    num_centroids: int,
    num_res_centroids: int,
    out_features: int,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> tuple[float, float]:
    """Benchmark the quantized GEMV operation.

    Args:
        data: Tuple containing the input data for the GEMV operation.
        vector_length: Length of each vector used in quantization.
        num_codebooks: Number of codebooks used in the quantization.
        num_centroids: Number of centroids for quantization.
        num_res_centroids: Number of residual centroids for quantization.
        out_features: Output feature dimension.
        num_warmup: Number of warmup iterations.
        num_runs: Number of benchmark runs.

    Returns:
        Tuple containing the mean and standard deviation of the benchmark times.

    """
    (
        act,
        main_indices,
        centroids,
        res_indices,
        res_indices,
        res_centroids,
        scale_weights,
        scale_bias,
    ) = data

    # Define common arguments for quant_gemv_v2 function
    gemv_args = {
        "x": act,
        "bias": None,  # no bias in this benchmark
        "indices": main_indices,
        "centroids": centroids,
        "residual_indices": res_indices,
        "residual_centroids": res_centroids,
        "scale_weights": scale_weights,
        "scale_bias": scale_bias,
        "vector_len": vector_length,
        "num_codebooks": num_codebooks,
        "num_centroids": num_centroids,
        "num_residual_centroids": num_res_centroids,
        "out_features": out_features,
    }

    # Warmup
    for _ in range(num_warmup):
        vptq.ops.quant_gemv_v2(**gemv_args)

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(num_runs):
        start.record()
        vptq.ops.quant_gemv_v2(**gemv_args)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times_tensor = torch.tensor(times)
    mean_time = times_tensor.mean().item()
    std_time = times_tensor.std().item()

    return mean_time, std_time


def gen_data(
    in_features: int,
    out_features: int,
    num_centroids: int,
    num_res_centroids: int,
    batch_size: int,
    length: int = 1,
    num_codebooks: int = 1,
    vec_len: int = 8,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Generate data for the quantized GEMV benchmark.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        num_centroids: Number of centroids for quantization.
        num_res_centroids: Number of residual centroids.
        batch_size: Batch size for the input tensor.
        length: Sequence length for the input tensor.
        num_codebooks: Number of codebooks.
        vec_len: Length of each vector in quantization.
        dtype: Data type for tensors.

    Returns:
        Tuple containing the generated data for benchmarking.
    """
    mean = 2e-2
    std = 0.5

    # Helper function for tensor creation
    def create_tensor(size: tuple[int, ...]) -> torch.Tensor:
        return torch.normal(
            mean=mean, std=std, size=size, device=device, dtype=dtype
        )

    # Create all tensors with consistent parameters
    act = create_tensor((batch_size, length, in_features))
    centroids = create_tensor((num_codebooks, num_centroids, vec_len))
    res_centroids = create_tensor((num_codebooks, num_res_centroids, vec_len))
    scale_weights = create_tensor((in_features, 1))
    scale_bias = create_tensor((in_features, 1))

    # Create indices tensors
    num_indices = in_features * out_features // vec_len
    main_indices = (
        torch.arange(num_centroids, device=device, dtype=torch.int32)
        .repeat(num_indices // num_centroids)
        .to(dtype=torch.uint16)
    )
    res_indices = (
        torch.arange(num_res_centroids, device=device, dtype=torch.int32)
        .repeat(num_indices // num_res_centroids)
        .to(dtype=torch.uint8)
    )

    return (
        act,
        main_indices,
        centroids,
        res_indices,
        res_indices,
        res_centroids,
        scale_weights,
        scale_bias,
    )


def run_benchmark(
    batch_size: int = 15,
    seq_len: int = 1,
    in_features: int = 4096,
    out_features: int = 4096,
    num_centroids: int = 8192,
    num_res_centroids: int = 256,
    num_codebooks: int = 1,
    vec_len: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> tuple[float, float]:
    """Benchmark the quantized GEMV operation.

    Args:
        batch_size: Batch size for the input tensor.
        seq_len: Sequence length for the input tensor.
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        num_centroids: Number of centroids for quantization.
        num_res_centroids: Number of residual centroids.
        num_codebooks: Number of codebooks.
        vec_len: Length of each vector in quantization.
        dtype: Data type for tensors.
        num_warmup: Number of warmup iterations.
        num_iters: Number of benchmark iterations.

    Returns:
        Tuple containing mean and std of execution times.
    """
    # Generate data for benchmark
    data = gen_data(
        in_features=in_features,
        out_features=out_features,
        num_centroids=num_centroids,
        num_res_centroids=num_res_centroids,
        batch_size=batch_size,
        length=seq_len,
        num_codebooks=num_codebooks,
        vec_len=vec_len,
        dtype=dtype,
    )
    return benchmark_quant_gemv(
        data,
        vec_len,
        num_codebooks,
        num_centroids,
        num_res_centroids,
        out_features,
        num_warmup,
        num_iters,
    )


@pytest.mark.parametrize("batch_size", [1, 8, 15])  # type: ignore
@pytest.mark.parametrize("features", [1024, 2048, 4096, 8192])  # type: ignore
@pytest.mark.parametrize(  #
    "out_features",
    [1024, 4096, 8192, 14336],
)  # type: ignore
def test_quant_gemv_performance(
    benchmark: pytest.fixture,
    batch_size: int,
    features: int,
    out_features: int,
) -> None:
    """Test the performance of quant_gemv with different hyperparameters.

    Args:
        benchmark: The pytest-benchmark fixture
        batch_size: Batch size for input tensor
        features: Feature dimension for input
        out_features: Feature dimension for output
    """
    benchmark(
        run_benchmark,
        batch_size=batch_size,
        in_features=features,
        out_features=out_features,
    )
