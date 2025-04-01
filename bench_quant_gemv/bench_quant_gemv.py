"""Benchmark the quantized GEMV operation."""

import pytest
import torch
import vptq
from data_utils import gen_gemv_data, gen_vptq_data
from torch_gemv import torch_gemv

torch.manual_seed(1234)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")


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


def benchmark_torch_gemv(
    data: tuple[torch.Tensor, torch.Tensor],
    num_warmup: int = 10,
    num_runs: int = 100,
) -> tuple[float, float]:
    """Benchmark the PyTorch GEMV operation.

    Args:
        data: Tuple containing the input tensors (act, weight).
        num_warmup: Number of warmup iterations.
        num_runs: Number of benchmark runs.

    Returns:
        Tuple containing the mean and standard deviation of the benchmark times.
    """
    act, weight = data

    # Warmup
    for _ in range(num_warmup):
        torch_gemv(act, weight)

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(num_runs):
        start.record()
        torch_gemv(act, weight)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times_tensor = torch.tensor(times)
    mean_time = times_tensor.mean().item()
    std_time = times_tensor.std().item()
    return mean_time, std_time


def run_benchmark(
    implementation: str = "vptq",
    batch_size: int = 15,
    seq_len: int = 1,
    in_features: int = 4096,
    out_features: int = 4096,
    num_centroids: int = 8192,
    num_res_centroids: int = 256,
    num_codebooks: int = 1,
    vec_len: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    device_str: str = "cuda",
    num_warmup: int = 10,
    num_iters: int = 100,
) -> tuple[float, float]:
    """Benchmark the GEMV operation.

    Args:
        implementation: Which implementation to benchmark ("vptq" or "torch").
        batch_size: Batch size for the input tensor.
        seq_len: Sequence length for the input tensor.
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        num_centroids: Number of centroids for quantization.
        num_res_centroids: Number of residual centroids.
        num_codebooks: Number of codebooks.
        vec_len: Length of each vector in quantization.
        dtype: Data type for tensors.
        device_str: Device to place tensors on (e.g. "cuda" or "cpu").
        num_warmup: Number of warmup iterations.
        num_iters: Number of benchmark iterations.

    Returns:
        Tuple containing mean and std of execution times.
    """
    if implementation != "vptq":
        # Generate data for torch benchmark
        data = gen_gemv_data(
            in_features=in_features,
            out_features=out_features,
            batch_size=batch_size,
            device_str=device_str,
            dtype=dtype,
        )
        return benchmark_torch_gemv(data, num_warmup, num_iters)

    # Generate data for vptq benchmark
    data_quant_gemv = gen_vptq_data(
        in_features=in_features,
        out_features=out_features,
        num_centroids=num_centroids,
        num_res_centroids=num_res_centroids,
        batch_size=batch_size,
        length=seq_len,
        num_codebooks=num_codebooks,
        vec_len=vec_len,
        dtype=dtype,
        device_str=device_str,
    )
    return benchmark_quant_gemv(
        data_quant_gemv,
        vec_len,
        num_codebooks,
        num_centroids,
        num_res_centroids,
        out_features,
        num_warmup,
        num_iters,
    )


@pytest.mark.parametrize("implementation", ["vptq", "torch"])  # type: ignore
@pytest.mark.parametrize("batch_size", [1])  # type: ignore
@pytest.mark.parametrize(  # type: ignore
    "in_features", [1024, 2048, 4096, 8192, 16384]
)
@pytest.mark.parametrize(  # type: ignore
    "out_features", [1024, 4096, 8192, 14336, 28672, 53248]
)
@pytest.mark.parametrize("num_centroids", [4096, 8192])  # type: ignore
@pytest.mark.parametrize("num_res_centroids", [256, 512])  # type: ignore
def test_gemv_performance(
    benchmark: pytest.fixture,
    implementation: str,
    batch_size: int,
    in_features: int,
    out_features: int,
    num_centroids: int,
    num_res_centroids: int,
) -> None:
    """Test the performance of GEMV implementations.

    Args:
        benchmark: The pytest-benchmark fixture
        implementation: Which implementation to benchmark
        batch_size: Batch size for input tensor
        in_features: Feature dimension for input
        out_features: Feature dimension for output
        num_centroids: Number of main centroids for quantization
        num_res_centroids: Number of residual centroids for quantization
    """
    benchmark(
        run_benchmark,
        implementation=implementation,
        batch_size=batch_size,
        in_features=in_features,
        out_features=out_features,
        num_centroids=num_centroids,
        num_res_centroids=num_res_centroids,
    )


if __name__ == "__main__":
    # test directly without using pytest
    batch_size = 1
    in_features = [1024, 2048, 4096, 8192, 16384]
    out_features = [1024, 4096, 8192, 14336, 28672, 53248]
    num_centroids = [4096, 8192]
    num_res_centroids = [256, 512]

    header = (
        "|count|batch_size|in_features|out_features|"
        "main_centroids|residual_centroids|"
        "vptq (ms)|torch (ms)|ratio|\n"
    )
    header += "|---|---|---|---|---|---|---|---|---|\n"
    print(header, end="")  # noqa: T201

    count = 1
    for num_res_centroid in num_res_centroids:
        for num_centroid in num_centroids:
            for out_feature in out_features:
                for in_feature in in_features:
                    mean1, std1 = run_benchmark(
                        implementation="vptq",
                        batch_size=batch_size,
                        in_features=in_feature,
                        out_features=out_feature,
                        num_centroids=num_centroid,
                        num_res_centroids=num_res_centroid,
                    )

                    mean2, std2 = run_benchmark(
                        implementation="torch",
                        batch_size=batch_size,
                        in_features=in_feature,
                        out_features=out_feature,
                    )

                    row = (
                        f"|{count}|{batch_size}|{in_feature}|"
                        f"{out_feature}|{num_centroid}|{num_res_centroid}|"
                        f"{mean1:.4f}|"
                        f"{mean2:.4f}|"
                        f"{mean1 / mean2:.2f}|\n"
                    )
                    print(row, end="")  # noqa: T201
                    count += 1
