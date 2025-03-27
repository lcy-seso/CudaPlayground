"""Benchmark the quantized GEMV operation."""

import torch
import vptq
from data_utils import gen_vptq_data

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
    num_runs: int = 50,
) -> None:
    """Benchmark the quantized GEMV operation.

    Args:
        data: Tuple containing the input data for the GEMV operation.
        vector_length: Length of each vector used in quantization.
        num_codebooks: Number of codebooks used in the quantization.
        num_centroids: Number of centroids for quantization.
        num_res_centroids: Number of residual centroids for quantization.
        out_features: Output feature dimension.
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
    for _ in range(num_runs):
        vptq.ops.quant_gemv_v2(**gemv_args)


if __name__ == "__main__":
    batch_size = 1
    in_feature = 1024
    out_feature = 8672
    num_centroid = 8192  # 2^13
    num_res_centroid = 256  # 2^8

    seq_len = 1
    num_codebook = 1
    vec_len = 8
    dtype = torch.bfloat16
    device_str = "cuda"
    num_iters = 10

    data_quant_gemv = gen_vptq_data(
        in_features=in_feature,
        out_features=out_feature,
        num_centroids=num_centroid,
        num_res_centroids=num_res_centroid,
        batch_size=batch_size,
        length=seq_len,
        num_codebooks=num_codebook,
        vec_len=vec_len,
        dtype=dtype,
        device_str=device_str,
    )

    benchmark_quant_gemv(
        data_quant_gemv,
        vec_len,
        num_codebook,
        num_centroid,
        num_res_centroid,
        out_feature,
        num_iters,
    )
    print("Benchmark completed")  # noqa: T201
