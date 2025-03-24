"""Utility functions for generating data for quantized GEMV benchmarking."""

import torch

__all__ = [
    "gen_vptq_data",
    "gen_gemv_data",
]


def gen_vptq_data(
    in_features: int,
    out_features: int,
    num_centroids: int,
    num_res_centroids: int,
    batch_size: int,
    length: int = 1,
    num_codebooks: int = 1,
    vec_len: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    device_str: str = "cuda",
) -> tuple[
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
        device_str: Device to place tensors on (e.g. "cuda" or "cpu").

    Returns:
        Tuple containing the generated data for benchmarking.
    """
    device = torch.device(device_str)

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
        res_centroids,
        scale_weights,
        scale_bias,
    )


def gen_gemv_data(
    in_features: int,
    out_features: int,
    batch_size: int,
    device_str: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate data for the GEMV benchmark.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        batch_size: Batch size for the input tensor.
        device_str: Device to place tensors on (e.g. "cuda" or "cpu").
        dtype: Data type for tensors.

    Returns:
        Tuple containing the generated data for benchmarking.
    """
    device = torch.device(device_str)

    act = torch.randn(batch_size, in_features, device=device, dtype=dtype)
    weight = torch.randn(in_features, out_features, device=device, dtype=dtype)

    return act, weight
