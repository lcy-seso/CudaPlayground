"""PyTorch implementation of GEMV operations."""

import torch

__all__ = [
    "torch_gemv",
]


def torch_gemv(
    act: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Perform a GEMV operation using PyTorch.

    Args:
        act: Input activation tensor of shape (batch_size, in_features)
        weight: Weight matrix of shape (out_features, in_features)

    Returns:
        Output tensor of shape (batch_size, out_features)

    Raises:
        ValueError: If batch size is greater than 15
    """
    batch_size, _ = act.shape

    if batch_size > 15:
        raise ValueError("Batch size must be less than 15")

    return act @ weight
