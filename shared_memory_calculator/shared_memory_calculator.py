"""Calculate the shared memory usage of vector-quantized kernels."""

import itertools


def to_kb(
    size: int,
    data_bit: int = 16,  # half/bf16 = 16, fp8 = 8
) -> float:
    """Convert the size in elements to kilobytes.

    Args:
        size: Number of elements.
        data_bit: Bit width of each element (default: 16 for half/bf16,
                  8 for fp8).

    Returns:
        Size in kilobytes.
    """
    return size * data_bit / 8 / 1024


def cal_shared_memory_usage(
    index_bit: int,
    res_index_bit: int,
    vector_bit: int,
) -> tuple[int, float]:
    """Calculate the shared memory usage of vector-quantized kernels."""
    vector_length = 8

    num_centroid = 2 << index_bit
    num_res_centroid = 2 << res_index_bit

    codebook_size = to_kb(num_centroid * vector_length, vector_bit)

    res_codebook_size = to_kb(num_res_centroid * vector_length, vector_bit)
    return int(codebook_size), res_codebook_size


if __name__ == "__main__":
    params = {
        "index_bit": [12, 11, 10],
        "res_index_bit": [4, 5, 6],
        "vector_bit": [16, 8],
    }

    header = (
        "|" + "|".join(params.keys()) + "|codebook (KB)|residual codebook (KB)|"
    )
    print(header)  # noqa: T201
    print("|:--:|:--:|:--:|:--:|:--:|")  # noqa: T201

    for param in itertools.product(
        params["index_bit"], params["res_index_bit"], params["vector_bit"]
    ):
        codebook_size, res_codebook_size = cal_shared_memory_usage(*param)
        print(  # noqa: T201
            f"|{param[0]}|{param[1]}|"
            f"{param[2]}|{codebook_size}|"
            f"{res_codebook_size}|"
        )
