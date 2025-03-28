"""Calculate the shared memory usage of vector-quantized kernels."""


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


def to_str(num: float) -> str:
    """Convert a float number to a string with minimal decimal places.

    Formats the number with 2 decimal places and removes trailing zeros
    and decimal point if not needed.

    Args:
        num: The float number to convert.

    Returns:
        A string representation of the number with minimal decimal places.
    """
    return f"{num:.2f}".rstrip("0").rstrip(".")


def shared_memory_calculator(
    index_bits: tuple[tuple[int, int], ...],
    vector_bits: tuple[int, ...] = (16, 8),
    vector_length: int = 8,
) -> None:
    """Calculate the shared memory usage of vector-quantized kernels.

    Computes and prints a formatted table showing memory usage for different
    configurations of vector-quantized kernels.

    Args:
        index_bits: List of tuples containing (index_bit, residual_index_bit)
                    pairs. index_bit represents the number of bits for the main
                    codebook indices. residual_index_bit represents the number
                    of bits for the residual codebook indices.
        vector_bits: List of bit widths for vector elements (e.g., 16 for half/
                     bf16, 8 for fp8).
        vector_length: Length of each vector in the codebook.

    Returns:
        None. Results are printed to stdout in a markdown table format.
    """
    header = (
        "|$B$|$B_{res}$|$C$|$C_{res}$|$b$"
        "|Codebook (KB)|Residual Codebook (KB)|Total (KB)|"
    )
    print(header)  # noqa: T201
    print("|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|")  # noqa: T201

    for vector_bit in vector_bits:
        for index_bit, res_index_bit in index_bits:

            num_centroid = 2 << (index_bit - 1)
            num_res_centroid = 2 << (res_index_bit - 1)

            codebook_size = to_kb(num_centroid * vector_length, vector_bit)
            res_codebook_size = to_kb(
                num_res_centroid * vector_length, vector_bit
            )
            total_size = codebook_size + res_codebook_size

            res = [
                f"|{index_bit}|{res_index_bit}|",
                f"{num_centroid}|{num_res_centroid}|",
                f"{vector_bit}|",
                f"{to_str(codebook_size)}|",
                f"{to_str(res_codebook_size)}|",
                f"{to_str(total_size)}|",
            ]
            print("".join(res))  # noqa: T201


if __name__ == "__main__":

    # a hardware friendly configuration
    shared_memory_calculator(
        (
            (12, 4),
            (11, 5),
            (10, 6),
        )
    )

    # a less hardware friendly configuration
    shared_memory_calculator(
        (
            (14, 10),  # 24 in total
            (14, 9),  # 23 in total
            (13, 9),  # 22 in total
            (13, 8),  # 21 in total
            (12, 7),  # 19 in total
        )
    )
