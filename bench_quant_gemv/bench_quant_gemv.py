import torch
import vptq

torch.manual_seed(1234)
dtype = torch.bfloat16
device = torch.device("cuda", 0)


def test_quant_gemv(x: torch.Tensor,
                    in_features: int,
                    out_features: int,
                    batch_size: int,
                    length: int,
                    num_centroids: int,
                    num_res_centroids: int,
                    num_codebooks: int = 1,
                    vector_length: int = 8):

    mean = 2e-2
    std = 0.5

    #====== generate data for unittest.  ======#
    # the activation tensor
    shape = (batch_size, length, in_features)
    x = torch.normal(mean=mean,
                     std=std,
                     size=shape,
                     device=device,
                     dtype=dtype)

    # generate indices for unittest.
    num_indices = in_features * out_features // vector_length
    num_repeats = num_indices // num_centroids
    main_indices = torch.as_tensor(list(range(num_centroids)) * num_repeats,
                                   device=device,
                                   dtype=torch.uint16)

    num_repeats = num_indices // num_res_centroids
    res_indices = torch.as_tensor(list(range(num_res_centroids)) * num_repeats,
                                  device=device,
                                  dtype=torch.uint8)

    shape = (num_codebooks, num_centroids, vector_length)
    centroids = torch.normal(mean=mean,
                             std=std,
                             size=shape,
                             device=device,
                             dtype=dtype)

    shape = (num_codebooks, num_res_centroids, vector_length)
    res_centroids = torch.normal(mean=mean,
                                 std=std,
                                 size=shape,
                                 device=device,
                                 dtype=dtype)

    shape = (in_features, 1)
    scale_weights = torch.normal(mean=mean,
                                 std=std,
                                 size=shape,
                                 device=device,
                                 dtype=dtype)
    scale_bias = torch.normal(mean=mean,
                              std=std,
                              size=shape,
                              device=device,
                              dtype=dtype)


if __name__ == "__main__":
    test_quant_gemv()
