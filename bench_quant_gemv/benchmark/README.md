## Overview

- [A100](A100.md)
- [H100](H100.md)

## Performance breakdown

Let's look into one specific case:

- batch_size = 1

- seq_len = 1

- in_feature = 10240

- out_feature = 81920

- num_centroid = 8192 = $2^{13}$

- num_res_centroid = 256 = $2^8$

- vec_len = 8

### A100

| No. | Step                          | Excluding this Step (ms) | Time Till This Step (ms) |
| :-: | :---------------------------- | :----------------------- | :----------------------- |
|  1  | Load codebook                 | 2.3540 $\pm$ 0.0032      | 0.4630 $\pm$ 0.0063      |
|  2  | Load tiled inputs             |                          | 1.4628 $\pm$ 0.0033      |
|  3  | Decode and compute over tiles | 1.6491 $\pm$ 0.0118      | 1.4963 $\pm$ 0.0031      |
|  4  | Accumulate between tiles      | 2.2847 $\pm$ 0.0054      | 2.5467 $\pm$ 0.0033      |
|  5  | Store results                 | 2.5471 $\pm$ 0.0035      | 2.5556 $\pm$ 0.0034      |
|     | Total                         | 2.5566 $\pm$ 0.0056      |                          |

### H100

| No. | Step                          | Elapsed Time (ms) |
| :-: | :---------------------------- | :---------------- |
|  1  | Load codebook                 |                   |
|  2  | Load tiled inputs             |                   |
|  3  | Decode and compute over tiles |                   |
|  4  | Accumulate between tiles      |                   |
|  5  | Store results                 |                   |
|     | Total                         |                   |
