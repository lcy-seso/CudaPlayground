## Hyper-parameters

| No. | Parameter Name               | Notation  |                                                    |
| :-: | :--------------------------- | :-------: | :------------------------------------------------: |
|  1  | input feature dimension      | $D_{in}$  |                                                    |
|  2  | output feature dimension     | $D_{out}$ |                                                    |
|  3  | number of centroids          |    $C$    |                                                    |
|  4  | number of residual centroids | $C_{res}$ |                                                    |
|  5  | index bit                    |    $B$    |                                                    |
|  6  | residual index bit           | $B_{res}$ |                                                    |
|  6  | number of codebooks          |    $N$    | ***preserved***, not usful at the current analysis |
|  7  | vector length                |    $V$    |                                                    |
|  8  | vector bit                   |    $b$    |            16 for half/bf16, 8 for fp8             |

## Shared memory usage

Shared memory per block:

- A100, 164 KB
- H100, 228 KB (39% increased)

### A hardware friendly configuration

| $B$ | $B_{res}$ | $C$  | $C_{res}$ | $b$ | Codebook (KB) | Residual Codebook (KB) | Total (KB) |
| :-: | :-------: | :--: | :-------: | :-: | :-----------: | :--------------------: | :--------: |
| 12  |     4     | 4096 |    16     | 16  |      64       |          0.25          |   64.25    |
| 11  |     5     | 2048 |    32     | 16  |      32       |          0.5           |    32.5    |
| 10  |     6     | 1024 |    64     | 16  |      16       |           1            |     17     |
| 12  |     4     | 4096 |    16     |  8  |      32       |          0.12          |   32.12    |
| 11  |     5     | 2048 |    32     |  8  |      16       |          0.25          |   16.25    |
| 10  |     6     | 1024 |    64     |  8  |       8       |          0.5           |    8.5     |

### A less hardware friendly configuration

| $B$ | $B_{res}$ |  $C$  | $C_{res}$ | $b$ | Codebook (KB) | Residual Codebook (KB) | Total (KB) |
| :-: | :-------: | :---: | :-------: | :-: | :-----------: | :--------------------: | :--------: |
| 13  |     9     | 8192  |    512    | 16  |      128      |           8            |    136     |
| 13  |     8     | 8192  |    256    | 16  |      128      |           4            |    132     |
| 12  |     7     | 4096  |    128    | 16  |      64       |           2            |     66     |
| 14  |    10     | 16384 |   1024    |  8  |      128      |           8            |    136     |
| 14  |     9     | 16384 |    512    |  8  |      128      |           4            |    132     |
| 13  |     9     | 8192  |    512    |  8  |      64       |           4            |     68     |
| 13  |     8     | 8192  |    256    |  8  |      64       |           2            |     66     |
| 12  |     7     | 4096  |    128    |  8  |      32       |           1            |     33     |
