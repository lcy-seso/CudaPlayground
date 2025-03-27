<p align="center">
<img src="figures/gemv.png"><br>
Fig. The vector-quantized GEMV operation.

hyper-parameters:

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

| index_bit | res_index_bit | vector_bit | codebook (KB) | residual codebook (KB) |
| :-------: | :-----------: | :--------: | :-----------: | :--------------------: |
|    12     |       4       |     16     |      128      |           0            |
|    12     |       4       |     8      |      64       |           0            |
|    12     |       5       |     16     |      128      |           1            |
|    12     |       5       |     8      |      64       |           0            |
|    12     |       6       |     16     |      128      |           2            |
|    12     |       6       |     8      |      64       |           1            |
|    11     |       4       |     16     |      64       |           0            |
|    11     |       4       |     8      |      32       |           0            |
|    11     |       5       |     16     |      64       |           1            |
|    11     |       5       |     8      |      32       |           0            |
|    11     |       6       |     16     |      64       |           2            |
|    11     |       6       |     8      |      32       |           1            |
|    10     |       4       |     16     |      32       |           0            |
|    10     |       4       |     8      |      16       |           0            |
|    10     |       5       |     16     |      32       |           1            |
|    10     |       5       |     8      |      16       |           0            |
|    10     |       6       |     16     |      32       |           2            |
|    10     |       6       |     8      |      16       |           1            |
