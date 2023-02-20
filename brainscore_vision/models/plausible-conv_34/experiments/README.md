# Notes on experiments

## CIFAR10
### Resnet18 + sparse connections + first layer is conv
#### Adam

Random channel-channel connectivity
| channel-channel sparsity |  lr  | weight decay | n epochs | batch size | **10% validation accuracy** |
|--------------------------|:----:|:------------:|:--------:|:----------:|:-----------------------:|
| 1/16                     | 5e-4 | 5e-4         | 300      | 64         | ~82%                    |
| 1/32                     | 5e-4 | 5e-4         | 300      | 128        | ~83%                    |
| 1/64                     | 5e-4 | 5e-4         | 300      | 256        | ~84%                    |

Semi-deterministic channel-channel connectivity (so all input channels are connected)

| channel-channel sparsity |  lr  | weight decay | n epochs | batch size | **10% validation accuracy** |
|--------------------------|:----:|:------------:|:--------:|:----------:|:-----------------------:|
| 1/64                     | 5e-4 | 1e-6         | 300      | 256        | ~85%                    |


### Resnet20CIFAR10 + sparse connections + first layer is conv
#### Adam

Semi-deterministic channel-channel connectivity (so all input channels are connected)
| channel-channel sparsity | n repetitions|  lr  | weight decay | (decrease->) n epochs | batch size | **10% validation accuracy** | commit |
|--------------------------|:------------:|:----:|:------------:|:--------:|:----------:|:-----------------------:|:-------:|
| 1/4                     |1| 5e-4 | 1e-6         | 300      | 128         | ~83%                    | |
| 1/4                     |1| 5e-4 | 1e-5         | 300      | 128         | ~82%                    | |
| 1/4                     |1| 5e-4 | 1e-4         | 300      | 128         | ~82%                    | |
| 1/4                     |1| 3e-4 | 1e-5         | 300      | 128         | ~82%                    | |
| 1/4                     |8| 3e-4 | 1e-5         | 300      | 128         | ~85%                    | |
| 1/4                     |1| 3e-4 | 1e-5         | 300      | 256         | ~83%                    | c91dc93ec2a7e756eb11b90ad51a99ee856e7f3e |
| 1/4                     |8| 3e-4 | 1e-5         | 300      | 256         | ~85-86%                    | |
| 1/4                     |16| 3e-4 | 1e-5         | 300      | 256         | ~85-86%                    | |
| 1/4  + eps=1.0                   |1| 3e-4 | 1e-5         | 300      | 256         | ~83%              | 5e2c499f53c6d025ed8f9166cec97babdfd1ba43     |
| conv                   |1| 3e-4 | 1e-5         | 300      | 256         | ~85-86%                    | f3758f903fd1f1c0cf544d84e539cb95f92a98c3 |
| conv                   |1| 5e-3 | 1e-5         | 300      | 128         | ~90%                    | 3a83b4d6d4ec437b9812971e9b0acd0895d4d585 |
| conv                   |1| 5e-3 | 1e-5         | 200 -> 300      | 128         | ~90%                    |  |
| 1/4                     |1| 1e-3 | 1e-5         | 200 -> 300      | 128         | ~84%                    | |
| 1/4                     |4| 1e-3 | 1e-5         | 200 -> 300      | 128         | ~85%                    | |
| 1/4                     |8| ?? | 1e-5         | 200 -> 300      | 128         | ~??%                    | |
| 1/4                     |16| ?? | 1e-5         | 200 -> 300      | 128         | ~??%                    | |
| 1/2                     |1| ?? | 1e-5         | 200 -> 300      | 128         | ~??%                    | |
| 1/2                     |4| ?? | 1e-5         | 200 -> 300      | 128         | ~??%                    | |
| 1/8                     |1| ?? | 1e-5         | 200 -> 300      | 128         | ~??%                    | |
| 1/8                     |4| ?? | 1e-5         | 200 -> 300      | 128         | ~??%                    | |
| 1/16                     |1| ?? | 1e-5         | 200 -> 300      | 128         | ~??%                    | |
| 1/16                     |4| ?? | 1e-5         | 200 -> 300      | 128         | ~??%                    | |

*equalizing the gradient at least with Adam led to divergence of the final loss. 
But maybe it just reflect equalization and not actual divergence in learning.

#### Depth experiment: 8 base width + AdamW + zero augmentation
Also the first layer is approx conv, and the init is relu kaiming normal

| type | base width | block length | ch-ch sparsity | n repetitions|  lr  | weight decay | (decrease->) n epochs | batch size | **10% validation accuracy** |
|:------------:|:------------:|:------------:|--------------------------|:------------:|:----:|:------------:|:--------:|:----------:|:-----------------------:|
| bp conv| 8 | 1 | 1                     |1| ? | ?        | 200->250->300      | 128         | ~??%   |
| bp lc| 8 | 1 | 1                     |1| ? | ?        | 200->250->300      | 128         | ~??%   |
| bp conv| 8 | 2 | 1                     |1| ? | ?        | 200->250->300      | 128         | ~??%   |
| bp lc| 8 | 2 | 1                     |1| ? | ?        | 200->250->300      | 128         | ~??%   |
| bp conv| 8 | 3 | 1                     |1| ? | ?        | 200->250->300      | 128         | ~??%   |
| bp lc| 8 | 3 | 1                     |1| ? | ?        | 200->250->300      | 128         | ~??%   |