# DistGPUGraph

## Required

NCCL >= 2.7.3

## Runs
```shell

CUDA_VISIBLE_DEVICES=0,1,2 python test_multi_cards.py


CUDA_VISIBLE_DEVICES=0,1 python test_gcn.py  # 默认Cora，仅在模型forward开头拉取特征


CUDA_VISIBLE_DEVICES=0,1 python test_gcn_update_layer.py  # 默认Cora，模型每一层都需拉取特征

```
