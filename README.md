# DistGPUGraph

## Required

NCCL >= 2.7.3

## Runs
```shell

# 首先用需要保存 partition 后的数据集.

python get_partition_dataset.py --dataset reddit --mode metis --num_procs 4

# 接着可以用 gcn, gat, unimp 模型来训.

(注意: train_unimp.py 里面实际跑的是阿杰修改的 ResGAT)

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_gcn.py --dataset reddit --mode metis --hidden_size 256 --num_layers 1

```
