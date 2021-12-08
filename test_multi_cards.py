import warnings
warnings.filterwarnings('ignore')

import numpy as np
import paddle
import paddle.distributed as dist

from gpu_shard_tool import ShardTool
from gather_scatter_layer import GatherScatter


"""
2-Cards Example:

GPU0:
node_info = {"node_sidx": 0, "node_eidx": 9}
forward_meta_info = {1: [4, 5, 7, 8, 9]} # 表示需要发送给1号卡 5,7,8,9 号节点.
backward_meta_info = {1: [11, 15, 16, 19]} # 表示需要将拿到的这几个节点的梯度发送给1号卡

GPU1:
node_info = {"node_sidx": 10, "node_eidx": 20}
forward_meta_info = {0: [11, 15, 16, 19]} # 表示需要发送给0号卡 11,15,16,19 号节点
backward_meta_info = {0: [4, 5, 7, 8, 9]} # 表示需要将拿到的这几个节点的梯度发送给0号卡

forward 和 backward 区别：
forward时只需要把recv到的emb按顺序concat起来，因此不需要考虑实际index；
但backward需要把recv到的grad按照实际节点index对应相加.

"""


def main():

    # run: CUDA_VISIBLE_DEVICES=0,1 python test.py

    if dist.get_world_size() > 1:
        dist.init_parallel_env()
    
    np.random.seed(5)

    # Multi-cards examples
    node_infos = [{"node_sidx": i * 10, "node_eidx": (i + 1) * 10 - 1} for i in range(dist.get_world_size())]
    forward_meta_infos = [{} for i in range(dist.get_world_size())]
    backward_meta_infos = [np.zeros(dist.get_world_size(), dtype="int32") for i in range(dist.get_world_size())]
    for i in range(dist.get_world_size()):
        for j in range(dist.get_world_size()):
            if j == i:
                continue
            forward_meta_infos[i][j] = paddle.to_tensor(
                np.unique(np.random.randint(node_infos[i]["node_sidx"], node_infos[i]["node_eidx"] + 1, size=8)))
            backward_meta_infos[j][i] = len(forward_meta_infos[i][j])
    
    proc_id = dist.get_rank()
    node_info = node_infos[proc_id]
    forward_meta_info = forward_meta_infos[proc_id]
    backward_meta_info = backward_meta_infos[proc_id]
    shard_tool = ShardTool(node_info, forward_meta_info, backward_meta_info)

    # emb_size = 3
    np.random.seed(dist.get_rank())
    data = paddle.to_tensor(np.random.randn(node_info["node_eidx"] - node_info["node_sidx"] + 1, 3))
    data.stop_gradient = False
    gather_scatter = GatherScatter()
    y = gather_scatter.apply(data, shard_tool)
    y.mean().backward()

    print("GPU: %d" % dist.get_rank(), data.grad.numpy())


if __name__ == "__main__":
    dist.spawn(main, nprocs=-1)
