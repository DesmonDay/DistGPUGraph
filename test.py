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

    if dist.get_rank() == 0:
        node_info = {"node_sidx": 0, "node_eidx": 9}
        forward_meta_info = {1: paddle.to_tensor([4, 5, 7, 8, 9])}
        backward_meta_info = {1: paddle.to_tensor([11, 15, 16, 19])}
        shard_tool = ShardTool(node_info, forward_meta_info, backward_meta_info)
    elif dist.get_rank() == 1:
        node_info = {"node_sidx": 10, "node_eidx": 20}
        forward_meta_info = {0: paddle.to_tensor([11, 15, 16, 19])}
        backward_meta_info = {0: paddle.to_tensor([4, 5, 7, 8, 9])}
        shard_tool = ShardTool(node_info, forward_meta_info, backward_meta_info)

    # emb_size = 3
    data = paddle.to_tensor(np.random.randn(node_info["node_eidx"] - node_info["node_sidx"] + 1, 3))
    data.stop_gradient = False
    gather_scatter = GatherScatter()
    y = gather_scatter.apply(data, shard_tool)

    print("GPU: ", dist.get_rank())
    print(y)

    y.mean().backward()
    print(data.grad)


if __name__ == "__main__":
    dist.spawn(main, nprocs=-1)
