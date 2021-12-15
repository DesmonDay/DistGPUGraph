import unittest

import numpy as np
import paddle
import paddle.distributed as dist

from gpu_shard_tool import ShardTool
from gather_scatter_layer import GatherScatter


class TestGatherScatter(unittest.TestCase):
    # Just check 2-cards.

    def setUp(self):
        dist.init_parallel_env() # Can only run once.
        # nodes: 10, emb_dim: 2, gpus: 2
 
        node_infos = [{"node_sidx": i * 5, "node_eidx": (i + 1) * 5 - 1} for i in range(2)]
        forward_meta_infos = [{} for i in range(2)]
        forward_meta_infos[0] = {1: paddle.to_tensor([0, 2, 3], dtype="int32")}
        forward_meta_infos[1] = {0: paddle.to_tensor([5, 8], dtype="int32")}
        backward_meta_infos = []
        backward_meta_infos.append(np.array([0, 2], dtype="int32")) # 0-card
        backward_meta_infos.append(np.array([3, 0], dtype="int32")) # 1-card

        proc_id = dist.get_rank()
        node_info = node_infos[proc_id]
        forward_meta_info = forward_meta_infos[proc_id]
        backward_meta_info = backward_meta_infos[proc_id]
        self.shard_tool = ShardTool(node_info, forward_meta_info, backward_meta_info)

    def test_forward_backward(self):
        if dist.get_rank() == 0:
            emb = paddle.to_tensor([[0,0],[1,1],[2,2],[3,3],[4,4]], dtype="float32")
            emb.stop_gradient = False
            gather_scatter = GatherScatter()
            out = gather_scatter.apply(emb, self.shard_tool)
            y = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[8,8]], dtype="float32")
            out.mean().backward()
            print(dist.get_rank(), out.grad, emb.grad)
        else:
            emb = paddle.to_tensor([[5,5],[6,6],[7,7],[8,8],[9,9]], dtype="float32")
            emb.stop_gradient = False
            gather_scatter = GatherScatter()
            out = gather_scatter.apply(emb, self.shard_tool)
            y = np.array([[0,0],[2,2],[3,3],[5,5],[6,6],[7,7],[8,8],[9,9]], dtype="float32")
            out.mean().backward()
            print(dist.get_rank(), out.grad, emb.grad)

        self.assertTrue((out.numpy() == y).all())


if __name__ == "__main__":
    dist.spawn(unittest.main, nprocs=-1)
