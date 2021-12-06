import numpy as np
import paddle
import paddle.nn as nn
import paddle.distributed as dist


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


class ShardTool(object):
    def __init__(self, node_info, forward_meta_info, backward_meta_info):
        self.node_sidx = node_info["node_sidx"]
        self.node_eidx = node_info["node_eidx"]
        self.forward_meta_info = forward_meta_info
        self.backward_meta_info = backward_meta_info

    def forward_gather(self, shard_node_emb): 
        self.send_forward_index()
        recv_meta = self.recv_forward_index()

        self.send_emb(shard_node_emb)
        forward_emb = self.recv_emb(shard_node_emb, recv_meta)
        forward_emb = paddle.concat(forward_emb, axis=0)
        return forward_emb

    def backward_scatter(self, grad):
        self.send_backward_index()
        recv_meta = self.recv_backward_index()

        own_start_idx, own_end_idx = self.send_grad(grad)
        recv_grads = self.recv_grad(grad, recv_meta)

        own_grad_idx = paddle.arange(own_start_idx, own_end_idx, step=1)
        backward_grad = paddle.gather(grad, own_grad_idx)
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                continue
            backward_grad[self.forward_meta_info[i] - self.node_sidx] += recv_grads[i]
        return backward_grad

    def send_forward_index(self):
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                continue
            size = paddle.to_tensor(self.forward_meta_info[i].shape[0], 
                dtype="int32")
            dist.send(size, dst=i)

    def recv_forward_index(self):
        recv_output = []
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                recv_output.append(paddle.to_tensor([0], dtype="int32")) # 实际上并不需要，仅占位用
                continue
            tensor_type = paddle.to_tensor([0], dtype="int32")
            dist.recv(tensor_type, src=i)
            recv_output.append(tensor_type)

        for o in recv_output:
            dist.wait(o)
        return recv_output

    def send_emb(self, shard_node_emb):
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                continue
            emb = paddle.gather(shard_node_emb, self.forward_meta_info[i] - self.node_sidx)
            dist.send(emb, dst=i)

    def recv_emb(self, shard_node_emb, recv_meta):
        recv_output = [] 
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                recv_output.append(shard_node_emb)
                continue
            tensor_type = paddle.zeros([recv_meta[i], shard_node_emb.shape[1]], 
                dtype=shard_node_emb.dtype)
            dist.recv(tensor_type, src=i)
            recv_output.append(tensor_type)

        for o in recv_output:
            dist.wait(o)
        return recv_output

    def send_backward_index(self):
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                continue
            size = paddle.to_tensor(self.backward_meta_info[i].shape[0], 
                dtype="int32")
            dist.send(size, dst=i)

    def recv_backward_index(self):
        recv_output = []
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                recv_output.append(paddle.to_tensor([0], dtype="int32")) # 实际上并不需要，仅占位用
                continue
            tensor_type = paddle.to_tensor([0], dtype="int32")
            dist.recv(tensor_type, src=i)
            recv_output.append(tensor_type)

        for o in recv_output:
            dist.wait(o)
        return recv_output

    def send_grad(self, grad):
        start_idx = 0
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                own_start_idx = start_idx
                start_idx += (self.node_eidx - self.node_sidx + 1)
                own_end_idx = start_idx
                continue
            end_idx = start_idx + self.backward_meta_info[i].shape[0]
            grad_i = paddle.gather(grad, paddle.arange(start_idx, end_idx, step=1))
            start_idx = end_idx
            dist.send(grad_i, dst=i)
        return own_start_idx, own_end_idx

    def recv_grad(self, grad, recv_meta):
        recv_output = []
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                recv_output.append(paddle.to_tensor([0], dtype="int32")) # 实际上并不需要，仅占位用
                continue
            tensor_type = paddle.zeros([recv_meta[i], grad.shape[1]],
                dtype=grad.dtype)
            dist.recv(tensor_type, src=i)
            recv_output.append(tensor_type)

        for o in recv_output:
            dist.wait(o)
        return recv_output

