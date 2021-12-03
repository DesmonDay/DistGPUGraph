import numpy as np
import paddle
import paddle.nn as nn
import paddle.distributed as dist


class ShardTool(object):
    def __init__(self, node_info, send_meta_info, recv_meta_info):
        self.node_sidx = node_info["node_sidx"]
        self.node_edix = node_info["node_eidx"]
        self.send_meta_info = send_meta_info
        self.recv_meta_info = recv_meta_info

    def forward_gather(self, shard_node_emb): 
        self.send_forward_index()
        recv_meta = self.recv_forward_index()

        self.send_emb(shard_node_emb)
        forward_emb = self.recv_emb(shard_node_emb, recv_meta)
        forward_emb = paddle.concat(forward_emb, axis=0)

        return forward_emb

    def backward_scatter(self, ):
        # Backward: 发送grad，接收grad
        pass

    def send_forward_index(self):
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                continue
            size = paddle.to_tensor(self.send_meta_info[i].shape[0], dtype="int32")
            dist.send(size, dst=i)

    def recv_forward_index(self):
        recv_output = []
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                recv_output.append(paddle.to_tensor([0], dtype="int32")) # 实际上并不需要，仅占位用
                continue
            tensor_type = paddle.zeros([0], dtype="int32")
            dist.recv(tensor_type, src=i)
            recv_output.append(tensor_type)

        for o in recv_output:
            dist.wait(o)

    def send_emb(self, shard_node_emb):
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                continue
            # 这里先暂时默认 send_meta_info 为 paddle.Tensor
            # 处理偏置问题
            emb = paddle.gather(shard_node_emb, self.send_meta_info[i] - self.node_sidx)
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

    def send_grad(self):
        pass

    def recv_grad(self):
        pass
