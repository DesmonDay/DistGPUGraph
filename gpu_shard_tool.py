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
Update: backward_meta_info实际上只需要保存对应卡号所需要的梯度长度，
        因为我在聚合 emb时是根据卡号顺序依次添加的，因此可以更新为
        backward_meta_info = {1: 4}

GPU1:
node_info = {"node_sidx": 10, "node_eidx": 20}
forward_meta_info = {0: [11, 15, 16, 19]} # 表示需要发送给0号卡 11,15,16,19 号节点
backward_meta_info = {0: [4, 5, 7, 8, 9]} # 表示需要将拿到的这几个节点的梯度发送给0号卡
Update: backward_meta_info = {0: 5}

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
        self.own_start_idx, self.own_end_idx, self.split_idx = self.get_split_idx()

    def get_split_idx(self):
        split_idx = {}
        start_idx = 0
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                own_start_idx = start_idx
                start_idx += self.node_eidx - self.node_sidx + 1
                own_end_idx = start_idx
            end_idx = start_idx + self.backward_meta_info[i] 
            split_idx[i] = (start_idx, end_idx)
            start_idx = end_idx
        return own_start_idx, own_end_idx, split_idx

    def forward_gather(self, shard_node_emb): 
        recv_meta = self.send_recv_forward_index()
        forward_emb = self.send_recv_emb(shard_node_emb, recv_meta)
        forward_emb = paddle.concat(forward_emb, axis=0)
        return forward_emb

    def backward_scatter(self, grad):
        recv_meta = self.send_recv_backward_index()
        recv_grads = self.send_recv_grad(grad, recv_meta)
        recv_grads = paddle.concat(recv_grads, 0)
        backward_grad = grad[self.own_start_idx : self.own_end_idx]
        init_backward_grad = paddle.zeros_like(backward_grad)
        forward_meta_infos = [self.forward_meta_info[i] - self.node_sidx 
                              for i in range(dist.get_world_size()) if i != dist.get_rank()]
        forward_meta_infos = paddle.concat(forward_meta_infos, 0)
        acc_backward_grad = paddle.scatter(init_backward_grad, forward_meta_infos,
                                     recv_grads, overwrite=False)
        return backward_grad + acc_backward_grad

    def send_recv_forward_index(self):
        recv_output = []
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                recv_output.append(paddle.zeros([1], dtype="int32")) # 占位
                continue
            if i < dist.get_rank():
                tensor_type = paddle.zeros([1], dtype="int32")
                dist.recv(tensor_type, src=i)
                recv_output.append(tensor_type)
                dist.wait(tensor_type)
                size = paddle.to_tensor(np.array([self.forward_meta_info[i].shape[0]], dtype="int32"))
                dist.send(size, dst=i)
            else:
                size = paddle.to_tensor(np.array([self.forward_meta_info[i].shape[0]], dtype="int32"))
                dist.send(size, dst=i)
                tensor_type = paddle.zeros([1], dtype="int32")
                dist.recv(tensor_type, src=i)
                recv_output.append(tensor_type)
                dist.wait(tensor_type)
        return recv_output

    def send_recv_emb(self, shard_node_emb, recv_meta):
        recv_output = []
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                recv_output.append(shard_node_emb)
                continue
            if i < dist.get_rank():
                tensor_type = paddle.zeros([recv_meta[i], shard_node_emb.shape[1]],
                    dtype=shard_node_emb.dtype)
                dist.recv(tensor_type, src=i)
                recv_output.append(tensor_type)
                dist.wait(tensor_type)
                emb = paddle.gather(shard_node_emb, self.forward_meta_info[i] - self.node_sidx)
                dist.send(emb, dst=i)
            else:
                emb = paddle.gather(shard_node_emb, self.forward_meta_info[i] - self.node_sidx)
                dist.send(emb, dst=i)
                tensor_type = paddle.zeros([recv_meta[i], shard_node_emb.shape[1]],
                    dtype=shard_node_emb.dtype)
                dist.recv(tensor_type, src=i)
                recv_output.append(tensor_type)
                dist.wait(tensor_type)
        return recv_output

    def send_recv_backward_index(self):
        recv_output = []
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                recv_output.append(paddle.zeros([1], dtype="int32")) # 占位
                continue
            if i < dist.get_rank():
                tensor_type = paddle.zeros([1], dtype="int32")
                dist.recv(tensor_type, src=i)
                recv_output.append(tensor_type)
                dist.wait(tensor_type)
                size = paddle.to_tensor(np.array(self.backward_meta_info[i], dtype="int32"))
                dist.send(size, dst=i)
            else:
                size = paddle.to_tensor(np.array(self.backward_meta_info[i], dtype="int32"))
                dist.send(size, dst=i)
                tensor_type = paddle.zeros([1], dtype="int32")
                dist.recv(tensor_type, src=i)
                recv_output.append(tensor_type)
                dist.wait(tensor_type)
        return recv_output

    def send_recv_grad(self, grad, recv_meta):
        recv_output = []
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                continue
            if i < dist.get_rank():
                tensor_type = paddle.zeros([recv_meta[i], grad.shape[1]],
                    dtype=grad.dtype)
                dist.recv(tensor_type, src=i)
                recv_output.append(tensor_type)
                dist.wait(tensor_type)
                grad_i = grad[self.split_idx[i][0] : self.split_idx[i][1]]
                dist.send(grad_i, dst=i)
            else:
                grad_i = grad[self.split_idx[i][0] : self.split_idx[i][1]]
                dist.send(grad_i, dst=i)
                tensor_type = paddle.zeros([recv_meta[i], grad.shape[1]],
                    dtype=grad.dtype)
                dist.recv(tensor_type, src=i)
                recv_output.append(tensor_type)
                dist.wait(tensor_type)
        return recv_output
                
