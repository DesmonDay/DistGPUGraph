import paddle.distributed as dist
import paddle
from tqdm import tqdm
import time

class Test(object):
    def __init__(self):
        shape = [10000 * dist.get_world_size(), 100]
        self.emb = paddle.uniform(shape, dtype='float32')

        shape = [40000 * dist.get_world_size(), 100]
        self.emb2 = paddle.uniform(shape, dtype='float32')

    def send_recv_emb(self):
        shard_node_emb = self.emb[10000 * dist.get_rank(): 10000 * (dist.get_rank() + 1)]
        recv_output = []
        send_output = []
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                send_output.append(None)
                recv_output.append(shard_node_emb)
            else:
                send_output.append(self.emb[10000 * i: 10000 * (i + 1)])
                tensor_type = paddle.zeros([10000, 100], dtype="float32")
                recv_output.append(tensor_type)


        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                continue
            if i < dist.get_rank():
                dist.recv(recv_output[i], src=i)
                dist.send(send_output[i], dst=i)
            else:
                dist.send(send_output[i], dst=i)
                dist.recv(recv_output[i], src=i)
        recv_output = paddle.concat(recv_output, 0)
        return recv_output

    def send_recv_emb_v2(self):
        shard_node_emb = self.emb[10000 * dist.get_rank(): 10000 * (dist.get_rank() + 1)]
        recv_output = []
        send_output = []
        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                send_output.append(shard_node_emb)
            else:
                send_output.append(self.emb[10000 * i: 10000 * (i + 1)])
        dist.alltoall(send_output, recv_output)
        recv_output = paddle.concat(recv_output, 0)
        return recv_output

    def send_recv_emb_v3(self):
        shard_node_emb = self.emb2
        return dist.all_reduce(shard_node_emb)


def main(args):
    if dist.get_world_size() > 1:
        dist.init_parallel_env()
    g = Test()
    start = time.time()
    for i in tqdm(range(100)):
        with paddle.no_grad():
            output = g.send_recv_emb()
            output = output.numpy()
    end = time.time()
    print("sp", (end - start) / 100)

    start = time.time()
    for i in tqdm(range(100)):
        with paddle.no_grad():
            output = g.send_recv_emb_v2()
            output = output.numpy()
    end = time.time()
    print("sp2", (end - start) / 100)
 

    start = time.time()
    for i in tqdm(range(100)):
        with paddle.no_grad():
            output = g.send_recv_emb_v3()
            output = output.numpy()
    end = time.time()
    print("sp3", (end - start) / 100)
 


if __name__ == "__main__":
    args = None
    dist.spawn(main, args=(args, ), nprocs=-1)
