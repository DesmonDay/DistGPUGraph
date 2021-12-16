import time
import pickle
import argparse
import tqdm
from easydict import EasyDict as edict

import numpy as np
import pgl
import paddle
import paddle.nn as nn
from pgl.utils.logger import log
from paddle.optimizer import Adam
import paddle.distributed as dist
from pgl.sampling.custom import subgraph
from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

from gpu_shard_tool import ShardTool
from gather_scatter_layer import GatherScatter
from new_conv import GCNConv


class GCN(nn.Layer):
    def __init__(self, input_size, num_class, num_layers=1, 
                 hidden_size=64, dropout=0.5, **kwargs):
        super(GCN, self).__init__()
       
        self.num_class = num_class
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.gcns = nn.LayerList()
        self.gather_scatter = GatherScatter()

        for i in range(self.num_layers):
            if i == 0:
                self.gcns.append(
                    GCNConv(
                        input_size,
                        self.hidden_size,
                        activation="relu",
                        norm=True))
            else:
                self.gcns.append(
                    GCNConv(
                        self.hidden_size,
                        self.hidden_size,
                        activation="relu",
                        norm=True))
            self.gcns.append(nn.Dropout(self.dropout))
        self.gcns.append(GCNConv(self.hidden_size, self.num_class))
       
    def forward(self, graph, feature, norm, shard_tool):
        for m in self.gcns:
            if isinstance(m, nn.Dropout):
                feature = m(feature)
            else:
                # feature = self.gather_scatter.apply(feature, shard_tool)
                feature = m(graph, feature, self.gather_scatter, shard_tool, norm)
                # feature = feature[shard_tool.own_start_idx : shard_tool.own_end_idx]
        return feature


def load_data(load_path, proc_id):
    load_path = load_path + "/part_%d" % proc_id
    cur_nodes = np.load(load_path + "/cur_nodes.npy")
    node_info = {"node_sidx": cur_nodes[0], "node_eidx": cur_nodes[-1]}
    f = open(load_path + "/forward_meta_info.pkl", "rb")
    forward_meta_info = pickle.load(f)
    for i in forward_meta_info:
        forward_meta_info[i] = paddle.to_tensor(forward_meta_info[i]) 
    backward_meta_info = np.load(load_path + "/backward_meta_info.npy")
    graph = pgl.Graph.load(load_path, mmap_mode=None)
    graph.tensor()
    feature = paddle.to_tensor(np.load(load_path + "/feature.npy"), dtype="float32")
    gcn_norm = paddle.to_tensor(np.load(load_path + "/gcn_norm.npy"), dtype="float32")   
 
    train_index = paddle.to_tensor(np.load(load_path + "/train_index.npy"))
    train_label = paddle.to_tensor(np.load(load_path + "/train_label.npy"))
    val_index = paddle.to_tensor(np.load(load_path + "/val_index.npy"))
    val_label = paddle.to_tensor(np.load(load_path + "/val_label.npy"))
    test_index = paddle.to_tensor(np.load(load_path + "/test_index.npy"))
    test_label = paddle.to_tensor(np.load(load_path + "/test_label.npy"))

    shard_tool = ShardTool(node_info, forward_meta_info, backward_meta_info)

    num_classes = np.load(load_path + "/num_classes.npy")
    return graph, feature, gcn_norm, shard_tool, num_classes, train_index, train_label, \
           val_index, val_label, test_index, test_label 


def train(node_index, node_label, loss_scale, shard_tool, model, graph, feature, gcn_norm, criterion, optim):
    model.train()

    with model.no_sync():
        pred = model(graph, feature, gcn_norm, shard_tool)
        pred = paddle.gather(pred, node_index)
        loss = criterion(pred, node_label)
        loss = loss * loss_scale
        # loss = loss / 所有卡的训练样本数.
        loss.backward()
   
    fused_allreduce_gradients(list(model.parameters()), None) 
    optim.step()
    optim.clear_grad()
    return loss


@paddle.no_grad()
def eval(node_index, node_label, shard_tool, model, graph, feature, gcn_norm, criterion):
    model.eval()

    with model.no_sync():
        pred = model(graph, feature, gcn_norm, shard_tool)
        pred = paddle.gather(pred, node_index)
        loss = criterion(pred, node_label)

    correct_num = paddle.sum(paddle.cast(paddle.argmax(pred, -1, keepdim=True) == node_label, dtype="float32"))
    total_num = paddle.sum(paddle.ones_like(node_label, dtype="float32"))

    correct_num = dist.all_reduce(correct_num)
    total_num = dist.all_reduce(total_num)
    
    acc = correct_num / total_num
    return loss, acc


def main(args):
    if dist.get_world_size() > 1:
        dist.init_parallel_env()

    load_path = "%s_data/save_%d_split_data_%s" % (args.dataset, dist.get_world_size(),
                                                   args.mode)
    graph, feature, gcn_norm, shard_tool, num_classes, train_index, train_label, \
        val_index, val_label, test_index, test_label = load_data(load_path, dist.get_rank())
    print(graph)

    # GCN model
    model = GCN(input_size=feature.shape[1],
                num_class=num_classes,
                num_layers=args.num_layers,
                hidden_size=args.hidden_size) # 暂时用默认参数 
    if dist.get_world_size() > 1:
        model = paddle.DataParallel(model) 

    criterion = paddle.nn.loss.CrossEntropyLoss(reduction='sum')
    optim = Adam(
        learning_rate=0.01,
        parameters=model.parameters())

    cal_val_acc = []
    cal_test_acc = []
    loss_scale = paddle.to_tensor([train_index.shape[0]], dtype="float32")
    dist.all_reduce(loss_scale)
    loss_scale = 1 / loss_scale

    for epoch in tqdm.tqdm(range(args.epoch)):
        train_loss = train(train_index, train_label, loss_scale, shard_tool,
                           model, graph, feature, gcn_norm, criterion, optim)
        val_loss, val_acc = eval(val_index, val_label, shard_tool, model,
                                 graph, feature, gcn_norm, criterion)
        test_loss, test_acc = eval(test_index, test_label, shard_tool, model,
                                   graph, feature, gcn_norm, criterion)
        cal_val_acc.append(val_acc.numpy())
        cal_test_acc.append(test_acc.numpy())

    best_test_acc = cal_test_acc[np.argmax(cal_val_acc)]
    log.info("GPU: %d, Test acc: %f" % (dist.get_rank(), best_test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='DistGPUGraph')
    parser.add_argument(
        "--dataset", type=str, default="cora", help="cora, pubmed, citeseer, reddit, arxiv")
    parser.add_argument(
        "--mode", type=str, default="metis", help="metis, random")
    parser.add_argument(
        "--epoch", type=int, default=200, help="Epoch")

    # Model
    parser.add_argument(
        "--hidden_size", type=int, default=64)
    parser.add_argument(
        "--num_layers", type=int, default=1)

    args = parser.parse_args()
    dist.spawn(main, args=(args, ), nprocs=-1)

