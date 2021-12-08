import time
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
                    pgl.nn.GCNConv(
                        input_size,
                        self.hidden_size,
                        activation="relu",
                        norm=True))
            else:
                self.gcns.append(
                    pgl.nn.GCNConv(
                        self.hidden_size,
                        self.hidden_size,
                        activation="relu",
                        norm=True))
            self.gcns.append(nn.Dropout(self.dropout))
        self.gcns.append(pgl.nn.GCNConv(self.hidden_size, self.num_class))
       
    def forward(self, graph, feature, shard_tool):
        feature = self.gather_scatter.apply(feature, shard_tool)
        for m in self.gcns:
            if isinstance(m, nn.Dropout):
                feature = m(feature)
            else:
                feature = m(graph, feature)
        feature = slice_feat(shard_tool, feature)
        return feature


def normalize(feat):
    return feat / np.maximum(np.sum(feat, -1, keepdims=True), 1)


def load(name="cora"):
    log.info("Begin to load %s dataset." % name)

    if name == "cora":
        dataset = pgl.dataset.CoraDataset()
    elif name == "pubmed":
        dataset = pgl.dataset.CitationDataset("pubmed", symmetry_edges=True)
    elif name == "citeseer":
        dataset = pgl.dataset.CitationDataset("citeseer", symmetry_edges=True)
    elif name == "reddit":
        dataset = pgl.dataset.RedditDataset()
    elif name == "arxiv":
        dataset = pgl.dataset.OgbnArxivDataset()
    else:
        raise ValueError(name + " dataset doesn't exists")

    if name in ["cora", "pubmed", "citeseer"]:
        feature = normalize(dataset.graph.node_feat["words"])
    else:
        feature = dataset.feature

    if name == "reddit":
        y = np.arange(0, dataset.graph.num_nodes)
        y[dataset.train_index] = dataset.train_label
        y[dataset.val_index] = dataset.val_label
        y[dataset.test_index] = dataset.test_label
        dataset.y = y

    # Information of dataset: graph, y, train_index, val_index, test_index
    return dataset, feature


def shuffle_data(dataset, feature):
    log.info("Begin to shuffle dataset.")
    # Random shuffle data
    np.random.seed(100)
    permutation = np.arange(0, dataset.graph.num_nodes)
    np.random.shuffle(permutation)
    
    edges = dataset.graph.edges
    reindex = {}
    for ind, node in enumerate(permutation):
        reindex[node] = ind
    new_edges = pgl.graph_kernel.map_edges(
        np.arange(
            len(edges), dtype="int64"), edges, reindex)
    graph = pgl.Graph(edges=new_edges, num_nodes=dataset.graph.num_nodes)

    # Shuffle dataset and feature.
    dataset.graph = graph
    feature = feature[permutation]
    dataset.y = dataset.y[permutation]
    
    train_mask = np.zeros(dataset.graph.num_nodes, dtype=np.int64)
    train_mask[dataset.train_index] = 1
    train_mask = train_mask[permutation]
    dataset.train_index = np.where(train_mask == 1)[0]

    val_mask = np.zeros(dataset.graph.num_nodes, dtype=np.int64)
    val_mask[dataset.val_index] = 1
    val_mask = val_mask[permutation]
    dataset.val_index = np.where(val_mask == 1)[0]

    test_mask = np.zeros(dataset.graph.num_nodes, dtype=np.int64)
    test_mask[dataset.test_index] = 1
    test_mask = test_mask[permutation]
    dataset.test_index = np.where(test_mask == 1)[0]
   
    return dataset, feature


def dispatch_data(dataset, feature):
    log.info("Begin to dispatch dataset.")

    # Get dataset
    graph = dataset.graph
    y = dataset.y
    train_index = dataset.train_index
    val_index = dataset.val_index
    test_index = dataset.test_index

    num_nodes = graph.num_nodes
    num_procs = dist.get_world_size()
    proc_id = dist.get_rank()

    # 目前先暂时均等分片
    split_nodes = np.array_split(np.arange(num_nodes), num_procs)
    cur_nodes = split_nodes[proc_id]
    
    ## Get node_info
    node_info = {"node_sidx": cur_nodes[0], "node_eidx": cur_nodes[-1]}

    ## Get forward_meta_info 
    succ_nodes = np.concatenate(graph.successor(nodes=cur_nodes), -1)
    send_nodes = np.setdiff1d(succ_nodes, cur_nodes) # 经测试，已去重和排序
    # 获取 send_nodes 各自对应的卡号，确定当前卡需要发给其他卡什么数据
    nodes_buffer = np.zeros(num_nodes)
    nodes_buffer[send_nodes] = 1
    split_nodes_buf = np.array_split(nodes_buffer, num_procs)
    forward_meta_info = {}
    for i, buf in enumerate(split_nodes_buf):
        if i == proc_id:
            continue
        card_i = split_nodes[i][np.where(buf == 1)[0]]
        pred_nodes = np.concatenate(graph.predecessor(nodes=card_i), -1) 
        # 取交集
        forward_meta_info[i] = paddle.to_tensor(np.intersect1d(pred_nodes, cur_nodes), dtype="int32")
 
    ## Get backward_meta_info
    pred_nodes, pred_eids = graph.predecessor(nodes=cur_nodes, return_eids=True) 
    pred_nodes = np.concatenate(pred_nodes, -1)
    send_grads = np.setdiff1d(pred_nodes, cur_nodes)
    # 获得不同卡的梯度节点个数
    nodes_buffer = np.zeros(num_nodes)
    nodes_buffer[send_grads] = 1
    split_nodes_buf = np.array_split(nodes_buffer, num_procs)
    backward_meta_info = np.zeros(num_procs, dtype="int32")
    for i, buf in enumerate(split_nodes_buf):
        backward_meta_info[i] = np.sum(buf)
    
    # 获取新graph的edge index和node index
    new_graph_nodes = np.sort(np.concatenate((send_grads, cur_nodes)))
    pred_eids = np.concatenate(pred_eids, -1)
    new_graph = subgraph(graph, nodes=new_graph_nodes, eid=pred_eids,
                         with_node_feat=False)

    shard_tool = ShardTool(node_info, forward_meta_info, backward_meta_info)

    # 划分训练集，测试集
    dataset.feature = paddle.to_tensor(feature[cur_nodes], dtype="float32")
    dataset.y = y[cur_nodes]
    if len(dataset.y.shape) == 2:
        dataset.y = np.squeeze(dataset.y, -1)

    dataset.graph = new_graph
    dataset.graph.tensor()
    index_buffer = np.zeros(num_nodes)

    # Get train index
    index_buffer[train_index] = 1
    split_index_buf = np.array_split(index_buffer, num_procs)
    train_index = np.where(split_index_buf[proc_id] == 1)[0]
    dataset.train_label = paddle.to_tensor(
        np.expand_dims(dataset.y[train_index], -1))
    dataset.train_index = paddle.to_tensor(np.expand_dims(train_index, -1))

    index_buffer[train_index] = 0
    # Get val index
    index_buffer[val_index] = 1
    split_index_buf = np.array_split(index_buffer, num_procs)
    val_index = np.where(split_index_buf[proc_id] == 1)[0]
    dataset.val_label = paddle.to_tensor(
        np.expand_dims(dataset.y[val_index], -1))
    dataset.val_index = paddle.to_tensor(np.expand_dims(val_index, -1))

    index_buffer[val_index] = 0    
    # Get test index
    index_buffer[test_index] = 1
    split_index_buf = np.array_split(index_buffer, num_procs)
    test_index = np.where(split_index_buf[proc_id] == 1)[0]
    dataset.test_label = paddle.to_tensor(
        np.expand_dims(dataset.y[test_index], -1))
    dataset.test_index = paddle.to_tensor(np.expand_dims(test_index, -1))

    return dataset, shard_tool


def slice_feat(shard_tool, feature):
    backward_meta_info = shard_tool.backward_meta_info
    start_idx = 0
    for i in range(dist.get_world_size()):
        if i == dist.get_rank():
            own_start_idx = start_idx
            own_end_idx = own_start_idx + (shard_tool.node_eidx - shard_tool.node_sidx + 1)
            break
        start_idx += backward_meta_info[i]
    feature = feature[own_start_idx : own_end_idx]
    return feature


def train(node_index, node_label, shard_tool, model, graph, feature, criterion, optim):
    model.train()

    with model.no_sync():
        pred = model(graph, feature, shard_tool)
        pred = paddle.gather(pred, node_index)
        loss = criterion(pred, node_label)
        loss.backward()
   
    fused_allreduce_gradients(list(model.parameters()), None) 
    acc = paddle.metric.accuracy(input=pred, label=node_label, k=1)
    optim.step()
    optim.clear_grad()
    return loss, acc


@paddle.no_grad()
def eval(node_index, node_label, shard_tool, model, graph, feature, criterion):
    model.eval()

    with model.no_sync():
        pred = model(graph, feature, shard_tool)
        pred = paddle.gather(pred, node_index)
        loss = criterion(pred, node_label)
    acc = paddle.metric.accuracy(input=pred, label=node_label, k=1)
    return loss, acc 


def main(args):
    if dist.get_world_size() > 1:
        dist.init_parallel_env()

    dataset, feature = load(name=args.dataset)
    dataset, feature = shuffle_data(dataset, feature)
    dataset, shard_tool = dispatch_data(dataset, feature)

    feature = dataset.feature
    graph = dataset.graph

    train_index = dataset.train_index
    train_label = dataset.train_label

    val_index = dataset.val_index
    val_label = dataset.val_label

    test_index = dataset.test_index
    test_label = dataset.test_label

    # GCN model
    model = GCN(input_size=feature.shape[1],
                num_class=dataset.num_classes) # 暂时用默认参数 
    if dist.get_world_size() > 1:
        model = paddle.DataParallel(model) 

    criterion = paddle.nn.loss.CrossEntropyLoss()
    optim = Adam(
        learning_rate=0.01,
        parameters=model.parameters())

    cal_val_acc = []
    cal_test_acc = []

    for epoch in tqdm.tqdm(range(args.epoch)):
        train_loss, train_acc = train(train_index, train_label, shard_tool,
                                      model, graph, feature, criterion, optim)
        val_loss, val_acc = eval(val_index, val_label, shard_tool, model,
                                 graph, feature, criterion)
        test_loss, test_acc = eval(test_index, test_label, shard_tool, model,
                                   graph, feature, criterion)
        cal_val_acc.append(val_acc.numpy())
        cal_test_acc.append(test_acc.numpy())

    print(dist.get_rank(), cal_test_acc[np.argmax(cal_val_acc)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='DistGPUGraph')
    parser.add_argument(
        "--dataset", type=str, default="cora", help="cora, pubmed, citeseer, reddit, arxiv")
    parser.add_argument(
        "--epoch", type=int, default=200, help="Epoch")
    args = parser.parse_args()
    dist.spawn(main, args=(args, ), nprocs=-1)

