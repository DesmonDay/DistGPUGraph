# Get random partition graph or metis partition graph.
import os
import time
import pickle
import argparse

import numpy as np
import pgl
from pgl.utils.logger import log
from pgl.sampling.custom import subgraph
from pgl.partition import metis_partition, random_partition


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


def get_permutation(dataset, num_procs, mode="random"):
    log.info("Begin to permute data.")    

    if mode == "random":
        np.random.seed(0)
        random_part = random_partition(dataset.graph, num_procs)
        permutation = np.argsort(random_part)
        part = np.zeros(num_procs + 1, dtype=np.int64)
        for i in range(num_procs):
            part[i + 1] = part[i] + len(np.where(random_part == i)[0])
    elif mode == "metis":
        metis_part = metis_partition(dataset.graph, num_procs)
        permutation = np.argsort(metis_part)
        part = np.zeros(num_procs + 1, dtype=np.int64)
        for i in range(num_procs):
            part[i + 1] = part[i] + len(np.where(metis_part == i)[0])
    else:
        raise ValueError("%s mode doesn't exists" % mode)

    return permutation, part   


def reindex_data(permutation, dataset, feature):
    edges = dataset.graph.edges
    reindex = {}
    for ind, node in enumerate(permutation):
        reindex[node] = ind
    new_edges = pgl.graph_kernel.map_edges(
        np.arange(
            len(edges), dtype="int64"), edges, reindex)
    graph = pgl.Graph(edges=new_edges, num_nodes=dataset.graph.num_nodes)

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


def dispatch_data(args, part, dataset, feature, num_procs):
    log.info("Begin to dispatch data.")

    # Get dataset
    train_index = dataset.train_index
    val_index = dataset.val_index
    test_index = dataset.test_index

    range_nodes = np.arange(dataset.graph.num_nodes)
    split_nodes = np.array_split(range_nodes, part[1:-1])

    print(split_nodes)
   
    if not os.path.exists("./%s_data/save_%d_split_data_%s" % (args.dataset, num_procs, args.mode)):
        os.mkdir("./%s_data/save_%d_split_data_%s" % (args.dataset, num_procs, args.mode))
    for proc_id in range(num_procs):  # 保存不同进程所需数据
        save_path = "./%s_data/save_%d_split_data_%s/part_%d" % (args.dataset, num_procs, args.mode, proc_id)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        cur_nodes = split_nodes[proc_id]  #  node_orig_index
        np.save(save_path + "/cur_nodes.npy", cur_nodes)  # Save
        feat = feature[cur_nodes]
        np.save(save_path + "/feature.npy", feat)  # Save
          
        # Get forward_meta_info
        succ_nodes = np.concatenate(dataset.graph.successor(nodes=cur_nodes), -1)
        send_nodes = np.setdiff1d(succ_nodes, cur_nodes) # 经测试，已去重和排序
        # 获取 send_nodes 各自对应的卡号，确定当前卡需要发给其他卡什么数据
        nodes_buffer = np.zeros(dataset.graph.num_nodes)
        nodes_buffer[send_nodes] = 1
        split_nodes_buf = np.array_split(nodes_buffer, part[1:-1])
        forward_meta_info = {}
        for i, buf in enumerate(split_nodes_buf):
            if i == proc_id:
                continue
            card_i = split_nodes[i][np.where(buf == 1)[0]]
            pred_nodes = np.concatenate(dataset.graph.predecessor(nodes=card_i), -1)
            # 取交集
            forward_meta_info[i] = np.intersect1d(pred_nodes, cur_nodes)
        f = open(save_path + "/forward_meta_info.pkl", "wb")
        pickle.dump(forward_meta_info, f)  # Save
        f.close() 
        
        # Get backward_meta_info
        pred_nodes, pred_eids = dataset.graph.predecessor(nodes=cur_nodes, return_eids=True)
        pred_nodes = np.concatenate(pred_nodes, -1)
        send_grads = np.setdiff1d(pred_nodes, cur_nodes)
        # 获得不同卡的梯度节点个数
        nodes_buffer = np.zeros(dataset.graph.num_nodes)
        nodes_buffer[send_grads] = 1
        split_nodes_buf = np.array_split(nodes_buffer, part[1:-1])
        backward_meta_info = np.zeros(num_procs, dtype="int32")
        for i, buf in enumerate(split_nodes_buf):
            backward_meta_info[i] = np.sum(buf)
        np.save(save_path + "/backward_meta_info.npy", backward_meta_info)  # Save

        # 获取新graph的edge index和node index
        new_graph_nodes = np.sort(np.concatenate((send_grads, cur_nodes)))
        pred_eids = np.concatenate(pred_eids, -1)
        new_graph = subgraph(dataset.graph, nodes=new_graph_nodes, eid=pred_eids,
                         with_node_feat=False)
        new_graph.adj_dst_index
        new_graph.dump(save_path)  # Save
        
        # 划分数据集
        index_buffer = np.zeros(dataset.graph.num_nodes)
        y = dataset.y[cur_nodes]
        y = np.squeeze(y, -1) if len(y.shape) == 2 else y
        # Get train index
        index_buffer[dataset.train_index] = 1
        split_index_buf = np.array_split(index_buffer, part[1:-1])
        train_index = np.where(split_index_buf[proc_id] == 1)[0]
        train_label = np.expand_dims(y[train_index], -1)
        index_buffer[dataset.train_index] = 0
        # Get val index
        index_buffer[dataset.val_index] = 1
        split_index_buf = np.array_split(index_buffer, part[1:-1])
        val_index = np.where(split_index_buf[proc_id] == 1)[0]
        val_label = np.expand_dims(y[val_index], -1)
        index_buffer[dataset.val_index] = 0
        # Get test index
        index_buffer[dataset.test_index] = 1
        split_index_buf = np.array_split(index_buffer, part[1:-1])
        test_index = np.where(split_index_buf[proc_id] == 1)[0]
        test_label = np.expand_dims(y[test_index], -1)

        np.save(save_path + "/train_index.npy", train_index)
        np.save(save_path + "/train_label.npy", train_label)
        np.save(save_path + "/val_index.npy", val_index)
        np.save(save_path + "/val_label.npy", val_label)
        np.save(save_path + "/test_index.npy", test_index)
        np.save(save_path + "/test_label.npy", test_label)

        num_classes = np.save(save_path + "/num_classes.npy", dataset.num_classes)


def main(args):
    if not os.path.exists("./%s_data" % args.dataset):
        os.mkdir("./%s_data" % args.dataset)
    dataset, feature = load(name=args.dataset)
    permutation, part = get_permutation(dataset, args.num_procs, args.mode)
    dataset, feature = reindex_data(permutation, dataset, feature)

    dispatch_data(args, part, dataset, feature, args.num_procs)

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GetSplitData')
    parser.add_argument(
        "--dataset", type=str, default="cora", help="cora, pubmed, citeseer, reddit, arxiv")
    parser.add_argument(
        "--num_procs", type=int, default=2, help="Number of train gpus")
    parser.add_argument(
        "--mode", type=str, default="metis", help="metis, random")
    args = parser.parse_args()
    main(args) 
