import time
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed as dist

import pgl
from pgl.nn import functional as GF


class GATConv(nn.Layer):
    """Implementation of graph attention networks (GAT)
    This is an implementation of the paper GRAPH ATTENTION NETWORKS
    (https://arxiv.org/abs/1710.10903).
    Args:
        input_size: The size of the inputs. 
        hidden_size: The hidden size for gat.
        activation: (default None) The activation for the output.
        num_heads: (default 1) The head number in gat.
        feat_drop: (default 0.6) Dropout rate for feature.
        attn_drop: (default 0.6) Dropout rate for attention.
        concat: (default True) Whether to concat output heads or average them.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 feat_drop=0.6,
                 attn_drop=0.6,
                 num_heads=1,
                 concat=True,
                 activation=None):
        super(GATConv, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.concat = concat

        self.res_linear = nn.Linear(input_size, num_heads * hidden_size)
        self.linear = nn.Linear(input_size, num_heads * hidden_size)
        self.weight_src = self.create_parameter(shape=[num_heads, hidden_size])
        self.weight_dst = self.create_parameter(shape=[num_heads, hidden_size])

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        if isinstance(activation, str):
            activation = getattr(F, activation)
        self.activation = activation

    def _send_attention(self, src_feat, dst_feat, edge_feat):
        alpha = src_feat["src"] + dst_feat["dst"]
        alpha = self.leaky_relu(alpha)
        return {"alpha": alpha, "h": src_feat["h"]}

    def _reduce_attention(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = paddle.reshape(alpha, [-1, self.num_heads, 1])
        if self.attn_drop > 1e-15:
            alpha = self.attn_dropout(alpha)

        feature = msg["h"]
        feature = paddle.reshape(feature,
                                 [-1, self.num_heads, self.hidden_size])
        feature = feature * alpha
        if self.concat:
            feature = paddle.reshape(feature,
                                     [-1, self.num_heads * self.hidden_size])
        else:
            feature = paddle.mean(feature, axis=1)

        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def forward(self, graph, feature, gather_scatter, shard_tool):
        """
         
        Args:
 
            graph: `pgl.Graph` instance.
            feature: A tensor with shape (num_nodes, input_size)
     
        Return:
            If `concat=True` then return a tensor with shape (num_nodes, hidden_size),
            else return a tensor with shape (num_nodes, hidden_size * num_heads) 
        """
        model_compute = 0
        nccl = 0
        start_time = time.time()
        if self.feat_drop > 1e-15:
            feature = self.feat_dropout(feature)

        res = self.res_linear(feature)
        feature = self.linear(feature)
        model_compute += (time.time() - start_time)
        start_time = time.time()
        feature = gather_scatter.apply(feature, shard_tool)
        nccl += (time.time() - start_time)
       
        start_time = time.time() 
        feature = paddle.reshape(feature,
                                 [-1, self.num_heads, self.hidden_size])

        attn_src = paddle.sum(feature * self.weight_src, axis=-1)
        attn_dst = paddle.sum(feature * self.weight_dst, axis=-1)

        new_start = time.time()
        msg = graph.send(
            self._send_attention,
            src_feat={"src": attn_src,
                      "h": feature},
            dst_feat={"dst": attn_dst})
        output = graph.recv(reduce_func=self._reduce_attention, msg=msg)
        print("GPU: %d, send_recv: %.4f" % (dist.get_rank(), time.time() - new_start))
        output = output[shard_tool.own_start_idx : shard_tool.own_end_idx]
        output = output + res

        if self.activation is not None:
            output = self.activation(output)
        model_compute += (time.time() - start_time)
       
        print("GPU: %d, model compute: %.4f" % (dist.get_rank(), model_compute))
        print("GPU: %d, nccl time: %.4f" % (dist.get_rank(), nccl))
        return output


class GCNConv(nn.Layer):
    """Implementation of graph convolutional neural networks (GCN)
    This is an implementation of the paper SEMI-SUPERVISED CLASSIFICATION
    WITH GRAPH CONVOLUTIONAL NETWORKS (https://arxiv.org/pdf/1609.02907.pdf).
    Args:
        input_size: The size of the inputs. 
        output_size: The size of outputs
        activation: The activation for the output.
        norm: If :code:`norm` is True, then the feature will be normalized.
    """

    def __init__(self, input_size, output_size, activation=None, norm=True):
        super(GCNConv, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size, bias_attr=False)
        self.bias = self.create_parameter(shape=[output_size], is_bias=True)
        self.norm = norm
        if isinstance(activation, str):
            activation = getattr(F, activation)
        self.activation = activation

    def forward(self, graph, feature, gather_scatter, shard_tool, norm=None):
        """
         
        Args:
 
            graph: `pgl.Graph` instance.
            feature: A tensor with shape (num_nodes, input_size)
            norm: (default None). If :code:`norm` is not None, then the feature will be normalized by given norm. If :code:`norm` is None and :code:`self.norm` is `true`, then we use `lapacian degree norm`.
     
        Return:
            A tensor with shape (num_nodes, output_size)
        """

        if self.norm and norm is None:
            norm = GF.degree_norm(graph)

        if self.input_size > self.output_size:
            feature = self.linear(feature)

        if norm is not None:
            feature = feature * norm

        feature = gather_scatter.apply(feature, shard_tool) 
        output = graph.send_recv(feature, "sum")
        output = output[shard_tool.own_start_idx : shard_tool.own_end_idx]

        if self.input_size <= self.output_size:
            output = self.linear(output)

        if norm is not None:
            output = output * norm

        output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output




class TransformerConv(nn.Layer):
    """Implementation of TransformerConv from UniMP
    This is an implementation of the paper Unified Message Passing Model for Semi-Supervised Classification
    (https://arxiv.org/abs/2009.03509).
    Args:
    
        input_size: The size of the inputs. 
 
        hidden_size: The hidden size for gat.
 
        activation: (default None) The activation for the output.
 
        num_heads: (default 4) The head number in transformerconv.
 
        feat_drop: (default 0.6) Dropout rate for feature.
 
        attn_drop: (default 0.6) Dropout rate for attention.
 
        concat: (default True) Whether to concat output heads or average them.
        skip_feat: (default True) Whether to add a skip conect from input to output.
        gate: (default False) Whether to use a gate function in skip conect.
        layer_norm: (default True) Whether to aply layer norm in output
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_heads=4,
                 feat_drop=0.6,
                 attn_drop=0.6,
                 concat=True,
                 skip_feat=True,
                 gate=False,
                 layer_norm=True,
                 activation='relu'):
        super(TransformerConv, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.concat = concat

        self.q = nn.Linear(input_size, num_heads * hidden_size)
        self.k = nn.Linear(input_size, num_heads * hidden_size)
        self.v = nn.Linear(input_size, num_heads * hidden_size)

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)

        if skip_feat:
            if concat:
                self.skip_feat = nn.Linear(input_size, num_heads * hidden_size)
            else:
                self.skip_feat = nn.Linear(input_size, hidden_size)
        else:
            self.skip_feat = None

        if gate:
            if concat:
                self.gate = nn.Linear(3 * num_heads * hidden_size, 1)
            else:
                self.gate = nn.Linear(3 * hidden_size, 1)
        else:
            self.gate = None

        if layer_norm:
            if self.concat:
                self.layer_norm = nn.LayerNorm(num_heads * hidden_size)
            else:
                self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = None

        if isinstance(activation, str):
            activation = getattr(F, activation)
        self.activation = activation

    def send_attention(self, src_feat, dst_feat, edge_feat):
        if "edge_feat" in edge_feat:
            alpha = dst_feat["q"] * (src_feat["k"] + edge_feat['edge_feat'])
            src_feat["v"] = src_feat["v"] + edge_feat["edge_feat"]
        else:
            alpha = dst_feat["q"] * src_feat["k"]
        alpha = paddle.sum(alpha, axis=-1)
        return {"alpha": alpha, "v": src_feat["v"]}

    def reduce_attention(self, msg):
        alpha = msg.reduce_softmax(msg["alpha"])
        alpha = paddle.reshape(alpha, [-1, self.num_heads, 1])
        if self.attn_drop > 1e-15:
            alpha = self.attn_dropout(alpha)

        feature = msg["v"]
        feature = feature * alpha
        if self.concat:
            feature = paddle.reshape(feature,
                                     [-1, self.num_heads * self.hidden_size])
        else:
            feature = paddle.mean(feature, axis=1)
        feature = msg.reduce(feature, pool_type="sum")
        return feature

    def send_recv(self, graph, q, k, v, edge_feat):
        if edge_feat is not None:
            msg = graph.send(
                self.send_attention,
                src_feat={'k': k,
                          'v': v},
                dst_feat={'q': q},
                edge_feat={'edge_feat': edge_feat})
        else:
            msg = graph.send(
                self.send_attention,
                src_feat={'k': k,
                          'v': v},
                dst_feat={'q': q})

        output = graph.recv(reduce_func=self.reduce_attention, msg=msg)
        return output

    def forward(self, graph, feature, gather_scatter, shard_tool, edge_feat=None):
        q = self.q(feature)
        q = q / (self.hidden_size**0.5)
        q = gather_scatter.apply(q, shard_tool)

        k = self.k(feature)
        k = gather_scatter.apply(k, shard_tool)

        v = self.v(feature)
        v = gather_scatter.apply(v, shard_tool)


        q = paddle.reshape(q, [-1, self.num_heads, self.hidden_size])
        k = paddle.reshape(k, [-1, self.num_heads, self.hidden_size])
        v = paddle.reshape(v, [-1, self.num_heads, self.hidden_size])

        output = self.send_recv(graph, q, k, v, edge_feat=edge_feat)
        output = output[shard_tool.own_start_idx : shard_tool.own_end_idx]

        if self.skip_feat is not None:
            skip_feat = self.skip_feat(feature)
            if self.gate is not None:
                gate = F.sigmoid(
                    self.gate(
                        paddle.concat(
                            [skip_feat, output, skip_feat - output], axis=-1)))
                output = gate * skip_feat + (1 - gate) * output
            else:
                output = skip_feat + output

        if self.layer_norm is not None:
            output = self.layer_norm(output)

        if self.activation is not None:
            output = self.activation(output)
        return output
