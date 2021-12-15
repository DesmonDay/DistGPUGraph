import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

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

        if self.feat_drop > 1e-15:
            feature = self.feat_dropout(feature)

        feature = self.linear(feature)
        feature = gather_scatter.apply(feature, shard_tool)
        feature = paddle.reshape(feature,
                                 [-1, self.num_heads, self.hidden_size])

        attn_src = paddle.sum(feature * self.weight_src, axis=-1)
        attn_dst = paddle.sum(feature * self.weight_dst, axis=-1)
        msg = graph.send(
            self._send_attention,
            src_feat={"src": attn_src,
                      "h": feature},
            dst_feat={"dst": attn_dst})
        output = graph.recv(reduce_func=self._reduce_attention, msg=msg)
        output = output[shard_tool.own_start_idx : shard_tool.own_end_idx]

        if self.activation is not None:
            output = self.activation(output)
        return output

