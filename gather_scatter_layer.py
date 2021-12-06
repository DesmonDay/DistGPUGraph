import paddle
from paddle.autograd import PyLayer


class GatherScatter(PyLayer):
    def __init__(self, shard_tool):
        super(GatherScatter, self).__init__()
        self.shard_tool = shard_tool

    @staticmethod
    def forward(ctx, x):
        y = self.shard_tool.forward_gather(x)
        return y
        
    @staticmethod
    def backward(ctx, dy):
        grad = self.shard_tool.backward_scatter(dy)
        return grad
