import paddle
from paddle.autograd import PyLayer


class GatherScatter(PyLayer):
    @staticmethod
    def forward(ctx, x, shard_tool):
        ctx.shard_tool = shard_tool
        y = shard_tool.forward_gather(x)
        return y
        
    @staticmethod
    def backward(ctx, dy):
        grad = ctx.shard_tool.backward_scatter(dy)
        return grad
