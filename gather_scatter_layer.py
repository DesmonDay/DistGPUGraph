import paddle
from paddle.autograd import PyLayer
import paddle.distributed as dist


class GatherScatter(PyLayer):
    @staticmethod
    def forward(ctx, x, shard_tool):
        ctx.shard_tool = shard_tool
        try:
            y = shard_tool.forward_gather(x)
        except Exception as e:
            print("Hung", dist.get_rank())
            raise e
        return y
        
    @staticmethod
    def backward(ctx, dy):
        grad = ctx.shard_tool.backward_scatter(dy)
        return grad

