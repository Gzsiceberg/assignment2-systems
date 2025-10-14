import torch
import torch.distributed as dist
from timeit import default_timer as tx

class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles: list[dist.Work] = []
        self.params = list(self.module.parameters())
        self.device = self.params[0].device if self.params else torch.device("cpu")
        # Broadcast parameters from rank 0 to all other ranks
        with torch.no_grad():
            for param in self.params:
                dist.broadcast(param, src=0)
        self.world_size = dist.get_world_size()
        self.hooks = []
        self.acc_all_reduce_time = 0.0
        self._register_hooks()

    def forward(self, *args, **kwargs):
        self.acc_all_reduce_time = 0.0
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        start = tx()
        for handle in self.handles:
            handle.wait()
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    param.grad.mul_(1.0 / self.world_size)
        self.handles = []
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        end = tx()
        old_time = self.acc_all_reduce_time
        self.acc_all_reduce_time = 0.0
        return end - start + old_time


    def _register_hooks(self):
        for param in self.params:
            if not param.requires_grad:
                continue
            h = param.register_post_accumulate_grad_hook(self._all_reduce_hook)
            self.hooks.append(h)

    def _all_reduce_hook(self, param) -> None:
        start = tx()
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        if handle is not None:
            self.handles.append(handle)
        end = tx()
        self.acc_all_reduce_time += end - start
