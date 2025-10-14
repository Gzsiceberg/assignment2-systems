from typing import Tuple
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
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        if handle is not None:
            self.handles.append(handle)
        end = tx()
        self.acc_all_reduce_time += end - start


class DDPBucketedParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.handles: list[Tuple[dist.Work, torch.Tensor, list[torch.Tensor], list]] = []
        self.params = list(self.module.parameters())
        self.device = self.params[0].device if self.params else torch.device("cpu")
        # Broadcast parameters from rank 0 to all other ranks
        with torch.no_grad():
            for param in self.params:
                dist.broadcast(param, src=0)
        self.world_size = dist.get_world_size()
        self.hooks = []
        self.acc_all_reduce_time = 0.0
        self.bucket_size_mb: float = bucket_size_mb
        self.bucketed_params: list[list[torch.nn.Parameter]] = []
        self.current_bucketed_size: int = 0
        self._register_hooks()

    def forward(self, *args, **kwargs):
        self.acc_all_reduce_time = 0.0
        return self.module(*args, **kwargs)
    
    def _handle_last_bucket(self):
        if self.current_bucketed_size == 0:
            return
        start = tx()
        last_bucket = self.bucketed_params[-1]
        grads = [param.grad for param in last_bucket if param.grad is not None]
        flat_grads = torch._utils._flatten_dense_tensors(grads) # type: ignore
        handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
        if handle is not None:
            self.handles.append((handle, flat_grads, grads, last_bucket))
        end = tx()
        self.acc_all_reduce_time += end - start
        self.bucketed_params.append([])

    def finish_gradient_synchronization(self):
        self._handle_last_bucket()
        start = tx()
        for handle, flat_grads, grads, last_bucket in self.handles:
            handle.wait()
            with torch.no_grad():
                flat_grads.mul_(1.0 / self.world_size)
                updated_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads) # type: ignore
                for param, updated_grad in zip(last_bucket, updated_grads):
                    if param.grad is not None:
                        param.grad = updated_grad
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
    
    def _add_to_bucket(self, param: torch.nn.Parameter) -> None:
        if not self.bucketed_params:
            self.bucketed_params.append([param])
        else:
            self.bucketed_params[-1].append(param)
        num_elements = param.numel()
        num_bytes = num_elements * param.element_size()
        self.current_bucketed_size += num_bytes

    def _all_reduce_hook(self, param: torch.nn.Parameter) -> None:
        start = tx()
        self._add_to_bucket(param)
        if self.current_bucketed_size < self.bucket_size_mb * 1024 * 1024:
            end = tx()
            self.acc_all_reduce_time += end - start
            return
        self._handle_last_bucket()
