import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Callable, Iterable, Type, Any, override


class OptimizerStateSharding(Optimizer):
    optimizer: Optimizer | None = None
    shared_params_groups: dict[int, list[torch.nn.Parameter]] = {}
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.current_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        for i in range(self.world_size):
            self.shared_params_groups[i] = []
        self.optimizer = optimizer_cls([{"params": []}], **kwargs)
        super().__init__(params, defaults=kwargs)

    @torch.no_grad()
    def step(self, closure: Callable | None = None, **kwargs): 
        if self.optimizer is not None:
            self.optimizer.step(closure, **kwargs)
        all_handles = []
        # naive broadcast to sync all parameters after each step
        for assigned_rank in range(self.world_size):
            params = self.shared_params_groups[assigned_rank]
            for p in params:
                h = dist.broadcast(p, src=assigned_rank, async_op=True)
                p.grad = None  # clear the gradient after broadcasting
                if h is not None:
                    all_handles.append(h)
        for h in all_handles:
            h.wait()
    

    @override   
    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)

        from rich import print
        params = param_group['params']
        total_nums = 0
        current_nums = 0
        # assign params to different ranks in a round-robin manner
        num_params_per_rank_dict = {i: 0 for i in range(self.world_size)}
        for i, p in enumerate(params):
            keys = list(num_params_per_rank_dict.keys())
            min_key = min(keys, key=lambda k: num_params_per_rank_dict[k])
            assigned_rank = min_key
            self.shared_params_groups[assigned_rank].append(p)
            num_params = p.numel()
            total_nums += num_params
            num_params_per_rank_dict[assigned_rank] += num_params
            if assigned_rank == self.current_rank:
                current_nums += num_params
        percent = current_nums / total_nums
        print(f"Rank={self.current_rank}: total_params={total_nums} current_params={current_nums} percent={percent:.2%}")

        if self.optimizer:
            current_rank_params = self.shared_params_groups[self.current_rank]
            param_group = {**param_group, 'params': current_rank_params}
            self.optimizer.add_param_group(param_group)