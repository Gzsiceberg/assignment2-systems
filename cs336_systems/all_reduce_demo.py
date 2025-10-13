import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rich import print


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def dist_demo(rank, world_size):
    setup(rank, world_size)
    
    data = torch.randint(0, 10, (3,))
    print(f"Rank {rank} data before all_reduce: {data.tolist()}")
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} data after all_reduce: {data.tolist()}")

if __name__ == "__main__":
    world_size = 4
    mp.spawn(dist_demo, args=(world_size,), nprocs=world_size, join=True)
