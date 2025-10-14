from dataclasses import dataclass
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rich import print
from timeit import default_timer as tx

@dataclass
class DistributedConfig:
    world_size: int = 4
    backend: str = "gloo"
    size: int = 1

def setup(rank, config: DistributedConfig):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(config.backend, rank=rank, world_size=config.world_size)

def dist_demo(rank, config: DistributedConfig):
    setup(rank, config)

    if config.backend == "nccl":
        torch.cuda.set_device(rank)
    device = torch.device(f'cuda' if config.backend == 'nccl' else 'cpu')
    
    data = torch.rand(config.size * 1000_000 // 4, dtype=torch.float32, device=device)
    warmup_iters = 1 if config.backend == "gloo" else 5
    for _ in range(warmup_iters):
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)  # Use SUM for demonstration
        if config.backend == "nccl":
            torch.cuda.synchronize()
    
    iters = 5
    start = tx()
    for _ in range(iters):
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
        if config.backend == "nccl":
            torch.cuda.synchronize()
    end = tx()
    avg_time = (end - start) * 1000.0 / iters
    # print(f"Rank {rank}: Average time per iteration: {avg_time:.6f} ms")

    all_times = [torch.zeros(1, device=device) for _ in range(config.world_size)]
    local_time = torch.tensor([avg_time], device=device)
    dist.all_gather(all_times, local_time)

    if rank == 0:
        all_times = torch.cat(all_times)
        print(f"world_size={config.world_size} backend={config.backend} size={config.size}MB mean={all_times.mean():.6f}ms")
    
    dist.barrier()
    dist.destroy_process_group()

def run_sweep():
    gpu_count = torch.cuda.device_count()
    for backend in ["gloo", "nccl"]:
        if backend == "nccl" and gpu_count < 2:
            print("Skipping NCCL benchmark, requires at least 2 GPUs")
            continue
        for world_size in [2, 4, 6]:
            if backend == "nccl" and world_size > gpu_count:
                print(f"Skipping NCCL benchmark for world_size={world_size}, requires {world_size} GPUs")
                continue
            for size in [1, 10, 100, 1000]:
                config = DistributedConfig(world_size=world_size, backend=backend, size=size)
                mp.spawn(dist_demo, args=(config,), nprocs=world_size, join=True)
    

if __name__ == "__main__":
    run_sweep()
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--world_size", type=int, default=4, help="Number of processes to spawn")
    # parser.add_argument("--backend", type=str, default="gloo", help="Distributed backend")
    # parser.add_argument("--size", type=int, default=1, help="Size of the tensor to reduce")
    # args = parser.parse_args()
    # world_size = args.world_size
    # config = DistributedConfig(world_size=world_size, backend=args.backend, size=args.size)
    # mp.spawn(dist_demo, args=(config,), nprocs=world_size, join=True)
