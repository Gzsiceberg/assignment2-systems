from contextlib import nullcontext
from dataclasses import dataclass
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rich import print
from timeit import default_timer as tx

from cs336_basics import model
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


@dataclass
class DistributedConfig:
    world_size: int = 4
    backend: str = "gloo"
    flat_grad: bool = False  # new flag to enable flat gradient synchronization


@dataclass
class LLMConfig:
    vocab_size: int = 100_00
    batch_size: int = 4
    context_length: int = 128
    rope_theta: int = 10_000
    d_model: int = 1600
    num_layers: int = 48
    num_heads: int = 25
    d_ff: int = 4 * 1600
    lr: float = 1e-6
    autocast: bool = True


def setup(rank, config: DistributedConfig):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(config.backend, rank=rank, world_size=config.world_size)


def setup_llm(config: LLMConfig):
    vocab_size: int = config.vocab_size
    context_length: int = config.context_length
    rope_theta: int = config.rope_theta
    d_model = config.d_model
    num_layers = config.num_layers
    num_heads = config.num_heads
    llm = model.BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=4 * d_model,
        rope_theta=rope_theta,
    )

    opt = AdamW(llm.parameters(), lr=config.lr)

    return llm, opt


def dist_main(rank, config: DistributedConfig, llm_config: LLMConfig):
    setup(rank, config)
    torch.manual_seed(0 + rank)
    device = torch.device(f"cuda:{rank}" if config.backend == "nccl" else "cpu")
    context_length = llm_config.context_length
    batch_size = llm_config.batch_size
    llm, opt = setup_llm(llm_config)
    # sync the model initialization across ranks
    with torch.no_grad():
        for param in llm.parameters():
            dist.broadcast(param, src=0)
    dist.barrier()

    input_ids = torch.randint(
        0, llm_config.vocab_size, (batch_size, context_length), device=device
    )
    targets = torch.randint(
        0, llm_config.vocab_size, (batch_size, context_length), device=device
    )
    autocast = llm_config.autocast

    start = tx()
    epochs = 200
    all_reduce_times = []
    warmup_epochs = 10
    for epoch in range(epochs):
        with (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if autocast
            else nullcontext()
        ):
            logits = llm(input_ids)
        loss = cross_entropy(logits, targets)
        loss.backward()

        # sync gradients
        with torch.no_grad():
            all_reduce_start = tx()
            sync_grads(config, llm)
            if config.backend == "nccl":
                torch.cuda.synchronize()
            all_reduce_end = tx()
            if epoch >= warmup_epochs:
                all_reduce_times.append(all_reduce_end - all_reduce_start)

        opt.step()
        opt.zero_grad()

        if rank == 0 and epoch % 20 == 0:
            print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item():.5f}")
        if epoch == warmup_epochs - 1:
            if config.backend == "nccl":
                torch.cuda.synchronize()
            start = tx()  # reset the timer after warmup
    if config.backend == "nccl":
        torch.cuda.synchronize()
    end = tx()
    elapsed = end - start
    per_train_time = elapsed / (epochs - warmup_epochs)
    avg_all_reduce_time = sum(all_reduce_times) / len(all_reduce_times)
    stats = torch.tensor([per_train_time, avg_all_reduce_time])
    dist.reduce(stats, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"Average training time per epoch: {stats[0].item() / config.world_size:.4f} seconds")
        print(f"Average all-reduce time per epoch: {stats[1].item() / config.world_size:.4f} seconds")
        percent_all_reduce = 100.0 * (stats[1].item() / stats[0].item())
        print(f"Percentage of time spent in all-reduce: {percent_all_reduce:.2f}%")

    dist.barrier()
    dist.destroy_process_group()

def sync_grads(config, llm):
    if config.flat_grad:
        # Flatten all gradients into a single tensor
        grads = [param.grad.view(-1) for param in llm.parameters() if param.grad is not None]
        flat_grad = torch.cat(grads)
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
        flat_grad.mul_(1.0 / config.world_size)

        # Unflatten the gradients back to their original shapes
        pointer = 0
        for param in llm.parameters():
            if param.grad is not None:
                numel = param.grad.numel()
                param.grad.copy_(flat_grad[pointer:pointer + numel].view_as(param.grad))
                pointer += numel
    else:
        for param in llm.parameters():
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
            param.grad.mul_(1.0 / config.world_size)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='gloo', choices=['gloo', 'nccl'], help='Distributed backend to use')
    parser.add_argument('--flat_grad', action='store_true', help='Enable flat gradient synchronization')
    args = parser.parse_args()

    if args.backend == "nccl":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, cannot use NCCL backend")
        gpus = torch.cuda.device_count()
        if gpus < 2:
            raise ValueError("NCCL backend requires at least 2 GPUs")

    config = DistributedConfig()
    config.world_size = 2
    config.backend = args.backend
    config.flat_grad = args.flat_grad

    llm_config = LLMConfig()
    if config.backend == "gloo":
        llm_config.batch_size = 1
        llm_config.d_model = 8
        llm_config.num_layers = 2
        llm_config.num_heads = 1
        llm_config.context_length = 16
        llm_config.vocab_size = 32
        llm_config.lr = 1e-3
        llm_config.autocast = False
    mp.spawn(dist_main, args=(config, llm_config), nprocs=config.world_size)
