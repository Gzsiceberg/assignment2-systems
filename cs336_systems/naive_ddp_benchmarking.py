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
from cs336_systems.ddp import DDPBucketedParameters, DDPIndividualParameters


@dataclass
class DistributedConfig:
    world_size: int = 4
    backend: str = "gloo"
    flat_grad: bool = False  # new flag to enable flat gradient synchronization
    ddp: bool = False  # new flag to enable PyTorch DDP
    bucket_size: int = 1  # bucket size in MB for gradient synchronization (only for naive DDP)


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
    if config.backend == "nccl":
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda" if config.backend == "nccl" else "cpu")
    context_length = llm_config.context_length
    batch_size = llm_config.batch_size
    if rank == 0:
        print(f"Config: {config}")
        print(f"LLM Config: {llm_config}")
    llm, opt = setup_llm(llm_config)
    if config.backend == "nccl":
        llm.to(device)
    if config.ddp:
        if config.bucket_size > 0:
            llm = DDPBucketedParameters(llm, bucket_size_mb=config.bucket_size)
        else:
            llm = DDPIndividualParameters(llm)
    else:
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
    epochs = 100
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

        if config.ddp:
            if config.backend == "nccl":
                torch.cuda.synchronize()
            sync_time = llm.finish_gradient_synchronization() # type: ignore
            if epoch >= warmup_epochs:
                all_reduce_times.append(sync_time)
        else:
            # sync gradients
            with torch.no_grad():
                if config.backend == "nccl":
                    torch.cuda.synchronize()

                all_reduce_start = tx()
                sync_grads(config, llm)

                if config.backend == "nccl":
                    torch.cuda.synchronize()

                all_reduce_end = tx()
                if epoch >= warmup_epochs:
                    all_reduce_times.append(all_reduce_end - all_reduce_start)

        opt.step()
        opt.zero_grad()

        if rank == 0 and epoch % 20 == 0 and epoch > warmup_epochs:
            elapsed = tx() - start
            per_train_time = elapsed / (epoch + 1 - min(epoch + 1, warmup_epochs))
            avg_all_reduce_time = sum(all_reduce_times) / len(all_reduce_times)
            percent_all_reduce = 100.0 * (avg_all_reduce_time / per_train_time)
            debug_info = ""
            if config.ddp:
                debug_info = llm.get_debug_info() # type: ignore
            print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item():.5f}, Per-Train Time: {per_train_time:.4f}s Avg All-Reduce Time: {avg_all_reduce_time:.4f}s ({percent_all_reduce:.2f}%) {debug_info}")
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
    stats = torch.tensor([per_train_time, avg_all_reduce_time], device=device)
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
        params_with_grads = [p for p in llm.parameters() if p.grad is not None]
        if not params_with_grads:
            return
        grads = [param.grad for param in params_with_grads]
        flat_grads = torch._utils._flatten_dense_tensors(grads) # type: ignore
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads.mul_(1.0 / config.world_size)

        updated_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads) # type: ignore
        for p, updated_grad in zip(params_with_grads, updated_grads):
            p.grad = updated_grad
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
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--ddp', action='store_true', help='Use PyTorch DDP instead of naive implementation (for comparison)')
    parser.add_argument('--bucket_size', type=int, default=0, help='Bucket size in MB for gradient synchronization (only for naive DDP)')
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
    config.ddp = args.ddp
    config.bucket_size = args.bucket_size

    llm_config = LLMConfig()
    llm_config.batch_size = args.batch_size
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
