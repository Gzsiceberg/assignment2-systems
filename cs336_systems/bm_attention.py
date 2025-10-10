
from contextlib import nullcontext
import gc
from timeit import default_timer as timer
from typing import Callable
import torch
import numpy as np
import torch.cuda.nvtx as nvtx
from torch.profiler import ProfilerActivity
from cs336_basics.model import scaled_dot_product_attention
from cs336_basics.optimizer import AdamW
import cs336_basics.model
from cs336_systems.benchmarking import benchmark

complied_attention = torch.compile(scaled_dot_product_attention)

mem_total = 0
def do_attention(context_length: int, d_model: int):
    global mem_total
    batch_size = 8
    dtype = torch.float32
    q = torch.rand(batch_size, context_length, d_model, dtype=dtype).cuda() - 0.5
    k = torch.rand(batch_size, context_length, d_model, dtype=dtype).cuda() - 0.5
    v = torch.rand(batch_size, context_length, d_model, dtype=dtype).cuda() - 0.5
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    out = scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    mem_total += torch.cuda.memory_allocated() / (1024 ** 2)
    out.sum().backward()
    torch.cuda.synchronize()


def do_attention_complied(context_length: int, d_model: int):
    global mem_total
    batch_size = 8
    dtype = torch.float32
    q = torch.rand(batch_size, context_length, d_model, dtype=dtype).cuda() - 0.5
    k = torch.rand(batch_size, context_length, d_model, dtype=dtype).cuda() - 0.5
    v = torch.rand(batch_size, context_length, d_model, dtype=dtype).cuda() - 0.5
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    out = complied_attention(q, k, v)
    mem_total += torch.cuda.memory_allocated() / (1024 ** 2)
    out.sum().backward()
    torch.cuda.synchronize()


if __name__ == "__main__":
    all_d_models = [128]
    all_context_lengths = [4096]
    for d_model in all_d_models:
        for context_length in all_context_lengths:
            description = f"{d_model}, {context_length}"
            def run():
                do_attention_complied(context_length, d_model)
            benchmark(description, run, num_warmups=2, num_trials=5)
            print(f"{description}: {mem_total / (2 + 5):.2f} MB")

    