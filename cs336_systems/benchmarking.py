
import gc
from timeit import default_timer as timer
from typing import Callable
import torch
import numpy as np
import torch.cuda.nvtx as nvtx

from cs336_basics.optimizer import AdamW


def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3) -> tuple[float, float]:
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    # Time it for real now!
    total_time = 0.0
    all_time: list[float] = []
    for _ in range(num_trials):
        start = timer()
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
        end = timer()
        all_time.append(end - start)
    mean_time: float = float(np.mean(all_time) * 1000)
    std_time: float = float(np.std(all_time) * 1000)
    print(f"{description}: mean={mean_time:.2f}ms std={std_time:.2f}ms")
    return mean_time, std_time

def benchmark_llm(description: str, all_result: dict, num_layers: int = 12, d_model: int = 768, num_heads: int = 12, d_ff: int = 4 * 768):
    from cs336_basics import model
    from cs336_basics.nn_utils import cross_entropy
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()

    warmup_runs = args.warmup
    num_trials = args.trials
    has_opt = args.opt
    has_back = args.back
    vocab_size: int = 100_00
    batch_size: int = 4
    context_length: int = args.context_length
    rope_theta: int = 10_000
    llm = model.BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )

    opt = AdamW(llm.parameters(), lr=1e-6)

    input_ids = torch.randint(0, vocab_size, (batch_size, context_length)).cuda()
    targets = torch.randint(0, vocab_size, (batch_size, context_length)).cuda()
    llm = llm.cuda()
    total_params = sum(p.numel() for p in llm.parameters())
    print("-" * 80)
    print(f"Benchmarking {description} model with {total_params/1e6:.1f}M parameters context_length={context_length}")

    for _ in range(warmup_runs):
        logits = llm(input_ids)
        loss = cross_entropy(logits, targets)
        if has_back:
            loss.backward()
        if has_opt:
            opt.step()
            opt.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    forward_times: list[float] = []
    backward_times: list[float] = []
    opt_step_times: list[float] = []
    for _ in range(num_trials):
        start = timer()
        with nvtx.range(f"{description}_forward"):
            logits = llm(input_ids)
            loss = cross_entropy(logits, targets)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  
        end = timer()
        forward_times.append(end - start)

        if has_back:
            start = timer()
            with nvtx.range(f"{description}_backward"):
                loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()  
            end = timer()
            backward_times.append(end - start)

        if has_opt:
            start = timer()
            with nvtx.range(f"{description}_opt_step"):
                opt.step()
                opt.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.synchronize()  
            end = timer()
            opt_step_times.append(end - start)

    all_result["type"].append(description)
    mean, std = np.mean(forward_times) * 1000, np.std(forward_times) * 1000
    print(f"{description} forward: mean={mean:.2f}ms std={std:.2f}ms")
    all_result["mean_forward"].append(mean)
    all_result["std_forward"].append(std)

    if has_back:
        mean2, std2 = np.mean(backward_times) * 1000, np.std(backward_times) * 1000
        print(f"{description} backward: mean={mean2:.2f}ms std={std2:.2f}ms")
        all_result["mean_backward"].append(mean2)
        all_result["std_backward"].append(std2)

    if has_opt:
        mean3, std3 = np.mean(opt_step_times) * 1000, np.std(opt_step_times) * 1000
        print(f"{description} opt_step: mean={mean3:.2f}ms std={std3:.2f}ms")
        all_result["mean_opt_step"].append(mean3)
        all_result["std_opt_step"].append(std3)


if __name__ == "__main__":
    import pandas as pd
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--all", type=int, default=3, help="1: small, 2: medium, 3: large, 4: xlarge, 5: 2.7B")
    parser.add_argument("-w", "--warmup", type=int, default=5, help="Number of warmup runs")
    parser.add_argument("-t", "--trials", type=int, default=10, help="Number of benchmark trials")
    parser.add_argument("--opt", action="store_true", help="Include optimizer step in benchmark")
    parser.add_argument("--back", action="store_true", help="Include optimizer step in benchmark")
    parser.add_argument("--context-length", type=int, default=128, help="Context length")
    parser.add_argument("-o", "--output", type=str, default="benchmark_results.md", help="Output markdown file")
    args = parser.parse_args()

    all_results = {}
    all_results["type"] = []
    all_results["mean_forward"] = []
    all_results["std_forward"] = []
    if args.back:
        all_results["mean_backward"] = []
        all_results["std_backward"] = []
    if args.opt:
        all_results["mean_opt_step"] = []
        all_results["std_opt_step"] = []

    if args.all >= 1:
        benchmark_llm("small", all_results, d_model=768, d_ff=4 * 768, num_layers=12, num_heads=12)
    if args.all >= 2:
        benchmark_llm("medium", all_results, d_model=1024, d_ff=4 * 1024, num_layers=24, num_heads=16)
    if args.all >= 3:
        benchmark_llm("large", all_results, d_model=1280, d_ff=4 * 1280, num_layers=36, num_heads=20)
    if args.all >= 4:
        benchmark_llm("xlarge", all_results, d_model=1600, d_ff=4 * 1600, num_layers=48, num_heads=25)
    if args.all >= 5:
        benchmark_llm("2.7B", all_results, d_model=2560, d_ff=10240, num_layers=32, num_heads=32)

    df = pd.DataFrame(all_results)
    df.to_markdown(args.output, index=False)
