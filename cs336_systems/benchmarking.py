
from timeit import default_timer as timer
from typing import Callable
import torch
import numpy as np


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

def benchmark_llm(description: str, num_layers: int = 12, d_model: int = 768, num_heads: int = 12, d_ff: int = 4 * 768):
    from cs336_basics import model
    from cs336_basics.nn_utils import cross_entropy
    warmup_runs = args.warmup
    num_trials = args.trials
    vocab_size: int = 100_00
    batch_size: int = 4
    context_length: int = 128
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

    input_ids = torch.randint(0, vocab_size, (batch_size, context_length)).cuda()
    targets = torch.randint(0, vocab_size, (batch_size, context_length)).cuda()
    llm = llm.cuda()
    total_params = sum(p.numel() for p in llm.parameters())
    print("-" * 80)
    print(f"Benchmarking {description} model with {total_params/1e6:.1f}M parameters")

    def run_foward():
        logits = llm(input_ids)
        loss = cross_entropy(logits, targets)
    
    def run_forward_backward():
        logits = llm(input_ids)
        loss = cross_entropy(logits, targets)
        loss.backward()

    mean, std = benchmark(f"{description}_forward", run_foward, num_warmups=warmup_runs, num_trials=num_trials)
    mean2, std2 = benchmark(f"{description}_forward_backward", run_forward_backward, num_warmups=warmup_runs, num_trials=num_trials)
    return (mean, std), (mean2, std2)


if __name__ == "__main__":
    import pandas as pd
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("-w", "--warmup", type=int, default=5, help="Number of warmup runs")
    parser.add_argument("-t", "--trials", type=int, default=10, help="Number of benchmark trials")
    args = parser.parse_args()

    all_results = {}
    all_results["type"] = []
    all_results["mean_forward"] = []
    all_results["std_forward"] = []
    all_results["mean_forward_backward"] = []
    all_results["std_forward_backward"] = []

    (mean, std), (mean2, std2) = benchmark_llm("small", d_model=768, d_ff=4 * 768, num_layers=12, num_heads=12)
    all_results["type"].append("small")
    all_results["mean_forward"].append(mean)
    all_results["std_forward"].append(std)
    all_results["mean_forward_backward"].append(mean2)
    all_results["std_forward_backward"].append(std2)

    (mean, std), (mean2, std2) = benchmark_llm("medium", d_model=1024, d_ff=4 * 1024, num_layers=24, num_heads=16)
    all_results["type"].append("medium")
    all_results["mean_forward"].append(mean)
    all_results["std_forward"].append(std)
    all_results["mean_forward_backward"].append(mean2)
    all_results["std_forward_backward"].append(std2)

    (mean, std), (mean2, std2) = benchmark_llm("large", d_model=1280, d_ff=4 * 1280, num_layers=36, num_heads=20)
    all_results["type"].append("large")
    all_results["mean_forward"].append(mean)
    all_results["std_forward"].append(std)
    all_results["mean_forward_backward"].append(mean2)
    all_results["std_forward_backward"].append(std2)

    if args.all:
        (mean, std), (mean2, std2) = benchmark_llm("xlarge", d_model=1600, d_ff=4 * 1600, num_layers=48, num_heads=25)
        all_results["type"].append("xlarge")
        all_results["mean_forward"].append(mean)
        all_results["std_forward"].append(std)
        all_results["mean_forward_backward"].append(mean2)
        all_results["std_forward_backward"].append(std2)

        (mean, std), (mean2, std2) = benchmark_llm("2.7B", d_model=2560, d_ff=10240, num_layers=32, num_heads=32)
        all_results["type"].append("2.7B")
        all_results["mean_forward"].append(mean)
        all_results["std_forward"].append(std)
        all_results["mean_forward_backward"].append(mean2)
        all_results["std_forward_backward"].append(std2)

    df = pd.DataFrame(all_results)
    df.to_markdown("benchmark_results.md", index=False)
