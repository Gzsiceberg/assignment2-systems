
from timeit import default_timer as timer
from typing import Callable
import torch


def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    # Time it for real now!
    total_time = 0.0
    for _ in range(num_trials):
        start = timer()
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
        end = timer()
        total_time += end - start
    mean_time = total_time / num_trials * 1000  # milliseconds
    print(f"{description}: {mean_time:.2f} milliseconds")
    return mean_time


if __name__ == "__main__":
    from cs336_basics import model
    from cs336_basics.nn_utils import cross_entropy
    vocab_size = 100_00
    batch_size = 4
    context_length = 200
    llm = model.BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=4 * 768,
        rope_theta=10_000,
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, context_length)).cuda()
    targets = torch.randint(0, vocab_size, (batch_size, context_length)).cuda()
    llm = llm.cuda()

    def run_foward():
        with torch.no_grad():
            logits = llm(input_ids)
            loss = cross_entropy(logits, targets)
    
    def run_forward_backward():
        logits = llm(input_ids)
        loss = cross_entropy(logits, targets)
        loss.backward()
        llm.zero_grad()
    
    benchmark("forward", run_foward, num_warmups=1, num_trials=3)
    benchmark("forward_backward", run_forward_backward, num_warmups=1, num_trials=3)