import triton
import triton.language as tl
import torch
from torch import Tensor
from cs336_basics.nn_utils import softmax
from benchmarking import benchmark, profile
from rich import print


@triton.jit
def triton_softmax_kernel(
    x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr
):
    assert num_cols <= BLOCK_SIZE, "num_cols must be less than or equal to BLOCK_SIZE"

    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    mask = col_offsets < num_cols
    x_row = tl.load(x_ptrs, mask=mask, other=float("-inf"))

    x_row = x_row - tl.max(x_row, axis=0)
    x_row_exp = tl.exp(x_row)
    denom = tl.sum(x_row_exp, axis=0)
    y_row = x_row_exp / denom

    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=mask)


def triton_softmax(x: Tensor) -> Tensor:
    assert x.dim() == 2, "Input must be a 2D tensor"
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert x.dtype in [
        torch.float16,
        torch.float32,
    ], "Input must be of type float16 or float32"

    num_rows, num_cols = x.shape
    y = torch.empty_like(x)
    block_size = triton.next_power_of_2(num_cols)
    # print(f"Using block size: {block_size}")

    triton_softmax_kernel[(num_rows,)](
        x_ptr=x,
        y_ptr=y,
        x_row_stride=x.stride(0),
        y_row_stride=y.stride(0),
        num_cols=num_cols,
        BLOCK_SIZE=block_size,
    )

    return y


@torch.compile()
def compiled_softmax(x: Tensor) -> Tensor:
    return softmax(x, dim=-1)

def softmax_demo():
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).cuda()
    y1 = softmax(x, dim=-1)
    y2 = torch.softmax(x, dim=-1)
    y3 = triton_softmax(x)
    y4 = compiled_softmax(x)

    assert torch.allclose(y1, y2, atol=1e-6), f"Results do not match: {y1} vs {y2}"
    assert torch.allclose(y1, y3, atol=1e-6), f"Results do not match: {y1} vs {y3}"
    assert torch.allclose(y1, y4, atol=1e-6), f"Results do not match: {y1} vs {y4}"

    dim = 10240
    x = torch.randn(dim, dim, device="cuda", dtype=torch.float32)
    benchmark(
        "PyTorch Softmax",
        lambda: torch.softmax(x, dim=-1),
        num_warmups=5,
        num_trials=20,
    )
    benchmark("My Softmax", lambda: softmax(x, dim=-1), num_warmups=5, num_trials=20)
    benchmark("Triton Softmax", lambda: triton_softmax(x), num_warmups=5, num_trials=20)
    benchmark(
        "Compiled Softmax",
        lambda: compiled_softmax(x),
        num_warmups=5,
        num_trials=20,
    )
    # table = profile("Triton Softmax Profile", lambda: triton_softmax(x))
    # print(table)
    
    # table = profile(
    #     "PyTorch Softmax Profile", lambda: torch.softmax(x, dim=-1)
    # )
    # print(table)


@triton.jit
def vec_add_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    mask = offsets < n_elements
    x_ptrs = x_ptr + offsets
    y_ptrs = y_ptr + offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = tl.load(y_ptrs, mask=mask, other=0.0)
    z = x + y
    z_ptrs = z_ptr + offsets
    tl.store(z_ptrs, z, mask=mask)

def vec_add(x: Tensor, y: Tensor) -> Tensor:
    z = torch.empty_like(x)
    n_elements = x.numel()
    block_size = 1024
    num_blocks = triton.cdiv(n_elements, block_size)

    vec_add_kernel[(num_blocks,)](
        x_ptr=x,
        y_ptr=y,
        z_ptr=z,
        n_elements=n_elements,
        BLOCK_SIZE=block_size,
    )
    return z
    

def vec_add_demo():
    dim = 1024 * 10000
    x = torch.randn(dim, device="cuda")
    y = torch.randn(dim, device="cuda")

    z = x + y  # PyTorch addition
    z_triton = vec_add(x, y)  # Triton addition
    assert torch.allclose(z, z_triton), "Results do not match"

    mean_ms = triton.testing.do_bench(lambda: vec_add(x, y), quantiles=[0.5])
    mean_ms2 = triton.testing.do_bench(lambda: x + y, quantiles=[0.5])
    print(f"PyTorch Vec Add (triton.testing): {mean_ms2:.2f}ms")
    print(f"Triton Vec Add (triton.testing): {mean_ms:.2f}ms")

    print("Using benchmarking.py:")
    benchmark(
        "PyTorch Vec Add",
        lambda: x + y,
        num_warmups=5,
        num_trials=20,
    )
    benchmark(
        "Triton Vec Add",
        lambda: vec_add(x, y),
        num_warmups=5,
        num_trials=20,
    )


if __name__ == "__main__":
    # softmax_demo()
    vec_add_demo()