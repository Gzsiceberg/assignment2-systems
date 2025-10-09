from typing import Tuple
import triton
import triton.language as tl
import torch
from torch import Tensor
from cs336_basics.nn_utils import softmax
from benchmarking import benchmark, profile
from rich import print
import numpy as np


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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**i for i in range(12, 28, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch"],  # Possible values for `line_arg`.
        line_names=["Triton", "Torch"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="vector-add-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark_demo(size, provider):
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vec_add(x, y), quantiles=quantiles
        )
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    return [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        # Good config for fp8 inputs.
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
    ]


def get_autotune_config():
    return get_cuda_autotune_config()


@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    a_row_stride,
    a_col_stride,
    b_row_stride,
    b_col_stride,
    c_row_stride,
    c_col_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m_pid = tl.program_id(0)
    n_pid = tl.program_id(1)

    matmul_kernel_internal(
        A_ptr,
        B_ptr,
        C_ptr,
        M,
        N,
        K,
        a_row_stride,
        a_col_stride,
        b_row_stride,
        b_col_stride,
        c_row_stride,
        c_col_stride,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        m_pid,
        n_pid,
    )


@triton.jit
def matmul_kernel_internal(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    a_row_stride,
    a_col_stride,
    b_row_stride,
    b_col_stride,
    c_row_stride,
    c_col_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    m_pid,
    n_pid,
):
    c_mat = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    block_m_offsets = tl.arange(0, BLOCK_M)
    block_n_offsets = tl.arange(0, BLOCK_N)
    block_k_offsets = tl.arange(0, BLOCK_K)
    a_row_ids = (m_pid * BLOCK_M + block_m_offsets) % M
    b_row_ids = (n_pid * BLOCK_N + block_n_offsets) % N
    a_ptrs = (
        A_ptr
        + a_row_ids[:, None] * a_row_stride
        + block_k_offsets[None, :] * a_col_stride
    )
    b_ptrs = (
        B_ptr
        + b_row_ids[:, None] * b_row_stride
        + block_k_offsets[None, :] * b_col_stride
    )
    for k in range(0, K, BLOCK_K):
        col_ids = k + block_k_offsets
        col_mask = col_ids[None, :] < K

        a_mat = tl.load(a_ptrs, mask=col_mask, other=0.0)
        b_mat = tl.load(b_ptrs, mask=col_mask, other=0.0)

        c_mat += tl.dot(a_mat, tl.trans(b_mat))
        a_ptrs += BLOCK_K * a_col_stride
        b_ptrs += BLOCK_K * b_col_stride

    c_row_ids = m_pid * BLOCK_M + block_m_offsets
    c_col_ids = n_pid * BLOCK_N + block_n_offsets
    c_row_offsets = c_row_ids * c_row_stride
    c_col_offsets = c_col_ids * c_col_stride
    offset = c_row_offsets[:, None] + c_col_offsets[None, :]
    c_mask = (c_row_ids[:, None] < M) & (c_col_ids[None, :] < N)
    c_ptrs = C_ptr + offset
    tl.store(c_ptrs, c_mat, mask=c_mask)


@triton.jit
def matmul_kernel_l2_cache(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    a_row_stride,
    a_col_stride,
    b_row_stride,
    b_col_stride,
    c_row_stride,
    c_col_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    group_pid = tl.program_id(0)
    n_pid = tl.program_id(1)
    group_row_pid = tl.program_id(2)
    m_pid = group_pid * GROUP_M + group_row_pid

    matmul_kernel_internal(
        A_ptr,
        B_ptr,
        C_ptr,
        M,
        N,
        K,
        a_row_stride,
        a_col_stride,
        b_row_stride,
        b_col_stride,
        c_row_stride,
        c_col_stride,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        m_pid,
        n_pid,
    )


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel_l2_cache_v2(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    a_row_stride,
    a_col_stride,
    b_row_stride,
    b_col_stride,
    c_row_stride,
    c_col_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    # Number of program ids along the M axis
    num_pid_m = tl.cdiv(M, BLOCK_M)
    # Number of programs ids along the N axis
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Number of programs in group
    num_pid_in_group = GROUP_M * num_pid_n
    # Id of the group this program is in
    group_id = pid // num_pid_in_group
    # Row-id of the first program in the group
    first_pid_m = group_id * GROUP_M
    # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    # *Within groups*, programs are ordered in a column-major order
    in_group_id = pid % num_pid_in_group
    # Row-id of the program in the *launch grid*
    m_pid = first_pid_m + ((in_group_id) % group_size_m)
    # Col-id of the program in the *launch grid*
    n_pid = (in_group_id) // group_size_m

    matmul_kernel_internal(
        A_ptr,
        B_ptr,
        C_ptr,
        M,
        N,
        K,
        a_row_stride,
        a_col_stride,
        b_row_stride,
        b_col_stride,
        c_row_stride,
        c_col_stride,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        m_pid,
        n_pid,
    )


def triton_matmul(A: Tensor, B: Tensor) -> Tensor:
    C = torch.empty((A.shape[0], B.shape[0]), device=A.device, dtype=A.dtype)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    assert A.shape[1] == B.shape[1], "Incompatible matrix shapes"

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        A_ptr=A,
        B_ptr=B,
        C_ptr=C,
        M=M,
        N=N,
        K=K,
        a_row_stride=A.stride(0),
        a_col_stride=A.stride(1),
        b_row_stride=B.stride(0),
        b_col_stride=B.stride(1),
        c_row_stride=C.stride(0),
        c_col_stride=C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


def triton_matmul_l2_cache(A: Tensor, B: Tensor) -> Tensor:
    C = torch.empty((A.shape[0], B.shape[0]), device=A.device, dtype=A.dtype)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    GROUP_M = 4

    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    assert A.shape[1] == B.shape[1], "Incompatible matrix shapes"

    group_num = triton.cdiv(M, BLOCK_M * GROUP_M)
    col_blocks = triton.cdiv(N, BLOCK_N)

    grid = (group_num, col_blocks, GROUP_M)

    matmul_kernel_l2_cache[grid](
        A_ptr=A,
        B_ptr=B,
        C_ptr=C,
        M=M,
        N=N,
        K=K,
        a_row_stride=A.stride(0),
        a_col_stride=A.stride(1),
        b_row_stride=B.stride(0),
        b_col_stride=B.stride(1),
        c_row_stride=C.stride(0),
        c_col_stride=C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
    )
    return C


def triton_matmul_l2_cache_v2(A: Tensor, B: Tensor) -> Tensor:
    C = torch.empty((A.shape[0], B.shape[0]), device=A.device, dtype=A.dtype)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    assert A.shape[1] == B.shape[1], "Incompatible matrix shapes"

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    matmul_kernel_l2_cache_v2[grid](
        A_ptr=A,
        B_ptr=B,
        C_ptr=C,
        M=M,
        N=N,
        K=K,
        a_row_stride=A.stride(0),
        a_col_stride=A.stride(1),
        b_row_stride=B.stride(0),
        b_col_stride=B.stride(1),
        c_row_stride=C.stride(0),
        c_col_stride=C.stride(1),
    )
    return C


def matmul_demo():
    dim = 512 * 10
    M, K, N = dim, dim, dim
    A = torch.rand((M, K), device="cuda", dtype=torch.float32) - 0.5
    B = torch.rand((N, K), device="cuda", dtype=torch.float32) - 0.5

    C1 = A @ B.T
    C2 = triton_matmul_l2_cache(A, B)

    assert torch.allclose(
        C1, C2, rtol=0, atol=1e-2 * np.sqrt(K / 512)
    ), f"Results do not match, {C1} vs {C2}"

    benchmark(
        "PyTorch MatMul",
        lambda: torch.matmul(A, B.T),
        num_warmups=5,
        num_trials=20,
    )
    benchmark(
        "Triton MatMul with L2 Cache",
        lambda: triton_matmul_l2_cache(A, B),
        num_warmups=5,
        num_trials=20,
    )
    benchmark(
        "Triton MatMul with L2 Cache v2",
        lambda: triton_matmul_l2_cache_v2(A, B),
        num_warmups=5,
        num_trials=20,
    )
    benchmark(
        "Triton MatMul",
        lambda: triton_matmul(A, B),
        num_warmups=5,
        num_trials=20,
    )
    # ms01 = triton.testing.do_bench(lambda: torch.matmul(A, B.T), quantiles=[0.5])
    # print(f"PyTorch MatMul (triton.testing): {ms01:.2f}ms")
    # ms02 = triton.testing.do_bench(lambda: triton_matmul(A, B), quantiles=[0.5])
    # print(f"Triton MatMul (triton.testing): {ms02:.2f}ms")

    compiled_kernel = list(matmul_kernel_l2_cache.cache[0].values())[0]
    # print all attributes of compiled_kernel that do not start with _
    attrs = [attr for attr in dir(compiled_kernel) if not attr.startswith("_")]
    print(compiled_kernel.metadata)
    print("Compiled kernel attributes:")
    for attr in attrs:
        print(f" - {attr}")


@triton.jit
def layer_norm_fwd_kernel(
    x_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    W_ptr,
    b_ptr,
    row_stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    x_start_ptr = x_ptr + pid * row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # compute mean
    mean = 0.0
    for k in range(0, N, BLOCK_SIZE):
        col_ids = k + col_offsets
        mask = col_ids < N
        x_ptrs = x_start_ptr + col_ids
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0) / N

    # compute variance
    var = 0.0
    for k in range(0, N, BLOCK_SIZE):
        col_ids = k + col_offsets
        mask = col_ids < N
        x_ptrs = x_start_ptr + col_ids
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        x_minus_mean = tl.where(mask, x - mean, 0.0)
        var += tl.sum(x_minus_mean * x_minus_mean, axis=0) / N

    mean_ptr = mean_ptr + pid
    tl.store(mean_ptr, mean)
    rstd_ptr = rstd_ptr + pid
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(rstd_ptr, rstd)

    # compute y
    y_start_ptr = y_ptr + pid * row_stride
    for k in range(0, N, BLOCK_SIZE):
        col_ids = k + col_offsets
        mask = col_ids < N
        x_ptrs = x_start_ptr + col_ids
        W_ptrs = W_ptr + col_ids
        b_ptrs = b_ptr + col_ids
        W = tl.load(W_ptrs, mask=mask, other=1.0)
        b = tl.load(b_ptrs, mask=mask, other=0.0)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        y = y * W + b

        y_ptrs = y_start_ptr + col_ids
        tl.store(y_ptrs, y, mask=mask)


def layer_norm_fwd(x: Tensor, W: Tensor, b: Tensor, eps=1e-5):
    assert x.dim() == 2, "Input must be a 2D tensor"
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert x.dtype in [
        torch.float16,
        torch.float32,
    ], "Input must be of type float16 or float32"
    assert (
        W.dim() == 1 and W.shape[0] == x.shape[1]
    ), "W must be a 1D tensor with shape (N,)"
    assert (
        b.dim() == 1 and b.shape[0] == x.shape[1]
    ), "b must be a 1D tensor with shape (N,)"
    assert W.is_cuda and b.is_cuda, "W and b must be CUDA tensors"
    assert (
        W.dtype == x.dtype and b.dtype == x.dtype
    ), "W and b must have the same dtype as x"

    num_rows, num_cols = x.shape
    y = torch.empty_like(x)
    mean = torch.empty((num_rows,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((num_rows,), device=x.device, dtype=torch.float32)
    block_size = triton.next_power_of_2(num_cols)
    # print(f"Using block size: {block_size}")

    layer_norm_fwd_kernel[(num_rows,)](
        x_ptr=x,
        y_ptr=y,
        mean_ptr=mean,
        rstd_ptr=rstd,
        W_ptr=W,
        b_ptr=b,
        row_stride=x.stride(0),
        N=num_cols,
        eps=eps,
        BLOCK_SIZE=block_size,
    )

    return y, mean, rstd


def layer_norm_fw_demo():
    N = 10240
    batch_size = 1000
    x = torch.rand((batch_size, N), device="cuda", dtype=torch.float32) - 0.5
    W = torch.rand((N,), device="cuda", dtype=torch.float32) - 0.5
    b = torch.rand((N,), device="cuda", dtype=torch.float32) - 0.5

    y1 = torch.layer_norm(x, (N,), weight=W, bias=b, eps=1e-5)
    y2, mean2, rstd2 = layer_norm_fwd(x, W, b, eps=1e-5)

    assert torch.allclose(y1, y2, atol=1e-6), f"Results do not match: {y1} vs {y2}"
    # assert torch.allclose(mean1, mean2, atol=1e-6), f"Results do not match: {mean1} vs {mean2}"
    # assert torch.allclose(rstd1, rstd2, atol=1e-6), f"Results do not match: {rstd1} vs {rstd2}"

    benchmark(
        "PyTorch LayerNorm",
        lambda: torch.layer_norm(x, (N,), weight=W, bias=b, eps=1e-5),
        num_warmups=5,
        num_trials=20,
    )
    benchmark(
        "Triton LayerNorm",
        lambda: layer_norm_fwd(x, W, b, eps=1e-5),
        num_warmups=5,
        num_trials=20,
    )


@triton.jit
def layer_norm_bwd_kernel(
    dy_ptr, x_ptr, mean_ptr, rstd_ptr,
    W_ptr, dx_ptr,
    dW_ptr, db_ptr,
    row_stride, N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mean_ptrs = mean_ptr + row
    rstd_ptrs = rstd_ptr + row
    mean = tl.load(mean_ptrs).to(tl.float32)
    rstd = tl.load(rstd_ptrs).to(tl.float32)

    # compute dx
    for k in range(0, N, BLOCK_SIZE):
        col_ids = k + col_offsets
        mask = col_ids < N 

        dy_ptrs = dy_ptr + row * row_stride + col_ids
        x_ptrs = x_ptr + row * row_stride + col_ids
        w_ptrs = W_ptr + col_ids
        dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd

        dx_c1_part = (1.0 / N) * tl.sum(x_hat * dy * w) * x_hat
        dx_c2_part = (1.0 / N) * tl.sum(dy * w, axis=0)
        dx_part = rstd * (dy * w - dx_c1_part - dx_c2_part)
        dx_ptrs = dx_ptr + row * row_stride + col_ids
        tl.store(dx_ptrs, dx_part, mask=mask)

        db_part = dy
        tl.atomic_add(db_ptr + col_ids, db_part, mask=mask)
        dW_part = dy * x_hat
        tl.atomic_add(dW_ptr + col_ids, dW_part, mask=mask)


def layer_norm_bwd(dy: Tensor, x: Tensor, mean: Tensor, rstd: Tensor, W: Tensor, b: Tensor, eps=1e-5) -> Tuple[Tensor, Tensor, Tensor]:
    num_rows, num_cols = x.shape
    dx = torch.empty_like(x)
    dW = torch.zeros_like(W)
    db = torch.zeros_like(b)
    block_size = 1024

    layer_norm_bwd_kernel[(num_rows,)](
        dy_ptr=dy,
        x_ptr=x,
        mean_ptr=mean,
        rstd_ptr=rstd,
        W_ptr=W,
        dx_ptr=dx,
        dW_ptr=dW,
        db_ptr=db,
        row_stride=x.stride(0),
        N=num_cols,
        BLOCK_SIZE=block_size,
    )

    return dx, dW, db


def layer_norm_bwd_demo():
    N = 10240 * 4
    batch_size = 100
    x = torch.rand((batch_size, N), device="cuda", dtype=torch.float32) - 0.5
    W = torch.rand((N,), device="cuda", dtype=torch.float32) - 0.5
    b = torch.rand((N,), device="cuda", dtype=torch.float32) - 0.5

    x.requires_grad = True
    W.requires_grad = True
    b.requires_grad = True

    y = torch.layer_norm(x, (N,), weight=W, bias=b, eps=1e-5)
    dy = torch.rand_like(y) - 0.5

    y.backward(dy, retain_graph=True)
    dx1 = Tensor(x.grad)
    dW1 = Tensor(W.grad)
    db1 = Tensor(b.grad)

    x.grad = None
    W.grad = None
    b.grad = None


    y2, mean, rstd = layer_norm_fwd(x, W, b, eps=1e-5)
    assert torch.allclose(y, y2, atol=1e-6), f"Results do not match: {y} vs {y2}"

    dx2, dW2, db2 = layer_norm_bwd(dy, x, mean, rstd, W, b, eps=1e-5)

    scale = np.sqrt(batch_size / 10)
    assert torch.allclose(dx1, dx2, atol=1e-2 * scale, rtol=0), f"dx Results do not match: {dx1} vs {dx2}"
    assert torch.allclose(dW1, dW2, atol=1e-2 * scale, rtol=0), f"dw Results do not match: {dW1} vs {dW2}"
    assert torch.allclose(db1, db2, atol=1e-2 * scale, rtol=0), f"db Results do not match: {db1} vs {db2}"

    def backward():
        y.backward(dy, retain_graph=True)
        x.grad = None
        W.grad = None
        b.grad = None

    benchmark(
        "PyTorch LayerNorm Backward",
        backward,
        num_warmups=5,
        num_trials=20,
    )

    benchmark(
        "Triton LayerNorm Backward",
        lambda: layer_norm_bwd(dy, x, mean, rstd, W, b, eps=1e-5),
        num_warmups=5,
        num_trials=20,
    )



if __name__ == "__main__":
    # softmax_demo()
    # vec_add_demo()
    # benchmark_demo.run(show_plots=False, print_data=True)
    # layer_norm_fw_demo()
    layer_norm_bwd_demo()
