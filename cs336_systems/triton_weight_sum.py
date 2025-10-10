from typing import Tuple
import triton
import triton.language as tl
import torch
from torch import Tensor
from cs336_basics.nn_utils import softmax
from benchmarking import benchmark, profile
from rich import print
import numpy as np
from einops import rearrange, einsum


def weighted_sum(x, weight):
    # Here, assume that x has n-dim shape [..., D], and weight has 1D shape [D]
    return (weight * x).sum(axis=-1)


@triton.jit
def _weight_sum_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    x_stride_row,
    x_stride_col,
    w_stride,
    out_stride,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(M, N),
        strides=(x_stride_row, x_stride_col),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        # - The order of the dimensions in memory from major to minor
        # axes (= np.argsort(strides)) for optimizations, especially useful on H100
        order=(1, 0),
    )

    w_block_ptr = tl.make_block_ptr(
        w_ptr,
        shape=(N,),
        strides=(w_stride,),
        offsets=(0,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        out_ptr,
        shape=(M,),
        strides=(out_stride,),
        offsets=(pid_m * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )

    output = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for _ in range(0, N, BLOCK_N):
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        w = tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero")
        output += tl.sum(x * w[None, :], axis=1)

        x_block_ptr = x_block_ptr.advance((0, BLOCK_N))
        w_block_ptr = w_block_ptr.advance((BLOCK_N,))

    tl.store(output_block_ptr, output, boundary_check=(0,))


class WeightSumTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor) -> Tensor:
        intput_shape = x.shape
        N = x.shape[-1]
        M = x.numel() // N
        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        assert (
            weight.dim() == 1 and weight.shape[0] == N
        ), f"weight should be 1D with shape ({N},), but got {weight.shape}"
        assert x.is_cuda and weight.is_cuda, "Triton kernel only supports CUDA tensors"
        assert (
            x.is_contiguous() and weight.is_contiguous()
        ), "Triton kernel only supports contiguous tensors"

        x = rearrange(x, "... D -> ( ... ) D")
        BLOCK_M = 16
        BLOCK_N = triton.next_power_of_2(N) // 16
        grid = (triton.cdiv(M, BLOCK_M),)
        _weight_sum_kernel[grid](
            x,
            weight,
            out,
            x_stride_row=x.stride(0),
            x_stride_col=x.stride(1),
            w_stride=weight.stride(0),
            out_stride=out.stride(0),
            M=M,
            N=N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        return out.view(intput_shape[:-1])



if __name__ == "__main__":
    N = 1024
    M = 4096
    x = torch.rand((M, N), device="cuda", dtype=torch.float32) - 0.5
    weight = torch.rand((N,), device="cuda", dtype=torch.float32) - 0.5

    out1 = weighted_sum(x, weight)
    out2 = WeightSumTriton.apply(x, weight)
    max_diff = torch.max(torch.abs(out1 - out2))
    print(f"Max diff: {max_diff:.5f}")
    assert torch.allclose(out1, out2, atol=1e-5, rtol=0), f"Results are not close! out1: {out1}, out2: {out2}"

