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
def _weight_sum_fw_kernel(
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

@triton.jit
def _weight_sum_bw_kernel(
    x_ptr, w_ptr, grad_out_ptr,
    grad_x_ptr, grad_w_ptr,
    x_stride_row, x_stride_col,
    w_stride, grad_out_stride,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    grad_out_block_ptr = tl.make_block_ptr(
        grad_out_ptr,
        shape=(M,),
        strides=(grad_out_stride,),
        offsets=(row * BLOCK_M,),
        block_shape=(BLOCK_M,),
        order=(0,),
    )

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(M, N),
        strides=(x_stride_row, x_stride_col),
        offsets=(row * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
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

    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(M, N),
        strides=(x_stride_row, x_stride_col),
        offsets=(row * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    num_programs = tl.num_programs(0)
    grad_w_block_ptr = tl.make_block_ptr(
        grad_w_ptr,
        shape=(num_programs, N),
        strides=(x_stride_row, x_stride_col),
        offsets=(row, 0),
        block_shape=(1, BLOCK_N),
        order=(1, 0),
    )

    for _ in range(0, N, BLOCK_N):
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        w = tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero")
        grad_out = tl.load(grad_out_block_ptr, boundary_check=(0,), padding_option="zero")

        grad_w_part = grad_out[:, None] * x
        grad_x_part = grad_out[:, None] * w[None, :]

        tl.store(grad_x_block_ptr, grad_x_part, boundary_check=(0, 1))
        grad_w_part = tl.sum(grad_w_part, axis=0, keep_dims=True)
        tl.store(grad_w_block_ptr, grad_w_part, boundary_check=(1,))

        x_block_ptr = x_block_ptr.advance((0, BLOCK_N))
        w_block_ptr = w_block_ptr.advance((BLOCK_N,))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, BLOCK_N))
        grad_w_block_ptr = grad_w_block_ptr.advance((0, BLOCK_N))



class WeightSumTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor) -> Tensor:
        intput_shape = x.shape
        N = x.shape[-1]
        M = x.numel() // N
        out = torch.empty((M,), device=x.device, dtype=torch.float32)
        assert out.dim() == 1, f"out should be 1D with shape ({M},), but got {out.shape}"

        assert (
            weight.dim() == 1 and weight.shape[0] == N
        ), f"weight should be 1D with shape ({N},), but got {weight.shape}"
        assert x.is_cuda and weight.is_cuda, "Triton kernel only supports CUDA tensors"
        assert (
            x.is_contiguous() and weight.is_contiguous()
        ), "Triton kernel only supports contiguous tensors"
        ctx.save_for_backward(x, weight)

        x = rearrange(x, "... D -> ( ... ) D")
        BLOCK_M = 16
        BLOCK_N = triton.next_power_of_2(N) // 16
        grid = (triton.cdiv(M, BLOCK_M),)
        _weight_sum_fw_kernel[grid](
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
    
    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tuple[Tensor, Tensor]:
        x, weight = ctx.saved_tensors
        input_shape = x.shape
        x = rearrange(x, "... D -> ( ... ) D")
        grad_out = rearrange(grad_out, "... -> (...) 1")

        # grad_x = grad_out * weight[None, :]
        # assert grad_x.shape == x.shape, f"grad_x shape {grad_x.shape} != x shape {x.shape}"

        # grad_weight = grad_out * x
        # grad_weight = torch.sum(grad_weight, dim=tuple(range(grad_weight.dim() - 1)))
        # assert grad_weight.shape == weight.shape, f"grad_weight shape {grad_weight.shape} != weight shape {weight.shape}"
        # return grad_x, grad_weight

        BLOCK_M = 16
        BLOCK_N = triton.next_power_of_2(N) // 16
        grid = (triton.cdiv(x.shape[0], BLOCK_M),)
        grad_x = torch.zeros_like(x)
        grad_weight = torch.zeros((grid[0], x.shape[1]), device=x.device, dtype=torch.float32)
        _weight_sum_bw_kernel[grid](
            x, weight, grad_out,
            grad_x, grad_weight,
            x_stride_row=x.stride(0),
            x_stride_col=x.stride(1),
            w_stride=weight.stride(0),
            grad_out_stride=grad_out.stride(0),
            M=x.shape[0],
            N=x.shape[1],
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        grad_weight = torch.sum(grad_weight, dim=0)
        return grad_x.view(input_shape), grad_weight


if __name__ == "__main__":
    N = 1024
    M = 4096
    x = torch.rand((M // 512, 512, N), device="cuda", dtype=torch.float32) - 0.5
    weight = torch.rand((N,), device="cuda", dtype=torch.float32) - 0.5
    weight.requires_grad = True
    x.requires_grad = True

    out1 = weighted_sum(x, weight)
    out2 = WeightSumTriton.apply(x, weight)
    max_diff = torch.max(torch.abs(out1 - out2))
    print(f"Max diff: {max_diff:.5f} out1.shape: {out1.shape}, out2.shape: {out2.shape}")
    assert torch.allclose(out1, out2, atol=1e-5, rtol=0), f"Results are not close! out1: {out1}, out2: {out2}"

    out1.sum().backward()
    grad_x1 = x.grad
    grad_weight1 = weight.grad

    x.grad = None
    weight.grad = None

    out2.sum().backward()
    grad_x2 = x.grad
    grad_weight2 = weight.grad

    print(f"grad_weight1: {grad_weight1} grad_weight2: {grad_weight2}")
    assert torch.allclose(grad_x1, grad_x2, atol=1e-4, rtol=0), f"grad_x are not close! grad_x1: {grad_x1}, grad_x2: {grad_x2}"
    max_diff = torch.max(torch.abs(grad_weight1 - grad_weight2))
    print(f"Max diff in grad_weight: {max_diff:.5f}")
    assert torch.allclose(grad_weight1, grad_weight2, atol=1e-4, rtol=0), f"grad_weight are not close! grad_weight1: {grad_weight1}, grad_weight2: {grad_weight2}"
    print("Gradient check passed!")
