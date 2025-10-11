from typing import Tuple
import triton
from triton import language as tl
import torch
from torch import Tensor
import numpy as np
from einops import rearrange, einsum


class NaiveAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, Q: Tensor, K: Tensor, V: Tensor, is_casual: bool = False
    ) -> Tensor:
        q_shape = Q.shape
        Q = rearrange(Q, "... s d -> (...) s d")
        K = rearrange(K, "... s d -> (...) s d")
        V = rearrange(V, "... s d -> (...) s d")
        batch_size, context_len, d_model = Q.shape
        sqrt_d = np.sqrt(d_model)
        scores = einsum(Q, K, "b i d, b j d -> b i j") / sqrt_d  # (B, S, S)
        row_max = torch.amax(scores, dim=-1, keepdim=True)  # (B, S, 1)
        p = torch.exp(scores - row_max)  # (B, S, S)
        row_sum = torch.sum(p, dim=-1, keepdim=True)  # (B, S, 1)
        p = p / row_sum  # (B, S, S)
        o = einsum(p, V, "b i j, b j d -> b i d")  # (B, S, D)
        return o.reshape(q_shape)

    @staticmethod
    def backward(ctx, dO: Tensor) -> Tuple[Tensor, Tensor, Tensor, None]:
        raise NotImplementedError(
            "FlashAttention backward pass is not implemented yet."
        )


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, Q: Tensor, K: Tensor, V: Tensor, is_casual: bool = False
    ) -> Tensor:
        block_rows, block_cols = 16, 16
        assert block_rows == block_cols
        q_shape = Q.shape
        Q = rearrange(Q, "... s d -> (...) s d")
        K = rearrange(K, "... s d -> (...) s d")
        V = rearrange(V, "... s d -> (...) s d")
        batch_size, context_len, d_model = Q.shape

        num_block_rows = triton.cdiv(context_len, block_rows)
        num_block_cols = triton.cdiv(context_len, block_cols)

        final_o = torch.empty_like(V)
        final_l = torch.empty((batch_size, context_len), device=Q.device)
        sqrt_d = np.sqrt(d_model)
        mask = torch.tril(torch.ones((block_rows, block_cols), device=Q.device)).unsqueeze(0)  # (1, block_rows, block_cols)
        mask = mask.to(torch.bool)
        for i in range(num_block_rows):
            q_i = Q[:, i * block_rows : (i + 1) * block_rows, :]  # (B, block_rows, D)
            assert q_i.shape == (batch_size, block_rows, d_model)
            o_i = torch.zeros_like(q_i)  # (B, block_rows, D)
            m_i = torch.full(
                (batch_size, block_rows), float("-inf"), device=Q.device
            )  # (B, block_rows)
            l_i = torch.zeros(
                (batch_size, block_rows), device=Q.device
            )  # (B, block_rows)
            for j in range(num_block_cols):
                k_j = K[
                    :, j * block_cols : (j + 1) * block_cols, :
                ]  # (B, block_cols, D)
                v_j = V[
                    :, j * block_cols : (j + 1) * block_cols, :
                ]  # (B, block_cols, D)
                assert k_j.shape == (batch_size, block_cols, d_model)
                assert v_j.shape == (batch_size, block_cols, d_model)
                if is_casual and j > i:
                    continue
                else:
                    s_ij = (
                        einsum(q_i, k_j, "b i d, b j d -> b i j") / sqrt_d
                    )  # (B, block_rows, block_cols)
                    assert s_ij.shape == (batch_size, block_rows, block_cols)
                if is_casual and j == i:
                    s_ij = torch.where(mask, s_ij, -1e6)

                row_max = torch.amax(s_ij, dim=-1)  # (B, block_rows)
                m_ij = torch.maximum(m_i, row_max)
                assert m_ij.shape == (batch_size, block_rows)

                p_ij = torch.exp(
                    s_ij - m_ij.unsqueeze(-1)
                )  # (B, block_rows, block_cols)
                assert p_ij.shape == (batch_size, block_rows, block_cols)

                row_sum_p = torch.sum(p_ij, dim=-1)  # (B, block_rows)
                assert row_sum_p.shape == (batch_size, block_rows)
                l_ij = torch.exp(m_i - m_ij) * l_i + row_sum_p  # (B, block_rows)
                assert l_ij.shape == (batch_size, block_rows)

                o_ij = einsum(p_ij, v_j, "b i j, b j d -> b i d")  # (B, block_rows, D)
                o_ij = (
                    o_i * torch.exp(m_i - m_ij).unsqueeze(-1) + o_ij
                )  # (B, block_rows, D)
                assert o_ij.shape == (batch_size, block_rows, d_model)

                # update
                m_i = m_ij
                l_i = l_ij
                o_i = o_ij
            o_i = o_i / l_i.unsqueeze(-1)
            final_o[:, i * block_rows : (i + 1) * block_rows, :] = o_i
            final_l[:, i * block_rows : (i + 1) * block_rows] = m_i + torch.log(l_i)

        ctx.save_for_backward(Q, K, V, final_o, final_l)
        ctx.is_casual = is_casual
        ctx.block_rows = block_rows
        ctx.block_cols = block_cols
        ctx.q_shape = q_shape
        return final_o.reshape(q_shape)

    @staticmethod
    def backward(ctx, dO: Tensor) -> Tuple[Tensor, Tensor, Tensor, None]:
        raise NotImplementedError(
            "FlashAttention backward pass is not implemented yet."
        )


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    is_casual: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_id * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(row_id * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )  # (Q_TILE_SIZE, D)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_id * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )  # (K_TILE_SIZE, D)

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_id * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )  # (K_TILE_SIZE, D)

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D)
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)  # (Q_TILE_SIZE, D)
    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)  # (Q_TILE_SIZE,)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)  # (Q_TILE_SIZE,)

    num_cols = tl.cdiv(N_KEYS, K_TILE_SIZE)
    mask = (tl.arange(0, Q_TILE_SIZE)[:, None]) >= (tl.arange(0, K_TILE_SIZE)[None, :])  # (K_TILE_SIZE, Q_TILE_SIZE)
    if is_casual:
        num_cols = min(num_cols, row_id + 1)
    for col_id in tl.range(num_cols):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D)
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D)

        s = tl.dot(q, tl.trans(k)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)
        if is_casual and (col_id == row_id):
            s = tl.where(mask, s, -1e6)
        row_max = tl.max(s, axis=1)  # (Q_TILE_SIZE,)
        m_new = tl.maximum(m, row_max)  # (Q_TILE_SIZE,)

        p = tl.exp(s - m_new[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)
        exp_m = tl.exp(m - m_new)  # (Q_TILE_SIZE,)
        l_new = exp_m * l + tl.sum(p, axis=1)  # (Q_TILE_SIZE,)

        p = tl.cast(p, v.dtype)  # (Q_TILE_SIZE, K_TILE_SIZE)
        o = tl.dot(p, v, acc=o * exp_m[:, None])  # (Q_TILE_SIZE, D)

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

        m = m_new
        l = l_new
    final_o = o / l[:, None]  # (Q_TILE_SIZE, D)
    final_o = tl.cast(final_o, V_block_ptr.type.element_ty)  # (Q_TILE_SIZE, D)
    final_l = m + tl.log(l)  # (Q_TILE_SIZE,)
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_id * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(row_id * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )  # (Q_TILE_SIZE, D)
    tl.store(O_block_ptr, final_o, boundary_check=(0, 1))

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_id * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(row_id * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )  # (Q_TILE_SIZE,)
    tl.store(L_block_ptr, final_l, boundary_check=(0,))
        

def flash_attention_fwd(
    Q: Tensor, K: Tensor, V: Tensor, block_rows: int, block_cols: int, is_casual: bool = False
) -> Tuple[Tensor, Tensor]:
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.shape == K.shape and K.shape == V.shape
    assert block_rows == block_cols
    batch_size, N_QUERIES, d_dim = Q.shape
    N_KEYS = K.shape[1]
    O = torch.empty_like(V)
    L = torch.empty((batch_size, N_QUERIES), device=Q.device, dtype=torch.float32)
    scale = 1.0 / np.sqrt(d_dim)
    grid = (triton.cdiv(N_QUERIES, block_rows), batch_size)

    flash_fwd_kernel[grid](
        Q, K, V, O, L,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        L.stride(0), L.stride(1),
        N_QUERIES, N_KEYS, scale,
        is_casual=is_casual,
        D=d_dim,
        Q_TILE_SIZE=block_rows,
        K_TILE_SIZE=block_cols,
    )
    return O, L

class FlashAttentionTriton(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, Q: Tensor, K: Tensor, V: Tensor, is_casual: bool = False
    ) -> Tensor:
        block_rows, block_cols = 16, 16
        q_shape = Q.shape
        Q = rearrange(Q, "... s d -> (...) s d")
        K = rearrange(K, "... s d -> (...) s d")
        V = rearrange(V, "... s d -> (...) s d")
        O, L = flash_attention_fwd(Q, K, V, block_rows, block_cols, is_casual=is_casual)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_casual = is_casual
        ctx.block_rows = block_rows
        ctx.block_cols = block_cols
        ctx.q_shape = q_shape
        return O.reshape(q_shape)

    @staticmethod
    def backward(ctx, dO: Tensor) -> Tuple[Tensor, Tensor, Tensor, None]:
        raise NotImplementedError(
            "FlashAttention backward pass is not implemented yet."
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    is_casual = True
    context_len = 1024
    d_model = 128
    batch_size = 64
    q = torch.rand(batch_size, context_len, d_model).cuda() - 0.5
    k = torch.rand(batch_size, context_len, d_model).cuda() - 0.5
    v = torch.rand(batch_size, context_len, d_model).cuda() - 0.5
    mask = torch.tril(torch.ones(context_len, context_len)).cuda()
    mask = mask.to(torch.bool)
    mask = rearrange(mask, "i j -> 1 i j")

    q_index = torch.arange(context_len)
    k_index = torch.arange(context_len)
    mask02 = q_index[:, None] >= k_index[None, :]
    mask02 = mask02.to(torch.bool).cuda()
    mask02 = rearrange(mask02, "i j -> 1 i j")
    assert torch.equal(mask, mask02)

    out1 = FlashAttention.apply(q, k, v, is_casual)
    out2 = FlashAttentionTriton.apply(q, k, v, is_casual)

    from cs336_basics.model import scaled_dot_product_attention
    out_bench = scaled_dot_product_attention(q, k, v, mask=mask if is_casual else None)

    assert torch.allclose(
        out1, out_bench, atol=1e-3
    ), f"Max diff FlashAttention: {(out1 - out_bench).abs().max():.6f}"
    print("correctness test passed!")

    assert torch.allclose(
        out2, out_bench, atol=1e-3
    ), f"Max diff FlashAttentionTriton: {(out2 - out_bench).abs().max():.6f}"
    print("correctness test passed!")


    mean_ms = triton.testing.do_bench(lambda: FlashAttentionTriton.apply(q, k, v, is_casual), quantiles=[0.5])
    print(f"FlashAttentionTriton: {mean_ms:.2f}ms")

    mean_ms = triton.testing.do_bench(lambda: scaled_dot_product_attention(q, k, v, mask=mask if is_casual else None), quantiles=[0.5])
    print(f"PyTorch Attention: {mean_ms:.2f}ms")