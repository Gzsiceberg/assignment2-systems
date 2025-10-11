from typing import Tuple
import triton
import torch
from torch import Tensor
import numpy as np
from einops import rearrange, einsum

class NaiveAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor, is_casual: bool = False) -> Tensor:
        q_shape = Q.shape
        Q = rearrange(Q, '... s d -> (...) s d')
        K = rearrange(K, '... s d -> (...) s d')
        V = rearrange(V, '... s d -> (...) s d')
        batch_size, context_len, d_model = Q.shape
        sqrt_d = np.sqrt(d_model)
        scores = einsum(Q, K, 'b i d, b j d -> b i j') / sqrt_d  # (B, S, S)
        row_max = torch.amax(scores, dim=-1, keepdim=True)  # (B, S, 1)
        p = torch.exp(scores - row_max)  # (B, S, S)
        row_sum = torch.sum(p, dim=-1, keepdim=True)  # (B, S, 1)
        p = p / row_sum  # (B, S, S)
        o = einsum(p, V, 'b i j, b j d -> b i d')  # (B, S, D)
        return o.reshape(q_shape)
    
    @staticmethod
    def backward(ctx, dO: Tensor) -> Tuple[Tensor, Tensor, Tensor, None]:
        raise NotImplementedError("FlashAttention backward pass is not implemented yet.")

class FlashAttention2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor, is_casual: bool = False) -> Tensor:
        block_rows, block_cols = 128, 128
        q_shape = Q.shape
        Q = rearrange(Q, '... s d -> (...) s d')
        K = rearrange(K, '... s d -> (...) s d')
        V = rearrange(V, '... s d -> (...) s d')
        batch_size, context_len, d_model = Q.shape

        num_block_rows = triton.cdiv(context_len, block_rows)
        num_block_cols = triton.cdiv(context_len, block_cols)

        final_o = torch.empty_like(V)
        final_l = torch.empty((batch_size, context_len), device=Q.device)
        sqrt_d = np.sqrt(d_model)
        for i in range(num_block_rows):
            q_i = Q[:, i * block_rows: (i + 1) * block_rows, :]  # (B, block_rows, D)
            assert q_i.shape == (batch_size, block_rows, d_model)
            o_i = torch.zeros_like(q_i)  # (B, block_rows, D)
            m_i = torch.full((batch_size, block_rows), float('-inf'), device=Q.device)  # (B, block_rows)
            l_i = torch.zeros((batch_size, block_rows), device=Q.device)  # (B, block_rows)
            for j in range(num_block_cols):
                k_j = K[:, j * block_cols: (j + 1) * block_cols, :]  # (B, block_cols, D)
                v_j = V[:, j * block_cols: (j + 1) * block_cols, :]  # (B, block_cols, D)
                assert k_j.shape == (batch_size, block_cols, d_model)
                assert v_j.shape == (batch_size, block_cols, d_model)
                if is_casual and j > i:
                    continue
                else:
                    s_ij = einsum(q_i, k_j, 'b i d, b j d -> b i j') / sqrt_d  # (B, block_rows, block_cols)
                    assert s_ij.shape == (batch_size, block_rows, block_cols)
                if is_casual and j == i:
                    mask = torch.tril(torch.ones((block_rows, block_cols), device=Q.device)).unsqueeze(0)  # (1, block_rows, block_cols)
                    s_ij = s_ij * mask + (1.0 - mask) * float('-inf')

                row_max = torch.amax(s_ij, dim=-1)  # (B, block_rows)
                m_ij = torch.maximum(m_i, row_max)
                assert m_ij.shape == (batch_size, block_rows)

                p_ij = torch.exp(s_ij - m_ij.unsqueeze(-1))  # (B, block_rows, block_cols)
                assert p_ij.shape == (batch_size, block_rows, block_cols)

                row_sum_p = torch.sum(p_ij, dim=-1)  # (B, block_rows)
                assert row_sum_p.shape == (batch_size, block_rows)
                l_ij = torch.exp(m_i - m_ij) * l_i + row_sum_p  # (B, block_rows)
                assert l_ij.shape == (batch_size, block_rows)

                o_ij = einsum(p_ij, v_j, 'b i j, b j d -> b i d') # (B, block_rows, D)
                o_ij = o_i * torch.exp(m_i - m_ij).unsqueeze(-1) + o_ij  # (B, block_rows, D)
                assert o_ij.shape == (batch_size, block_rows, d_model)

                # update
                m_i = m_ij
                l_i = l_ij
                o_i = o_ij
            o_i = o_i / l_i.unsqueeze(-1)
            final_o[:, i * block_rows: (i + 1) * block_rows, :] = o_i
            final_l[:, i * block_rows: (i + 1) * block_rows] = m_i + torch.log(l_i)
        
        ctx.save_for_backward(Q, K, V, final_o, final_l)
        ctx.is_casual = is_casual
        ctx.block_rows = block_rows
        ctx.block_cols = block_cols
        ctx.q_shape = q_shape
        return final_o.reshape(q_shape)
    
    @staticmethod
    def backward(ctx, dO: Tensor) -> Tuple[Tensor, Tensor, Tensor, None]:
        raise NotImplementedError("FlashAttention backward pass is not implemented yet.")
    


if __name__ == "__main__":
    torch.manual_seed(0)
    q = torch.rand(2, 128, 2).cuda() - 0.5
    k = torch.rand(2, 128, 2).cuda() - 0.5
    v = torch.rand(2, 128, 2).cuda() - 0.5

    out = FlashAttention2.apply(q, k, v, False)

    from cs336_basics.model import scaled_dot_product_attention
    out2 = scaled_dot_product_attention(q, k, v)

    assert torch.allclose(out, out2, atol=1e-4), f"Max diff: {(out - out2).abs().max():.6f}"
    print("correctness test passed!")





            
                