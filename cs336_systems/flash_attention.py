from typing import Tuple
import triton
from triton import language as tl
import torch
from torch import Tensor
import numpy as np
from einops import rearrange, einsum
from cs336_basics.model import scaled_dot_product_attention


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
        if is_casual:
            mask = torch.tril(torch.ones((context_len, context_len), device=Q.device)).unsqueeze(0)  # (1, S, S)
            mask = mask.to(torch.bool)
            scores = torch.where(mask, scores, float("-inf"))
        row_max = torch.amax(scores, dim=-1, keepdim=True)  # (B, S, 1)
        p = torch.exp(scores - row_max)  # (B, S, S)
        row_sum = torch.sum(p, dim=-1, keepdim=True)  # (B, S, 1)
        p = p / row_sum  # (B, S, S)
        o = einsum(p, V, "b i j, b j d -> b i d")  # (B, S, D)
        ctx.save_for_backward(Q, K, V, p)
        ctx.is_casual = is_casual
        ctx.q_shape = q_shape
        return o.reshape(q_shape)

    @staticmethod
    def backward(ctx, dO: Tensor) -> Tuple[Tensor, Tensor, Tensor, None]:
        Q, K, V, P = ctx.saved_tensors
        is_casual = ctx.is_casual
        q_shape = ctx.q_shape
        dO = rearrange(dO, "... s d -> (...) s d")
        batch_size, context_len, d_model = Q.shape

        sqrt_d = np.sqrt(d_model)
        dV = P.transpose(-2, -1) @ dO  # (B, D, query) @ (B, query, D) -> (B, D, D)
        assert dV.shape == (batch_size, context_len, d_model)
        dP = dO @ V.transpose(-2, -1)  # (B, query, key)
        assert dP.shape == (batch_size, context_len, context_len)

        dS = P * (dP  - torch.sum(dP * P, dim=-1, keepdim=True))  # (B, query, key)
        if is_casual:
            mask = torch.tril(torch.ones((context_len, context_len), device=Q.device)).unsqueeze(0)  # (1, S, S)
            mask = mask.to(torch.bool)
            dS = torch.where(mask, dS, 0.0)
        dQ = dS @ K / sqrt_d  # (B, query, D)
        assert dQ.shape == (batch_size, context_len, d_model)
        dK = dS.transpose(-2, -1) @ Q / sqrt_d  # (B, key, D)
        assert dK.shape == (batch_size, context_len, d_model)

        return dQ.reshape(q_shape), dK.reshape(q_shape), dV.reshape(q_shape), None




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
            for j in range(num_block_cols if not is_casual else min(i + 1, num_block_cols)):
                k_j = K[
                    :, j * block_cols : (j + 1) * block_cols, :
                ]  # (B, block_cols, D)
                v_j = V[
                    :, j * block_cols : (j + 1) * block_cols, :
                ]  # (B, block_cols, D)
                assert k_j.shape == (batch_size, block_cols, d_model)
                assert v_j.shape == (batch_size, block_cols, d_model)
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

        ctx.save_for_backward(Q, K, V, final_l)
        ctx.is_casual = is_casual
        ctx.block_rows = block_rows
        ctx.block_cols = block_cols
        ctx.q_shape = q_shape
        return final_o.reshape(q_shape)

    @staticmethod
    def backward(ctx, dO: Tensor) -> Tuple[Tensor, Tensor, Tensor, None]:
        Q, K, V, L = ctx.saved_tensors
        is_casual = ctx.is_casual
        block_rows = ctx.block_rows
        block_cols = ctx.block_cols
        q_shape = ctx.q_shape
        dO = rearrange(dO, "... s d -> (...) s d")
        dQ, dK, dV = FlashAttention._backward(Q, K, V, L, is_casual, dO)
        return dQ.reshape(q_shape), dK.reshape(q_shape), dV.reshape(q_shape), None

    
    @staticmethod
    @torch.compile
    def _backward(Q: Tensor, K: Tensor, V: Tensor, L: Tensor, is_casual: bool, dO: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, context_len, d_model = Q.shape
        sqrt_d = np.sqrt(d_model)
        S = Q @ K.transpose(-2, -1) / sqrt_d  # (B, S, S)
        if is_casual:
            mask = torch.tril(torch.ones((context_len, context_len), device=Q.device)).unsqueeze(0)  # (1, S, S)
            mask = mask.to(torch.bool)
            S = torch.where(mask, S, float("-inf"))
        P = torch.exp(S - L.unsqueeze(-1))  # (B, S, S)
        assert P.shape == (batch_size, context_len, context_len)
        dV = P.transpose(-2, -1) @ dO  # (B, D, S) @ (B, S, D) -> (B, D, D)

        dP = dO @ V.transpose(-2, -1)  # (B, S, D) @ (B, D, S) -> (B, S, S)
        dS = P * (dP  - torch.sum(dP * P, dim=-1, keepdim=True))  # (B, S, S)
        if is_casual:
            mask = torch.tril(torch.ones((context_len, context_len), device=Q.device)).unsqueeze(0)  # (1, S, S)
            mask = mask.to(torch.bool)
            dS = torch.where(mask, dS, 0.0)

        dQ = dS @ K / sqrt_d  # (B, S, D)
        dK = dS.transpose(-2, -1) @ Q / sqrt_d  # (B, S, D)
        return dQ, dK, dV


@triton.jit
def flash_bkw_vk_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS, scale,
    is_casual: tl.constexpr,
    D_MODEL: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    col_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_id * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )  # (Q_TILE_SIZE, D)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_id * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(col_id * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )  # (K_TILE_SIZE, D)

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_id * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(col_id * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )  # (K_TILE_SIZE, D)

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_id * stride_ob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )  # (Q_TILE_SIZE, D)

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_id * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )  # (Q_TILE_SIZE,)

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_id * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )  # (Q_TILE_SIZE,)

    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D_MODEL)
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D_MODEL)
    row_nums = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    start_row = 0
    if is_casual:
        start_row = col_id
    mask = (tl.arange(0, Q_TILE_SIZE)[:, None]) >= (tl.arange(0, K_TILE_SIZE)[None, :])  # (K_TILE_SIZE, Q_TILE_SIZE)
    dV = tl.zeros((K_TILE_SIZE, D_MODEL), dtype=tl.float32)  # (K_TILE_SIZE, D_MODEL)
    dK = tl.zeros((K_TILE_SIZE, D_MODEL), dtype=tl.float32)  # (K_TILE_SIZE, D_MODEL)
    for row in range(start_row, row_nums):
        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D_MODEL)
        dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D_MODEL)
        l = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")  # (Q_TILE_SIZE,)
        d = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")  # (Q_TILE_SIZE,)

        s = tl.dot(q, tl.trans(k)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)
        if is_casual and (row == col_id):
            s = tl.where(mask, s, -1e6)
        p = tl.exp(s - l[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)
        dV = tl.dot(tl.trans(p), dO, acc=dV)  # (K_TILE_SIZE, D)
        dP = tl.dot(dO, tl.trans(v))  # (Q_TILE_SIZE, K_TILE_SIZE)
        if is_casual and (row == col_id):
            dP = tl.where(mask, dP, 0.0)
        dS = p * (dP - d[:, None]) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)
        dK = tl.dot(tl.trans(dS), q, acc=dK)  # (K_TILE_SIZE, D)

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    k_dtype = K_block_ptr.type.element_ty
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_id * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(col_id * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )  # (K_TILE_SIZE, D_MODEL)
    tl.store(dK_block_ptr, dK.to(k_dtype), boundary_check=(0, 1))

    v_dtype = V_block_ptr.type.element_ty
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_id * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(col_id * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )  # (K_TILE_SIZE, D_MODEL)
    tl.store(dV_block_ptr, dV.to(v_dtype), boundary_check=(0, 1))

@triton.jit
def flash_bkw_q_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS, scale,
    is_casual: tl.constexpr,
    D_MODEL: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_id * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )  # (K_TILE_SIZE, D_MODEL)

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_id * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )  # (K_TILE_SIZE, D_MODEL)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_id * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(row_id * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )  # (Q_TILE_SIZE, D_MODEL)

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_id * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(row_id * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )  # (Q_TILE_SIZE, D_MODEL)

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_id * stride_ob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_oq, stride_od),
        offsets=(row_id * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )  # (Q_TILE_SIZE, D_MODEL)

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_id * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(row_id * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )  # (Q_TILE_SIZE,)

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_id * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(row_id * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )  # (Q_TILE_SIZE,)

    col_nums = tl.cdiv(N_KEYS, K_TILE_SIZE)
    if is_casual:
        col_nums = min(col_nums, row_id + 1)
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D_MODEL)
    dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D_MODEL)
    l = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")  # (Q_TILE_SIZE,)
    d = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")  # (Q_TILE_SIZE,)
    mask = (tl.arange(0, Q_TILE_SIZE)[:, None]) >= (tl.arange(0, K_TILE_SIZE)[None, :])  # (K_TILE_SIZE, Q_TILE_SIZE)

    dQ = tl.zeros((Q_TILE_SIZE, D_MODEL), dtype=tl.float32)  # (Q_TILE_SIZE, D_MODEL)
    for col_id in range(col_nums):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D_MODEL)
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D_MODEL)
        s = tl.dot(q, tl.trans(k)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)
        if is_casual and (col_id == row_id):
            s = tl.where(mask, s, -1e6)
        p = tl.exp(s - l[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)

        dP = tl.dot(dO, tl.trans(v))  # (Q_TILE_SIZE, K_TILE_SIZE)
        if is_casual and (col_id == row_id):
            dP = tl.where(mask, dP, 0.0)
        dS = p * (dP - d[:, None]) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)
        dQ = tl.dot(dS, k, acc=dQ)  # (Q_TILE_SIZE, D_MODEL)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    q_dtype = Q_block_ptr.type.element_ty
    tl.store(dQ_block_ptr, dQ.to(q_dtype), boundary_check=(0, 1))

def flash_attention_bkw(
    Q: Tensor, K: Tensor, V: Tensor, L: Tensor, dO: Tensor, O: Tensor, block_rows: int, block_cols: int, is_casual: bool = False
) -> Tuple[Tensor, Tensor, Tensor]:
    batch_size, N_QUERIES, d_dim = Q.shape
    _, N_KEYS, _ = K.shape
    N_KEYS = K.shape[1]
    D = (dO * O).sum(dim=-1)  # (B, S)
    scale = 1.0 / np.sqrt(d_dim)

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    grid = (triton.cdiv(N_KEYS, block_cols), batch_size)
    flash_bkw_vk_kernel[grid](
        Q, K, V, dO, L, D, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1),
        D.stride(0), D.stride(1),
        N_QUERIES, 
        N_KEYS, 
        scale,
        is_casual=is_casual,
        D_MODEL=d_dim,
        Q_TILE_SIZE=block_rows,
        K_TILE_SIZE=block_cols,
    )

    grid = (triton.cdiv(N_QUERIES, block_rows), batch_size)
    flash_bkw_q_kernel[grid](
        Q, K, V, dO, L, D, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1),
        D.stride(0), D.stride(1),
        N_QUERIES, 
        N_KEYS, 
        scale,
        is_casual=is_casual,
        D_MODEL=d_dim,
        Q_TILE_SIZE=block_rows,
        K_TILE_SIZE=block_cols,
    )
    return dQ, dK, dV


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
        context_len = Q.shape[1]
        if context_len < 4096:
            block_rows, block_cols = 16, 16
        else:
            block_rows, block_cols = 32, 32
        O, L = flash_attention_fwd(Q, K, V, block_rows, block_cols, is_casual=is_casual)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_casual = is_casual
        ctx.block_rows = block_rows
        ctx.block_cols = block_cols
        ctx.q_shape = q_shape
        return O.reshape(q_shape)

    @staticmethod
    def backward(ctx, dO: Tensor) -> Tuple[Tensor, Tensor, Tensor, None]:
        Q, K, V, O, L = ctx.saved_tensors
        is_casual = ctx.is_casual
        block_rows = ctx.block_rows
        block_cols = ctx.block_cols
        q_shape = ctx.q_shape
        dO = rearrange(dO, "... s d -> (...) s d")
        # dQ, dK, dV = FlashAttention._backward(Q, K, V, L, is_casual, dO)
        dQ, dK, dV = flash_attention_bkw(Q, K, V, L, dO, O, block_rows, block_cols, is_casual=is_casual)
        return dQ.reshape(q_shape), dK.reshape(q_shape), dV.reshape(q_shape), None
    

def test_backward():
    is_casual = args.is_casual
    context_len = 512
    d_model = 64
    batch_size = 64
    q = torch.rand(batch_size, context_len, d_model).cuda() - 0.5
    k = torch.rand(batch_size, context_len, d_model).cuda() - 0.5
    v = torch.rand(batch_size, context_len, d_model).cuda() - 0.5
    mask = torch.tril(torch.ones(context_len, context_len)).cuda()
    mask = mask.to(torch.bool)
    mask = rearrange(mask, "i j -> 1 i j")
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    from cs336_basics.model import scaled_dot_product_attention
    out_bench = scaled_dot_product_attention(q, k, v, mask=mask if is_casual else None)

    dO = torch.rand_like(out_bench)
    out_bench.backward(dO)
    dq_bench, dk_bench, dv_bench = q.grad, k.grad, v.grad
    q.grad = None
    k.grad = None
    v.grad = None

    # out = NaiveAttention.apply(q, k, v, is_casual)
    # assert torch.allclose(
    #     out, out_bench, atol=1e-3
    # ), f"Max diff NaiveAttention: {(out - out_bench).abs().max():.6f}"
    # print("NaiveAttention forward correctness test passed!")

    # out.backward(dO)
    # dq, dk, dv = q.grad, k.grad, v.grad
    # q.grad = None
    # k.grad = None
    # v.grad = None
    # assert torch.allclose(
    #     dq, dq_bench, atol=1e-3
    # ), f"Max diff dQ: {(dq - dq_bench).abs().max():.6f}"
    # assert torch.allclose(
    #     dk, dk_bench, atol=1e-3
    # ), f"Max diff dK: {(dk - dk_bench).abs().max():.6f}"
    # assert torch.allclose(
    #     dv, dv_bench, atol=1e-3
    # ), f"Max diff dV: {(dv - dv_bench).abs().max():.6f}"
    # print("NaiveAttention backward correctness test passed!")


    # out1 = FlashAttention.apply(q, k, v, is_casual)
    # assert torch.allclose(
    #     out1, out_bench, atol=1e-3
    # ), f"Max diff FlashAttention: {(out1 - out_bench).abs().max():.6f}"
    # print("FlashAttention forward correctness test passed!")

    # out1.backward(dO)
    # dq1, dk1, dv1 = q.grad, k.grad, v.grad
    # q.grad = None
    # k.grad = None
    # v.grad = None
    # assert torch.allclose(
    #     dv1, dv_bench, atol=1e-3
    # ), f"Max diff dV FlashAttention: {(dv1 - dv_bench).abs().max():.6f}"
    # assert torch.allclose(
    #     dq1, dq_bench, atol=1e-3
    # ), f"Max diff dQ FlashAttention: {(dq1 - dq_bench).abs().max():.6f}"
    # assert torch.allclose(  
    #     dk1, dk_bench, atol=1e-3
    # ), f"Max diff dK FlashAttention: {(dk1 - dk_bench).abs().max():.6f}"
    # print("FlashAttention backward correctness test passed!")

    out2 = FlashAttentionTriton.apply(q, k, v, is_casual)
    assert torch.allclose(
        out2, out_bench, atol=1e-3
    ), f"Max diff FlashAttentionTriton: {(out2 - out_bench).abs().max():.6f}"
    print("FlashAttentionTriton forward correctness test passed!")
    out2.backward(dO)
    dq2, dk2, dv2 = q.grad, k.grad, v.grad
    q.grad = None
    k.grad = None
    v.grad = None
    assert torch.allclose(
        dv2, dv_bench, atol=1e-3
    ), f"Max diff dV FlashAttentionTriton: {(dv2 - dv_bench).abs().max():.6f}"
    assert torch.allclose(  
        dk2, dk_bench, atol=1e-3
    ), f"Max diff dK FlashAttentionTriton: {(dk2 - dk_bench).abs().max():.6f}"
    assert torch.allclose(
        dq2, dq_bench, atol=1e-3
    ), f"Max diff dQ FlashAttentionTriton: {(dq2 - dq_bench).abs().max():.6f}"
    print("FlashAttentionTriton backward correctness test passed!")


def test_forward():
    torch.manual_seed(0)
    is_casual = args.is_casual
    context_len = 2048
    d_model = 64
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
    out3 = NaiveAttention.apply(q, k, v, is_casual)

    from cs336_basics.model import scaled_dot_product_attention
    out_bench = scaled_dot_product_attention(q, k, v, mask=mask if is_casual else None)

    assert torch.allclose(
        out1, out_bench, atol=1e-3
    ), f"Max diff FlashAttention: {(out1 - out_bench).abs().max():.6f}"
    print("FlashAttention correctness test passed!")

    assert torch.allclose(
        out2, out_bench, atol=1e-3
    ), f"Max diff FlashAttentionTriton: {(out2 - out_bench).abs().max():.6f}"
    print("FlashAttentionTriton correctness test passed!")

    assert torch.allclose(
        out3, out_bench, atol=1e-3
    ), f"Max diff NaiveAttention: {(out3 - out_bench).abs().max():.6f}"
    print("NaiveAttention correctness test passed!")


    mean_ms = triton.testing.do_bench(lambda: FlashAttentionTriton.apply(q, k, v, is_casual), quantiles=[0.5])
    print(f"FlashAttentionTriton: {mean_ms:.2f}ms")

    mean_ms = triton.testing.do_bench(lambda: scaled_dot_product_attention(q, k, v, mask=mask if is_casual else None), quantiles=[0.5])
    print(f"PyTorch Attention: {mean_ms:.2f}ms")


def do_benchmark(context_len, d_model, is_casual, all_data: dict, dtype=torch.float32):
    batch_size = 1
    q = torch.rand(batch_size, context_len, d_model, dtype=dtype).cuda() - 0.5
    k = torch.rand(batch_size, context_len, d_model, dtype=dtype).cuda() - 0.5
    v = torch.rand(batch_size, context_len, d_model, dtype=dtype).cuda() - 0.5
    mask = torch.tril(torch.ones(context_len, context_len)).cuda()
    mask = mask.to(torch.bool)
    mask = rearrange(mask, "i j -> 1 i j")

    mean_ms = triton.testing.do_bench(lambda: FlashAttentionTriton.apply(q, k, v, is_casual), quantiles=[0.5])
    print(f"FlashAttentionTriton: {mean_ms:.2f}ms")
    all_data["flash_attention_fw"].append(mean_ms)

    mean_ms = triton.testing.do_bench(lambda: scaled_dot_product_attention(q, k, v, mask=mask if is_casual else None), quantiles=[0.5])
    print(f"PyTorch Attention: {mean_ms:.2f}ms")
    all_data["pytorch_fw"].append(mean_ms)

    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    out = FlashAttentionTriton.apply(q, k, v, is_casual)
    dO = torch.rand_like(out)
    def fn01():
        out.backward(dO, retain_graph=True)
        q.grad = None
        k.grad = None
        v.grad = None
    mean_ms = triton.testing.do_bench(fn01, quantiles=[0.5])
    print(f"FlashAttentionTriton backward: {mean_ms:.2f}ms")
    all_data["flash_attention_bw"].append(mean_ms)

    out = scaled_dot_product_attention(q, k, v, mask=mask if is_casual else None)
    def fn02():
        out.backward(dO, retain_graph=True)
        q.grad = None
        k.grad = None
        v.grad = None
    mean_ms = triton.testing.do_bench(fn02, quantiles=[0.5])
    print(f"PyTorch Attention backward: {mean_ms:.2f}ms")
    all_data["pytorch_bw"].append(mean_ms)

    def fn03():
        out = FlashAttentionTriton.apply(q, k, v, is_casual)
        out.backward(dO)
        q.grad = None
        k.grad = None
        v.grad = None
    mean_ms = triton.testing.do_bench(fn03, quantiles=[0.5])
    print(f"FlashAttentionTriton total: {mean_ms:.2f}ms")
    all_data["flash_attention_total"].append(mean_ms)

    def fn04():
        out = scaled_dot_product_attention(q, k, v, mask=mask if is_casual else None)
        out.backward(dO)
        q.grad = None
        k.grad = None
        v.grad = None
    mean_ms = triton.testing.do_bench(fn04, quantiles=[0.5])
    print(f"PyTorch Attention total: {mean_ms:.2f}ms")
    all_data["pytorch_total"].append(mean_ms)


def benchmark():
    torch.manual_seed(0)
    is_casual = args.is_casual
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    all_data = {
        "context_len": [],
        "d_model": [],
        "flash_attention_fw": [],
        "pytorch_fw": [],
        "flash_attention_bw": [],
        "pytorch_bw": [],
        "flash_attention_total": [],
        "pytorch_total": [],

    }
    context_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    # context_lens = [8192,]
    d_models = [16, 32, 64, 128]
    # d_models = [64,]
    for context_len in context_lens:
        for d_model in d_models:
            print("-" * 80)
            print(f"Benchmarking context_len={context_len}, d_model={d_model}")
            all_data["context_len"].append(context_len)
            all_data["d_model"].append(d_model)
            do_benchmark(context_len, d_model, is_casual, all_data, dtype=dtype)
    import pandas as pd
    df = pd.DataFrame(all_data)
    df["FWD"] = df["pytorch_fw"] / df["flash_attention_fw"]
    df["BWD"] = df["pytorch_bw"] / df["flash_attention_bw"]
    df["TOTAL"] = df["pytorch_total"] / df["flash_attention_total"]
    pd.set_option("display.float_format", "{:.2f}".format)
    print(df)

    if args.output:
        df = df.round(2)
        df.to_markdown(args.output, index=False)


def test_timing_flash_forward_backward():
    n_heads = 1
    d_head = 64
    sequence_length = 16384
    q, k, v = torch.randn(
        3, n_heads, sequence_length, d_head, device='cuda', dtype=torch.bfloat16, requires_grad=True
    )

    def flash_forward_backward():
        o = FlashAttentionTriton.apply(q, k, v, True)
        loss = o.sum()
        loss.backward()

    # results = triton.testing.do_bench(flash_forward_backward, rep=10000, warmup=1000)
    results = triton.testing.do_bench(flash_forward_backward)
    print(f"FlashAttentionTriton forward + backward: {results:.2f}ms")



if __name__ == "__main__":
    import argparse
    from rich import print
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--test_forward", action="store_true")
    parser.add_argument("-b", "--test_backward", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--is_casual", action="store_true")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        test_timing_flash_forward_backward()
    if args.benchmark:
        benchmark()
    if args.test_forward:
        test_forward()
    if args.test_backward:
        test_backward()