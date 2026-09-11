# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Dense MHA regression kernel, matching boltOps flash_attention_wdra."""
import triton
import triton.language as tl

@triton.jit
def _flash_attention_fwd_kernel(
    Q,
    K,
    V,
    Out,
    Lse,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_lb,
    stride_lh,
    stride_ls,
    N_CTX: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_Q_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    query_block = pid % NUM_Q_BLOCKS
    head_batch = pid // NUM_Q_BLOCKS
    head = head_batch % NUM_HEADS
    batch = head_batch // NUM_HEADS

    offs_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_valid = offs_m < N_CTX

    q_ptrs = (
        Q
        + batch * stride_qb
        + head * stride_qh
        + offs_m[:, None] * stride_qs
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)
    # exp2 is substantially cheaper than exp; keep max/sum in log2 space.
    q = (q * (sm_scale * 1.4426950408889634)).to(q.dtype)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    for start_n in tl.range(
        0, N_CTX, BLOCK_N, warp_specialize=WARP_SPECIALIZE
    ):
        token = start_n + offs_n
        kv_valid = token < N_CTX
        k_ptrs = (
            K
            + batch * stride_kb
            + head * stride_kh
            + token[:, None] * stride_ks
            + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            V
            + batch * stride_vb
            + head * stride_vh
            + token[:, None] * stride_vs
            + offs_d[None, :] * stride_vd
        )
        k = tl.load(k_ptrs, mask=kv_valid[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k))
        valid = kv_valid[None, :]
        if CAUSAL:
            valid = valid & (token[None, :] <= offs_m[:, None])
        scores = tl.where(valid, scores, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp2(m_i - m_ij)
        p = tl.exp2(scores - m_ij[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptrs, mask=kv_valid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    out = acc / l_i[:, None]
    out_ptrs = (
        Out
        + batch * stride_ob
        + head * stride_oh
        + offs_m[:, None] * stride_os
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out, mask=q_valid[:, None])
    lse_ptrs = Lse + batch * stride_lb + head * stride_lh + offs_m * stride_ls
    # Convert the log2-domain online-softmax state back to natural logarithms.
    tl.store(lse_ptrs, (m_i + tl.log2(l_i)) * 0.6931471805599453, mask=q_valid)
