import math
import os

import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_hip
from triton.backends.amd.compiler import HIPBackend
import triton.knobs

# Per-matrix byte span: strictly between the old 2GiB cap and the new 4GiB-1 cap.
OLD_CAP_BYTES = 2**31 - 1
NEW_CAP_BYTES = 2**32 - 1

# float32 GEMM with M=K=24000 -> 24000*24000*4 = 2_304_000_000 bytes (~2.15 GiB) per operand.
GEMM_M = 24000
GEMM_N = 512
GEMM_K = 24000
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32

# Flash-attention: K is >3GiB; Q/V stay small (cross-attn). Layout (batch, seqlen, nheads, headdim).
FA_BATCH = 1
FA_NHEADS = 4
FA_D_HEAD = 128
FA_SEQLEN_Q = 512
# 3145856 * 4 * 128 * 2 = 3_221_225_472 bytes (~3.00 GiB) for K/V.
FA_SEQLEN_K = (3 * 2**30) // (FA_BATCH * FA_NHEADS * FA_D_HEAD * 2) + 128
FA_MIN_K_BYTES = 3 * 2**30

# Same default as other third_party/hcu/test scripts; no pytest device fixture.
_HCU_TEST_DEVICE = "cuda"


def _config_pointer_range_indices(cfg0):
    """Best-effort: configs[0] may be an object with .pointer_range_32 or a dict from cache_hook."""
    if hasattr(cfg0, "pointer_range_32"):
        return cfg0.pointer_range_32
    if isinstance(cfg0, dict):
        return [k for k, v in cfg0.items() if isinstance(v, list) and ["tt.pointer_range", 32] in v]
    return []


def _arg_index_for_pointer_meta(k):
    if isinstance(k, int):
        return k
    if isinstance(k, tuple) and k and isinstance(k[0], int):
        return k[0]
    return None


def _tensor_bytes(t: torch.Tensor) -> int:
    return HIPBackend.get_tensor_physical_size(t)


def _reference_flash_attention(q, k, v, sm_scale, block_m=64, block_n=4096):
    """Chunked flash-attention reference (BSHD layout)."""
    qf = q.float()
    kf = k.float()
    vf = v.float()
    b, sq, nh, d = qf.shape
    sk = kf.shape[1]
    out = torch.zeros(b, sq, nh, d, device=qf.device, dtype=torch.float32)
    for bi in range(b):
        for hi in range(nh):
            for m0 in range(0, sq, block_m):
                m1 = min(m0 + block_m, sq)
                q_tile = qf[bi, m0:m1, hi]
                m_i = torch.full((m1 - m0, ), -float("inf"), device=qf.device)
                lse_i = torch.full((m1 - m0, ), -float("inf"), device=qf.device)
                acc = torch.zeros(m1 - m0, d, device=qf.device)
                for n0 in range(0, sk, block_n):
                    n1 = min(n0 + block_n, sk)
                    k_tile = kf[bi, n0:n1, hi]
                    v_tile = vf[bi, n0:n1, hi]
                    qk = (q_tile @ k_tile.T) * sm_scale
                    m_ij = torch.maximum(torch.amax(qk, dim=-1), m_i)
                    p = torch.exp(qk - m_ij.unsqueeze(-1))
                    l_ij = torch.sum(p, dim=-1)
                    acc_o_scale = torch.exp(m_i - m_ij)
                    acc = acc * acc_o_scale.unsqueeze(-1)
                    acc = acc + p @ v_tile
                    lse_i = m_ij + torch.log(torch.exp(lse_i - m_ij) + l_ij)
                    m_i = m_ij
                o_scale = torch.exp(m_i - lse_i)
                out[bi, m0:m1, hi] = acc * o_scale.unsqueeze(-1)
    return out.to(dtype=q.dtype)


def _fa_mem_required() -> int:
    elem = torch.float16.itemsize
    q_bytes = FA_BATCH * FA_SEQLEN_Q * FA_NHEADS * FA_D_HEAD * elem
    kv_bytes = FA_BATCH * FA_SEQLEN_K * FA_NHEADS * FA_D_HEAD * elem
    return q_bytes + 2 * kv_bytes + q_bytes + 512 * 1024 * 1024


def _matrix_bytes(m, k, dtype=torch.float32) -> int:
    return m * k * dtype.itemsize


def _gpu_free_bytes() -> int:
    free, _ = torch.cuda.mem_get_info()
    return free


def _gemm_mem_required() -> int:
    elem = torch.float32.itemsize
    return (GEMM_M * GEMM_K + GEMM_K * GEMM_N + GEMM_M * GEMM_N) * elem


@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tl.assume(stride_am >= 0)
    tl.assume(stride_ak >= 0)
    tl.assume(stride_bk >= 0)
    tl.assume(stride_bn >= 0)
    tl.assume(stride_cm >= 0)
    tl.assume(stride_cn >= 0)
    tl.assume(M >= 0)
    tl.assume(N >= 0)
    tl.assume(K >= 0)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fa_fwd_kernel(
    Q,
    K,
    V,
    Bias,
    Out,
    Lse,
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.assume(stride_qb >= 0)
    tl.assume(stride_qh >= 0)
    tl.assume(stride_qm >= 0)
    tl.assume(stride_kb >= 0)
    tl.assume(stride_kh >= 0)
    tl.assume(stride_kn >= 0)
    tl.assume(stride_vb >= 0)
    tl.assume(stride_vh >= 0)
    tl.assume(stride_vn >= 0)
    tl.assume(stride_ob >= 0)
    tl.assume(stride_oh >= 0)
    tl.assume(stride_om >= 0)
    tl.assume(stride_bb >= 0)
    tl.assume(stride_bh >= 0)
    tl.assume(stride_bm >= 0)
    tl.assume(seqlen_q >= 0)
    tl.assume(seqlen_k >= 0)
    tl.assume(seqlen_q_rounded >= 0)
    tl.assume(nheads >= 0)
    tl.assume(headdim >= 0)
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k)) # , trans_b=True
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator --
        # BUG: have to store and immediately load
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        # update acc_o
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    # BUG: have to store and immediately load
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )


def _fa_flash_attn_forward(q, k, v, bias=None, causal=False, softmax_scale=None):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 4
    if bias is None:
        bias = torch.empty(0, device=q.device, dtype=q.dtype)
    grid = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)
    _fa_fwd_kernel[grid](
        q,
        k,
        v,
        bias,
        o,
        lse,
        tmp,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated


@pytest.mark.skipif(reason="within_4g is a HIP specific optimization", condition=not is_hip())
def test_gemm_buffer_ops_between_2g_and_4g():
    """GEMM on >2GiB matrices verifies buffer ops correctness under tt.pointer_range=32."""
    device = _HCU_TEST_DEVICE
    a_bytes = _matrix_bytes(GEMM_M, GEMM_K)
    assert OLD_CAP_BYTES < a_bytes <= NEW_CAP_BYTES

    mem_need = _gemm_mem_required() + 512 * 1024 * 1024
    if _gpu_free_bytes() < mem_need:
        pytest.skip(f"need ~{mem_need // 2**30}GiB free device memory for GEMM operands")

    default_buffer_ops = os.environ.get("AMDGCN_USE_BUFFER_OPS", "1")
    pointer_range_32 = None
    try:
        os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"

        def cache_hook(*args, **kwargs):
            nonlocal pointer_range_32
            pointer_range_32 = _config_pointer_range_indices(kwargs["compile"]["configs"][0])

        triton.knobs.runtime.jit_cache_hook = cache_hook

        torch.manual_seed(0)
        a = torch.randn(GEMM_M, GEMM_K, dtype=torch.float32, device=device)
        b = torch.randn(GEMM_K, GEMM_N, dtype=torch.float32, device=device)
        c = torch.empty(GEMM_M, GEMM_N, dtype=torch.float32, device=device)

        grid = (triton.cdiv(GEMM_M, BLOCK_M), triton.cdiv(GEMM_N, BLOCK_N))
        _gemm_kernel[grid](
            a,
            b,
            c,
            GEMM_M,
            GEMM_N,
            GEMM_K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

        if pointer_range_32:
            idxs = {i for i in (_arg_index_for_pointer_meta(k) for k in pointer_range_32) if i is not None}
            assert idxs == {0, 1, 2}

        ref = torch.matmul(a, b)
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    finally:
        triton.knobs.runtime.jit_cache_hook = None
        os.environ["AMDGCN_USE_BUFFER_OPS"] = default_buffer_ops


@pytest.mark.skipif(reason="within_4g is a HIP specific optimization", condition=not is_hip())
def test_flash_attn_buffer_ops_k_over_3g():
    """Flash attention with K > 3GiB (Q/V small) under buffer ops + tt.pointer_range=32."""
    device = _HCU_TEST_DEVICE
    mem_need = _fa_mem_required()
    if _gpu_free_bytes() < mem_need:
        pytest.skip(f"need ~{mem_need // 2**30}GiB free device memory for flash-attention operands")

    default_buffer_ops = os.environ.get("AMDGCN_USE_BUFFER_OPS", "1")
    pointer_range_32 = None
    try:
        os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"

        def cache_hook(*args, **kwargs):
            nonlocal pointer_range_32
            pointer_range_32 = _config_pointer_range_indices(kwargs["compile"]["configs"][0])

        triton.knobs.runtime.jit_cache_hook = cache_hook

        dtype = torch.float16
        torch.manual_seed(1)
        q = torch.randn(FA_BATCH, FA_SEQLEN_Q, FA_NHEADS, FA_D_HEAD, dtype=dtype, device=device)
        k = torch.randn(FA_BATCH, FA_SEQLEN_K, FA_NHEADS, FA_D_HEAD, dtype=dtype, device=device)
        v = torch.randn(FA_BATCH, FA_SEQLEN_K, FA_NHEADS, FA_D_HEAD, dtype=dtype, device=device)

        k_bytes = _tensor_bytes(k)
        assert k_bytes > FA_MIN_K_BYTES
        assert k_bytes <= NEW_CAP_BYTES
        assert _tensor_bytes(q) < FA_MIN_K_BYTES

        sm_scale = FA_D_HEAD**-0.5
        o, _, _ = _fa_flash_attn_forward(q, k, v, causal=False, softmax_scale=sm_scale)

        if pointer_range_32:
            idxs = {i for i in (_arg_index_for_pointer_meta(k) for k in pointer_range_32) if i is not None}
            assert 1 in idxs  # K

        ref = _reference_flash_attention(q, k, v, sm_scale)
        torch.testing.assert_close(o, ref, rtol=1e-2, atol=1e-2)
    finally:
        triton.knobs.runtime.jit_cache_hook = None
        os.environ["AMDGCN_USE_BUFFER_OPS"] = default_buffer_ops
