# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Portions adapted from ROCm/aiter (https://github.com/ROCm/aiter/pull/5750):
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Upstream vLLM/SGLang dot_scaled kernels for compile-only regression.

Function bodies are retained from the commits linked below. Global function
names have a module prefix to avoid collisions; no address, mask, scale or dot
expression is changed. These kernels do not import either framework.
"""
import triton
import triton.language as tl


# Source: https://github.com/vllm-project/vllm/blob/6376fda7a95904abf8977011af526c693c4c3004/vllm/model_executor/kernels/linear/mxfp8/rocm_native.py#L30
@triton.jit
def vllm_linear_mxfp8_linear_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_xsm,
    stride_xsk,
    stride_wn,
    stride_wk,
    stride_wsn,
    stride_wsk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    xs_ptrs = xs_ptr + offs_m[:, None] * stride_xsm + offs_sk[None, :] * stride_xsk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    ws_ptrs = ws_ptr + offs_n[:, None] * stride_wsn + offs_sk[None, :] * stride_wsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
        xs = tl.load(xs_ptrs, mask=m_mask[:, None], other=0)
        ws = tl.load(ws_ptrs, mask=n_mask[:, None], other=0)
        acc += tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3")
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        xs_ptrs += (BLOCK_K // 32) * stride_xsk
        ws_ptrs += (BLOCK_K // 32) * stride_wsk

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :]
    )


# Source: https://github.com/vllm-project/vllm/blob/6376fda7a95904abf8977011af526c693c4c3004/vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py#L101
@triton.jit(do_not_specialize=["M"])
def vllm_block32_block32_tiled_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    slot_ptr,
    count_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    N_FIRST: tl.constexpr,
    EVEN_K: tl.constexpr,
    FUSED_SPLITS: tl.constexpr,
):
    if FUSED_SPLITS > 1:
        pid_m, pid_n, tile, pid_k = vllm_block32_split_tile(N, BLOCK_N, FUSED_SPLITS)
        if pid_m * BLOCK_M >= M:
            return
    else:
        # N_FIRST launches output-column tiles fastest, so consecutive
        # programs share an activation tile.
        if N_FIRST:
            pid_n, pid_m = tl.program_id(0), tl.program_id(1)
        else:
            pid_m, pid_n = tl.program_id(0), tl.program_id(1)
        pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    k0 = pid_k * K_PER_SPLIT
    offs_k = k0 + tl.arange(0, BLOCK_K)
    offs_sk = k0 // 32 + tl.arange(0, BLOCK_K // 32)
    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    xs_ptrs = xs_ptr + offs_m[:, None] * (K // 32) + offs_sk[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]
    ws_ptrs = ws_ptr + (offs_n[:, None] // 32) * (K // 32) + offs_sk[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        if EVEN_K:
            x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            xs = tl.load(xs_ptrs, mask=m_mask[:, None], other=0)
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
            ws = tl.load(ws_ptrs, mask=n_mask[:, None], other=0)
        else:
            k_ok = (offs_k + kk) < K
            s_ok = (offs_sk + kk // 32) < K // 32
            x = tl.load(x_ptrs, mask=m_mask[:, None] & k_ok[None, :], other=0.0)
            xs = tl.load(xs_ptrs, mask=m_mask[:, None] & s_ok[None, :], other=0)
            w = tl.load(w_ptrs, mask=n_mask[:, None] & k_ok[None, :], other=0.0)
            ws = tl.load(ws_ptrs, mask=n_mask[:, None] & s_ok[None, :], other=0)
        acc = tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3", acc=acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
        xs_ptrs += BLOCK_K // 32
        ws_ptrs += BLOCK_K // 32
    out_mask = m_mask[:, None] & n_mask[None, :]
    if FUSED_SPLITS > 1:
        vllm_block32_sum_splits(
            acc,
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
            out_mask,
            slot_ptr + tile * (FUSED_SPLITS * BLOCK_M * BLOCK_N),
            count_ptr + tile,
            pid_k,
            FUSED_SPLITS,
        )
    else:
        out_ptrs = (
            out_ptr + pid_k * M * N + offs_m[:, None] * stride_om + offs_n[None, :]
        )
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


# Source: https://github.com/vllm-project/vllm/blob/6376fda7a95904abf8977011af526c693c4c3004/vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py#L67
@triton.jit
def vllm_block32_split_tile(N: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr):
    """(pid_m, pid_n, tile, split) of a 1D grid that keeps a tile's splits on
    one XCD, so they meet in that XCD's L2."""
    pid = tl.program_id(0)
    tile = pid // (8 * SPLITS) * 8 + pid % 8
    grid_n: tl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    return tile // grid_n, tile % grid_n, tile, pid // 8 % SPLITS


# Source: https://github.com/vllm-project/vllm/blob/6376fda7a95904abf8977011af526c693c4c3004/vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py#L77
@triton.jit
def vllm_block32_sum_splits(
    acc, out_ptrs, out_mask, slot_ptr, count_ptr, split, SPLITS: tl.constexpr
):
    """Publish this split's partial; the tile's last arrival sums all of them
    in split order and re-arms the counter."""
    BM: tl.constexpr = acc.shape[0]
    BN: tl.constexpr = acc.shape[1]
    local = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    tl.store(slot_ptr + split * BM * BN + local, acc)
    # The partial must be in L2 before the arrival is counted. The splits
    # share an XCD, so no device-scope release (an L2 writeback) is needed.
    tl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)", "=v,v", [split], dtype=tl.int32, is_pure=False, pack=1
    )
    tl.debug_barrier()
    if tl.atomic_add(count_ptr, 1, sem="acq_rel", scope="cta") == SPLITS - 1:
        total = tl.zeros((BM, BN), dtype=tl.float32)
        for s in tl.static_range(SPLITS):
            total += tl.load(slot_ptr + s * BM * BN + local, cache_modifier=".cv")
        tl.store(out_ptrs, total.to(out_ptrs.dtype.element_ty), mask=out_mask)
        tl.store(count_ptr, 0)


# Source: https://github.com/vllm-project/vllm/blob/6376fda7a95904abf8977011af526c693c4c3004/vllm/model_executor/kernels/linear/mxfp8/rocm_block32_gemm.py#L182
@triton.jit(do_not_specialize=["M"])
def vllm_block32_block32_packed_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    slot_ptr,
    count_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_PACK: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    EVEN_K: tl.constexpr,
    FUSED_SPLITS: tl.constexpr,
):
    if FUSED_SPLITS > 1:
        pid_m, pid_n, tile, pid_k = vllm_block32_split_tile(N, BLOCK_N, FUSED_SPLITS)
        if pid_m * BLOCK_M >= M:
            return
        k_start = pid_k * K_PER_SPLIT
        k_stop = tl.minimum(k_start + K_PER_SPLIT, K)
    else:
        pid_m, pid_n, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        if K_PER_SPLIT >= K:
            k_start = 0
            k_stop = K
        else:
            k_start = pid_k * K_PER_SPLIT
            k_stop = tl.minimum(k_start + K_PER_SPLIT, K)
    # Tile row r is token r // K_PACK reading K panel r % K_PACK (and likewise
    # for weight rows), so one dot covers K_PACK consecutive BLOCK_K slices.
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M * K_PACK) // K_PACK
    x_panel = tl.arange(0, BLOCK_M * K_PACK) % K_PACK
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N * K_PACK) // K_PACK
    w_panel = tl.arange(0, BLOCK_N * K_PACK) % K_PACK
    ks = tl.arange(0, BLOCK_K)
    gs = tl.arange(0, BLOCK_K // 32)
    row_ok = rows < M
    col_ok = cols < N
    acc = tl.zeros((BLOCK_M * K_PACK, BLOCK_N * K_PACK), dtype=tl.float32)
    for base in range(k_start, k_stop, BLOCK_K * K_PACK):
        xk = base + x_panel[:, None] * BLOCK_K + ks[None, :]
        wk = base + w_panel[:, None] * BLOCK_K + ks[None, :]
        xg = base // 32 + x_panel[:, None] * (BLOCK_K // 32) + gs[None, :]
        wg = base // 32 + w_panel[:, None] * (BLOCK_K // 32) + gs[None, :]
        x_mask = row_ok[:, None]
        w_mask = col_ok[:, None]
        xs_mask = row_ok[:, None]
        ws_mask = col_ok[:, None]
        if not EVEN_K:
            x_mask = x_mask & (xk < K)
            w_mask = w_mask & (wk < K)
            xs_mask = xs_mask & (xg < K // 32)
            ws_mask = ws_mask & (wg < K // 32)
        x = tl.load(x_ptr + rows[:, None] * K + xk, mask=x_mask, other=0.0)
        w = tl.load(w_ptr + cols[:, None] * K + wk, mask=w_mask, other=0.0)
        xs = tl.load(xs_ptr + rows[:, None] * (K // 32) + xg, mask=xs_mask, other=0)
        ws = tl.load(
            ws_ptr + (cols[:, None] // 32) * (K // 32) + wg, mask=ws_mask, other=0
        )
        acc = tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3", acc=acc)
    # Keep only products of matching K panels: the block diagonal.
    blocks = acc.reshape(BLOCK_M, K_PACK, BLOCK_N, K_PACK).trans(0, 2, 1, 3)
    pair = tl.arange(0, K_PACK)
    diagonal = tl.where(
        pair[None, None, :, None] == pair[None, None, None, :], blocks, 0.0
    )
    y = tl.sum(tl.sum(diagonal, 3), 2)
    out_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + out_rows[:, None] * stride_om + out_cols[None, :]
    out_mask = (out_rows[:, None] < M) & (out_cols[None, :] < N)
    if FUSED_SPLITS > 1:
        vllm_block32_sum_splits(
            y,
            out_ptrs,
            out_mask,
            slot_ptr + tile * (FUSED_SPLITS * BLOCK_M * BLOCK_N),
            count_ptr + tile,
            pid_k,
            FUSED_SPLITS,
        )
    else:
        tl.store(
            out_ptrs + pid_k * M * N, y.to(out_ptr.dtype.element_ty), mask=out_mask
        )


# Source: https://github.com/vllm-project/vllm/blob/6376fda7a95904abf8977011af526c693c4c3004/vllm/model_executor/layers/fused_moe/experts/mxfp8_native_moe.py#L57
@triton.jit
def vllm_moe_mxfp8_grouped_gemm_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    EM,
    N,
    K,
    num_valid_tokens,
    top_k,
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
    stride_be,
    stride_bn,
    stride_bk,
    stride_bse,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    A_DIV: tl.constexpr,
    MUL_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # ---- Grid-swizzle (super-grouping over M) so consecutive program-ids cover a
    # GROUP_SIZE_M x grid_n super-block. This keeps a touched expert's A rows + its
    # B-weight column-tiles L2/MALL-resident across the N-tiles they share, killing
    # most of the redundant A re-read (the single largest redundant-traffic source:
    # dot_scaled keeps operands in registers, so A is otherwise re-fetched per N-tile).
    # ``num_pid_m`` uses EM (= sorted_token_ids.shape[0]), the SAME bound the grid
    # is sized from, NOT the runtime ``num_tokens_post_padded`` (<= EM). This keeps
    # the swizzle consistent with the grid: ``group_size_m`` is always >= 1 for every
    # launched program, so the ``% group_size_m`` / ``// group_size_m`` below can never
    # hit modulo/division-by-zero. ``num_tokens_post_padded`` is loaded afterwards and
    # only gates the early-return. Mirrors the reference ``fused_moe_kernel``.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if GROUP_SIZE_M == 1:
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    num_post = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_post:
        return

    offs_tid = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_tid).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    a_row = offs_token // A_DIV

    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    as_ptrs = a_scale_ptr + a_row[:, None] * stride_asm + offs_sk[None, :] * stride_ask
    b_ptrs = (
        b_ptr
        + off_e * stride_be
        + offs_n[:, None] * stride_bn
        + offs_k[None, :] * stride_bk
    )
    bs_ptrs = (
        b_scale_ptr
        + off_e * stride_bse
        + offs_n[:, None] * stride_bsn
        + offs_sk[None, :] * stride_bsk
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_mask = offs_n < N
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=n_mask[:, None], other=0.0)
        asc = tl.load(as_ptrs, mask=token_mask[:, None], other=0)
        bsc = tl.load(bs_ptrs, mask=n_mask[:, None], other=0)
        acc += tl.dot_scaled(a, asc, "e4m3", b.T, bsc, "e4m3")

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        as_ptrs += (BLOCK_K // 32) * stride_ask
        bs_ptrs += (BLOCK_K // 32) * stride_bsk

    if MUL_WEIGHT:
        w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc = acc * w[:, None]

    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=token_mask[:, None] & n_mask[None, :],
    )


# Source: https://github.com/vllm-project/vllm/blob/6376fda7a95904abf8977011af526c693c4c3004/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py#L1937
@triton.jit
def vllm_mla_mxfp8_wo_a_bmm_kernel(
    a_ptr,  # [T, G * K] e4m3
    as_ptr,  # [T, G * K // 32] E8M0
    w_ptr,  # [G * R, K] e4m3
    ws_ptr,  # [G * R // SCALE_ROWS, K // 32] E8M0
    out_ptr,  # [T, G * R]
    num_tokens,
    stride_at,
    stride_ast,
    stride_wn,
    stride_wsn,
    stride_out,
    R: tl.constexpr,
    K: tl.constexpr,
    SCALE_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Output columns are group-major, so a BLOCK_N tile never straddles two
    # groups (R % BLOCK_N == 0) and picks its group's K slice of A.
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    group = (pid_n * BLOCK_N) // R
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < num_tokens
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)

    a_ptrs = a_ptr + offs_m[:, None] * stride_at + (group * K + offs_k)[None, :]
    as_ptrs = (
        as_ptr + offs_m[:, None] * stride_ast + (group * (K // 32) + offs_sk)[None, :]
    )
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
    ws_ptrs = ws_ptr + (offs_n // SCALE_ROWS)[:, None] * stride_wsn + offs_sk[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(K // BLOCK_K):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        a_s = tl.load(as_ptrs, mask=m_mask[:, None], other=127)
        w = tl.load(w_ptrs)
        w_s = tl.load(ws_ptrs)
        acc += tl.dot_scaled(a, a_s, "e4m3", w.T, w_s, "e4m3")
        a_ptrs += BLOCK_K
        as_ptrs += BLOCK_K // 32
        w_ptrs += BLOCK_K
        ws_ptrs += BLOCK_K // 32

    tl.store(
        out_ptr + offs_m[:, None] * stride_out + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None],
    )


# Source: https://github.com/vllm-project/vllm/blob/6376fda7a95904abf8977011af526c693c4c3004/vllm/v1/attention/ops/ultraquant/triton_unified_attention.py#L247
@triton.jit
def vllm_attention_kernel_ultraquant_unified_attention_2d(
    output_ptr,
    query_ptr,  # FP8 E4M3, already Hadamard-rotated
    KV_cache_ptr,  # uint8 view
    block_tables_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    sinks_ptr,
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    stride_cache_block: tl.int64,
    stride_cache_pos: tl.int64,
    stride_cache_head: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    num_seqs: tl.int32,
    K_SCALES_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALES_OFFSET: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
    USE_SINKS: tl.constexpr = 0,
    SLIDING_WINDOW: tl.constexpr = 0,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = vllm_attention_find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_d_half = tl.arange(0, HEAD_SIZE_PADDED // 2)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    half_mask = tl.where(offs_d_half * 2 < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=tl.zeros([], tl.float8e4nv),
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        M = tl.load(
            sinks_ptr + query_offset_1,
            mask=query_mask_1,
            other=float("-inf"),
        ).to(tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len

    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = tl.cdiv(max_seq_prefix_len, TILE_SIZE)

    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    query_abs_pos = context_len + query_pos[:, None]
    dummy_tile_mask = tl.full([TILE_SIZE], 1, tl.int1)

    for j in range(tile_start, tile_end - 1):
        seq_offset = j * TILE_SIZE + offs_t
        if TILE_SIZE == BLOCK_SIZE:
            physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j).to(
                tl.int64
            )
            slot_within_block = offs_t.to(tl.int64)
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)
            slot_within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)
        block_base = physical_block_idx * stride_cache_block
        data_bases, k_scales_addrs, val_bases, v_scales_addrs = vllm_attention_ultraquant_slot_addrs(
            block_base,
            slot_within_block,
            kv_head_idx,
            stride_cache_pos,
            stride_cache_head,
            K_SCALES_OFFSET=K_SCALES_OFFSET,
            V_CODES_OFFSET=V_CODES_OFFSET,
            V_SCALES_OFFSET=V_SCALES_OFFSET,
        )

        S = scale * vllm_attention_ultraquant_qk(
            Q,
            KV_cache_ptr,
            data_bases,
            k_scales_addrs,
            offs_d_half,
            half_mask,
            dummy_tile_mask,
            BLOCK_D=HEAD_SIZE_PADDED,
            HEAD_DIM=HEAD_SIZE,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=True,
        )
        seq_mask = seq_offset[None, :] <= query_abs_pos
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (
                (query_abs_pos - seq_offset[None, :]) < SLIDING_WINDOW
            )
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += vllm_attention_ultraquant_pv(
            P,
            KV_cache_ptr,
            val_bases,
            v_scales_addrs,
            offs_d,
            dim_mask,
            dummy_tile_mask,
            HEAD_DIM=HEAD_SIZE,
            BLOCK_D=HEAD_SIZE_PADDED,
            BLOCK_M=BLOCK_M,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=True,
            UE8M0_BIAS_C=UE8M0_BIAS_C,
        )

    # Tail tile (masked)
    if tile_end > tile_start:
        j = tile_end - 1
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        if TILE_SIZE == BLOCK_SIZE:
            physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j).to(
                tl.int64
            )
            slot_within_block = offs_t.to(tl.int64)
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)
            slot_within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)
        block_base = physical_block_idx * stride_cache_block
        data_bases, k_scales_addrs, val_bases, v_scales_addrs = vllm_attention_ultraquant_slot_addrs(
            block_base,
            slot_within_block,
            kv_head_idx,
            stride_cache_pos,
            stride_cache_head,
            K_SCALES_OFFSET=K_SCALES_OFFSET,
            V_CODES_OFFSET=V_CODES_OFFSET,
            V_SCALES_OFFSET=V_SCALES_OFFSET,
        )

        S = scale * vllm_attention_ultraquant_qk(
            Q,
            KV_cache_ptr,
            data_bases,
            k_scales_addrs,
            offs_d_half,
            half_mask,
            tile_mask,
            BLOCK_D=HEAD_SIZE_PADDED,
            HEAD_DIM=HEAD_SIZE,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=False,
        )
        seq_mask = seq_offset[None, :] <= query_abs_pos
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (
                (query_abs_pos - seq_offset[None, :]) < SLIDING_WINDOW
            )
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += vllm_attention_ultraquant_pv(
            P,
            KV_cache_ptr,
            val_bases,
            v_scales_addrs,
            offs_d,
            dim_mask,
            tile_mask,
            HEAD_DIM=HEAD_SIZE,
            BLOCK_D=HEAD_SIZE_PADDED,
            BLOCK_M=BLOCK_M,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=False,
            UE8M0_BIAS_C=UE8M0_BIAS_C,
        )

    acc = acc / L[:, None]

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )
    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


# Source: https://github.com/vllm-project/vllm/blob/6376fda7a95904abf8977011af526c693c4c3004/vllm/v1/attention/ops/turboquant_soa/triton_turboquant_unified_attention.py#L53
@triton.jit
def vllm_attention_find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


# Source: https://github.com/vllm-project/vllm/blob/6376fda7a95904abf8977011af526c693c4c3004/vllm/v1/attention/ops/ultraquant/triton_unified_attention.py#L218
@triton.jit
def vllm_attention_ultraquant_slot_addrs(
    block_base,
    slot_within_block,
    kv_head_idx,
    stride_cache_pos: tl.int64,
    stride_cache_head: tl.int64,
    K_SCALES_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALES_OFFSET: tl.constexpr,
):
    data_bases = (
        block_base
        + slot_within_block * stride_cache_pos
        + tl.cast(kv_head_idx, tl.int64) * stride_cache_head
    )
    return (
        data_bases,
        data_bases + K_SCALES_OFFSET,
        data_bases + V_CODES_OFFSET,
        data_bases + V_SCALES_OFFSET,
    )


# Source: https://github.com/vllm-project/vllm/blob/6376fda7a95904abf8977011af526c693c4c3004/vllm/v1/attention/ops/ultraquant/triton_unified_attention.py#L139
@triton.jit
def vllm_attention_ultraquant_pv(
    P,
    KV_cache_ptr,
    val_bases,
    v_scales_addrs,
    d_offs,
    d_mask,
    tile_mask,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    UNMASKED: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
):
    V = vllm_attention_ultraquant_load_v_tile(
        KV_cache_ptr,
        val_bases,
        v_scales_addrs,
        d_offs,
        d_mask,
        tile_mask,
        OUT_DTYPE=tl.bfloat16,
        HEAD_DIM=HEAD_DIM,
        BLOCK_D=BLOCK_D,
        GROUP_SIZE_C=GROUP_SIZE_C,
        N_GROUPS_C=N_GROUPS_C,
        TILE_SIZE=TILE_SIZE,
        UNMASKED=UNMASKED,
        UE8M0_BIAS_C=UE8M0_BIAS_C,
    )
    _ = BLOCK_M
    return tl.dot(P.to(tl.bfloat16), V, out_dtype=tl.float32)


# Source: https://github.com/vllm-project/vllm/blob/6376fda7a95904abf8977011af526c693c4c3004/vllm/v1/attention/ops/ultraquant/triton_unified_attention.py#L92
@triton.jit
def vllm_attention_ultraquant_load_v_tile(
    KV_cache_ptr,
    val_bases,
    v_scales_addrs,
    d_offs,
    d_mask,
    tile_mask,
    OUT_DTYPE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    UNMASKED: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
):
    """Load and dequant a V tile with arithmetic FP4 decode."""
    half_idx = d_offs // 2
    nibble_shift = (d_offs % 2) * 4
    addrs = val_bases[:, None] + half_idx[None, :]
    if UNMASKED:
        byte_raw = tl.load(KV_cache_ptr + addrs, mask=d_mask[None, :], other=0).to(
            tl.int32
        )
    else:
        byte_raw = tl.load(
            KV_cache_ptr + addrs,
            mask=tile_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
    codes = (byte_raw >> nibble_shift[None, :]) & 0xF
    grp = tl.arange(0, N_GROUPS_C)
    scale_addrs = v_scales_addrs[:, None] + grp[None, :]
    if UNMASKED:
        scale_bytes = tl.load(KV_cache_ptr + scale_addrs).to(tl.int32)
    else:
        scale_bytes = tl.load(
            KV_cache_ptr + scale_addrs, mask=tile_mask[:, None], other=0
        ).to(tl.int32)
    scale_exp = scale_bytes - UE8M0_BIAS_C
    fp4_vals = vllm_attention_ultraquant_fp4_decode_arith(codes)
    scales = tl.where(scale_bytes == 0, 0.0, tl.exp2(tl.cast(scale_exp, tl.float32)))
    V_g = tl.reshape(fp4_vals, [TILE_SIZE, N_GROUPS_C, GROUP_SIZE_C])
    return tl.reshape(V_g * scales[:, :, None], [TILE_SIZE, BLOCK_D]).to(OUT_DTYPE)


# Source: https://github.com/vllm-project/vllm/blob/6376fda7a95904abf8977011af526c693c4c3004/vllm/v1/attention/ops/ultraquant/triton_unified_attention.py#L39
@triton.jit
def vllm_attention_ultraquant_fp4_decode_arith(codes):
    """Arithmetic FP4 E2M1 decode, bit-exact to format.FP4_BITS_TO_VALUE."""
    mag = codes & 7
    e = mag >> 1
    m = mag & 1
    exp_field = 126 + e
    mant_field = tl.where(e != 0, m, 0) << 22
    bits = (exp_field << 23) | mant_field
    magval = bits.to(tl.float32, bitcast=True)
    magval = tl.where(mag == 0, 0.0, magval)
    signf = tl.where((codes & 8) != 0, -1.0, 1.0)
    return magval * signf


# Source: https://github.com/vllm-project/vllm/blob/6376fda7a95904abf8977011af526c693c4c3004/vllm/v1/attention/ops/ultraquant/triton_unified_attention.py#L177
@triton.jit
def vllm_attention_ultraquant_qk(
    Q,
    KV_cache_ptr,
    data_bases,
    k_scales_addrs,
    offs_d_half,
    half_mask,
    tile_mask,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    UNMASKED: tl.constexpr,
):
    K_T_packed, K_scales = vllm_attention_ultraquant_load_k_packed(
        KV_cache_ptr,
        data_bases,
        k_scales_addrs,
        offs_d_half,
        half_mask,
        tile_mask,
        BLOCK_D=BLOCK_D,
        HEAD_DIM=HEAD_DIM,
        GROUP_SIZE_C=GROUP_SIZE_C,
        N_GROUPS_C=N_GROUPS_C,
        TILE_SIZE=TILE_SIZE,
        UNMASKED=UNMASKED,
    )
    return tl.dot_scaled(
        Q,
        None,
        "e4m3",
        K_T_packed,
        K_scales,
        "e2m1",
        out_dtype=tl.float32,
    )


# Source: https://github.com/vllm-project/vllm/blob/6376fda7a95904abf8977011af526c693c4c3004/vllm/v1/attention/ops/ultraquant/triton_unified_attention.py#L54
@triton.jit
def vllm_attention_ultraquant_load_k_packed(
    KV_cache_ptr,
    data_bases,
    k_scales_addrs,
    d_half_offs,
    half_mask,
    tile_mask,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    UNMASKED: tl.constexpr,
):
    """Load packed FP4 K codes and UE8M0 scales for scaled QK MFMA."""
    addrs = data_bases[:, None] + d_half_offs[None, :]
    if UNMASKED:
        K_codes = tl.load(KV_cache_ptr + addrs, mask=half_mask[None, :], other=0)
    else:
        K_codes = tl.load(
            KV_cache_ptr + addrs,
            mask=tile_mask[:, None] & half_mask[None, :],
            other=0,
        )
    K_T_packed = tl.trans(K_codes)
    grp = tl.arange(0, N_GROUPS_C)
    scale_addrs = k_scales_addrs[:, None] + grp[None, :]
    if UNMASKED:
        K_scales = tl.load(KV_cache_ptr + scale_addrs)
    else:
        K_scales = tl.load(KV_cache_ptr + scale_addrs, mask=tile_mask[:, None], other=0)
    _ = HEAD_DIM
    _ = GROUP_SIZE_C
    _ = BLOCK_D
    return K_T_packed, K_scales


# Source: https://github.com/vllm-project/vllm/blob/6376fda7a95904abf8977011af526c693c4c3004/vllm/v1/attention/ops/ultraquant/triton_unified_attention.py#L537
@triton.jit
def vllm_attention_kernel_ultraquant_unified_attention_3d(
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,  # FP8 E4M3, already Hadamard-rotated
    KV_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    sinks_ptr,
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    stride_cache_block: tl.int64,
    stride_cache_pos: tl.int64,
    stride_cache_head: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    num_seqs: tl.int32,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    K_SCALES_OFFSET: tl.constexpr,
    V_CODES_OFFSET: tl.constexpr,
    V_SCALES_OFFSET: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
    N_GROUPS_C: tl.constexpr,
    UE8M0_BIAS_C: tl.constexpr,
    USE_SINKS: tl.constexpr = 0,
    SLIDING_WINDOW: tl.constexpr = 0,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    seq_idx = vllm_attention_find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    tiles_per_segment = tl.cdiv(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_d_half = tl.arange(0, HEAD_SIZE_PADDED // 2)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    half_mask = tl.where(offs_d_half * 2 < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=tl.zeros([], tl.float8e4nv),
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS and segm_idx == 0:
        M = tl.load(
            sinks_ptr + query_offset_1,
            mask=query_mask_1,
            other=float("-inf"),
        ).to(tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = tl.cdiv(max_seq_prefix_len, TILE_SIZE)

    tile_lo = segm_idx * tiles_per_segment
    tile_hi = tl.minimum((segm_idx + 1) * tiles_per_segment, num_tiles)

    if SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        swa_tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        swa_tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)
        tile_lo = tl.maximum(tile_lo, swa_tile_start)
        tile_hi = tl.minimum(tile_hi, swa_tile_end)

    query_abs_pos = context_len + query_pos[:, None]
    dummy_tile_mask = tl.full([TILE_SIZE], 1, tl.int1)
    tail = tile_hi - 1

    for j in range(tile_lo, tile_hi - 1):
        seq_offset = j * TILE_SIZE + offs_t
        if TILE_SIZE == BLOCK_SIZE:
            physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j).to(
                tl.int64
            )
            slot_within_block = offs_t.to(tl.int64)
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)
            slot_within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)
        block_base = physical_block_idx * stride_cache_block
        data_bases, k_scales_addrs, val_bases, v_scales_addrs = vllm_attention_ultraquant_slot_addrs(
            block_base,
            slot_within_block,
            kv_head_idx,
            stride_cache_pos,
            stride_cache_head,
            K_SCALES_OFFSET=K_SCALES_OFFSET,
            V_CODES_OFFSET=V_CODES_OFFSET,
            V_SCALES_OFFSET=V_SCALES_OFFSET,
        )

        S = scale * vllm_attention_ultraquant_qk(
            Q,
            KV_cache_ptr,
            data_bases,
            k_scales_addrs,
            offs_d_half,
            half_mask,
            dummy_tile_mask,
            BLOCK_D=HEAD_SIZE_PADDED,
            HEAD_DIM=HEAD_SIZE,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=True,
        )
        seq_mask = seq_offset[None, :] <= query_abs_pos
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (
                (query_abs_pos - seq_offset[None, :]) < SLIDING_WINDOW
            )
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += vllm_attention_ultraquant_pv(
            P,
            KV_cache_ptr,
            val_bases,
            v_scales_addrs,
            offs_d,
            dim_mask,
            dummy_tile_mask,
            HEAD_DIM=HEAD_SIZE,
            BLOCK_D=HEAD_SIZE_PADDED,
            BLOCK_M=BLOCK_M,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=True,
            UE8M0_BIAS_C=UE8M0_BIAS_C,
        )

    # Tail
    if tile_lo < tile_hi:
        j = tail
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        if TILE_SIZE == BLOCK_SIZE:
            physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j).to(
                tl.int64
            )
            slot_within_block = offs_t.to(tl.int64)
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)
            slot_within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)
        block_base = physical_block_idx * stride_cache_block
        data_bases, k_scales_addrs, val_bases, v_scales_addrs = vllm_attention_ultraquant_slot_addrs(
            block_base,
            slot_within_block,
            kv_head_idx,
            stride_cache_pos,
            stride_cache_head,
            K_SCALES_OFFSET=K_SCALES_OFFSET,
            V_CODES_OFFSET=V_CODES_OFFSET,
            V_SCALES_OFFSET=V_SCALES_OFFSET,
        )

        S = scale * vllm_attention_ultraquant_qk(
            Q,
            KV_cache_ptr,
            data_bases,
            k_scales_addrs,
            offs_d_half,
            half_mask,
            tile_mask,
            BLOCK_D=HEAD_SIZE_PADDED,
            HEAD_DIM=HEAD_SIZE,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=False,
        )
        seq_mask = seq_offset[None, :] <= query_abs_pos
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (
                (query_abs_pos - seq_offset[None, :]) < SLIDING_WINDOW
            )
        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += vllm_attention_ultraquant_pv(
            P,
            KV_cache_ptr,
            val_bases,
            v_scales_addrs,
            offs_d,
            dim_mask,
            tile_mask,
            HEAD_DIM=HEAD_SIZE,
            BLOCK_D=HEAD_SIZE_PADDED,
            BLOCK_M=BLOCK_M,
            GROUP_SIZE_C=GROUP_SIZE_C,
            N_GROUPS_C=N_GROUPS_C,
            TILE_SIZE=TILE_SIZE,
            UNMASKED=False,
            UE8M0_BIAS_C=UE8M0_BIAS_C,
        )

    # Write segment partials for stage-2 reduce.
    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


# Source: https://github.com/sgl-project/sglang/blob/610a74d6485539b07aa772e31199257ca3d7c1d0/python/sglang/kernels/ops/gemm/mxfp8_amd_gfx95.py#L36
@triton.jit
def sglang_linear_mxfp8_linear_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_xsm,
    stride_xsk,
    stride_wn,
    stride_wk,
    stride_wsn,
    stride_wsk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    xs_ptrs = xs_ptr + offs_m[:, None] * stride_xsm + offs_sk[None, :] * stride_xsk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    ws_ptrs = ws_ptr + offs_n[:, None] * stride_wsn + offs_sk[None, :] * stride_wsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
        xs = tl.load(xs_ptrs, mask=m_mask[:, None], other=0)
        ws = tl.load(ws_ptrs, mask=n_mask[:, None], other=0)
        acc += tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3")
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        xs_ptrs += (BLOCK_K // 32) * stride_xsk
        ws_ptrs += (BLOCK_K // 32) * stride_wsk

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        o_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


# Source: https://github.com/sgl-project/sglang/blob/610a74d6485539b07aa772e31199257ca3d7c1d0/python/sglang/kernels/ops/gemm/mxfp8_native_amd_gfx95.py#L207
@triton.jit
def sglang_shuffled_mxfp8_shuffled_gemm_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xsm,
    stride_wsn,
    stride_om,
    k_per_split,
    K_PAD,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUT_F32: tl.constexpr,
    K_TAIL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # split-K partition; partials are summed in fixed order outside
    pid_k = tl.program_id(2)
    k0 = pid_k * k_per_split
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    m_mask = offs_m < M
    n_mask = offs_n < N

    # a 2 KB lane-order block is (g1, g0, row, half, 16 B) with lane 16 * (2 g1 + g0) + row holding
    # K [32 g1 + 16 g0 + 64 half, +16); put it back into row-major [BLOCK_N, BLOCK_K] in registers
    T: tl.constexpr = BLOCK_N // 16
    S: tl.constexpr = BLOCK_K // 128
    nsteps = K_PAD // 128
    blk = tl.arange(0, T * S)
    blk_tile = pid_n * T + blk // S
    blk_mask = blk_tile < N // 16  # the last N block may hold fewer 16-row tiles
    blk_off = blk_tile * (nsteps * 2048) + (blk % S + k0 // 128) * 2048
    w_ptrs = w_ptr + blk_off[:, None] + tl.arange(0, 2048)[None, :]

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k0 + offs_k[None, :]
    xs_ptrs = xs_ptr + offs_m[:, None] * stride_xsm + k0 // 32 + offs_sk[None, :]
    ws_ptrs = (
        ws_ptr + (offs_n // 32)[:, None] * stride_wsn + k0 // 32 + offs_sk[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for it in range(0, k_per_split // BLOCK_K):
        if K_TAIL:
            # the weight's zero-padded K tail meets a masked activation
            k_cur = k0 + it * BLOCK_K
            x_mask = m_mask[:, None] & (k_cur + offs_k < K)[None, :]
            xs_mask = m_mask[:, None] & (k_cur // 32 + offs_sk < K // 32)[None, :]
        else:
            x_mask = m_mask[:, None]
            xs_mask = m_mask[:, None]
        x = tl.load(x_ptrs, mask=x_mask, other=0)
        w_raw = tl.load(w_ptrs, mask=blk_mask[:, None], other=0)  # [T * S, 2048]
        w7 = tl.reshape(w_raw, (T, S, 2, 2, 16, 2, 16))  # (t, s, g1, g0, row, half, e)
        w7 = tl.permute(w7, (0, 4, 1, 5, 2, 3, 6))  # (t, row, s, half, g1, g0, e)
        w = tl.reshape(w7, (BLOCK_N, BLOCK_K))
        xs = tl.load(xs_ptrs, mask=xs_mask, other=127)
        ws = tl.load(ws_ptrs, mask=n_mask[:, None], other=127)
        acc = tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3", acc)
        x_ptrs += BLOCK_K
        xs_ptrs += BLOCK_K // 32
        w_ptrs += S * 2048
        ws_ptrs += BLOCK_K // 32

    o_ptrs = (
        out_ptr
        + pid_k.to(tl.int64) * M * N
        + offs_m[:, None] * stride_om
        + offs_n[None, :]
    )
    if OUT_F32:
        tl.store(o_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])
    else:
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


# Source: https://github.com/sgl-project/sglang/blob/610a74d6485539b07aa772e31199257ca3d7c1d0/python/sglang/kernels/ops/moe/mxfp8_moe_amd_gfx95.py#L33
@triton.jit
def sglang_moe_mxfp8_grouped_gemm_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    E,
    N,
    K,
    num_valid_tokens,
    top_k,
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
    stride_be,
    stride_bn,
    stride_bk,
    stride_bse,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    A_DIV: tl.constexpr,
    MUL_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_post = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_post:
        return

    offs_tid = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_tid).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    valid_expert = (off_e >= 0) & (off_e < E)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    a_row = offs_token // A_DIV

    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    as_ptrs = a_scale_ptr + a_row[:, None] * stride_asm + offs_sk[None, :] * stride_ask
    b_ptrs = (
        b_ptr
        + off_e * stride_be
        + offs_n[:, None] * stride_bn
        + offs_k[None, :] * stride_bk
    )
    bs_ptrs = (
        b_scale_ptr
        + off_e * stride_bse
        + offs_n[:, None] * stride_bsn
        + offs_sk[None, :] * stride_bsk
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_mask = offs_n < N
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=valid_expert & n_mask[:, None], other=0.0)
        asc = tl.load(as_ptrs, mask=token_mask[:, None], other=0)
        bsc = tl.load(bs_ptrs, mask=valid_expert & n_mask[:, None], other=0)
        acc += tl.dot_scaled(a, asc, "e4m3", b.T, bsc, "e4m3")

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        as_ptrs += (BLOCK_K // 32) * stride_ask
        bs_ptrs += (BLOCK_K // 32) * stride_bsk

    if MUL_WEIGHT:
        w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc = acc * w[:, None]

    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=token_mask[:, None] & n_mask[None, :],
    )
