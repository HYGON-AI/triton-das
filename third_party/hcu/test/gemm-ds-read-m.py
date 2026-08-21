# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""ds_read_m buffer-load GEMM correctness (default-on).

Covers:
  - fp16 / bf16 / int8 / fp8, BN in {32,64,128}, various num_warps
  - BK=32/64 multi-tile (BK independent of the old BK==32 gate)
  - panel-major preferred path, flat fallback, and multi-slot pipeline allocation
  - row-col B keeps the generic local-load fallback
  - final ISA must contain ds_read_m when the case is expected to hit the fast path

Run:
  pytest third_party/hcu/test/gemm-ds-read-m.py -q
  # or: python3 third_party/hcu/test/gemm-ds-read-m.py
"""

from __future__ import annotations

import os
import re

os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "1")
# Default-on: do not force-enable; clear any explicit disable/enable override.
os.environ.pop("TRITON_HCU_ENABLE_DS_READ_M", None)

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hip

DS_READ_M_RE = re.compile(
    r"ds_read_m32x16_b16|ds_read_m32x32_b8|llvm\.hcu\.ds\.read\.m",
    re.IGNORECASE,
)
PANEL_LAYOUT_DEF_RE = re.compile(
    r"(#[A-Za-z0-9_.$-]+)\s*=\s*#ttg\.shared_linear<")


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  stride_am: tl.constexpr, stride_ak: tl.constexpr,
                  stride_bk: tl.constexpr, stride_bn: tl.constexpr,
                  stride_cm: tl.constexpr, stride_cn: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr, ACC_DTYPE: tl.constexpr,
                  DOT_OUT_DTYPE: tl.constexpr, STORE_DTYPE: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=DOT_OUT_DTYPE)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(STORE_DTYPE)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def _has_ds_read_m(asm) -> bool:
    for key in ("amdgcn", "isa"):
        text = asm.get(key) if asm else None
        if isinstance(text, str) and DS_READ_M_RE.search(text):
            return True
    return False


def _ttgir(asm) -> str:
    text = asm.get("ttgir") if asm else None
    return text if isinstance(text, str) else ""


def _has_panel_allocation(ttgir: str, block_k: int, block_n: int,
                          num_buffers=None) -> bool:
    panel_layouts = set(PANEL_LAYOUT_DEF_RE.findall(ttgir))
    if not panel_layouts:
        return False
    leading_shape = (f"{num_buffers}x" if num_buffers is not None
                     else r"(?:\d+x)?")
    shape = re.compile(
        rf"!ttg\.memdesc<{leading_shape}{block_k}x{block_n}x")
    for line in ttgir.splitlines():
        if "ttg.local_alloc" not in line or not shape.search(line):
            continue
        if any(f", {layout}," in line for layout in panel_layouts):
            return True
    return False


def run_case(dtype, block_m, block_n, block_k, num_warps, *, multi_tile=False,
             require_ds_read_m=True, num_stages=1,
             require_pipeline_buffers=None, b_k_contiguous=False,
             expect_panel=None):
    if multi_tile:
        m = block_m * 2
        n = block_n * 2
        k = max(block_k * 4, 256)
    else:
        m, n, k = block_m, block_n, block_k

    is_fp8 = dtype in (torch.float8_e5m2, torch.float8_e4m3fn)
    if dtype == torch.int8:
        a = torch.randint(-3, 4, (m, k), device="cuda", dtype=torch.int8)
        b = torch.randint(-3, 4, (k, n), device="cuda", dtype=torch.int8)
        c = torch.empty((m, n), device="cuda", dtype=torch.int32)
        acc_dtype = tl.int32
        dot_out_dtype = tl.int32
        store_dtype = tl.int32
    elif is_fp8:
        a = (
            torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.25
        ).to(dtype)
        b = (
            torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.25
        ).to(dtype)
        c = torch.empty((m, n), device="cuda", dtype=torch.float16)
        acc_dtype = tl.float32
        dot_out_dtype = tl.float32
        store_dtype = tl.float16
    else:
        a = torch.randn((m, k), device="cuda", dtype=dtype)
        b = torch.randn((k, n), device="cuda", dtype=dtype)
        c = torch.empty((m, n), device="cuda", dtype=dtype)
        acc_dtype = tl.float32
        dot_out_dtype = tl.float32
        store_dtype = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16

    if b_k_contiguous:
        # Keep logical B[K,N] unchanged while making K the contiguous physical
        # dimension, matching the row-col GEMM load direction.
        b = b.T.contiguous().T

    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    h = matmul_kernel[grid](
        a, b, c, m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        ACC_DTYPE=acc_dtype,
        DOT_OUT_DTYPE=dot_out_dtype,
        STORE_DTYPE=store_dtype,
        num_warps=num_warps, num_stages=num_stages)

    if require_ds_read_m:
        assert _has_ds_read_m(h.asm), (
            f"expected ds_read_m in asm for "
            f"dtype={dtype} BM={block_m} BN={block_n} BK={block_k} warps={num_warps}")
    if expect_panel is None:
        panel_capacity = (block_m // 16) * (block_n // 16)
        power_of_two_tile = (
            block_k > 0 and block_k & (block_k - 1) == 0 and
            block_n > 0 and block_n & (block_n - 1) == 0
        )
        expect_panel = (
            require_ds_read_m and not b_k_contiguous and power_of_two_tile and
            num_warps <= panel_capacity
        )
    has_panel = _has_panel_allocation(_ttgir(h.asm), block_k, block_n)
    if expect_panel:
        assert has_panel, (
                f"expected panel-major shared_linear layout for "
                f"dtype={dtype} BM={block_m} BN={block_n} "
                f"BK={block_k} warps={num_warps}")
    else:
        assert not has_panel, (
            f"expected non-panel fallback for dtype={dtype} BM={block_m} "
            f"BN={block_n} BK={block_k} warps={num_warps}")
    if require_pipeline_buffers is not None:
        assert _has_panel_allocation(
            _ttgir(h.asm), block_k, block_n, require_pipeline_buffers
        ), (
            f"expected a {require_pipeline_buffers}-slot panel allocation for "
            f"dtype={dtype} BM={block_m} BN={block_n} "
            f"BK={block_k} warps={num_warps}")

    if dtype == torch.int8:
        ref = torch.matmul(a.cpu().to(torch.int32), b.cpu().to(torch.int32)).cuda()
        torch.testing.assert_close(c, ref, rtol=0, atol=0)
    elif is_fp8:
        ref = torch.matmul(a.to(torch.float16), b.to(torch.float16))
        torch.testing.assert_close(c, ref, rtol=5e-2, atol=1e-1)
    elif dtype == torch.bfloat16:
        ref = torch.matmul(a, b)
        rtol = atol = 5e-2 if multi_tile else 3e-2
        torch.testing.assert_close(c, ref, rtol=rtol, atol=atol)
    else:
        ref = torch.matmul(a, b)
        rtol = atol = 2e-2 if multi_tile else 1e-2
        torch.testing.assert_close(c, ref, rtol=rtol, atol=atol)


# Single-CTA tile: fp16 + int8 + fp8, cover BN / warps.
DS_READ_M_CASES = [
    # fp16 / b16 (10)
    (torch.float16, 32, 1),
    (torch.float16, 32, 2),
    (torch.float16, 32, 4),
    (torch.float16, 64, 2),
    (torch.float16, 64, 4),
    (torch.float16, 64, 8),
    (torch.float16, 128, 1),
    (torch.float16, 128, 2),
    (torch.float16, 128, 4),
    (torch.float16, 128, 8),
    # bf16 / b16
    (torch.bfloat16, 64, 4),
    (torch.bfloat16, 128, 8),
    # int8 / b8 (10)
    (torch.int8, 32, 1),
    (torch.int8, 32, 2),
    (torch.int8, 32, 4),
    (torch.int8, 64, 2),
    (torch.int8, 64, 4),
    (torch.int8, 64, 8),
    (torch.int8, 128, 1),
    (torch.int8, 128, 2),
    (torch.int8, 128, 4),
    (torch.int8, 128, 8),
    # fp8 / b8
    (torch.float8_e5m2, 64, 4),
    (torch.float8_e4m3fn, 64, 4),
]

# Multi-tile: BK=32/64, 2/4/8 warps. int8+BK=64 catches B8 nRep/kTile packing.
DS_READ_M_BLOCK_K_CASES = [
    (torch.float16, 64, 64, 32, 2),
    (torch.float16, 64, 64, 64, 2),
    (torch.float16, 128, 128, 32, 4),
    (torch.float16, 128, 128, 64, 4),
    (torch.float16, 256, 128, 64, 4),
    (torch.float16, 128, 128, 32, 8),
    (torch.float16, 128, 128, 64, 8),
    (torch.float16, 128, 256, 32, 8),
    (torch.float16, 128, 256, 64, 8),
    (torch.float16, 256, 128, 32, 8),
    (torch.float16, 256, 128, 64, 8),
    (torch.float16, 256, 256, 32, 8),
    (torch.float16, 256, 256, 64, 8),
    (torch.bfloat16, 128, 128, 32, 8),
    (torch.bfloat16, 256, 256, 64, 8),
    (torch.int8, 64, 64, 64, 2),
    # (torch.int8, 128, 128, 64, 4),
    (torch.int8, 32, 128, 64, 4),
    (torch.int8, 64, 128, 128, 4),
    (torch.float8_e5m2, 128, 128, 64, 4),
    (torch.float8_e4m3fn, 128, 128, 64, 4),
]


@pytest.mark.skipif(not is_hip(), reason="ds_read_m gemm test is HIP/HCU-specific")
@pytest.mark.parametrize("dtype,n,num_warps", DS_READ_M_CASES)
def test_ds_read_m_gemm(dtype, n, num_warps):
    torch.manual_seed(0)
    # Single tile: M=16, K=32 (b16/b8 instr K coverage).
    run_case(dtype, block_m=16, block_n=n, block_k=32, num_warps=num_warps)


@pytest.mark.skipif(not is_hip(), reason="ds_read_m gemm test is HIP/HCU-specific")
@pytest.mark.parametrize("dtype,block_m,block_n,block_k,num_warps", DS_READ_M_BLOCK_K_CASES)
def test_ds_read_m_block_k(dtype, block_m, block_n, block_k, num_warps):
    torch.manual_seed(0)
    run_case(dtype, block_m, block_n, block_k, num_warps, multi_tile=True)


@pytest.mark.skipif(not is_hip(), reason="ds_read_m gemm test is HIP/HCU-specific")
@pytest.mark.parametrize("dtype,num_stages,num_buffers", [
    (torch.float16, 3, 2),
    (torch.bfloat16, 3, 2),
    (torch.int8, 3, 2),
    (torch.float16, 4, 3),
    (torch.int8, 4, 3),
])
def test_ds_read_m_panel_pipeline_buffers(dtype, num_stages, num_buffers):
    torch.manual_seed(0)
    block_k = 32 if dtype == torch.float16 else 64
    run_case(dtype, 128, 128, block_k, 8, multi_tile=True,
             num_stages=num_stages, require_pipeline_buffers=num_buffers)


@pytest.mark.skipif(not is_hip(), reason="ds_read_m gemm test is HIP/HCU-specific")
def test_ds_read_m_row_col_fallback():
    torch.manual_seed(0)
    run_case(torch.float16, 128, 128, 32, 8, multi_tile=True,
             num_stages=2, require_ds_read_m=False,
             b_k_contiguous=True, expect_panel=False)


def main():
    torch.manual_seed(0)
    for dtype, n, num_warps in DS_READ_M_CASES:
        run_case(dtype, block_m=16, block_n=n, block_k=32, num_warps=num_warps)
        print(f"PASS: dtype={dtype}, M=16, N={n}, K=32, num_warps={num_warps}")
    for dtype, bm, bn, bk, nw in DS_READ_M_BLOCK_K_CASES:
        run_case(dtype, bm, bn, bk, nw, multi_tile=True)
        print(f"PASS: multi-tile dtype={dtype} BM={bm} BN={bn} BK={bk} warps={nw}")


if __name__ == "__main__":
    main()
