"""ds_read_m buffer-load GEMM correctness (default-on).

Covers:
  - fp16 / bf16 / int8 / fp8, BN in {32,64,128}, various num_warps
  - BK=32/64 multi-tile (BK independent of the old BK==32 gate)
  - panel-major staging, f32 b32 flat staging, and multi-slot pipeline allocation
  - row-col B keeps the generic K-major staging fallback
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
# Anchor to the MFMA attribute: blocked load layouts also carry a
# warpsPerCTA field that must not be confused with the dot's warp split.
MFMA_WARPS_RE = re.compile(
    r"#ttg\.amd_mfma<\{[^}]*?warpsPerCTA\s*=\s*\[([0-9]+)\s*,\s*([0-9]+)\]")

PANEL_LAYOUT_DEF_RE = re.compile(
    r"(#[A-Za-z0-9_.$-]+)\s*=\s*#ttg\.shared_linear<")
SHARED_LINEAR_DEF_RE = re.compile(
    r"(#[A-Za-z0-9_.$-]+)\s*=\s*#ttg\.shared_linear<\{offset\s*=\s*\[(.*?)\]\}")
FLAT_B16_LAYOUT_DEF_RE = re.compile(
    r"(#[A-Za-z0-9_.$-]+)\s*=\s*#ttg\.swizzled_shared<"
    r"\{vec\s*=\s*8,\s*perPhase\s*=\s*2,\s*maxPhase\s*=\s*2,\s*"
    r"order\s*=\s*\[1,\s*0\]\}>")
FLAT_B8_LAYOUT_DEF_RE = re.compile(
    r"(#[A-Za-z0-9_.$-]+)\s*=\s*#ttg\.swizzled_shared<"
    r"\{vec\s*=\s*16,\s*perPhase\s*=\s*1,\s*maxPhase\s*=\s*[1-4],\s*"
    r"order\s*=\s*\[1,\s*0\]\}>")
FLAT_B32_LAYOUT_DEF_RE = re.compile(
    r"(#[A-Za-z0-9_.$-]+)\s*=\s*#ttg\.swizzled_shared<"
    r"\{vec\s*=\s*4,\s*perPhase\s*=\s*1,\s*maxPhase\s*=\s*1,\s*"
    r"order\s*=\s*\[1,\s*0\]\}>")


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  stride_am: tl.constexpr, stride_ak: tl.constexpr,
                  stride_bk: tl.constexpr, stride_bn: tl.constexpr,
                  stride_cm: tl.constexpr, stride_cn: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr, ACC_DTYPE: tl.constexpr,
                  DOT_OUT_DTYPE: tl.constexpr, STORE_DTYPE: tl.constexpr,
                  INPUT_PRECISION: tl.constexpr = "tf32"):
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
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) &
                    (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) &
                    (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=DOT_OUT_DTYPE,
                     input_precision=INPUT_PRECISION)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    c = acc.to(STORE_DTYPE)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def _asm_text(asm) -> str:
    """Combined assembly text for opcode checks (amdgcn or isa key)."""
    text = ""
    for key in ("amdgcn", "isa"):
        value = asm.get(key) if asm else None
        if isinstance(value, str):
            text += value
    return text


def _has_ds_read_m(asm) -> bool:
    return bool(DS_READ_M_RE.search(_asm_text(asm)))


def _has_expected_ds_read_m(asm, element_bits: int) -> bool:
    opcode = {
        8: r"ds_read_m32x32_b8|llvm\.hcu\.ds\.read\.m32x32\.b8",
        16: r"ds_read_m32x16_b16|llvm\.hcu\.ds\.read\.m32x16\.b16",
        32: r"ds_read_m32x8_b32|llvm\.hcu\.ds\.read\.m32x8\.b32",
    }[element_bits]
    return bool(re.search(opcode, _asm_text(asm), re.IGNORECASE))


def _dtypes_for(dtype):
    """(acc_dtype, dot_out_dtype, store_dtype) for a GEMM of `dtype`."""
    if dtype == torch.int8:
        return tl.int32, tl.int32, tl.int32
    if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        return tl.float32, tl.float32, tl.float16
    if dtype == torch.bfloat16:
        return tl.float32, tl.float32, tl.bfloat16
    if dtype == torch.float32:
        return tl.float32, tl.float32, tl.float32
    return tl.float32, tl.float32, tl.float16


def _ref_tolerances(dtype, multi_tile: bool):
    if dtype == torch.int8:
        return 0, 0
    if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        return 5e-2, 1e-1
    if dtype == torch.bfloat16:
        return (5e-2, 5e-2) if multi_tile else (3e-2, 3e-2)
    return (2e-2, 2e-2) if multi_tile else (1e-2, 1e-2)


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


def _has_canonical_flat_allocation(ttgir: str, block_k: int, block_n: int,
                                   element_bits: int, num_buffers=None) -> bool:
    layout_re = (FLAT_B32_LAYOUT_DEF_RE if element_bits == 32 else
                 FLAT_B16_LAYOUT_DEF_RE if element_bits == 16 else
                 FLAT_B8_LAYOUT_DEF_RE)
    flat_layouts = set(layout_re.findall(ttgir))
    if not flat_layouts:
        return False
    leading_shape = (f"{num_buffers}x" if num_buffers is not None
                     else r"(?:\d+x)?")
    shape = re.compile(
        rf"!ttg\.memdesc<{leading_shape}{block_k}x{block_n}x")
    return any(
        "ttg.local_alloc" in line and shape.search(line) and
        any(f", {layout}," in line for layout in flat_layouts)
        for line in ttgir.splitlines()
    )


def run_case(dtype, block_m, block_n, block_k, num_warps, *, multi_tile=False,
             require_ds_read_m=True, num_stages=1,
             require_pipeline_buffers=None, b_k_contiguous=False,
             expect_panel=None, expect_flat=None,
             problem_m=None, problem_n=None, problem_k=None,
             input_precision="tf32"):
    if multi_tile:
        m = block_m * 2
        n = block_n * 2
        k = max(block_k * 4, 256)
    else:
        m, n, k = block_m, block_n, block_k
    m = problem_m if problem_m is not None else m
    n = problem_n if problem_n is not None else n
    k = problem_k if problem_k is not None else k

    is_fp8 = dtype in (torch.float8_e5m2, torch.float8_e4m3fn)
    acc_dtype, dot_out_dtype, store_dtype = _dtypes_for(dtype)
    if dtype == torch.int8:
        a = torch.randint(-3, 4, (m, k), device="cuda", dtype=torch.int8)
        b = torch.randint(-3, 4, (k, n), device="cuda", dtype=torch.int8)
        c = torch.empty((m, n), device="cuda", dtype=torch.int32)
    elif is_fp8:
        a = (
            torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.25
        ).to(dtype)
        b = (
            torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.25
        ).to(dtype)
        c = torch.empty((m, n), device="cuda", dtype=torch.float16)
    else:
        a = torch.randn((m, k), device="cuda", dtype=dtype)
        b = torch.randn((k, n), device="cuda", dtype=dtype)
        c = torch.empty((m, n), device="cuda", dtype=dtype)

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
        INPUT_PRECISION=input_precision,
        num_warps=num_warps, num_stages=num_stages)

    failures = []
    element_bits = 8 if dtype == torch.int8 or is_fp8 else (
        32 if dtype == torch.float32 else 16)
    panel_shape = (
        block_k > 0 and block_k & (block_k - 1) == 0 and
        block_n > 0 and block_n & (block_n - 1) == 0 and
        not b_k_contiguous
    )
    # Staging is only selected while the output still has an MMAC tile per
    # warp; past that capacity the surplus warps duplicate another warp's work,
    # so AccelerateHCUMatmul leaves the generic contract in place.
    capacity_ok = num_warps <= (block_m // 16) * (block_n // 16)
    if expect_panel is None:
        expect_panel = require_ds_read_m and panel_shape and capacity_ok
    if expect_flat is None:
        # AccelerateHCUMatmul only materializes flat N-major staging for the
        # proved f32 (b32) contract: a scalar transposed flat store is slower
        # than the generic K-major staging that streams through vector
        # cp.async.  But that generic b16/b8 staging is bit-for-bit the
        # canonical flat encoding, so whenever a panel plan is declined on an
        # otherwise panel-shaped tile the lowering still proves a flat matrix
        # read off it -- expect the flat allocation, not a fallback.
        expect_flat = (require_ds_read_m and not expect_panel and
                       (element_bits == 32 or panel_shape))
    has_panel = _has_panel_allocation(_ttgir(h.asm), block_k, block_n)
    if expect_panel:
        if not has_panel:
            failures.append(
                f"expected panel-major shared_linear layout for "
                f"dtype={dtype} BM={block_m} BN={block_n} "
                f"BK={block_k} warps={num_warps}")
    else:
        if has_panel:
            failures.append(
                f"expected non-panel fallback for dtype={dtype} BM={block_m} "
                f"BN={block_n} BK={block_k} warps={num_warps}")
    has_flat = _has_canonical_flat_allocation(
        _ttgir(h.asm), block_k, block_n, element_bits)
    if expect_flat:
        if not has_flat:
            failures.append(
                f"expected canonical flat shared allocation for "
                f"dtype={dtype} BM={block_m} BN={block_n} "
                f"BK={block_k} warps={num_warps}")
    else:
        if has_flat:
            failures.append(
                f"expected no canonical flat shared allocation for "
                f"dtype={dtype} BM={block_m} BN={block_n} "
                f"BK={block_k} warps={num_warps}")
    if require_pipeline_buffers is not None:
        has_pipeline_allocation = (
            _has_canonical_flat_allocation(
                _ttgir(h.asm), block_k, block_n, element_bits,
                require_pipeline_buffers)
            if expect_flat else _has_panel_allocation(
                _ttgir(h.asm), block_k, block_n, require_pipeline_buffers))
        layout_name = "canonical flat" if expect_flat else "panel"
        if not has_pipeline_allocation:
            failures.append(
                f"expected a {require_pipeline_buffers}-slot {layout_name} allocation for "
                f"dtype={dtype} BM={block_m} BN={block_n} "
                f"BK={block_k} warps={num_warps}")
    if require_ds_read_m and not _has_expected_ds_read_m(h.asm, element_bits):
        failures.append(
            f"expected ds_read_m{element_bits} opcode in asm for "
            f"dtype={dtype} BM={block_m} BN={block_n} BK={block_k} warps={num_warps}")
    assert not failures, "\n".join(failures)

    if dtype == torch.int8:
        ref = torch.matmul(a.cpu().to(torch.int32), b.cpu().to(torch.int32)).cuda()
    elif is_fp8:
        ref = torch.matmul(a.to(torch.float16), b.to(torch.float16))
    else:
        ref = torch.matmul(a, b)
    rtol, atol = _ref_tolerances(dtype, multi_tile)
    torch.testing.assert_close(c, ref, rtol=rtol, atol=atol)
    return h


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
@pytest.mark.parametrize("dtype", [
    torch.float16,
    torch.bfloat16,
    torch.int8,
])
def test_ds_read_m_non_power_outer_pipeline(dtype):
    torch.manual_seed(0)
    # Global K=192 and N=160 are non-power-of-two. Full 64x128 physical B
    # tiles are masked at the outer edges, retaining a valid panel contract.
    run_case(dtype, 128, 128, 64, 4, num_stages=3,
             require_pipeline_buffers=2, problem_n=160, problem_k=192)


@pytest.mark.skipif(not is_hip(), reason="ds_read_m gemm test is HIP/HCU-specific")
def test_ds_read_m_row_col_fallback():
    torch.manual_seed(0)
    # K-contiguous B keeps the generic K-major staging (vector cp.async);
    # a scalar transposed flat staging store is measurably slower there.
    run_case(torch.float16, 128, 128, 32, 8, multi_tile=True,
             num_stages=3,
             b_k_contiguous=True, expect_panel=False, expect_flat=False,
             require_ds_read_m=False)


@pytest.mark.skipif(not is_hip(), reason="ds_read_m gemm test is HIP/HCU-specific")
def test_ds_read_m_f32_b32():
    torch.manual_seed(0)
    # B32 (m32x8.b32) flat N-major staging is the default f32 contract
    # (measured +3%..+12% win); the A operand keeps the generic path.
    run_case(torch.float32, 16, 32, 16, 1, input_precision="tf32",
             expect_panel=False, expect_flat=True)


@pytest.mark.skipif(not is_hip(), reason="ds_read_m gemm test is HIP/HCU-specific")
def test_ds_read_m_f32_b32_k8():
    torch.manual_seed(0)
    # Minimum K8 tile exercises the single-fragment B32 packing path.
    run_case(torch.float32, 16, 32, 8, 1, input_precision="tf32",
             expect_panel=False, expect_flat=True)


@pytest.mark.skipif(not is_hip(), reason="ds_read_m gemm test is HIP/HCU-specific")
def test_ds_read_m_f32_b32_multi_warp():
    torch.manual_seed(0)
    # B32 flat staging is independent of the warp partition: M-split,
    # N-split, and mixed M+N warps all keep the canonical flat N-major
    # contract.
    run_case(torch.float32, 32, 32, 16, 2, input_precision="tf32",
             expect_panel=False, expect_flat=True)
    run_case(torch.float32, 64, 32, 16, 4, input_precision="tf32",
             expect_panel=False, expect_flat=True)
    run_case(torch.float32, 16, 64, 16, 2, input_precision="tf32",
             expect_panel=False, expect_flat=True)
    run_case(torch.float32, 32, 64, 16, 4, input_precision="tf32",
             expect_panel=False, expect_flat=True)


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
