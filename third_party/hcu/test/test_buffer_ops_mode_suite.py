"""Buffer-ops correctness / mode suite (#830, Case5, convert smoke).

Default product knobs:

  AMDGCN_USE_BUFFER_OPS=1
  AMDGCN_BUFFER_OPS_USE_RANGE_ANALYSIS=0  # nonneg
  AMDGCN_ANALYZE_SMALL_TENSOR_RANGE=0     # pointer_range fast path allowed

Override those env vars before pytest to exercise nonneg vs RA.
``TRITON_ALWAYS_COMPILE=1`` is forced so flips take effect.

Coverage
--------
* Conversion smoke under ANALYZE=1 (baseline / pid / dyn-loop)
* Issue #830 fold-U overflow (ANALYZE=0; expect PASS with small-tensor no-fold)
* Case5 ~4GiB i32 byte-wrap (PASS under RA; xfail known wrap under nonneg)

Branch if/select lives in ``test_buffer_ops_branch_offset.py``.

Listed in ``regression.sh``.

Run:

  pytest third_party/hcu/test/test_buffer_ops_mode_suite.py -s
  AMDGCN_BUFFER_OPS_USE_RANGE_ANALYSIS=1 pytest ... -k case5
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_hip

_HCU_TEST_DEVICE = "cuda"
BLOCK = 1024

_KNOB_ENVS = (
    "AMDGCN_USE_BUFFER_OPS",
    "AMDGCN_ANALYZE_SMALL_TENSOR_RANGE",
    "AMDGCN_BUFFER_OPS_USE_RANGE_ANALYSIS",
    "TRITON_ALWAYS_COMPILE",
)


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "on", "yes")


def _use_ra() -> bool:
    return _env_truthy("AMDGCN_BUFFER_OPS_USE_RANGE_ANALYSIS", "0")


def _analyze_global() -> bool:
    return _env_truthy("AMDGCN_ANALYZE_SMALL_TENSOR_RANGE", "0")


@contextmanager
def _buffer_ops_env(
    *,
    analyze: bool | None = None,
    use_ra: bool | None = None,
) -> Iterator[None]:
    prev = {k: os.environ.get(k) for k in _KNOB_ENVS}
    try:
        os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        os.environ["AMDGCN_ANALYZE_SMALL_TENSOR_RANGE"] = (
            "1" if (analyze if analyze is not None else _analyze_global()) else "0"
        )
        os.environ["AMDGCN_BUFFER_OPS_USE_RANGE_ANALYSIS"] = (
            "1" if (use_ra if use_ra is not None else _use_ra()) else "0"
        )
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _has_buffer_load(ttgir: str) -> bool:
    return ("amdg.buffer_load" in ttgir) or ("amdgpu.buffer_load" in ttgir)


def _assert_buffer_load_only(ttgir: str) -> None:
    assert _has_buffer_load(ttgir), f"expected buffer_load, got:\n{ttgir[:2000]}"
    assert "tt.load" not in ttgir, f"expected no tt.load, got:\n{ttgir[:2000]}"


# ---------------------------------------------------------------------------
# Convert smoke kernels
# ---------------------------------------------------------------------------


@triton.jit
def _kernel_baseline(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs))


@triton.jit
def _kernel_pid(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask, other=0.0), mask=mask)


@triton.jit
def _kernel_dyn_loop(x_ptr, out_ptr, tiles, BLOCK: tl.constexpr):
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(0, tiles):
        offs = k * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + offs)
    tl.store(out_ptr + tl.arange(0, BLOCK), acc)


# ---------------------------------------------------------------------------
# #830
# ---------------------------------------------------------------------------

_ROW = 8_388_608  # 1 << 23
_XBLOCK = 1024
_XNUMEL = 64 * _ROW


@triton.jit
def _kernel_830_a(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    x1 = xindex // 8388608
    x0 = xindex % 8388608
    active = (x1 >= 32) & (x1 < 40) & xmask
    d = -2113929216 + x0 + 67108864 * x1
    val = tl.load(in_ptr0 + d, mask=active, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + xindex, val, mask=xmask)


@triton.jit
def _kernel_830_b(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    x1 = xindex // 8388608
    x0 = xindex % 8388608
    active = (x1 >= 40) & (x1 < 48) & xmask
    d = 33554432 + x0 + 67108864 * ((-40) + x1)
    val = tl.load(in_ptr0 + d, mask=active, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + xindex, val, mask=xmask)


def _i32(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.int32)


def _ref_830(xindex: torch.Tensor, case: str):
    x1 = torch.div(xindex, _ROW, rounding_mode="floor")
    x0 = torch.remainder(xindex, _ROW)
    if case == "A":
        active = (x1 >= 32) & (x1 < 40)
        d = (
            _i32(torch.full((), -2113929216, device=xindex.device))
            + _i32(x0)
            + _i32(torch.full((), 67108864, device=xindex.device)) * _i32(x1)
        )
    else:
        active = (x1 >= 40) & (x1 < 48)
        d = (
            _i32(torch.full((), 33554432, device=xindex.device))
            + _i32(x0)
            + _i32(torch.full((), 67108864, device=xindex.device))
            * (_i32(torch.full((), -40, device=xindex.device)) + _i32(x1))
        )
    return active, d.to(torch.int64)


def _run_830_case(case: str) -> None:
    kernel = _kernel_830_a if case == "A" else _kernel_830_b
    lo, hi = (32 * _ROW, 40 * _ROW) if case == "A" else (40 * _ROW, 48 * _ROW)
    device = _HCU_TEST_DEVICE
    inp = torch.randn((65536, 8192), device=device, dtype=torch.bfloat16)
    out = torch.zeros(_XNUMEL, device=device, dtype=torch.float32)
    inp_flat = inp.view(-1)
    grid = (triton.cdiv(_XNUMEL, _XBLOCK),)
    kernel[grid](inp, out, _XNUMEL, XBLOCK=_XBLOCK)
    torch.cuda.synchronize()

    xindex = torch.arange(lo, hi, device=device, dtype=torch.int64)
    active, d = _ref_830(xindex, case)
    valid = active & (d >= 0) & (d < inp_flat.numel())
    exp = torch.zeros(hi - lo, device=device, dtype=torch.float32)
    exp[valid] = inp_flat[d[valid]].float()
    torch.testing.assert_close(out[lo:hi], exp, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Case5
# ---------------------------------------------------------------------------

_CASE5_BASE = 1 << 30
_CASE5_BLOCK = 64


@triton.jit
def _kernel_case5(x_ptr, out_ptr, N, BLOCK: tl.constexpr, BASE: tl.constexpr):
    r = tl.arange(0, BLOCK)
    offs = tl.arange(BASE, BASE + BLOCK)
    mask = (offs >= 0) & (offs < N)
    tl.store(out_ptr + r, tl.load(x_ptr + offs, mask=mask, other=0.0), mask=mask)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not is_hip(), reason="buffer ops conversion is HIP/HCU-specific"
)


def test_convert_baseline_analyze():
    """ANALYZE=1: plain arange always converts."""
    with _buffer_ops_env(analyze=True):
        device = _HCU_TEST_DEVICE
        x = torch.randn(BLOCK, device=device)
        out = torch.empty(BLOCK, device=device)
        compiled = _kernel_baseline[(1,)](x, out, BLOCK=BLOCK)
        torch.cuda.synchronize()
        _assert_buffer_load_only(compiled.asm.get("ttgir", "") or "")
        torch.testing.assert_close(out, x)


def test_convert_pid_analyze():
    """ANALYZE=1: pid*BLOCK+arange — nonneg converts; RA may refuse."""
    with _buffer_ops_env(analyze=True):
        device = _HCU_TEST_DEVICE
        n = BLOCK
        x = torch.randn(n, device=device)
        out = torch.empty(n, device=device)
        compiled = _kernel_pid[(1,)](x, out, n, BLOCK=BLOCK)
        torch.cuda.synchronize()
        ttgir = compiled.asm.get("ttgir", "") or ""
        torch.testing.assert_close(out, x)
        if _use_ra():
            assert _has_buffer_load(ttgir) or "tt.load" in ttgir
        else:
            _assert_buffer_load_only(ttgir)


def test_convert_dyn_loop_analyze():
    """ANALYZE=1: dynamic loop bound — nonneg converts; RA may refuse."""
    with _buffer_ops_env(analyze=True):
        device = _HCU_TEST_DEVICE
        tiles = 8
        x = torch.randn(BLOCK * tiles, device=device)
        out = torch.empty(BLOCK, device=device)
        compiled = _kernel_dyn_loop[(1,)](x, out, tiles, BLOCK=BLOCK)
        torch.cuda.synchronize()
        ttgir = compiled.asm.get("ttgir", "") or ""
        ref = sum(x[k * BLOCK:(k + 1) * BLOCK] for k in range(tiles))
        torch.testing.assert_close(out, ref)
        if _use_ra():
            assert _has_buffer_load(ttgir) or "tt.load" in ttgir
        else:
            _assert_buffer_load_only(ttgir)


@pytest.mark.parametrize("case", ["A", "B"])
def test_issue_830_fold_u(case):
    """#830: small-tensor no-fold must stay numerically correct."""
    with _buffer_ops_env(analyze=False):
        _run_830_case(case)


def test_case5_byte_offset_wrap():
    """Case5: i32 byte-off wrap under nonneg buffer convert.

    Default JIT+nonneg: xfail (known silent wrap).
    USE_RA=1: expect numerical PASS.
    """
    if not _use_ra():
        pytest.xfail(
            "Case5 silent i32 byte-wrap under nonneg; "
            "set AMDGCN_BUFFER_OPS_USE_RANGE_ANALYSIS=1 for correct behavior"
        )

    with _buffer_ops_env(analyze=False):
        device = _HCU_TEST_DEVICE
        n = _CASE5_BASE + _CASE5_BLOCK
        x = torch.zeros(n, device=device, dtype=torch.float32)
        x[:_CASE5_BLOCK] = 1000.0 + torch.arange(
            _CASE5_BLOCK, device=device, dtype=torch.float32
        )
        x[_CASE5_BASE:_CASE5_BASE + _CASE5_BLOCK] = 5000.0 + torch.arange(
            _CASE5_BLOCK, device=device, dtype=torch.float32
        )
        ref = x[_CASE5_BASE:_CASE5_BASE + _CASE5_BLOCK].clone()
        out = torch.zeros(_CASE5_BLOCK, device=device, dtype=torch.float32)
        compiled = _kernel_case5[(1,)](
            x, out, n, BLOCK=_CASE5_BLOCK, BASE=_CASE5_BASE
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        ttgir = compiled.asm.get("ttgir", "") or ""
        assert "tt.load" in ttgir or _has_buffer_load(ttgir)


def test_report_active_knobs():
    with _buffer_ops_env():
        from triton import knobs

        print(
            "buffer_ops mode suite knobs:",
            f"use_ra={bool(getattr(knobs.amd, 'buffer_ops_use_range_analysis', False))}",
            f"analyze={bool(knobs.amd.buffer_ops_analyze_small_tensor_range)}",
        )
