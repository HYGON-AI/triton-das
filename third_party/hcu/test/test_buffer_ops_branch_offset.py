"""Buffer-ops conversion for if/select joined offsets.

Mirrors range-analysis.mlir @ifOp / @select. Under TTGIR these lower to
``scf.if`` / ``arith.select``. With ANALYZE=1 (skip pointer_range fast path),
nonneg fallback walks both arms. Expect ``buffer_load`` + numerical correctness
under default JIT + USE_RA=0.

Knob overrides (optional):
  AMDGCN_BUFFER_OPS_USE_RANGE_ANALYSIS

Listed in ``regression.sh``.

Run:
  pytest third_party/hcu/test/test_buffer_ops_branch_offset.py -s
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


@contextmanager
def _buffer_ops_env(*, analyze: bool = True) -> Iterator[None]:
    prev = {k: os.environ.get(k) for k in _KNOB_ENVS}
    try:
        os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        os.environ["AMDGCN_ANALYZE_SMALL_TENSOR_RANGE"] = "1" if analyze else "0"
        # Keep caller/product default for RA when already set.
        os.environ.setdefault("AMDGCN_BUFFER_OPS_USE_RANGE_ANALYSIS", "0")
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _assert_buffer_load_only(ttgir: str) -> None:
    has_buf = ("amdg.buffer_load" in ttgir) or ("amdgpu.buffer_load" in ttgir)
    assert has_buf, f"expected buffer_load in ttgir, got:\n{ttgir[:2000]}"
    assert "tt.load" not in ttgir, f"expected no tt.load, got:\n{ttgir[:2000]}"


@triton.jit
def _kernel_if_offset(x_ptr, out_ptr, take_range, BLOCK: tl.constexpr):
    """Python ``if`` → TTGIR ``scf.if`` join on offsets."""
    if take_range:
        offs = tl.arange(0, BLOCK)
    else:
        offs = tl.zeros((BLOCK,), dtype=tl.int32)
    tl.store(out_ptr + tl.arange(0, BLOCK), tl.load(x_ptr + offs))


@triton.jit
def _kernel_where_offset(x_ptr, out_ptr, take_range, BLOCK: tl.constexpr):
    """``tl.where`` → TTGIR ``arith.select`` on offsets."""
    pred = take_range != 0
    ar = tl.arange(0, BLOCK)
    z = tl.zeros((BLOCK,), dtype=tl.int32)
    offs = tl.where(pred, ar, z)
    tl.store(out_ptr + tl.arange(0, BLOCK), tl.load(x_ptr + offs))


@triton.jit
def _kernel_no_branch(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs))


pytestmark = pytest.mark.skipif(
    not is_hip(), reason="buffer ops conversion is HIP/HCU-specific"
)


@pytest.mark.parametrize("take_range", [True, False])
def test_if_offset_converts_to_buffer_load(take_range):
    with _buffer_ops_env(analyze=True):
        device = _HCU_TEST_DEVICE
        x = (1000.0 + torch.arange(BLOCK, device=device, dtype=torch.float32)).contiguous()
        out = torch.empty(BLOCK, device=device, dtype=torch.float32)
        ref = x.clone() if take_range else x[0].expand(BLOCK).clone()
        compiled = _kernel_if_offset[(1,)](x, out, take_range, BLOCK=BLOCK)
        torch.cuda.synchronize()
        _assert_buffer_load_only(compiled.asm.get("ttgir", "") or "")
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("take_range", [True, False])
def test_where_offset_converts_to_buffer_load(take_range):
    with _buffer_ops_env(analyze=True):
        device = _HCU_TEST_DEVICE
        x = (1000.0 + torch.arange(BLOCK, device=device, dtype=torch.float32)).contiguous()
        out = torch.empty(BLOCK, device=device, dtype=torch.float32)
        ref = x.clone() if take_range else x[0].expand(BLOCK).clone()
        compiled = _kernel_where_offset[(1,)](x, out, take_range, BLOCK=BLOCK)
        torch.cuda.synchronize()
        _assert_buffer_load_only(compiled.asm.get("ttgir", "") or "")
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_no_branch_converts_under_nonneg():
    with _buffer_ops_env(analyze=True):
        device = _HCU_TEST_DEVICE
        x = (1000.0 + torch.arange(BLOCK, device=device, dtype=torch.float32)).contiguous()
        out = torch.empty(BLOCK, device=device, dtype=torch.float32)
        compiled = _kernel_no_branch[(1,)](x, out, BLOCK=BLOCK)
        torch.cuda.synchronize()
        _assert_buffer_load_only(compiled.asm.get("ttgir", "") or "")
        torch.testing.assert_close(out, x, rtol=0, atol=0)
