"""E2E tests for optional buffer-op offset asserts."""

import os
import subprocess
import sys
import tempfile

import pytest
import torch
import triton
import triton.language as tl

from triton._internal_testing import is_hip

_HCU_TEST_DEVICE = "cuda"
_ASSERT_ENV = "TRITON_AMDGPU_EMIT_BUFFER_OPS_OFFSET_ASSERT"

GEMM_M = 256
GEMM_N = 256
GEMM_K = 256
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
# Per-lane element stride: lane i has offset i * STRIDE; byte off exceeds 2^32-2.
OOB_ELEM_STRIDE = 1 << 30


@triton.jit
def _buffer_load_with_assert_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    """1D splat+offset pattern: canonicalize -> buffer_load + offset assert."""
    offs = tl.arange(0, BLOCK)
    vals = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, vals)


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


def _run_gemm(device, enable_assert: bool):
    prev_assert = os.environ.get(_ASSERT_ENV)
    prev_buf = os.environ.get("AMDGCN_USE_BUFFER_OPS", "1")
    try:
        if enable_assert:
            os.environ[_ASSERT_ENV] = "1"
        else:
            os.environ.pop(_ASSERT_ENV, None)
        os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"

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
        return a, b, c
    finally:
        if prev_assert is None:
            os.environ.pop(_ASSERT_ENV, None)
        else:
            os.environ[_ASSERT_ENV] = prev_assert
        os.environ["AMDGCN_USE_BUFFER_OPS"] = prev_buf


@pytest.mark.skipif(reason="buffer op offset assert is HIP-specific", condition=not is_hip())
def test_buffer_ops_offset_assert_gemm_ok():
    device = _HCU_TEST_DEVICE
    a, b, c = _run_gemm(device, enable_assert=True)
    torch.testing.assert_close(c, torch.matmul(a, b), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(reason="buffer op offset assert is HIP-specific", condition=not is_hip())
def test_buffer_ops_offset_assert_disabled():
    device = _HCU_TEST_DEVICE
    a, b, c = _run_gemm(device, enable_assert=False)
    torch.testing.assert_close(c, torch.matmul(a, b), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(reason="buffer op offset assert is HIP-specific", condition=not is_hip())
def test_buffer_ops_offset_assert_emits_ttir():
    prev_assert = os.environ.get(_ASSERT_ENV)
    prev_buf = os.environ.get("AMDGCN_USE_BUFFER_OPS", "1")
    try:
        os.environ[_ASSERT_ENV] = "1"
        os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
        device = _HCU_TEST_DEVICE
        x = torch.randn(64, dtype=torch.float32, device=device)
        out = torch.empty(64, dtype=torch.float32, device=device)
        compiled = _buffer_load_with_assert_kernel[(1,)](x, out, BLOCK=64)
        ttgir = compiled.asm["ttgir"]
        assert "tt.assert" in ttgir
        # staging uses amdg.*; release/3.x uses amdgpu.*
        assert "amdgpu.buffer_load" in ttgir or "amdg.buffer_load" in ttgir
    finally:
        if prev_assert is None:
            os.environ.pop(_ASSERT_ENV, None)
        else:
            os.environ[_ASSERT_ENV] = prev_assert
        os.environ["AMDGCN_USE_BUFFER_OPS"] = prev_buf


@pytest.mark.skipif(reason="buffer op offset assert is HIP-specific", condition=not is_hip())
def test_buffer_ops_offset_assert_catches_oob():
    script = f"""
import os
import torch
import triton
import triton.language as tl

os.environ["{_ASSERT_ENV}"] = "1"
os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"

BLOCK = 64

@triton.jit
def kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    # Tensor offset (not folded into base): lane i -> i * STRIDE elements * 4 B.
    offs = tl.arange(0, BLOCK) * {OOB_ELEM_STRIDE}
    vals = tl.load(x_ptr + offs)
    tl.store(out_ptr + tl.arange(0, BLOCK), vals)

x = torch.randn(BLOCK, dtype=torch.float32, device="cuda")
out = torch.empty(BLOCK, dtype=torch.float32, device="cuda")
kernel[(1,)](x, out, BLOCK=BLOCK)
torch.cuda.synchronize()
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    env = os.environ.copy()
    env[_ASSERT_ENV] = "1"
    env["AMDGCN_USE_BUFFER_OPS"] = "1"
    try:
        r = subprocess.run([sys.executable, path], env=env, capture_output=True, text=True)
    finally:
        os.unlink(path)
    out = r.stdout + r.stderr
    assert r.returncode != 0
    assert "device assertion failed" in out
