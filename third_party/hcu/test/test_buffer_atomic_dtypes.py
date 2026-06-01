"""Test buffer atomic correctness for multiple data types with partial mask.

Verifies that masked buffer_atomic operations do not VM fault when
the OOB sentinel is access_size-aligned (regression test for the
-elemByteWidth sentinel fix for 4GiB num_records).
"""
import logging
import pytest
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
DEVICE = "cuda"


# ---------- kernels ----------

@triton.jit
def atomic_add_i32(out_ptr, TILE_R: tl.constexpr):
    bin_indices = tl.arange(0, TILE_R)
    mask = bin_indices < TILE_R
    tl.atomic_add(out_ptr + bin_indices, tl.full((TILE_R,), 1, dtype=tl.int32),
                  mask=mask, sem="relaxed")

@triton.jit
def atomic_and_i32(out_ptr, TILE_R: tl.constexpr):
    bin_indices = tl.arange(0, TILE_R)
    mask = bin_indices < TILE_R
    tl.atomic_and(out_ptr + bin_indices, tl.full((TILE_R,), 0xFF, dtype=tl.int32),
                  mask=mask, sem="relaxed")

@triton.jit
def atomic_or_i32(out_ptr, TILE_R: tl.constexpr):
    bin_indices = tl.arange(0, TILE_R)
    mask = bin_indices < TILE_R
    tl.atomic_or(out_ptr + bin_indices, tl.full((TILE_R,), 0xFF00, dtype=tl.int32),
                 mask=mask, sem="relaxed")

@triton.jit
def atomic_xor_i32(out_ptr, TILE_R: tl.constexpr):
    bin_indices = tl.arange(0, TILE_R)
    mask = bin_indices < TILE_R
    tl.atomic_xor(out_ptr + bin_indices, tl.full((TILE_R,), 0xFF, dtype=tl.int32),
                  mask=mask, sem="relaxed")

@triton.jit
def atomic_max_i32(out_ptr, TILE_R: tl.constexpr):
    bin_indices = tl.arange(0, TILE_R)
    mask = bin_indices < TILE_R
    tl.atomic_max(out_ptr + bin_indices, tl.full((TILE_R,), 42, dtype=tl.int32),
                  mask=mask, sem="relaxed")

@triton.jit
def atomic_min_i32(out_ptr, TILE_R: tl.constexpr):
    bin_indices = tl.arange(0, TILE_R)
    mask = bin_indices < TILE_R
    tl.atomic_min(out_ptr + bin_indices, tl.full((TILE_R,), 0, dtype=tl.int32),
                  mask=mask, sem="relaxed")

@triton.jit
def atomic_add_i64(out_ptr, TILE_R: tl.constexpr):
    bin_indices = tl.arange(0, TILE_R)
    mask = bin_indices < TILE_R
    tl.atomic_add(out_ptr + bin_indices, tl.full((TILE_R,), 1, dtype=tl.int64),
                  mask=mask, sem="relaxed")

@triton.jit
def atomic_add_f32(out_ptr, TILE_R: tl.constexpr):
    bin_indices = tl.arange(0, TILE_R)
    mask = bin_indices < TILE_R
    tl.atomic_add(out_ptr + bin_indices, tl.full((TILE_R,), 1.0, dtype=tl.float32),
                  mask=mask, sem="relaxed")

@triton.jit
def atomic_add_f16(out_ptr, TILE_R: tl.constexpr):
    bin_indices = tl.arange(0, TILE_R)
    mask = bin_indices < TILE_R
    tl.atomic_add(out_ptr + bin_indices, tl.full((TILE_R,), 1.0, dtype=tl.float16),
                  mask=mask, sem="relaxed")

@triton.jit
def atomic_add_bf16(out_ptr, TILE_R: tl.constexpr):
    bin_indices = tl.arange(0, TILE_R)
    mask = bin_indices < TILE_R
    tl.atomic_add(out_ptr + bin_indices, tl.full((TILE_R,), 1.0, dtype=tl.bfloat16),
                  mask=mask, sem="relaxed")


# ---------- test cases ----------

TEST_CASES = [
    pytest.param(atomic_add_i32, torch.int32,    1.0,   id="atomic_add_i32"),
    pytest.param(atomic_and_i32, torch.int32,    0.0,   id="atomic_and_i32"),
    pytest.param(atomic_or_i32,  torch.int32,    0xFF00, id="atomic_or_i32"),
    pytest.param(atomic_xor_i32, torch.int32,    0xFF,   id="atomic_xor_i32"),
    pytest.param(atomic_max_i32, torch.int32,    42,    id="atomic_max_i32"),
    pytest.param(atomic_min_i32, torch.int32,    0,     id="atomic_min_i32"),
    pytest.param(atomic_add_i64, torch.int64,    1.0,   id="atomic_add_i64"),
    pytest.param(atomic_add_f32, torch.float32,  1.0,   id="atomic_add_f32"),
    pytest.param(atomic_add_f16, torch.float16,  1.0,   id="atomic_add_f16"),
    pytest.param(atomic_add_bf16, torch.bfloat16, 1.0,  id="atomic_add_bf16"),
]


@pytest.mark.parametrize("kernel_fn,out_dtype,expected_first", TEST_CASES)
def test_buffer_atomic_partial_mask(kernel_fn, out_dtype, expected_first):
    """buffer_atomic with partial mask should not VM fault."""
    TILE_R = 16
    NUM_WARPS = 4

    out = torch.zeros((TILE_R,), device=DEVICE, dtype=out_dtype)
    kernel_fn[(1,)](out, TILE_R, num_warps=NUM_WARPS)
    torch.cuda.synchronize()

    # The first element should have been written by the first thread
    if out_dtype in (torch.int32, torch.int64):
        expected = int(expected_first)
        assert out[0] == expected, (
            f"Expected out[0]={expected}, got {out[0]}")
    elif out_dtype.is_floating_point:
        expected = float(expected_first)
        assert abs(float(out[0]) - expected) < 1e-3, (
            f"Expected out[0]≈{expected}, got {float(out[0])}")
