# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def kernel_i64_f64(
    x_i64_ptr,  # *i64 input
    z_i64_ptr,  # *i64 output
    x_f64_ptr,  # *f64 input
    z_f64_ptr,  # *f64 output
    n_elements,  # number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Grid-stride loop
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load and store i64 values
    x_i64 = tl.load(x_i64_ptr + offsets, mask=mask)
    tl.store(z_i64_ptr + offsets, x_i64, mask=mask)

    # Load and store f64 values
    x_f64 = tl.load(x_f64_ptr + offsets, mask=mask)
    tl.store(z_f64_ptr + offsets, x_f64, mask=mask)


@pytest.mark.parametrize('size', [1024, 2048])
def test_kernel_i64_f64(size):
    # Initialize input tensors
    x_i64 = torch.randint(-2**63, 2**63-1, (size,), dtype=torch.int64, device='cuda')
    x_f64 = torch.randn(size, dtype=torch.float64, device='cuda')

    # Initialize output tensors
    z_i64 = torch.empty(size, dtype=torch.int64, device='cuda')
    z_f64 = torch.empty(size, dtype=torch.float64, device='cuda')

    # Grid and block sizes
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)

    # Launch kernel
    kernel_i64_f64[grid](
        x_i64, z_i64,
        x_f64, z_f64,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Check results
    torch.testing.assert_close(z_i64, x_i64)
    torch.testing.assert_close(z_f64, x_f64)

test_kernel_i64_f64(1024)