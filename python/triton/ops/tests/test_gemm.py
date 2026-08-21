# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

import pytest
import torch
import triton

from triton.ops.gemm import gemm


def run_triton(x, weight):
    return gemm(x, weight)


def run_torch(x, weight):
    return torch.matmul(x, weight)


def get_x_vals():
    x_vals = [
        (1, 576, 256),
        ]
    return x_vals


@pytest.mark.parametrize(
    "M, N, K, device",
    [
        (*shape, device)
        for shape in get_x_vals()
        for device in ['cuda']
    ],
)
def test_gemm(M, N, K, device):
    # Generate test data
    input_tensor = torch.rand(
        (M, K),
        dtype=torch.float16,
        device=device
    )
    weight = torch.rand(
        (K, N),
        dtype=torch.float16,
        device=device
    )

    assert torch.allclose(run_torch(input_tensor, weight),
                          run_triton(input_tensor, weight))
