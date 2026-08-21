# Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from vLLM @ cb080f32 (vllm/model_executor/layers/quantization/awq_triton.py)


"""AWQ Triton Implementation.

This module contains the AWQ (Activation-Weight Quantization) implementation using Triton.
Cloned from vllm main branch (commit:cb080f32) and modified to fit HCU.
Original file path: vllm/model_executor/layers/quantization/awq_triton.py
"""

import torch
import triton
import triton.language as tl
import pytest
import os
import json
import numpy as np

from awq_triton import (
    awq_dequantize_triton,
    awq_gemm_triton,
    AWQ_TRITON_SUPPORTED_GROUP_SIZES,
)

device = "cuda"

def reverse_awq_order(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse the AWQ order of the given tensor.
    
    Args:
        tensor: Input tensor to reorder
        
    Returns:
        Reordered tensor with bits masked to 4 bits
    """
    bits = 4
    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    reverse_order_tensor = torch.arange(
        tensor.shape[-1],
        dtype=torch.int32,
        device=tensor.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    tensor = tensor[:, reverse_order_tensor] & 0xF
    return tensor


def awq_dequantize_torch(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Dequantize weights using PyTorch implementation.
    
    Args:
        qweight: Quantized weight tensor
        scales: Scale factors tensor
        qzeros: Zero points tensor
        group_size: Size of groups for quantization
        
    Returns:
        Dequantized tensor
    """
    if group_size == -1:
        group_size = qweight.shape[0]

    bits = 4
    shifts = torch.arange(0, 32, bits, device=qzeros.device)

    iweights = torch.bitwise_right_shift(
        qweight[:, :, None],
        shifts[None, None, :],
    ).to(torch.int8)
    iweights = iweights.view(iweights.shape[0], -1)

    zeros = torch.bitwise_right_shift(
        qzeros[:, :, None],
        shifts[None, None, :],
    ).to(torch.int8)
    zeros = zeros.view(qzeros.shape[0], -1)
    zeros = reverse_awq_order(zeros)

    iweights = reverse_awq_order(iweights)

    iweights = torch.bitwise_and(iweights, (2**bits) - 1)
    zeros = torch.bitwise_and(zeros, (2**bits) - 1)

    scales = scales.repeat_interleave(group_size, dim=0)
    zeros = zeros.repeat_interleave(group_size, dim=0)
    return (iweights - zeros) * scales


@pytest.mark.parametrize("qweight_rows", [3584, 18944, 128, 256, 512, 1024])
@pytest.mark.parametrize("qweight_cols", [448, 576, 4736, 16, 32, 64, 128])
@pytest.mark.parametrize("group_size", AWQ_TRITON_SUPPORTED_GROUP_SIZES)
def test_dequantize(qweight_rows, qweight_cols, group_size):
    """Test AWQ dequantization implementations."""
    if group_size == -1:
        group_size = qweight_rows

    qweight_dtype = torch.int32
    scales_rows = qweight_rows // group_size
    scales_cols = qweight_cols * 8
    scales_dtype = torch.float16
    zeros_rows = scales_rows
    zeros_cols = qweight_cols
    zeros_dtype = torch.int32

    torch.manual_seed(0)

    qweight = torch.randint(0, torch.iinfo(torch.int32).max,
                          (qweight_rows, qweight_cols),
                          dtype=qweight_dtype, device=device)
    scales = torch.rand(scales_rows, scales_cols,
                       dtype=scales_dtype, device=device)
    zeros = torch.randint(0, torch.iinfo(torch.int32).max,
                         (zeros_rows, zeros_cols),
                         dtype=zeros_dtype, device=device)

    iweights_triton = awq_dequantize_triton(qweight, scales, zeros)
    assert not torch.any(torch.isinf(iweights_triton)) and not torch.any(torch.isnan(iweights_triton))

    iweights_torch = awq_dequantize_torch(qweight, scales, zeros, group_size)
    torch.testing.assert_close(iweights_triton, iweights_torch)

AWQ_GEMM_KNG_CASES = [
    # "K, N, G"
    (256, 576, 64),
    (256, 1536, 64),
    (256, 3072, 64),
    (256, 4096, 64),
    (256, 4608, 64),
    (256, 7168, 64),

    (512, 576, 64),
    (512, 1536, 64),
    (512, 3072, 64),
    (512, 4096, 64),
    (512, 4608, 64),
    (512, 7168, 64),

    (1536, 576, 64),
    (1536, 1536, 64),
    (1536, 3072, 64),
    (1536, 4096, 64),
    (1536, 4608, 64),
    (1536, 7168, 64),

    (2048, 512, 64),
    (2048, 1536, 64),
    (2048, 3072, 64),
    (2048, 4096, 64),
    (2048, 4608, 64),
    (2048, 7168, 64),

    (2304, 512, 64),
    (2304, 1536, 64),
    (2304, 3072, 64),
    (2304, 4096, 64),
    (2304, 4608, 64),
    (2304, 7168, 64),

    (7168, 512, 64),
    (7168, 1536, 64),
    (7168, 3072, 64),
    (7168, 4096, 64),
    (7168, 4608, 64),
    (7168, 7168, 64),
]

AWQ_GEMM_TEST_CASES = [
    # "M, K, N, G"
    (M, *KNG) for M in range(1, 129) for KNG in AWQ_GEMM_KNG_CASES
]
AWQ_GEMM_TEST_CASES.append((2, 7168, 576, 64))
AWQ_GEMM_TEST_CASES.append((10, 7168, 576, 64))
AWQ_GEMM_TEST_CASES.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

# input   - [M, K]
# qweight - [K, N // 8]
# qzeros  - [K // G, N // 8]
# scales  - [K // G, N]
@pytest.mark.parametrize("M, K, N, G", AWQ_GEMM_TEST_CASES)
def test_gemm(N, K, M, G):
    G = G if G != -1 else K
    device = "cuda"
    input_rows = M
    input_cols = K
    input_dtype = torch.float16
    qweight_rows = input_cols
    qweight_cols = N // 8
    scales_rows = qweight_rows // G
    scales_cols = N
    scales_dtype = torch.float16
    qzeros_rows = scales_rows
    qzeros_cols = qweight_cols

    torch.manual_seed(0)

    input = torch.rand((input_rows, input_cols),
                      dtype=input_dtype,
                      device=device)
    qweight = torch.randint(0,
                          torch.iinfo(torch.int32).max,
                          (qweight_rows, qweight_cols),
                          device=device)
    qzeros = torch.randint(0,
                         torch.iinfo(torch.int32).max,
                         (qzeros_rows, qzeros_cols),
                         device=device)
    scales = torch.rand((scales_rows, scales_cols),
                       dtype=scales_dtype,
                       device=device)

    output_triton = awq_gemm_triton(input, qweight, scales, qzeros)
    assert (not torch.any(torch.isinf(output_triton))
            and not torch.any(torch.isnan(output_triton)))

    dequantized_weights = awq_dequantize_torch(qweight, scales, qzeros, G)
    output_torch = torch.matmul(input, dequantized_weights)
    assert (not torch.any(torch.isinf(output_torch))
            and not torch.any(torch.isnan(output_torch)))
    
    # Move tensors to CPU for comparison
    output_triton_cpu = output_triton.cpu()
    output_torch_cpu = output_torch.cpu()

    # Calculate the tolerance bound based on torch.testing.assert_close formula
    atol = 1e-1
    rtol = 1e-1
    EXPECTED = output_torch_cpu.to(torch.float16)
    ACTUAL = output_triton_cpu.to(torch.float16)
    tolerance = atol + rtol * torch.abs(EXPECTED)
    abs_diff = torch.abs(ACTUAL - EXPECTED)
    
    # Find elements where absolute difference exceeds the tolerance
    mask = abs_diff > tolerance
    
    '''
    if torch.any(mask):
        print("\nElements that exceed torch.testing.assert_close tolerance:")
        mismatched_indices = torch.nonzero(mask)
        for idx in mismatched_indices:
            i, j = idx[0].item(), idx[1].item()
            actual = ACTUAL[i,j]
            expected = EXPECTED[i,j]
            abs_difference = abs_diff[i,j]
            tolerance_at_point = tolerance[i,j]
            print(f"Position [{i},{j}]:")
            print(f"  Actual (Triton): {actual:.6f}")
            print(f"  Expected (Torch): {expected:.6f}")
            print(f"  |actual - expected|: {abs_difference:.6f}")
            print(f"  tolerance (atol + rtol*|expected|): {tolerance_at_point:.6f}")
    '''
            
    # Original assertion
    torch.testing.assert_close(ACTUAL,
                             EXPECTED,
                             atol=atol,
                             rtol=rtol)


# Performance benchmarking configurations
AWQ_PERF_MODEL_CASES = [
    (4096, 4096, 128),
    (4096, 8192, 128),
    (8192, 4096, 128),
]

configs = [
    triton.testing.Benchmark(
        x_names=['NUM_ROWS', 'NUM_COLS', 'GROUP_SIZE'],
        x_vals=AWQ_PERF_MODEL_CASES,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('red', '-'), ('blue', '--')],
        ylabel='TOPS',
        xlabel='Matrix Dimensions (Rows x Cols)',
        plot_name='AWQ Dequantize Performance',
        args={'dtype': torch.float16, 'device': device},
    )
]

@triton.testing.perf_report(configs)
def bench_awq_dequantize(NUM_ROWS, NUM_COLS, GROUP_SIZE, provider,
                        dtype=torch.float16, device="cuda"):
    """Benchmark AWQ dequantization performance."""
    warmup = 25
    rep = 10

    qweight = torch.randint(0, torch.iinfo(torch.int32).max,
                          (NUM_ROWS, NUM_COLS//8),
                          dtype=torch.int32, device=device)
    scales = torch.rand((NUM_ROWS//GROUP_SIZE, NUM_COLS),
                       dtype=dtype, device=device)
    zeros = torch.randint(0, torch.iinfo(torch.int32).max,
                         (NUM_ROWS//GROUP_SIZE, NUM_COLS//8),
                         dtype=torch.int32, device=device)

    if provider == "triton":
        fn = lambda: awq_dequantize_triton(qweight, scales, zeros,
                                         block_size_x=32, block_size_y=32)
    else:  # provider == "torch"
        fn = lambda: awq_dequantize_torch(qweight, scales, zeros, GROUP_SIZE)

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    ops_per_element = 3
    total_elements = NUM_ROWS * NUM_COLS
    total_ops = total_elements * ops_per_element
    return total_ops / ms * 1e-9  # Return TOPS

# Benchmark configurations
bench_configs = [
    triton.testing.Benchmark(
        x_names=['M', 'K', 'N', 'G'],
        x_vals=AWQ_GEMM_TEST_CASES,
        line_arg='Provider',
        line_vals=["Triton"],
        line_names=['Triton'],
        styles=[('red', '-'), ('blue', '--')],
        ylabel='ms',
        xlabel='Matrix Dimensions (M×K×N)',
        plot_name='AWQ GEMM Performance',
        args={'device': device}
    )
]

@triton.testing.perf_report(bench_configs)
def bench_awq_gemm(M, K, N, G, Provider, device="cuda"):
    """Benchmark AWQ GEMM performance.
    
    Args:
        M: Number of rows in input matrix
        K: Number of columns in input matrix
        N: Number of columns in weight matrix (pre-quantization)
        G: AWQ group size
        provider: Implementation provider ('triton' or 'torch')
        device: Device to run on
    """
    warmup = 25
    rep = 10
    G = G if G != -1 else K

    # Generate test data
    input_tensor = torch.rand(
        (M, K),
        dtype=torch.float16,
        device=device
    )
    qweight = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (K, N // 8),
        dtype=torch.int32,
        device=device
    )
    qzeros = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (K // G, N // 8),
        dtype=torch.int32,
        device=device
    )
    scales = torch.rand(
        (K // G, N),
        dtype=torch.float16,
        device=device
    )

    #if provider == "triton":
    if True:
        fn = lambda: awq_gemm_triton(
            input_tensor,
            qweight,
            scales,
            qzeros,
        )
        if int(os.getenv("TRITON_COMPILE_ONLY", 0)) == 1:
            ms = triton.testing.do_bench_compile_only(fn, warmup=warmup, rep=rep)
        else:
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms
        
    '''
    else:  # provider == "torch"
        fn = lambda: torch.matmul(
            input_tensor, 
            awq_dequantize_torch(qweight, scales, qzeros, G)
        )
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms
    '''

if __name__ == "__main__":
    bench_awq_dequantize.run(print_data=True)
    bench_awq_gemm.run(print_data=True)
