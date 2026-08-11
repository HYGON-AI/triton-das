"""Regression coverage for five gfx946 MMAC instruction variants."""

import re

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hip


_MMAC_INSTRUCTIONS = {
    "f16_f16": r"\bv_mmac_16x16x16_f16\b",
    "f16_bf16": r"\bv_mmac_bf16_16x16x16_f16\b",
    "bf16_f16": r"\bv_mmac_f16_16x16x16_bf16\b",
    "bf16_bf16": r"\bv_mmac_16x16x16_bf16\b",
    "f64_f64": r"\bv_mmac_16x16x4_f64\b",
}


def is_hip_gfx946():
    if not is_hip():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.arch == "gfx946"


@triton.jit
def floating_mmac_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

    for k_base in range(0, K, BLOCK_K):
        offs_k = k_base + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc = tl.dot(a, b, acc, out_dtype=ACC_DTYPE)

    c = acc.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c)


def matmul_ref(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """CPU reference; f16/bf16 values accumulate in fp32 before conversion."""
    a = a.detach().cpu()
    b = b.detach().cpu()
    m, k = a.shape
    n = b.shape[1]
    acc_dtype = torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
    result = torch.empty((m, n), dtype=out_dtype)
    for row in range(m):
        for column in range(n):
            value = torch.zeros((), dtype=acc_dtype)
            for reduction in range(k):
                value += a[row, reduction] * b[reduction, column]
            result[row, column] = value.to(out_dtype)
    return result


def is_equal_default(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    if actual.dtype == torch.float16:
        epsilon = 2**-10
    elif actual.dtype == torch.bfloat16:
        epsilon = 2**-7
    elif actual.dtype == torch.float32:
        epsilon = 2**-23
    elif actual.dtype == torch.float64:
        epsilon = 2**-52
    else:
        return actual == expected
    return torch.abs(actual - expected) <= 2.0 * torch.abs(actual + expected) * epsilon


def assert_mmac_result(actual: torch.Tensor, expected: torch.Tensor):
    ulp_match = is_equal_default(actual, expected)
    failed_ulp = ~ulp_match
    if not bool(torch.any(failed_ulp).item()):
        return

    if actual.dtype == torch.bfloat16:
        atol = 4e-2
        rtol = 4e-2
    elif actual.dtype == torch.float16:
        atol = 2e-2
        rtol = 2e-2
    elif actual.dtype == torch.float32:
        atol = rtol = 1e-5
    elif actual.dtype == torch.float64:
        atol = rtol = 1e-10
    else:
        atol = rtol = 0

    torch.testing.assert_close(
        actual[failed_ulp], expected[failed_ulp], atol=atol, rtol=rtol
    )


def _assert_instruction(program, instruction: str):
    amdgcn = program.asm["amdgcn"]
    assert re.search(_MMAC_INSTRUCTIONS[instruction], amdgcn), (
        f"expected {_MMAC_INSTRUCTIONS[instruction]!r} in generated AMDGCN assembly"
    )


@pytest.mark.skipif(
    not is_hip_gfx946(), reason="MMAC instruction tests require the HCU gfx946 target"
)
@pytest.mark.parametrize(
    "name,input_dtype,output_dtype,block_k,asm_instruction",
    [
        ("f16_f16", torch.float16, torch.float16, 16, "f16_f16"),
        ("f16_bf16", torch.float16, torch.bfloat16, 16, "f16_bf16"),
        ("bf16_f16", torch.bfloat16, torch.float16, 16, "bf16_f16"),
        ("bf16_bf16", torch.bfloat16, torch.bfloat16, 16, "bf16_bf16"),
        ("f64_f64", torch.float64, torch.float64, 4, "f64_f64"),
    ],
)
def test_floating_mmac_instructions(name, input_dtype, output_dtype, block_k, asm_instruction):
    M = 32
    N = 32
    K = 32
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=input_dtype)
    b = torch.randn((K, N), device="cuda", dtype=input_dtype)
    c = torch.empty((M, N), device="cuda", dtype=output_dtype)
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = block_k
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    program = floating_mmac_kernel[grid](
        a, b, c,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, ACC_DTYPE={
            torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float64: tl.float64,
        }[output_dtype],
        num_warps=4,
    )
    assert_mmac_result(c.cpu(), matmul_ref(a, b, output_dtype))
    _assert_instruction(program, asm_instruction)
