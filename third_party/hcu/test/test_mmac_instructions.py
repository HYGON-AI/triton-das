"""Regression coverage for gfx946 MMAC variants and acc16 fallback."""

import re

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_hip


_MMAC_INSTRUCTIONS = {
    "f16_acc_f32": r"\bv_mmac_f32_16x16x16_f16\b",
    "f16_f16": r"\bv_mmac_16x16x16_f16\b",
    "f16_bf16": r"\bv_mmac_bf16_16x16x16_f16\b",
    "bf16_f16": r"\bv_mmac_f16_16x16x16_bf16\b",
    "bf16_bf16": r"\bv_mmac_16x16x16_bf16\b",
    "f64_f64": r"\bv_mmac_16x16x4_f64\b",
}

_HCU_ARCHES_WITHOUT_ACC16 = {"gfx928", "gfx936", "gfx938", "gfx92a"}


def is_hip_gfx946():
    if not is_hip():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.arch == "gfx946"


def is_hip_hcu_without_acc16():
    if not is_hip():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.arch in _HCU_ARCHES_WITHOUT_ACC16


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
        f"expected {_MMAC_INSTRUCTIONS[instruction]!r} in generated GCN assembly"
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


@pytest.mark.skipif(
    not is_hip_hcu_without_acc16(),
    reason="MMAC acc16 fallback test requires an HCU target without acc16 support",
)
@pytest.mark.parametrize(
    "dtype,acc_dtype,fallback_instruction,native_instruction",
    [
        (torch.float16, tl.float16, "f16_acc_f32", "f16_f16"),
    ],
)
def test_mmac_acc16_falls_back_to_f32_accumulation(
    dtype, acc_dtype, fallback_instruction, native_instruction
):
    M = 32
    N = 32
    K = 32
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    c = torch.empty((M, N), device="cuda", dtype=dtype)
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    program = floating_mmac_kernel[grid](
        a,
        b,
        c,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        K=K,
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=16,
        ACC_DTYPE=acc_dtype,
        num_warps=4,
    )

    assert_mmac_result(c.cpu(), matmul_ref(a, b, dtype))
    _assert_instruction(program, fallback_instruction)
    assert not re.search(_MMAC_INSTRUCTIONS[native_instruction], program.asm["amdgcn"]), (
        "unexpected native 16-bit accumulation MMAC on a target without acc16 support"
    )


# HCU arches that expose the K=8 f32 MMAC (ieee: v_mmac_16x16x8_f32, tf32:
# v_mmac_f32_16x16x8_tf32); the other HCU arches only expose the K=4
# scalar-operand variant (v_mmac_16x16x4_f32).
_HCU_F32_K8_ARCHES = {"gfx928", "gfx936", "gfx938"}
_HCU_ARCHES = _HCU_F32_K8_ARCHES | {"gfx92a", "gfx946"}


def is_hip_hcu():
    if not is_hip():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.arch in _HCU_ARCHES


# HCU arches with the MMAC_LAYOUT feature (gfx938/gfx946): the only arches
# that honor mmac_layout_force. Other HCU arches fall back to the LEGACY
# layout with a diagnostic, so forcing a mode there exercises nothing.
_HCU_MMAC_LAYOUT_ARCHES = {"gfx938", "gfx946"}


def is_hip_hcu_mmac_layout():
    if not is_hip_hcu():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.arch in _HCU_MMAC_LAYOUT_ARCHES


def _current_arch():
    return triton.runtime.driver.active.get_current_target().arch


@triton.jit
def f32_kpack_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_base in range(0, K, BLOCK_K):
        offs_k = k_base + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)
    c = acc.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c)


@pytest.mark.skipif(not is_hip_hcu(), reason="MMAC kpack tests require an HCU target")
@pytest.mark.parametrize("kpack", [1, 2])
@pytest.mark.parametrize("block_k", [4, 8])
def test_fp32_kpack_mmac_operand_types(kpack, block_k):
    """fp32 dots must lower to MMAC with correctly typed A/B operands for any kPack.

    Regression test: with kPack=2 the dot-operand kWidth doubles (2 for the
    K=4 scalar-operand f32 MMAC, 4 for the K=8 f32 MMAC), but the scalar
    v_mmac_16x16x4_f32 still consumes one float per lane per instruction.
    The lowering used to key the scalar path on kWidth==1, so kPack=2
    produced an i32 operand and LLVM aborted with "Calling a function with
    a bad signature!" in CallInst::init.
    """
    M = N = 64
    K = 32
    BLOCK_K = block_k
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    program = f32_kpack_kernel[grid](
        a, b, c,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        K=K, BLOCK_M=16, BLOCK_N=16, BLOCK_K=BLOCK_K, INPUT_PRECISION="ieee",
        num_warps=4, kpack=kpack,
    )
    assert_mmac_result(c.cpu(), matmul_ref(a, b, torch.float32))
    amdgcn = program.asm["amdgcn"]
    # K=4 always selects the scalar-operand MMAC (v_mmac_16x16x4_f32); K=8
    # selects the K=8 MMAC (v_mmac_16x16x8_f32) only on arches that expose
    # MMAC_F32_K8, and the K=4 variant otherwise.
    expected = r"\bv_mmac_16x16x8_f32\b" if (
        BLOCK_K == 8 and _current_arch() in _HCU_F32_K8_ARCHES
    ) else r"\bv_mmac_16x16x4_f32\b"
    assert re.search(expected, amdgcn), (
        f"expected {expected} in generated GCN assembly"
    )


@pytest.mark.skipif(not is_hip_hcu(), reason="MMAC kpack tests require an HCU target")
@pytest.mark.parametrize(
    "input_dtype,block_k", [(torch.float32, 8), (torch.float16, 16)]
)
def test_fp32_kpack2_mmac_kloop_accuracy(input_dtype, block_k):
    """kPack=2 must not dilute the carried-in accumulator in multi-iter K loops.

    kPack=2 widens kWidth to 2*kBase, so the fragment K extent
    (kWidth * kGroups) is twice the BLOCK_K covered by one dot. Each dot
    contribution is therefore accumulated twice and has to be rescaled back
    by 1/2. Rescaling the whole accumulator instead of (acc - c) progressively
    dilutes the partial sums of the previous K-loop iterations.
    """
    M = N = 64
    K = 64
    assert K % block_k == 0 and K // block_k > 1
    torch.manual_seed(0)
    if input_dtype == torch.float16:
        a = (torch.randn((M, K), device="cuda") * 0.3).to(torch.float16)
        b = (torch.randn((K, N), device="cuda") * 0.3).to(torch.float16)
    else:
        a = torch.randn((M, K), device="cuda", dtype=torch.float32)
        b = torch.randn((K, N), device="cuda", dtype=torch.float32)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    # fp32 on HCU defaults to tf32 precision; for f16 the precision is unused.
    precision = "tf32" if input_dtype == torch.float32 else "ieee"
    f32_kpack_kernel[grid](
        a, b, c,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        K=K, BLOCK_M=16, BLOCK_N=16, BLOCK_K=block_k, INPUT_PRECISION=precision,
        num_warps=4, kpack=2,
    )
    ref_dtype = torch.float32
    expected = matmul_ref(a, b, ref_dtype)
    atol = rtol = 1e-2 if input_dtype == torch.float16 else 5e-2
    torch.testing.assert_close(c.cpu(), expected, atol=atol, rtol=rtol)


@triton.jit
def int8_kloop_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k_base in range(0, K, BLOCK_K):
        offs_k = k_base + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)


@pytest.mark.skipif(not is_hip_hcu(), reason="MMAC kpack tests require an HCU target")
def test_int8_kpack2_mmac_kloop_accuracy():
    """kPack=2 must not dilute the integer accumulator in multi-iter K loops.

    Companion to test_fp32_kpack2_mmac_kloop_accuracy for the integer
    rescale path (arithmetic shift of (acc - c) instead of (acc * 1/rate)):
    with K=64 and BLOCK_K=32 the dot runs two K-loop iterations, so a
    diluting implementation would lose half of the first iteration's sum.
    """
    M = N = 64
    K = 64
    BLOCK_K = 32
    assert K // BLOCK_K > 1
    torch.manual_seed(0)
    a = torch.randint(-100, 100, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-100, 100, (K, N), device="cuda", dtype=torch.int8)
    c = torch.empty((M, N), device="cuda", dtype=torch.int32)
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    program = int8_kloop_kernel[grid](
        a, b, c,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        K=K, BLOCK_M=16, BLOCK_N=16, BLOCK_K=BLOCK_K,
        num_warps=4, kpack=2,
    )
    # Exact integer reference: the rescale must be bit-exact.
    torch.testing.assert_close(c.cpu(), a.cpu().int() @ b.cpu().int())
    assert re.search(r"\bv_mmac_i32_16x16x32_i8\b", program.asm["amdgcn"]), (
        "expected v_mmac_i32_16x16x32_i8 in generated GCN assembly"
    )
