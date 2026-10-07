"""
Block Scaled Matrix Multiplication
==================================
This tutorial demonstrates a Triton implementation of block scaled matrix multiplication on Hygon GPUs.
The tutorial supports OCP microscaling formats such as mxfp4. These matrix multiplications are hardware-accelerated
using v_mmac_scale_fp32_16x16x64_fp4 which is suppoted on gfx946.

.. code-block:: bash

    # FP4
    pytest block-scaled-matmul.py

Future updates to this sample which support mixed precision block scaled matmul are planned.
"""

# %%
# Background
# ----------
#
# HCU gfx946 devices that support utlize block scaled matrix multiply instructions.
# In order for low latency access to these scale factors in the fast
# inner loop over tensor core MMAs, it is important to ensure that the blocked
# scale factors are stored in a contiguous memory layout according to their access
# pattern.
#
# The block scaled matmul tensor core instructions compute the following product:
#
#     C = (A * scale_a) @ (B * scale_b)
#
# where scale_a and scale_b are the blocked scale factors for the A and B matrices.
# Under block scaled matmul, each scale factor is broadcast and multiplied across a
# vector of elements from the A and B matrices, usually along their respective K axes.
# The number of elements of A and B over which each scale factor is broadcast is herein
# refered to as the vector size (VEC_SIZE).
#
# In a linear row-major layout, the scale factors would take the shape
#
#     (M, K // VEC_SIZE) and (N, K // VEC_SIZE)
#
# in global memory.
#
# In order to conform with Triton's language semantics for dot_scaled, the scale factors
# are prepared in the above 2D layout expected by tl.dot_scaled.
#
# On gfx946 GPUs, scaled MMAC instructions natively support scaled matrix multiplication.
# Since it only supports OCP microscaling formats each scale is an 8-bit value that scales
# 32 elements from A or B operand tensors.
# Scales are stored as 8-bit tensors. Since MMAC instructions are warp-level instructions, that
# means that each thread provides a fixed set of operand values to MFMA instructions.
#
# For example, in an MMAC instruction with shape 16x16x64:
# - 4 threads contribute elements along the K dimension.
# - 16 threads contribute elements along the M or N dimension.
#
# From the perspective of the scales tensor, even if the K dimension is stored contiguously in
# shared memory, each thread sees its elements along K dim as strided due to interleaving with
# other threads. This striding limits the ability to load scale values using vectorized memory
# access.
#
# We have two MMAC cases: one with K dimension 32, and one with 64.
# K dimension 32 case (16x16x32.f8f6f4) is planned to support.
#
# For 16x16x64 scaled MMAC instruction, the minimum tile size is 16x16x32xi8 (2 fp4 pack to 1 int8).
# For example, for a 32x256 operand tile, the corresponding scale tensor has shape 32x8,
# where each scale covers 32 elements along the K dimension.
#
# Case: v_mmac_scaled_fp32_16x16x64_fp4
#
#            K = 32       K = 32
#        +------------+ +------------+
#    M=16|  MMAC op 0 | |  MMAC op 1 |
#        +------------+ +------------+
#    M=16|  MMAC op 2 | |  MMAC op 3 |
#        +------------+ +------------+
#

import torch
import pytest
import triton
import triton.language as tl
from triton.tools.mxfp import MXScaleTensor, MXFP4Tensor

SCALE_BLOCK_SIZE = 32


def supports_block_scaling():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.arch in ['gfx946']


def to_tl_dtype(dtype):
    if dtype == torch.float32:
        return tl.float32
    elif dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    else:
        raise NotImplementedError


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # scale pointers
    a_scale_ptr, b_scale_ptr,
    # Matrix dimensions
    M, N, K,
    scale_K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
    A_SCALE_DOT_DTYPE: tl.constexpr,
    B_SCALE_DOT_DTYPE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    BLOCK_SCALE_K: tl.constexpr,
):
    """Kernel for computing the scaled matmul D = (A * a_scale) x (B * b_scale) + C.
    A has shape (M, K), B has shape (K, N) and C/D has shape (M, N)
    """

    tl.assume(stride_am >= 0)
    tl.assume(stride_ak >= 0)
    tl.assume(stride_bk >= 0)
    tl.assume(stride_bn >= 0)
    tl.assume(stride_cm >= 0)
    tl.assume(stride_cn >= 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_ks = tl.arange(0, BLOCK_SCALE_K)
    a_scale_ptrs = a_scale_ptr + offs_am[:, None] * stride_asm + offs_ks[None, :] * stride_ask
    b_scale_ptrs = b_scale_ptr + offs_bn[:, None] * stride_bsn + offs_ks[None, :] * stride_bsk

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=OUTPUT_DTYPE)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0)
        b = tl.load(b_ptrs, mask=(offs_bn[None, :] < N) & (offs_k[:, None] < K - k * BLOCK_SIZE_K), other=0)

        a_scale = tl.load(
            a_scale_ptrs,
            mask=(offs_am[:, None] < M) & (offs_ks[None, :] < scale_K - k * BLOCK_SCALE_K),
            other=0,
        )
        b_scale = tl.load(
            b_scale_ptrs,
            mask=(offs_bn[:, None] < N) & (offs_ks[None, :] < scale_K - k * BLOCK_SCALE_K),
            other=0,
        )

        accumulator = tl.dot_scaled(
            a, a_scale, A_SCALE_DOT_DTYPE,
            b, b_scale, B_SCALE_DOT_DTYPE,
            accumulator,
        )

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        a_scale_ptrs += BLOCK_SCALE_K * stride_ask
        b_scale_ptrs += BLOCK_SCALE_K * stride_bsk

    c = leaky_relu(accumulator) if ACTIVATION == "leaky_relu" else accumulator

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a, b, a_scale, b_scale, configs, activation="",
           a_layout="row", b_layout="col",
           a_scale_dot_dtype="e2m1", b_scale_dot_dtype="e2m1",
           output_dtype=torch.float32):
    """
    a_scale: shape [M, cdiv(K, 32)], dtype=torch.uint8 (e8m0 code)
    b_scale: shape [N, cdiv(K, 32)], dtype=torch.uint8 (e8m0 code)
    """
    assert (a_layout in ["row", "col"] and b_layout in ["row", "col"])

    if a_layout == 'row':
        M, aK = a.shape
        stride_am, stride_ak = a.stride(0), a.stride(1)
    else:
        aK, M = a.shape
        stride_am, stride_ak = a.stride(1), a.stride(0)

    if b_layout == 'row':
        bK, N = b.shape
        stride_bk, stride_bn = b.stride(0), b.stride(1)
    else:
        N, bK = b.shape
        stride_bk, stride_bn = b.stride(1), b.stride(0)

    assert aK == bK, f"K dimension mismatch: A K={aK}, B K={bK}"
    K = aK

    a_scale_K = a_scale.shape[-1]
    b_scale_K = b_scale.shape[-1]
    assert a_scale_K == b_scale_K, f"Scale K dimension mismatch: A scale K={a_scale_K}, B scale K={b_scale_K}"
    scale_K = a_scale_K

    stride_asm, stride_ask = a_scale.stride(0), a_scale.stride(1)
    stride_bsn, stride_bsk = b_scale.stride(0), b_scale.stride(1)

    c = torch.empty((M, N), device=a.device, dtype=output_dtype)

    configs['BLOCK_SCALE_K'] = (configs['BLOCK_SIZE_K'] * 2) // SCALE_BLOCK_SIZE
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    matmul_kernel[grid](
        a, b, c,
        a_scale, b_scale,
        M, N, K,
        scale_K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        c.stride(0), c.stride(1),
        stride_asm, stride_ask,
        stride_bsn, stride_bsk,
        ACTIVATION=activation,
        A_SCALE_DOT_DTYPE=a_scale_dot_dtype,
        B_SCALE_DOT_DTYPE=b_scale_dot_dtype,
        OUTPUT_DTYPE=to_tl_dtype(output_dtype),
        **configs,
    )
    return c


def create_inputs(M, N, K, a_layout, b_layout, a_scale_dot_dtype,
                   b_scale_dot_dtype):
    assert (a_layout in ["row", "col"] and b_layout in ["row", "col"])

    torch.manual_seed(42)

    if a_layout == 'row':
        a_mxfp = MXFP4Tensor(size=(M, K)).random()
        a_scale = MXScaleTensor(size=(M, K // SCALE_BLOCK_SIZE)).random(low=0.5, high=8.0)
        a = a_mxfp.to_packed_tensor(1)
        ref_a = a_mxfp.to(torch.float32) * a_scale.to(torch.float32).repeat_interleave(SCALE_BLOCK_SIZE, dim=1)
    else:
        raise NotImplementedError

    if b_layout == 'row':
        raise NotImplementedError
    else:
        b_mxfp = MXFP4Tensor(size=(N, K)).random()
        b_scale = MXScaleTensor(size=(N, K // SCALE_BLOCK_SIZE)).random(low=0.5, high=8.0)
        b = b_mxfp.to_packed_tensor(1)
        ref_b = b_mxfp.to(torch.float32) * b_scale.to(torch.float32).repeat_interleave(SCALE_BLOCK_SIZE, dim=1)

    return a.cuda(), b.cuda(), a_scale.data.cuda(), b_scale.data.cuda(), ref_a, ref_b


def get_hip_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': m, 'BLOCK_SIZE_N': n, 'BLOCK_SIZE_K': k,
             'GROUP_SIZE_M': g, 'waves_per_eu': 1},
            num_warps=w, num_stages=s)
                for m in [16, 32, 64, 128, 256]
                for n in [16, 32, 64, 128, 256]
                for k in [32, 64, 128]
                for w in [1, 2, 4, 8, 16]
                for s in [1]
                for g in [1]
    ]


@pytest.mark.skipif(
    not supports_block_scaling(), reason="Blocked scale matmul require gfx946 target"
)
@pytest.mark.parametrize('M, N, K', [(256, 256, 256)])
@pytest.mark.parametrize("configs", [c.all_kwargs() for c in get_hip_autotune_config()])
@pytest.mark.parametrize('a_layout', ['row'])
@pytest.mark.parametrize('b_layout', ['col'])
@pytest.mark.parametrize('a_scale_dot_dtype', ["e2m1"])
@pytest.mark.parametrize('b_scale_dot_dtype', ["e2m1"])
@pytest.mark.parametrize('output_dtype', [torch.float32])
def test_matmul(M, N, K, configs, a_layout, b_layout, a_scale_dot_dtype,
                b_scale_dot_dtype, output_dtype):
    torch.manual_seed(0)
    a, b, a_scale, b_scale, ref_a, ref_b = create_inputs(
        M, N, K, a_layout, b_layout, a_scale_dot_dtype, b_scale_dot_dtype)

    triton_output = matmul(a, b, a_scale, b_scale, configs,
                           a_layout=a_layout, b_layout=b_layout,
                           a_scale_dot_dtype=a_scale_dot_dtype,
                           b_scale_dot_dtype=b_scale_dot_dtype,
                           output_dtype=output_dtype)

    torch_output_cpu = torch.matmul(ref_a, ref_b.T).to(triton_output.dtype)
    triton_output_cpu = triton_output.cpu()

    assert torch.allclose(triton_output_cpu, torch_output_cpu, atol=1e-2, rtol=1e-2)
