"""AWQ Triton Implementation.

This module contains the AWQ (Activation-Weight Quantization) implementation using Triton.
Cloned from vllm main branch (commit:cb080f32) and modified to fit HCU.
Original file path: vllm/model_executor/layers/quantization/awq_triton.py
"""

# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl
import pytest

# Device to run computations on
device = "cuda"

# Supported group sizes for AWQ Triton implementation
AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]


@triton.jit
def awq_dequantize_kernel(
        qweight_ptr,  # quantized matrix
        scales_ptr,  # scales, per group
        zeros_ptr,  # zeros, per group
        group_size,  # Should always be one of the supported group sizes
        result_ptr,  # Output matrix
        num_cols,  # input num cols in qweight
        num_rows,  # input num rows in qweight
        BLOCK_SIZE_X: tl.constexpr,
        BLOCK_SIZE_Y: tl.constexpr):
    # Setup the pids.
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Compute offsets and masks for qweight_ptr.
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets = num_cols * offsets_y[:, None] + offsets_x[None, :]

    masks_y = offsets_y < num_rows
    masks_x = offsets_x < num_cols

    masks = masks_y[:, None] & masks_x[None, :]

    # Compute offsets and masks for result output ptr.
    result_offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    result_offsets = (8 * num_cols * result_offsets_y[:, None] +
                     result_offsets_x[None, :])

    result_masks_y = result_offsets_y < num_rows
    result_masks_x = result_offsets_x < num_cols * 8
    result_masks = result_masks_y[:, None] & result_masks_x[None, :]

    # Load the weights.
    iweights = tl.load(qweight_ptr + offsets, masks, 0.0)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)
    iweights = tl.interleave(iweights, iweights)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = ((tl.arange(0, 2) * 4)[None, :] +
                               tl.arange(0, 4)[:, None]).reshape(8)

    # Use this to compute a set of shifts that can be used to unpack and
    # reorder the values in iweights and zeros.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_Y * BLOCK_SIZE_X, 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    iweights = (iweights >> shifts) & 0xF

    # Compute zero offsets and masks.
    zero_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]

    zero_masks_y = zero_offsets_y < num_rows // group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y[:, None] & zero_masks_x[None, :]

    # Load the zeros.
    zeros = tl.load(zeros_ptr + zero_offsets, zero_masks, 0.0)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.interleave(zeros, zeros)
    zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Unpack and reorder: shift out the correct 4-bit value and mask.
    zeros = (zeros >> shifts) & 0xF

    # Compute scale offsets and masks.
    scale_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    scale_offsets_x = (pid_x * BLOCK_SIZE_X * 8 +
                      tl.arange(0, BLOCK_SIZE_X * 8))
    scale_offsets = (num_cols * 8 * scale_offsets_y[:, None] +
                    scale_offsets_x[None, :])
    scale_masks_y = scale_offsets_y < num_rows // group_size
    scale_masks_x = scale_offsets_x < num_cols * 8
    scale_masks = scale_masks_y[:, None] & scale_masks_x[None, :]

    # Load the scales.
    scales = tl.load(scales_ptr + scale_offsets, scale_masks, 0.0)
    scales = tl.broadcast_to(scales, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))

    # Dequantize.
    iweights = (iweights - zeros) * scales
    iweights = iweights.to(result_ptr.type.element_ty)

    # Finally, store.
    tl.store(result_ptr + result_offsets, iweights, result_masks)

@triton.autotune(
   configs=[
       triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=2),
       triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=8),
       triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=2),
       triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=1),
       triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=2),
       triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=8),
       triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=2),
       triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=2),
       triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=1, num_warps=8),
   ],
   key=["M", "N", "K", "GROUP_SIZE", "SPLIT_K"],
   perf_debug=True
)
@triton.jit
def awq_gemm_kernel(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, K,
                    group_size, BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                    SPLIT_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = c_ptr.type.element_ty

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                          dtype=accumulator_dtype)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = ((tl.arange(0, 2) * 4)[None, :] +
                               tl.arange(0, 4)[:, None]).reshape(8)

    # Create the necessary shifts to use to unpack.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :],
                           (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_am = offsets_am < M

    offsets_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_bn = offsets_bn < N // 8

    offsets_zn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_zn = offsets_zn < N // 8

    offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    masks_sn = offsets_sn < N

    offsets_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offsets_a = K * offsets_am[:, None] + offsets_k[None, :]
    offsets_b = (N // 8) * offsets_k[:, None] + offsets_bn[None, :]

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a, other=0.0)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b, other=0.0)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        # Dequantize b.
        offsets_szk = (
            (BLOCK_SIZE_K * SPLIT_K * k + pid_z * BLOCK_SIZE_K) // group_size +
            tl.arange(0, 1))
        offsets_z = (N // 8) * offsets_szk[:, None] + offsets_zn[None, :]
        masks_zk = offsets_szk < K // group_size
        masks_z = masks_zk[:, None] & masks_zn[None, :]
        zeros_ptrs = zeros_ptr + offsets_z
        zeros = tl.load(zeros_ptrs, mask=masks_z, other=0.0)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        offsets_s = N * offsets_szk[:, None] + offsets_sn[None, :]
        masks_sk = offsets_szk < K // group_size
        masks_s = masks_sk[:, None] & masks_sn[None, :]
        scales_ptrs = scales_ptr + offsets_s
        scales = tl.load(scales_ptrs, mask=masks_s, other=0.0)
        scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        b = (b - zeros) * scales
        b = b.to(c_ptr.type.element_ty)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K * SPLIT_K
        a_ptrs += BLOCK_SIZE_K * SPLIT_K
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * (N // 8)

    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + pid_z * N * M + N * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def awq_dequantize_triton(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    block_size_x: int = 32,
    block_size_y: int = 32,
) -> torch.Tensor:
    """Dequantize weights using Triton implementation.

    Args:
        qweight: Quantized weight tensor
        scales: Scale factors tensor
        zeros: Zero points tensor
        block_size_x: Block size for x dimension
        block_size_y: Block size for y dimension

    Returns:
        Dequantized tensor
    """
    K = qweight.shape[0]
    M = scales.shape[1]
    group_size = qweight.shape[0] // scales.shape[0]

    assert K > 0 and M > 0
    assert scales.shape[0] == K // group_size and scales.shape[1] == M
    assert zeros.shape[0] == K // group_size and zeros.shape[1] == M // 8
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    # Result tensor:
    # number of rows = same as input tensor
    # number of cols = 8 x input tensor num cols
    result = torch.empty(
        qweight.shape[0],
        qweight.shape[1] * 8,
        device=qweight.device,
        dtype=scales.dtype,
    )

    Y = qweight.shape[0]  # num rows
    X = qweight.shape[1]  # num cols

    def grid(META):
        return (triton.cdiv(X, META['BLOCK_SIZE_X']), triton.cdiv(Y, META['BLOCK_SIZE_Y']))

    awq_dequantize_kernel[grid](
        qweight,
        scales,
        zeros,
        group_size,
        result,
        X,
        Y,
        BLOCK_SIZE_X=block_size_x,
        BLOCK_SIZE_Y=block_size_y,
    )

    return result


def awq_gemm_triton(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
) -> torch.Tensor:
    """Perform AWQ GEMM operation using Triton implementation.

    Args:
        input: Input tensor
        qweight: Quantized weight tensor
        scales: Scale factors tensor
        qzeros: Zero points tensor
        split_k_iters: Number of split K iterations

    Returns:
        Result tensor from GEMM operation
    """
    M, K = input.shape
    N = qweight.shape[1] * 8
    group_size = qweight.shape[0] // qzeros.shape[0]

    assert N > 0 and K > 0 and M > 0
    assert qweight.shape[0] == K and qweight.shape[1] == N // 8
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 8
    assert scales.shape[0] == K // group_size and scales.shape[1] == N
    assert split_k_iters & (split_k_iters - 1) == 0 and split_k_iters != 0
    assert split_k_iters <= 32
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            split_k_iters,
        )

    result = torch.zeros(
        (split_k_iters, M, N),
        dtype=scales.dtype,
        device=input.device,
    )

    awq_gemm_kernel[grid](
        input,
        qweight,
        result,
        qzeros,
        scales,
        M,
        N,
        K,
        group_size,
        SPLIT_K=split_k_iters,
    )

    result = result.sum(0)
    return result


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

AWQ_GEMM_TEST_CASES = [
    # "M, K, N, G, splitK"
    (10, 2048, 512, 64, 8),
    (2, 7168, 512, 64, 8),
    (5, 7168, 512, 64, 8),
    (10, 7168, 512, 64, 8),
    (21, 7168, 512, 64, 8),
    (22, 7168, 512, 64, 8),
    (23, 7168, 512, 64, 8),
    (24, 7168, 512, 64, 8),
    (25, 7168, 512, 64, 8),
    (26, 7168, 512, 64, 8),
    (27, 7168, 512, 64, 8),
    (28, 7168, 512, 64, 8),
    (29, 7168, 512, 64, 8),
    (35, 7168, 512, 64, 8),
    (36, 7168, 512, 64, 8),
    (37, 7168, 512, 64, 8),
    (38, 7168, 512, 64, 8),
    (39, 7168, 512, 64, 8),
    (40, 7168, 512, 64, 8),
    (41, 7168, 512, 64, 8),
    (42, 7168, 512, 64, 8),
    (10, 1536, 576, 64, 8),
    (2, 7168, 576, 64, 8),
    (5, 7168, 576, 64, 8),
    (10, 7168, 576, 64, 8),
    (21, 7168, 576, 64, 8),
    (22, 7168, 576, 64, 8),
    (23, 7168, 576, 64, 8),
    (24, 7168, 576, 64, 8),
    (25, 7168, 576, 64, 8),
    (26, 7168, 576, 64, 8),
    (27, 7168, 576, 64, 8),
    (28, 7168, 576, 64, 8),
    (29, 7168, 576, 64, 8),
    (35, 7168, 576, 64, 8),
    (36, 7168, 576, 64, 8),
    (37, 7168, 576, 64, 8),
    (38, 7168, 576, 64, 8),
    (39, 7168, 576, 64, 8),
    (40, 7168, 576, 64, 8),
    (41, 7168, 576, 64, 8),
    (42, 7168, 576, 64, 8),
    (10, 256, 1536, 64, 8),
    (2, 7168, 1536, 64, 8),
    (5, 7168, 1536, 64, 8),
    (10, 7168, 1536, 64, 8),
    (21, 7168, 1536, 64, 8),
    (22, 7168, 1536, 64, 8),
    (23, 7168, 1536, 64, 8),
    (24, 7168, 1536, 64, 8),
    (25, 7168, 1536, 64, 8),
    (26, 7168, 1536, 64, 8),
    (27, 7168, 1536, 64, 8),
    (28, 7168, 1536, 64, 8),
    (29, 7168, 1536, 64, 8),
    (35, 7168, 1536, 64, 8),
    (36, 7168, 1536, 64, 8),
    (37, 7168, 1536, 64, 8),
    (38, 7168, 1536, 64, 8),
    (39, 7168, 1536, 64, 8),
    (40, 7168, 1536, 64, 8),
    (41, 7168, 1536, 64, 8),
    (42, 7168, 1536, 64, 8),
    (2, 1536, 3072, 64, 8),
    (5, 1536, 3072, 64, 8),
    (10, 1536, 3072, 64, 8),
    (21, 1536, 3072, 64, 8),
    (22, 1536, 3072, 64, 8),
    (23, 1536, 3072, 64, 8),
    (24, 1536, 3072, 64, 8),
    (25, 1536, 3072, 64, 8),
    (26, 1536, 3072, 64, 8),
    (27, 1536, 3072, 64, 8),
    (28, 1536, 3072, 64, 8),
    (29, 1536, 3072, 64, 8),
    (35, 1536, 3072, 64, 8),
    (36, 1536, 3072, 64, 8),
    (37, 1536, 3072, 64, 8),
    (38, 1536, 3072, 64, 8),
    (39, 1536, 3072, 64, 8),
    (40, 1536, 3072, 64, 8),
    (41, 1536, 3072, 64, 8),
    (42, 1536, 3072, 64, 8),
    (10, 7168, 3072, 64, 8),
    (2, 512, 4096, 64, 8),
    (5, 512, 4096, 64, 8),
    (10, 512, 4096, 64, 8),
    (21, 512, 4096, 64, 8),
    (22, 512, 4096, 64, 8),
    (23, 512, 4096, 64, 8),
    (24, 512, 4096, 64, 8),
    (25, 512, 4096, 64, 8),
    (26, 512, 4096, 64, 8),
    (27, 512, 4096, 64, 8),
    (28, 512, 4096, 64, 8),
    (29, 512, 4096, 64, 8),
    (35, 512, 4096, 64, 8),
    (36, 512, 4096, 64, 8),
    (37, 512, 4096, 64, 8),
    (38, 512, 4096, 64, 8),
    (39, 512, 4096, 64, 8),
    (40, 512, 4096, 64, 8),
    (41, 512, 4096, 64, 8),
    (42, 512, 4096, 64, 8),
    (2, 7168, 4608, 64, 8),
    (5, 7168, 4608, 64, 8),
    (10, 7168, 4608, 64, 8),
    (21, 7168, 4608, 64, 8),
    (22, 7168, 4608, 64, 8),
    (23, 7168, 4608, 64, 8),
    (24, 7168, 4608, 64, 8),
    (25, 7168, 4608, 64, 8),
    (26, 7168, 4608, 64, 8),
    (27, 7168, 4608, 64, 8),
    (28, 7168, 4608, 64, 8),
    (29, 7168, 4608, 64, 8),
    (35, 7168, 4608, 64, 8),
    (36, 7168, 4608, 64, 8),
    (37, 7168, 4608, 64, 8),
    (38, 7168, 4608, 64, 8),
    (39, 7168, 4608, 64, 8),
    (40, 7168, 4608, 64, 8),
    (41, 7168, 4608, 64, 8),
    (42, 7168, 4608, 64, 8),
    (2, 256, 7168, 64, 8),
    (5, 256, 7168, 64, 8),
    (10, 256, 7168, 64, 8),
    (21, 256, 7168, 64, 8),
    (22, 256, 7168, 64, 8),
    (23, 256, 7168, 64, 8),
    (24, 256, 7168, 64, 8),
    (25, 256, 7168, 64, 8),
    (26, 256, 7168, 64, 8),
    (27, 256, 7168, 64, 8),
    (28, 256, 7168, 64, 8),
    (29, 256, 7168, 64, 8),
    (35, 256, 7168, 64, 8),
    (36, 256, 7168, 64, 8),
    (37, 256, 7168, 64, 8),
    (38, 256, 7168, 64, 8),
    (39, 256, 7168, 64, 8),
    (40, 256, 7168, 64, 8),
    (41, 256, 7168, 64, 8),
    (42, 256, 7168, 64, 8),
    (10, 512, 7168, 64, 8),
    (2, 2048, 7168, 64, 8),
    (5, 2048, 7168, 64, 8),
    (10, 2048, 7168, 64, 8),
    (21, 2048, 7168, 64, 8),
    (22, 2048, 7168, 64, 8),
    (23, 2048, 7168, 64, 8),
    (24, 2048, 7168, 64, 8),
    (25, 2048, 7168, 64, 8),
    (26, 2048, 7168, 64, 8),
    (27, 2048, 7168, 64, 8),
    (28, 2048, 7168, 64, 8),
    (29, 2048, 7168, 64, 8),
    (35, 2048, 7168, 64, 8),
    (36, 2048, 7168, 64, 8),
    (37, 2048, 7168, 64, 8),
    (38, 2048, 7168, 64, 8),
    (39, 2048, 7168, 64, 8),
    (40, 2048, 7168, 64, 8),
    (41, 2048, 7168, 64, 8),
    (42, 2048, 7168, 64, 8),
    (2, 2304, 7168, 64, 8),
    (5, 2304, 7168, 64, 8),
    (10, 2304, 7168, 64, 8),
    (21, 2304, 7168, 64, 8),
    (22, 2304, 7168, 64, 8),
    (23, 2304, 7168, 64, 8),
    (24, 2304, 7168, 64, 8),
    (25, 2304, 7168, 64, 8),
    (26, 2304, 7168, 64, 8),
    (27, 2304, 7168, 64, 8),
    (28, 2304, 7168, 64, 8),
    (29, 2304, 7168, 64, 8),
    (35, 2304, 7168, 64, 8),
    (36, 2304, 7168, 64, 8),
    (37, 2304, 7168, 64, 8),
    (38, 2304, 7168, 64, 8),
    (39, 2304, 7168, 64, 8),
    (40, 2304, 7168, 64, 8),
    (41, 2304, 7168, 64, 8),
    (42, 2304, 7168, 64, 8),
]

# input   - [M, K]
# qweight - [K, N // 8]
# qzeros  - [K // G, N // 8]
# scales  - [K // G, N]
@pytest.mark.parametrize('M, K, N, G, splitK', AWQ_GEMM_TEST_CASES)
def test_gemm(N, K, M, G, splitK):
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

    output_triton = awq_gemm_triton(input, qweight, scales, qzeros, splitK)

    assert (not torch.any(torch.isinf(output_triton))
            and not torch.any(torch.isnan(output_triton)))

    dequantized_weights = awq_dequantize_triton(qweight, scales, qzeros)
    output_torch = torch.matmul(input, dequantized_weights)

    assert (not torch.any(torch.isinf(output_torch))
            and not torch.any(torch.isnan(output_torch)))

    torch.testing.assert_close(output_triton.cpu(),
                             output_torch.cpu(),
                             atol=1e-1,
                             rtol=1e-1)


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
        x_names=['M', 'K', 'N', 'G', 'splitK'],
        x_vals=AWQ_GEMM_TEST_CASES,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('red', '-'), ('blue', '--')],
        ylabel='ms',
        xlabel='Matrix Dimensions (M×K×N)',
        plot_name='AWQ GEMM Performance',
        args={'device': 'cuda'}
    )
]

@triton.testing.perf_report(bench_configs)
def bench_awq_gemm(M, K, N, G, splitK, provider, device="cuda"):
    """Benchmark AWQ GEMM performance.

    Args:
        M: Number of rows in input matrix
        K: Number of columns in input matrix
        N: Number of columns in weight matrix (pre-quantization)
        G: AWQ group size
        splitK: Parallel split factor
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
        device=device
    )
    scales = torch.rand(
        (K // G, N),
        dtype=torch.float16,
        device=device
    )

    if provider == "triton":
        fn = lambda: awq_gemm_triton(
            input_tensor,
            qweight,
            scales,
            qzeros,
            splitK
        )
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    else:  # provider == "torch"
        fn = lambda: torch.matmul(
            input_tensor,
            awq_dequantize_torch(qweight, scales, qzeros, G)
        )
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

if __name__ == "__main__":
    #bench_awq_dequantize.run(print_data=True)
    bench_awq_gemm.run(print_data=True)