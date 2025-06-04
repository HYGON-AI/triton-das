# SPDX-License-Identifier: Apache-2.0
import os
import json
import logging
import functools
from functools import partial

import torch
import triton
import triton.language as tl

from typing import Any, Dict, List, Optional, Tuple
from vllm.platforms import current_platform

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
    result_offsets_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(
        0, BLOCK_SIZE_X * 8)
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

@triton.jit
def awq_gemm_kernel_inner(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, tile_idx, iter_begin, iter_end,
                          M, N, K, GROUP_SIZE: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                          BLOCK_SIZE_K: tl.constexpr, NUM_GROUPS: tl.constexpr, A_LOAD_ORDER: tl.constexpr = 0):
    tl.assume(tile_idx >= 0)
    tl.assume(iter_begin >= 0)
    tl.assume(iter_end >= 0)
    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(K > 0)

    num_tile_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tile_n = tl.cdiv(N, BLOCK_SIZE_N)

    tile_idx_m = tile_idx // num_tile_n
    tile_idx_n = tile_idx % num_tile_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
    # that will map given indices to the correct order.
    reverse_awq_order_tensor = ((tl.arange(0, 2) * 4)[None, :] + tl.arange(0, 4)[:, None]).reshape(8)

    # Create the necessary shifts to use to unpack.
    shifts = reverse_awq_order_tensor * 4
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8))
    shifts = tl.reshape(shifts, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    # Offsets and masks.
    offsets_am = tile_idx_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_am = offsets_am < M

    offsets_bzn = tile_idx_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
    offsets_bzn = tl.max_contiguous(tl.multiple_of(offsets_bzn, BLOCK_SIZE_N // 8), BLOCK_SIZE_N // 8)
    masks_bzn = offsets_bzn < N // 8

    offsets_sn = tile_idx_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_sn = tl.max_contiguous(tl.multiple_of(offsets_sn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    masks_sn = offsets_sn < N

    offsets_k = iter_begin * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offsets_ak = tl.max_contiguous(tl.multiple_of(offsets_k, BLOCK_SIZE_K), BLOCK_SIZE_K)
    offsets_a = K * offsets_am[:, None] + offsets_ak[None, :]
    offsets_b = (N // 8) * offsets_k[:, None] + offsets_bzn[None, :]

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b
    for k in range(iter_end - iter_begin):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        masks_b = masks_k[:, None] & masks_bzn[None, :]
        other_bzs = 0.0
        a = tl.load(a_ptrs, mask=masks_a, other=0.)
        b = tl.load(b_ptrs, masks_b, other_bzs)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)

        # Dequantize b.
        offsets_szk = ((BLOCK_SIZE_K * k + iter_begin * BLOCK_SIZE_K) // GROUP_SIZE + tl.arange(0, NUM_GROUPS))
        masks_szk = offsets_szk < K // GROUP_SIZE
        masks_z = masks_szk[:, None] & masks_bzn[None, :]
        masks_s = masks_szk[:, None] & masks_sn[None, :]

        offsets_z = (N // 8) * offsets_szk[:, None] + offsets_bzn[None, :]
        zeros_ptrs = zeros_ptr + offsets_z
        zeros = tl.load(zeros_ptrs, mask=masks_z, other=other_bzs)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)
        zeros = tl.interleave(zeros, zeros)

        offsets_s = N * offsets_szk[:, None] + offsets_sn[None, :]
        scales_ptrs = scales_ptr + offsets_s
        scales = tl.load(scales_ptrs, mask=masks_s, other=other_bzs)

        if NUM_GROUPS == 1:
            # Original efficient implementation for single group
            zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))
            scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))
        else:
            # Reshape to (NUM_GROUPS, 1, N) then broadcast to (NUM_GROUPS, group_size_in_block, N)
            zeros = tl.broadcast_to(zeros[:, None, :], (NUM_GROUPS, GROUP_SIZE, BLOCK_SIZE_N))
            scales = tl.broadcast_to(scales[:, None, :], (NUM_GROUPS, GROUP_SIZE, BLOCK_SIZE_N))
            # Reshape back to (BLOCK_SIZE_K, N)
            zeros = tl.reshape(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))
            scales = tl.reshape(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))

        b = (b >> shifts) & 0xF
        zeros = (zeros >> shifts) & 0xF
        b = (b - zeros) * scales
        b = b.to(a_ptr.type.element_ty)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)

        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * (N // 8)

    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = tile_idx_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = tile_idx_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_c = N * offs_cm[:, None] + offs_cn[None, :]
    c_ptrs = c_ptr + offs_c
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if c_ptr.type.element_ty == tl.float16:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)

@triton.jit
def awq_gemm_kernel_streamk(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, K,
                    GROUP_SIZE: tl.constexpr, NUM_CUS: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
                    BLOCK_SIZE_K: tl.constexpr, NUM_GROUPS: tl.constexpr, A_LOAD_ORDER: tl.constexpr = 0):

    pid = tl.program_id(axis=0)

    tiles_M = tl.cdiv(M, BLOCK_SIZE_M)
    tiles_N = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = tiles_M * tiles_N
    dangling_tiles = total_tiles % NUM_CUS
    dp_tiles = total_tiles - dangling_tiles

    if pid < dp_tiles:
        iters_per_cta = tl.cdiv(K, BLOCK_SIZE_K)
        awq_gemm_kernel_inner(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, pid, 0, iters_per_cta,
                              M, N, K, GROUP_SIZE, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_GROUPS, A_LOAD_ORDER)
    else:
        iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
        total_iters = iters_per_tile * dangling_tiles
        iters_per_cta = tl.cdiv(total_iters, NUM_CUS)

        iter_begin = (pid - dp_tiles) * iters_per_cta
        iter_end = iter_begin + iters_per_cta

        while iter_begin < iter_end:
            tile_idx = iter_begin // iters_per_tile + dp_tiles
            tile_iter_begin = (tile_idx - dp_tiles) * iters_per_tile
            tile_iter_end = tile_iter_begin + iters_per_tile
            local_iter_begin = iter_begin - tile_iter_begin
            local_iter_end = tl.minimum(iter_end, tile_iter_end) - tile_iter_begin
            awq_gemm_kernel_inner(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, tile_idx, local_iter_begin, local_iter_end,
                                M, N, K, GROUP_SIZE, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_GROUPS, A_LOAD_ORDER)
            iter_begin = tile_iter_end

@triton.jit
def awq_gemm_kernel_splitk(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, K,
                    GROUP_SIZE: tl.constexpr, NUM_CUS: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr, SPLITK: tl.constexpr, NUM_GROUPS: tl.constexpr, A_LOAD_ORDER: tl.constexpr = 0):

    pid = tl.program_id(axis=0)

    tiles_M = tl.cdiv(M, BLOCK_SIZE_M)
    tiles_N = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = tiles_M * tiles_N
    tile_idx = pid % total_tiles

    iters_per_cta = tl.cdiv(K, BLOCK_SIZE_K * SPLITK)
    iter_begin = pid // total_tiles * iters_per_cta
    iter_end = iter_begin + iters_per_cta

    awq_gemm_kernel_inner(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, tile_idx, iter_begin, iter_end,
                          M, N, K, GROUP_SIZE, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_GROUPS, A_LOAD_ORDER)

'''
@triton.autotune(
    configs=[
        triton.Config({
            "BLOCK_SIZE_M": M,
            "BLOCK_SIZE_N": N,
            "BLOCK_SIZE_K": K,
            "SCHEDULER": SCHEDULER,
            "SPLITK": SPLITK
        }, num_warps=num_warps, num_stages=num_stages)
        for M in [128, 64, 32] for N in [128, 64, 32] for K in [64, 32]\
        for SCHEDULER in [0, 1] for SPLITK in [1, 2, 4, 8]\
        for num_warps in [1, 2, 4, 8, 16] for num_stages in [1, 2]
    ],
    key=["M", "K", "N", "GROUP_SIZE"],
    perf_debug=True,
    prune_configs_by={
        "early_config_prune": lambda configs, nargs, **kwargs: [
            config for config in configs
            # SCHEDULE=1 代表 STREAMK，不需要遍历那么多 SPLITK 的值
            if config.all_kwargs()["SCHEDULER"] == 0 or (config.all_kwargs()["SCHEDULER"] == 1 and config.all_kwargs()["SPLITK"] == 1)
        ]
    } 
)
'''
@triton.heuristics(values={
    "NUM_GROUPS": lambda args: triton.cdiv(args["BLOCK_SIZE_K"], args["GROUP_SIZE"])
})
@triton.jit
def awq_gemm_kernel(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, K,
                    GROUP_SIZE: tl.constexpr, NUM_CUS: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, NUM_GROUPS: tl.constexpr,
                    SPLITK: tl.constexpr, SCHEDULER: tl.constexpr, A_LOAD_ORDER: tl.constexpr = 0):
        if SCHEDULER == 0:
            return awq_gemm_kernel_splitk(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, K, GROUP_SIZE, NUM_CUS, BLOCK_SIZE_M,
                                          BLOCK_SIZE_N, BLOCK_SIZE_K, SPLITK, NUM_GROUPS, A_LOAD_ORDER)
        else:
            return awq_gemm_kernel_streamk(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, K, GROUP_SIZE, NUM_CUS, BLOCK_SIZE_M,
                                           BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_GROUPS, A_LOAD_ORDER)

# qweights - [K     , M // 8], int32
# scales   - [K // G, M     ], float16
# zeros    - [K // G, M // 8], int32
def awq_dequantize_triton(qweight: torch.Tensor,
                          scales: torch.Tensor,
                          zeros: torch.Tensor,
                          block_size_x: int = 32,
                          block_size_y: int = 32) -> torch.Tensor:
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
    result = torch.empty(qweight.shape[0],
                         qweight.shape[1] * 8,
                         device=qweight.device,
                         dtype=scales.dtype)

    Y = qweight.shape[0]  # num rows
    X = qweight.shape[1]  # num cols

    grid = lambda META: (
        triton.cdiv(X, META['BLOCK_SIZE_X']),
        triton.cdiv(Y, META['BLOCK_SIZE_Y']),
    )
    awq_dequantize_kernel[grid](qweight,
                                scales,
                                zeros,
                                group_size,
                                result,
                                X,
                                Y,
                                BLOCK_SIZE_X=block_size_x,
                                BLOCK_SIZE_Y=block_size_y)

    return result

@functools.lru_cache
def get_w4a16_awq_gemm_config_filepath(N: int, K: int, GROUP_SIZE: int, **kwargs) -> str:
    device_name = current_platform.get_device_name().replace(" ", "_")
    json_file_name = f"awq_gemm_N={N},K={K},device_name={device_name},dtype=w4a16,group_size={GROUP_SIZE}.json"

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )

    return config_file_path

@functools.lru_cache
def get_w4a16_awq_gemm_configs(
    N: int, K: int, GROUP_SIZE: int
) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the w8a8 block fp8 kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the w8a8 block fp8 kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """
    config_file_path = get_w4a16_awq_gemm_config_filepath(N, K, GROUP_SIZE)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            print(
                "\nUsing configuration from {} for W4A16 AWQ GEMM kernel.".format(config_file_path),
            )
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    print(
        (
            "\nUsing default W4A16 AWQ GEMM kernel config. Performance might "
            "be sub-optimal! Config file not found at {}"
        ).format(config_file_path)
    )
    return None

# input   - [M, K]
# qweight - [K, N // 8]
# qzeros  - [K // G, N // 8]
# scales  - [K // G, N]
# split_k_iters - parallelism along K-dimension, int, power of 2.
def awq_gemm_triton(input: torch.Tensor,
                    qweight: torch.Tensor,
                    scales: torch.Tensor,
                    qzeros: torch.Tensor) -> torch.Tensor:
    M, K = input.shape
    N = qweight.shape[1] * 8
    group_size = qweight.shape[0] // qzeros.shape[0]

    assert N > 0 and K > 0 and M > 0
    assert qweight.shape[0] == K and qweight.shape[1] == N // 8
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 8
    assert scales.shape[0] == K // group_size and scales.shape[1] == N
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    configs = get_w4a16_awq_gemm_configs(N, K, group_size)
    if configs:
        # If an optimal configuration map has been found, look up the
        # optimal config
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        # Default config
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "SCHEDULER": 1,
            "SPLITK": 1
        }

    NUM_CUS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def grid(META):
        tiles_M = (M + META["BLOCK_SIZE_M"] - 1) // META["BLOCK_SIZE_M"]
        tiles_N = (N + META["BLOCK_SIZE_N"] - 1) // META["BLOCK_SIZE_N"]
        total_tiles = tiles_M * tiles_N
        if META["SCHEDULER"] == 0:
            # splitk
            return (total_tiles * META["SPLITK"],)
        else:
            # streamk
            dangling_tiles = total_tiles % NUM_CUS
            dp_tiles = total_tiles - dangling_tiles
            return (dp_tiles + NUM_CUS * (dangling_tiles > 0),)

    if config["SCHEDULER"] == 0:
        c_dtype = torch.float16 if config["SPLITK"] == 1 else torch.float32
    else:
        tiles_M = (M + config["BLOCK_SIZE_M"] - 1) // config["BLOCK_SIZE_M"]
        tiles_N = (N + config["BLOCK_SIZE_N"] - 1) // config["BLOCK_SIZE_N"]
        total_tiles = tiles_M * tiles_N
        dangling_tiles = total_tiles % NUM_CUS
        c_dtype = torch.float16 if dangling_tiles == 0 else torch.float32
    result = torch.zeros((M, N), dtype=c_dtype, device=input.device)

    # A = input, B = qweight, C = result
    # A = M x K, B = K x N, C = M x N
    if int(os.getenv("TRITON_COMPILE_ONLY", 0)) == 1:
        func = partial(awq_gemm_kernel.warmup, grid=grid)
    else:
        func = awq_gemm_kernel[grid]
    func(input,
         qweight,
         result,
         qzeros,
         scales,
         M,
         N,
         K,
         group_size,
         NUM_CUS,
         **config)

    result = result.to(torch.float16)

    return result
