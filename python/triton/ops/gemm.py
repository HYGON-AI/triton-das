# SPDX-License-Identifier: Apache-2.0
import os
import json
import functools

import torch
import triton
import triton.language as tl

from typing import Optional
from .utils.config_utils import str_to_tuple


@triton.jit
def gemm_kernel_inner(a_ptr, b_ptr, c_ptr, tile_idx, k_idx, iter_begin, iter_end,
                      M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                      BLOCK_SIZE_K: tl.constexpr, RASTER: tl.constexpr,
                      SWIZZLE_SIZE: tl.constexpr,
                      M_DIVISIBLE_BY_BLOCK_SIZE_M: tl.constexpr,
                      N_DIVISIBLE_BY_BLOCK_SIZE_N: tl.constexpr,
                      K_DIVISIBLE_BY_BLOCK_SIZE_K: tl.constexpr,
                      A_LOAD_ORDER: tl.constexpr = 0):

    tl.assume(tile_idx >= 0)
    tl.assume(iter_begin >= 0)
    tl.assume(iter_end >= 0)
    tl.assume(k_idx >= 0)
    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(K > 0)

    num_tile_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tile_n = tl.cdiv(N, BLOCK_SIZE_N)

    if RASTER == 0:
        # along M
        num_pid_in_group = num_tile_m * SWIZZLE_SIZE
        group_idx = tile_idx // num_pid_in_group
        group_size_n = min(num_tile_n - group_idx * SWIZZLE_SIZE, SWIZZLE_SIZE)
        tile_idx_m = (tile_idx % num_pid_in_group) // group_size_n
        tile_idx_n = group_idx * SWIZZLE_SIZE + tile_idx % group_size_n
    else:
        # along N
        num_pid_in_group = num_tile_n * SWIZZLE_SIZE
        group_idx = tile_idx // num_pid_in_group
        group_size_m = min(num_tile_m - group_idx * SWIZZLE_SIZE, SWIZZLE_SIZE)
        tile_idx_n = (tile_idx % num_pid_in_group) // group_size_m
        tile_idx_m = group_idx * SWIZZLE_SIZE + tile_idx % group_size_m

    tile_idx_m = tile_idx // num_tile_n
    tile_idx_n = tile_idx % num_tile_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Offsets and masks.
    offsets_am = tile_idx_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_am = offsets_am < M

    offsets_bn = tile_idx_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_bn = tl.max_contiguous(tl.multiple_of(offsets_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    masks_bn = offsets_bn < N

    offsets_k = iter_begin * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offsets_ak = tl.max_contiguous(tl.multiple_of(offsets_k, BLOCK_SIZE_K), BLOCK_SIZE_K)
    offsets_a = K * offsets_am[:, None] + offsets_ak[None, :]
    offsets_b = N * offsets_k[:, None] + offsets_bn[None, :]

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b
    for k in range(iter_end - iter_begin):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        masks_b = masks_k[:, None] & masks_bn[None, :]
        if M_DIVISIBLE_BY_BLOCK_SIZE_M & K_DIVISIBLE_BY_BLOCK_SIZE_K:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=masks_a, other=0.)
        if N_DIVISIBLE_BY_BLOCK_SIZE_N & K_DIVISIBLE_BY_BLOCK_SIZE_K:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=masks_b, other=0.)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)

        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N

    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = tile_idx_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = tile_idx_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_c = M * N * k_idx + N * offs_cm[:, None] + offs_cn[None, :]
    c_ptrs = c_ptr + offs_c
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if M_DIVISIBLE_BY_BLOCK_SIZE_M & N_DIVISIBLE_BY_BLOCK_SIZE_N:
        tl.store(c_ptrs, c)
    else:
        tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def gemm_kernel_streamk(a_ptr, b_ptr, c_ptr, M, N, K, NUM_CUS: tl.constexpr,
                       BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                       BLOCK_SIZE_K: tl.constexpr,
                       DP_TILES: tl.constexpr, DANGLING_TILES: tl.constexpr,
                       RASTER: tl.constexpr, SWIZZLE_SIZE: tl.constexpr,
                       M_DIVISIBLE_BY_BLOCK_SIZE_M: tl.constexpr,
                       N_DIVISIBLE_BY_BLOCK_SIZE_N: tl.constexpr,
                       K_DIVISIBLE_BY_BLOCK_SIZE_K: tl.constexpr,
                       A_LOAD_ORDER: tl.constexpr = 0):

    pid = tl.program_id(axis=0)

    if pid < DP_TILES:
        iters_per_cta = tl.cdiv(K, BLOCK_SIZE_K)
        gemm_kernel_inner(a_ptr, b_ptr, c_ptr, pid, 0, 0, iters_per_cta, M, N, K,
                          BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, RASTER, SWIZZLE_SIZE, M_DIVISIBLE_BY_BLOCK_SIZE_M,
                          N_DIVISIBLE_BY_BLOCK_SIZE_N, K_DIVISIBLE_BY_BLOCK_SIZE_K, A_LOAD_ORDER)
    else:
        iters_per_tile = tl.cdiv(K, BLOCK_SIZE_K)
        total_iters = iters_per_tile * DANGLING_TILES
        iters_per_cta = tl.cdiv(total_iters, NUM_CUS)

        iter_begin = (pid - DP_TILES) * iters_per_cta
        iter_end = tl.minimum(iter_begin + iters_per_cta, total_iters)

        while iter_begin < iter_end:
            tile_idx = iter_begin // iters_per_tile + DP_TILES
            tile_iter_begin = (tile_idx - DP_TILES) * iters_per_tile
            tile_iter_end = tile_iter_begin + iters_per_tile
            local_iter_begin = iter_begin - tile_iter_begin
            local_iter_end = tl.minimum(iter_end, tile_iter_end) - tile_iter_begin
            k_idx = tl.cdiv(local_iter_begin, iters_per_cta)
            gemm_kernel_inner(a_ptr, b_ptr, c_ptr, tile_idx, k_idx, local_iter_begin,
                              local_iter_end, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                              RASTER, SWIZZLE_SIZE, M_DIVISIBLE_BY_BLOCK_SIZE_M,
                              N_DIVISIBLE_BY_BLOCK_SIZE_N, K_DIVISIBLE_BY_BLOCK_SIZE_K, A_LOAD_ORDER)
            iter_begin = tile_iter_end


@triton.jit
def gemm_kernel_splitk(a_ptr, b_ptr, c_ptr, M, N, K, NUM_CUS: tl.constexpr,
                       BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                       SPLITK: tl.constexpr, RASTER: tl.constexpr, SWIZZLE_SIZE: tl.constexpr,
                       M_DIVISIBLE_BY_BLOCK_SIZE_M: tl.constexpr,
                       N_DIVISIBLE_BY_BLOCK_SIZE_N: tl.constexpr, K_DIVISIBLE_BY_BLOCK_SIZE_K: tl.constexpr,
                       A_LOAD_ORDER: tl.constexpr = 0):

    pid = tl.program_id(axis=0)

    tiles_M = tl.cdiv(M, BLOCK_SIZE_M)
    tiles_N = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = tiles_M * tiles_N
    tile_idx = pid % total_tiles

    iters_per_cta = tl.cdiv(K, BLOCK_SIZE_K * SPLITK)
    iter_begin = pid // total_tiles * iters_per_cta
    iter_end = iter_begin + iters_per_cta
    k_idx = tl.cdiv(iter_begin, iters_per_cta)

    gemm_kernel_inner(a_ptr, b_ptr, c_ptr, tile_idx, k_idx, iter_begin, iter_end,
                          M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                          RASTER, SWIZZLE_SIZE, M_DIVISIBLE_BY_BLOCK_SIZE_M,
                          N_DIVISIBLE_BY_BLOCK_SIZE_N, K_DIVISIBLE_BY_BLOCK_SIZE_K, A_LOAD_ORDER)


@triton.heuristics(values={
    "M_DIVISIBLE_BY_BLOCK_SIZE_M": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0,
    "N_DIVISIBLE_BY_BLOCK_SIZE_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
    "K_DIVISIBLE_BY_BLOCK_SIZE_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
})
@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                NUM_CUS: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                BLOCK_SIZE_K: tl.constexpr, DP_TILES: tl.constexpr, DANGLING_TILES: tl.constexpr,
                SPLITK: tl.constexpr, SCHEDULER: tl.constexpr, RASTER: tl.constexpr, SWIZZLE_SIZE: tl.constexpr,
                M_DIVISIBLE_BY_BLOCK_SIZE_M: tl.constexpr = False,
                N_DIVISIBLE_BY_BLOCK_SIZE_N: tl.constexpr = False,
                K_DIVISIBLE_BY_BLOCK_SIZE_K: tl.constexpr = False,
                A_LOAD_ORDER: tl.constexpr = 0):
        if SCHEDULER == 0 or DANGLING_TILES == 0:
            # splitk, including dp (splik=1)
            return gemm_kernel_splitk(a_ptr, b_ptr, c_ptr, M, N, K, NUM_CUS, BLOCK_SIZE_M, BLOCK_SIZE_N,
                                      BLOCK_SIZE_K, SPLITK, RASTER, SWIZZLE_SIZE, M_DIVISIBLE_BY_BLOCK_SIZE_M,
                                      N_DIVISIBLE_BY_BLOCK_SIZE_N, K_DIVISIBLE_BY_BLOCK_SIZE_K, A_LOAD_ORDER)
        else:
            return gemm_kernel_streamk(a_ptr, b_ptr, c_ptr, M, N, K, NUM_CUS, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                                       DP_TILES, DANGLING_TILES, RASTER, SWIZZLE_SIZE, M_DIVISIBLE_BY_BLOCK_SIZE_M,
                                       N_DIVISIBLE_BY_BLOCK_SIZE_N, K_DIVISIBLE_BY_BLOCK_SIZE_K, A_LOAD_ORDER)


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
):
    if not hasattr(_get_config, "_config_dict"):
        from triton.ops import TRITON_OP_CONFIGS_DIR
        from triton.utils import get_gpu_label
        # load config from file
        fpath = f"{TRITON_OP_CONFIGS_DIR}/gemm_kernel-{get_gpu_label()}-fp16.json"
        with open(fpath) as f:
            data = json.load(f)
        # # or load config from cache
        # data = triton.utils.get_config_cache("gemm_kernel", get_gpu_label())
        _get_config._config_dict = data

    total_M = [str_to_tuple(k)[0] for k in data.keys()]
    m = min(total_M, key=lambda x: abs(x - M))
    return _get_config._config_dict[str(tuple([m, K, N]))]


# input  - [M, K]
# weight - [K, N]
def gemm(input: torch.Tensor, weight: torch.Tensor, config: Optional[dict] = None) -> torch.Tensor:
    M, K = input.shape
    N = weight.shape[1]

    assert N > 0 and K > 0 and M > 0
    assert weight.shape[0] == K

    NUM_CUS = torch.cuda.get_device_properties("cuda").multi_processor_count

    if config is None:
        config = _get_config(M, N, K)

    if config["SCHEDULER"] == 1:
        tiles_M = (M + config["BLOCK_SIZE_M"] - 1) // config["BLOCK_SIZE_M"]
        tiles_N = (N + config["BLOCK_SIZE_N"] - 1) // config["BLOCK_SIZE_N"]
        total_tiles = tiles_M * tiles_N
        dangling_tiles = max(0, total_tiles - NUM_CUS) % total_tiles
        dp_tiles = total_tiles - dangling_tiles
        config["DP_TILES"] = dp_tiles
        config["DANGLING_TILES"] = dangling_tiles
    else:
        config["DP_TILES"] = 0
        config["DANGLING_TILES"] = 0

    def grid(META):
        tiles_M = (M + META["BLOCK_SIZE_M"] - 1) // META["BLOCK_SIZE_M"]
        tiles_N = (N + META["BLOCK_SIZE_N"] - 1) // META["BLOCK_SIZE_N"]
        total_tiles = tiles_M * tiles_N
        if META["SCHEDULER"] == 0:
            # splitk
            return (total_tiles * META["SPLITK"],)
        else:
            # streamk
            if META["DANGLING_TILES"] == 0:
                return (total_tiles,)
            iters_per_tile = (K + META["BLOCK_SIZE_K"] - 1) // META["BLOCK_SIZE_K"]
            dangling_iters = iters_per_tile * META["DANGLING_TILES"]
            dangling_iters_per_cu = (dangling_iters + NUM_CUS - 1) // NUM_CUS
            num_cus_streamk = (dangling_iters + dangling_iters_per_cu - 1) // dangling_iters_per_cu
            return (META["DP_TILES"] + num_cus_streamk,)

    if (config["SCHEDULER"] == 0 and config["SPLITK"] == 1) or config["DANGLING_TILES"] == 0:
        # dp
        result = torch.zeros((M, N), dtype=torch.float16, device=input.device)
    elif config["SCHEDULER"] == 0 and config["SPLITK"] > 1:
        # splitk
        result = torch.zeros((config["SPLITK"], M, N), dtype=torch.float32, device=input.device)
    else:
        # streamk
        iters_per_tile = (K + config["BLOCK_SIZE_K"] - 1) // config["BLOCK_SIZE_K"]
        dangling_iters = iters_per_tile * dangling_tiles
        dangling_iters_per_cu = (dangling_iters + NUM_CUS - 1) // NUM_CUS
        num_cu_per_dangling_tile = (iters_per_tile + dangling_iters_per_cu - 1) // dangling_iters_per_cu + 1
        result = torch.zeros((num_cu_per_dangling_tile, M, N), dtype=torch.float32, device=input.device)

    # A = input, B = qweight, C = result
    # A = M x K, B = K x N, C = M x N
    gemm_kernel[grid](input, weight, result, M, N, K, NUM_CUS, **config)

    if result.ndim == 3:
        result = result.sum(0)

    result = result.to(torch.float16)

    return result
