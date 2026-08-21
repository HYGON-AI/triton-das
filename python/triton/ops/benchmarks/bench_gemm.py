# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""GEMM Triton Implementation"""

import torch
import triton
import os
import numpy as np

from triton.ops.gemm import gemm_kernel


device_id = 0
torch.cuda.set_device(device_id)
device = "cuda"

GEMM_KN_CASES = [
    # "K, N"
    (256, 576),
    (256, 1536),
    (256, 3072),
    (256, 4096),
    (256, 4608),
    (256, 7168),

    (512, 576),
    (512, 1536),
    (512, 3072),
    (512, 4096),
    (512, 4608),
    (512, 7168),

    (1536, 576),
    (1536, 1536),
    (1536, 3072),
    (1536, 4096),
    (1536, 4608),
    (1536, 7168),

    (2048, 512),
    (2048, 1536),
    (2048, 3072),
    (2048, 4096),
    (2048, 4608),
    (2048, 7168),

    (2304, 512),
    (2304, 1536),
    (2304, 3072),
    (2304, 4096),
    (2304, 4608),
    (2304, 7168),

    (7168, 512),
    (7168, 1536),
    (7168, 3072),
    (7168, 4096),
    (7168, 4608),
    (7168, 7168),
]

GEMM_TEST_CASES = [
    # "M, K, N"
    (M, *KN) for M in range(1, 129, 10) for KN in GEMM_KN_CASES
]
GEMM_TEST_CASES.sort(key=lambda x: (x[0], x[1], x[2]))


def is_bad_heuristics(M, K, N, heuristics):
    _ceil_div = lambda x, y: (x + y - 1) // y
    num_cus = torch.cuda.get_device_properties("cuda").multi_processor_count
    tiles_m = _ceil_div(M, heuristics["BLOCK_SIZE_M"])
    tiles_n = _ceil_div(N, heuristics["BLOCK_SIZE_N"])
    total_tiles = tiles_m * tiles_n
    last_wave_tiles = total_tiles % num_cus
    iters_per_tile = _ceil_div(K, heuristics["BLOCK_SIZE_K"])

    # 如果本来 dp 就没有 idle，那么不需要尝试 splitk
    if last_wave_tiles == num_cus and \
       (heuristics["SCHEDULER"] == 0 and heuristics["SPLITK"] > 1):
       return True

    # 也不需要尝试 streamk
    if last_wave_tiles == num_cus and heuristics["SCHEDULER"] == 1:
        return True

    # 如果 splitk 没有减少 idle，不要尝试了
    if heuristics["SCHEDULER"] == 0 and heuristics["SPLITK"] > 1:
        total_tiles_ = total_tiles * heuristics["SPLITK"]
        last_wave_tiles_ = total_tiles % num_cus
        if last_wave_tiles_ <= last_wave_tiles:
            return True

    # 如果 tile 数量不足以做 dp+1streamk，不要尝试
    if total_tiles <= num_cus and heuristics["SCHEDULER"] == 1:
        return True

    # 如果 streamk 没有减少 idle，不要尝试
    if heuristics["SCHEDULER"] == 1:
        dangling_tiles = max(0, total_tiles - num_cus) % num_cus
        total_dangling_iters = dangling_tiles * iters_per_tile
        iters_per_cu = _ceil_div(total_dangling_iters, num_cus)
        num_cus_to_stream = _ceil_div(total_dangling_iters, iters_per_cu)
        if num_cus_to_stream <= last_wave_tiles:
            return True

    # streamk 的时候不要尝试其他 splitk
    if heuristics["SPLITK"] > 1 and heuristics["SCHEDULER"] == 1:
        return True


def gemm(input: torch.Tensor, weight: torch.Tensor, heuristics: dict = {}) -> torch.Tensor:
    M, K = input.shape
    N = weight.shape[1]

    assert N > 0 and K > 0 and M > 0
    assert weight.shape[0] == K

    NUM_CUS = torch.cuda.get_device_properties("cuda").multi_processor_count

    default_config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "SCHEDULER": 0,
        "SPLITK": 1,
    }
    config = default_config
    config.update(heuristics)

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

    # delete re-defined meta-params in Hcutuner
    for k in heuristics:
        del config[k]

    # A = input, B = qweight, C = result
    # A = M x K, B = K x N, C = M x N
    BLOCK_SIZE_M = heuristics['BLOCK_SIZE_M']
    BLOCK_SIZE_N = heuristics['BLOCK_SIZE_N']

    def prune_configs(configs, nargs, **kwargs):
        def _ceil_div(x, y):
            return (x + y - 1) // y

        M, N, NUM_CUS = nargs["M"], nargs["N"], nargs["NUM_CUS"]
        num_block_m = _ceil_div(M, BLOCK_SIZE_M)
        num_block_n = _ceil_div(N, BLOCK_SIZE_N)

        def _prune(config):
            _config = config.all_kwargs()

            # 如果 M 和 N 方向的 block 数量相等，只选择 M 方向 RASTER，删除掉 N 方向的
            if num_block_m == num_block_n and _config["RASTER"] == 1:
                return True

            # 如果 SWIZZLE_SIZE 超出了对应的方向的 block 数量，那么删除掉
            if (_config["RASTER"] == 0 and _config["SWIZZLE_SIZE"] > num_block_n) \
                or (_config["RASTER"] == 1 and _config["SWIZZLE_SIZE"] > num_block_m):
                return True

        remained = [c for c in configs if not _prune(c)]
        return remained

    fn = triton.utils.hcutune(
            configs=[
                triton.Config({
                    **heuristics,
                    "RASTER": RASTER,
                    "SWIZZLE_SIZE": SWIZZLE_SIZE, "kpack": kpack, "instruction_sched_variant": instruction_sched_variant
                }, num_warps=num_warps, num_stages=num_stages)
                for RASTER in [0, 1] for SWIZZLE_SIZE in [1, 2, 4, 8, 16, 32]\
                for num_warps in [1, 2, 4, 8, 16] for num_stages in [1, 2] for kpack in [1, 2] for instruction_sched_variant in ['none', 'llvm-iglp-1', 'local-prefetch']
            ],
            key=["M", "K", "N", "GROUP_SIZE"],
            perf_debug=True,
            prune_configs_by={
                "early_config_prune": prune_configs
            }
        )(gemm_kernel)

    fn[grid](input, weight, result, M, N, K, NUM_CUS, **config)

    if result.ndim == 3:
        result = result.sum(0)

    result = result.to(torch.float16)
    return result


# Benchmark configurations
bench_configs = [
    triton.testing.Benchmark(
        x_names=['M', 'K', 'N'],
        x_vals=GEMM_TEST_CASES[:1],
        line_arg='Provider',
        line_vals=["Triton"],
        line_names=['Triton'],
        styles=[('red', '-'), ('blue', '--')],
        ylabel='ms',
        xlabel='Matrix Dimensions (M×K×N)',
        plot_name='GEMM Performance',
        args={'device': device}
    )
]


# @triton.testing.perf_report(bench_configs)
@triton.utils.dist_perf_report(bench_configs)
def bench_gemm(M, K, N, Provider, device="cuda", heuristics={}):
    """Benchmark GEMM performance.

    Args:
        M: Number of rows in input matrix
        K: Number of columns in input matrix
        N: Number of columns in weight matrix (pre-quantization)
        provider: Implementation provider ('triton' or 'torch')
        device: Device to run on
    """
    warmup = 25
    rep = 10

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

    if heuristics and is_bad_heuristics(M, K, N, heuristics):
        # Give bad heuristic a pretty bad performance
        return 9999

    fn = lambda: gemm(input_tensor, weight, heuristics)
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


if __name__ == "__main__":

    os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"

    all_heuristics = [
        {
            "BLOCK_SIZE_M": BLOCK_SIZE_M,
            "BLOCK_SIZE_N": BLOCK_SIZE_N,
            "BLOCK_SIZE_K": BLOCK_SIZE_K,
            "SCHEDULER": SCHEDULER,
            "SPLITK": SPLITK
        }
        for BLOCK_SIZE_M in [128, 64, 32, 16]
        for BLOCK_SIZE_N in [128, 64, 32]
        for BLOCK_SIZE_K in [64, 32]
        for SCHEDULER in [0, 1]
        for SPLITK in [1, 2, 4, 8]
    ]

    from tqdm import tqdm
    for heuristics in tqdm(all_heuristics, "Benching heuristics"):
        bench_gemm.run(print_data=True, heuristics=heuristics)
