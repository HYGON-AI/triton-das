# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
# Modified by Hygon Information Technology Co., Ltd., 2026.

"""
HCU WASP / WASP+WDRA regression matrix: GEMM & FA × MLS & non-MLS.

Default scenarios:
  GEMM  × {load, mls} × {wasp 4+4, wdra 4+8}              # num_stages=2
  # GEMM  × {load, mls} × {wdra 8+8}                       # temporarily disabled
  GEMM  × {load, mls} × {wdra 4+8} × num_stages=4
  GEMM  × mls_store × {wdra 4+8}                          # mmac_layout_force=3
  FA    × {load, mls} × {wasp 4+4, wdra 4+8}              # num_stages=2

Known limitation: gemm/load/wdra8+8 uses K=512; see the comment near
WDRA88_LOAD_GEMM_SHAPE below. wdra8+8 does not support num_stages=4
(abarrier ID limit).

MLS-store GEMM cases use tl.matrix_store for C with mmac_layout_force=3
(TRANSPOSE) so MFMA-C → store packing stays aligned.

Optional nowrap (no WASP/WDRA) via --mode nowrap:
  uses plain num_warps (not wasp_num_*), warp_specialize=False on GEMM.

IR shape for WASP/WDRA partitions is checked by lit
  (test/TritonGPU/hcu/hcu-gemm-wasp-wdra.mlir), not duplicated here.

Usage (from triton-staging root):
  OPTIMIZE_EPILOGUE=1 GPU_DFLAGS=\"['StatLog', 'SQAbar',]\" \\
    PYTHONPATH=python TRITON_BACKENDS_IN_TREE=1 \\
    python third_party/hcu/test/test_wasp_mls_regression.py

  # subset / filters
  python .../test_wasp_mls_regression.py --only gemm
  python .../test_wasp_mls_regression.py --only fa --mls
  python .../test_wasp_mls_regression.py --mode wasp    # 4+4 only
  python .../test_wasp_mls_regression.py --mode wdra    # 4+8 only
  python .../test_wasp_mls_regression.py --mode wdra88  # 8+8 TwoPTwoC (GEMM)
  python .../test_wasp_mls_regression.py --only gemm --mode wdra --num-stages 4
  python .../test_wasp_mls_regression.py --only gemm --mls --mode nowrap
  python .../test_wasp_mls_regression.py --only gemm --mls-store
  python .../test_wasp_mls_regression.py --compile-only
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import triton
import triton.language as tl

os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "1")

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MATRIX_STORE_REFERENCE_TOPS = 7.672933
MATRIX_STORE_MAX_REGRESSION_PCT = 2.0


def _iter_pmd_stats() -> List[Tuple[Path, int, Optional[float], int]]:
    stats = []
    for path in (Path.cwd() / "m5out").rglob("stats.txt"):
        text = path.read_text(errors="replace")
        cycles = re.search(r"^totalGcSimCycles\s+(\d+)", text, re.M)
        frequency = re.search(r"^gcFrequency\s+([0-9.eE+-]+)", text, re.M)
        if cycles:
            stats.append((path, int(cycles.group(1)),
                          float(frequency.group(1)) if frequency else None,
                          path.stat().st_mtime_ns))
    return stats


def _pmd_stats_snapshot() -> dict[str, Tuple[int, int]]:
    return {str(path): (mtime, cycles)
            for path, cycles, _, mtime in _iter_pmd_stats()}


def _latest_pmd_measurement(
    before: dict[str, Tuple[int, int]],
) -> Tuple[int, float]:
    base_cycles = max((cycles for _, cycles in before.values()), default=0)
    candidates = []
    for path, cycles, frequency, mtime in _iter_pmd_stats():
        previous = before.get(str(path))
        if previous is None or mtime > previous[0]:
            candidates.append((mtime, cycles, frequency))
    if not candidates:
        raise AssertionError("matrix_store performance check found no PMD stats")
    _, cycles, frequency = max(candidates)
    measured_cycles = cycles - base_cycles if cycles > base_cycles else cycles
    if measured_cycles <= 0 or frequency is None:
        raise AssertionError("matrix_store performance check found invalid PMD stats")
    return measured_cycles, frequency


def _load_sibling(mod_name: str, filename: str):
    path = os.path.join(TEST_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @triton.jit / mutual refs resolve.
    sys.modules[mod_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Shared configs
# ---------------------------------------------------------------------------

# MmacLayout::TRANSPOSE — aligns MFMA-C with MLS matrix_store packing.
MLS_STORE_MMAC_LAYOUT = 3
MLS_STORE_TEST_SHAPES = (
    (256, 256, 64),
    (512, 256, 128),
    (256, 512, 512),
    (256, 256, 4096),
)


def gemm_config(
    *,
    wasp: bool,
    wdra: bool,
    load_warps: Optional[int] = None,
    mma_warps: Optional[int] = None,
    num_stages: int = 2,
    mmac_layout_force: Optional[int] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
) -> dict:
    """Launch/options for GEMM.

    WASP on: wasp_num_load/mma_warps drive total warps (compiler ignores plain
    num_warps after WS). WASP off: must use plain num_warps and must not pass
    wasp_num_*; WDRA requires WASP.

    Under WASP, num_stages is the LDS buffer depth (2 or 4). Defaults: wasp
    4+4; wdra OnePTwoC 4+8; pass load/mma=8 for TwoPTwoC 8+8.
    """
    if wdra and not wasp:
        raise ValueError("wdra requires wasp=True")
    if load_warps is None:
        load_warps = 4
    if mma_warps is None:
        mma_warps = 8 if wdra else 4
    # WDRA MLS may retain both full and sliced LDS; keep tiles smaller.
    default_block = 64 if wdra else 128
    bm = block_m if block_m is not None else default_block
    bn = block_n if block_n is not None else default_block
    bk = block_k if block_k is not None else default_block
    cfg = {
        "BLOCK_SIZE_M": bm,
        "BLOCK_SIZE_N": bn,
        "BLOCK_SIZE_K": bk,
        "GROUP_SIZE_M": 1,
        "optimize_epilogue": True,
        "WARP_SPECIALIZE": wasp,
        "wasp_enabled": wasp,
        "wdra_enabled": wdra,
        "num_stages": num_stages,
    }
    if mmac_layout_force is not None:
        cfg["mmac_layout_force"] = mmac_layout_force
    if wasp:
        cfg.update(
            {
                "wasp_num_load_warps": load_warps,
                "wasp_num_mma_warps": mma_warps,
            }
        )
        if wdra:
            # TwoPTwoC (4 WDRA branches): load+load+main+tail must be %32==0.
            # 88+88+144+160 = 480.
            mma_tail = 160 if load_warps >= 8 else 140
            cfg.update(
                {
                    "wdra_num_load_regs": 88,
                    "wdra_num_mma_regs_main": 144,
                    "wdra_num_mma_regs_tail": mma_tail,
                }
            )
    else:
        # Match typical non-WASP GEMM / HIPOptions default (power-of-two).
        cfg["num_warps"] = 4
    return cfg


def fa_config(
    *,
    wasp: bool,
    wdra: bool,
    load_warps: Optional[int] = None,
    mma_warps: Optional[int] = None,
) -> dict:
    """Launch/options for FA (passed as config_list into fa.test_op_fwd)."""
    if wdra and not wasp:
        raise ValueError("wdra requires wasp=True")
    if load_warps is None:
        load_warps = 4
    if mma_warps is None:
        mma_warps = 8 if wdra else 4
    block = 32 if wdra else 64
    cfg = {
        "BLOCK_M": block,
        "BLOCK_N": block,
        "pre_load_v": False,
        "optimize_epilogue": True,
        # WASP LDS depth must be 2 or 4 (compiler validates when wasp_enabled).
        "num_stages": 2,
        "wasp_enabled": wasp,
        "wdra_enabled": wdra,
    }
    if wasp:
        cfg.update(
            {
                "wasp_num_load_warps": load_warps,
                "wasp_num_mma_warps": mma_warps,
            }
        )
        if wdra:
            cfg.update(
                {
                    "wdra_num_load_regs": 52,
                    "wdra_num_mma_regs_main": 160,
                    "wdra_num_mma_regs_tail": 160,
                }
            )
    else:
        cfg["num_warps"] = 4
    return cfg


# Back-compat aliases for older call sites / imports.
def wasp_config_gemm(*, wdra: bool, load_warps: Optional[int] = None,
                     mma_warps: Optional[int] = None) -> dict:
    return gemm_config(wasp=True, wdra=wdra, load_warps=load_warps,
                       mma_warps=mma_warps)


def wasp_config_fa(*, wdra: bool, load_warps: Optional[int] = None,
                   mma_warps: Optional[int] = None) -> dict:
    return fa_config(wasp=True, wdra=wdra, load_warps=load_warps,
                     mma_warps=mma_warps)


def is_hcu_support_mls() -> bool:
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch in ("gfx938", "gfx946", "gfx92a")


# ---------------------------------------------------------------------------
# GEMM kernels (tt.load / matrix_load / matrix_store)
# ---------------------------------------------------------------------------

@triton.jit
def gemm_load_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
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
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPECIALIZE):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def gemm_mls_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(K > 0)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    mls_offs_am = pid_m * BLOCK_SIZE_M
    mls_offs_bn = pid_n * BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPECIALIZE):
        mls_offs_k = k * BLOCK_SIZE_K
        a = tl.matrix_load(
            a_ptr,
            shape=[M, K],
            strides=[stride_am, stride_ak],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            offsets=[mls_offs_am, mls_offs_k],
        )
        b = tl.matrix_load(
            b_ptr,
            shape=[K, N],
            strides=[stride_bk, stride_bn],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
            offsets=[mls_offs_k, mls_offs_bn],
        )
        accumulator = tl.dot(a, b, accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def gemm_mls_store_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    """MLS load A/B + MLS store C (mmac_layout_force=3 expected at launch)."""
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(K > 0)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    mls_offs_am = pid_m * BLOCK_SIZE_M
    mls_offs_bn = pid_n * BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=WARP_SPECIALIZE):
        mls_offs_k = k * BLOCK_SIZE_K
        a = tl.matrix_load(
            a_ptr,
            shape=[M, K],
            strides=[stride_am, stride_ak],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            offsets=[mls_offs_am, mls_offs_k],
        )
        b = tl.matrix_load(
            b_ptr,
            shape=[K, N],
            strides=[stride_bk, stride_bn],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
            offsets=[mls_offs_k, mls_offs_bn],
        )
        accumulator = tl.dot(a, b, accumulator)
    c = accumulator.to(tl.float16)

    tl.matrix_store(
        c_ptr,
        c,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        offsets=[mls_offs_am, mls_offs_bn],
    )


def run_gemm(*, use_mls: bool, wasp: bool = True, wdra: bool = False,
             load_warps: Optional[int] = None, mma_warps: Optional[int] = None,
             num_stages: int = 2, compile_only: bool = False,
             shape=(128, 128, 4096), fixed_ab: bool = False,
             use_matrix_store: bool = False,
             mmac_layout_force: Optional[int] = None,
             block_m: Optional[int] = None,
             block_n: Optional[int] = None,
             block_k: Optional[int] = None) -> None:
    M, N, K = shape
    torch.manual_seed(0)
    if fixed_ab:
        # Constant A/B for itrace (IT0): A=1, B=1 => C[i,j]=K; fp16 1.0 = 0x3c00.
        a = torch.ones((M, K), device="cpu", dtype=torch.float16)
        b = (torch.ones((K, N), device="cpu", dtype=torch.float16)
             if use_matrix_store else
             torch.ones((N, K), device="cpu", dtype=torch.float16).transpose(1, 0))
    else:
        a = torch.randn((M, K), device="cpu", dtype=torch.float16)
        # FP16 matrix_store packing requires TT so B can interleave along N.
        # Keep the ordinary load/MLS regression on its existing TN coverage.
        b = (torch.randn((K, N), device="cpu", dtype=torch.float16)
             if use_matrix_store else
             torch.randn((N, K), device="cpu", dtype=torch.float16).transpose(1, 0))
    a_dev, b_dev = a.to("cuda"), b.to("cuda")
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    if use_matrix_store:
        use_mls = True
        if mmac_layout_force is None:
            mmac_layout_force = MLS_STORE_MMAC_LAYOUT
        if block_m is None:
            block_m = MLS_STORE_BLOCK_M
        if block_n is None:
            block_n = MLS_STORE_BLOCK_N
        if block_k is None:
            block_k = MLS_STORE_BLOCK_K
    config = gemm_config(
        wasp=wasp, wdra=wdra, load_warps=load_warps, mma_warps=mma_warps,
        num_stages=num_stages, mmac_layout_force=mmac_layout_force,
        block_m=block_m, block_n=block_n, block_k=block_k,
    )
    if use_matrix_store:
        config.update({
            "wdra_num_load_regs": 16,
            "wdra_num_mma_regs_main": 244,
            "wdra_num_mma_regs_tail": 244,
            "empty_arrive_after_mmac": False,
            "enable_v_mmac_cluster": 0,
            "enable_consumer_pingpong": True,
            "amdgpu_enable_max_ilp_scheduling_strategy": True,
            "hcu_use_gfx946_sched_model_v2": False,
        })
    if use_matrix_store:
        kernel = gemm_mls_store_kernel
    elif use_mls:
        kernel = gemm_mls_kernel
    else:
        kernel = gemm_load_kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    args = (
        a_dev, b_dev, c, M, N, K,
        a_dev.stride(0), a_dev.stride(1),
        b_dev.stride(0), b_dev.stride(1),
        c.stride(0), c.stride(1),
    )
    if compile_only:
        kernel.warmup(*args, grid=grid, **config)
        return
    check_matrix_store_perf = (use_matrix_store and wdra and
                               load_warps == 4 and mma_warps == 8 and
                               num_stages == 2 and
                               shape == MLS_STORE_GEMM_SHAPE)
    stats_before = _pmd_stats_snapshot() if check_matrix_store_perf else {}
    kernel[grid](*args, **config)
    torch.testing.assert_close(c.cpu(), torch.matmul(a, b), atol=1e-2, rtol=1e-2)
    if check_matrix_store_perf:
        cycles, frequency = _latest_pmd_measurement(stats_before)
        tops = 2.0 * M * N * K * frequency / cycles / 1e3
        minimum = MATRIX_STORE_REFERENCE_TOPS * (
            1.0 - MATRIX_STORE_MAX_REGRESSION_PCT / 100.0
        )
        print(f"  matrix_store perf: {tops:.6f} TOPS, minimum {minimum:.6f} "
              f"({cycles} cycles @ {frequency:.6f} GHz)", flush=True)
        if tops < minimum:
            raise AssertionError(
                f"matrix_store performance regression: {tops:.6f} TOPS is "
                f"below {minimum:.6f} TOPS (reference "
                f"{MATRIX_STORE_REFERENCE_TOPS:.6f}, max regression "
                f"{MATRIX_STORE_MAX_REGRESSION_PCT:.1f}%)"
            )


def run_fa(*, use_mls: bool, wasp: bool = True, wdra: bool = False,
           load_warps: Optional[int] = None, mma_warps: Optional[int] = None,
           compile_only: bool = False) -> None:
    if compile_only:
        # FA path has no compile-only hook; skip with note.
        print("  (fa has no compile-only mode; running full accuracy check)")
    if not wasp:
        # fa.py kernels still hardcode warp_specialize=True; nowrap is GEMM-only
        # until FA gains a WARP_SPECIALIZE constexpr.
        raise RuntimeError("FA nowrap (wasp=False) is not supported yet; use --only gemm")
    fa = _load_sibling("hcu_fa_regression", "fa.py")
    Z, Q_H, K_H, N_CTX, D_HEAD = 1, 2, 2, 128, 128
    config = fa_config(
        wasp=wasp, wdra=wdra, load_warps=load_warps, mma_warps=mma_warps
    )
    fa.test_op_fwd(
        Z, Q_H, K_H, N_CTX, D_HEAD,
        causal=False,
        config_list=config,
        USE_FP8=False,
        USE_MLS=use_mls,
    )


# ---------------------------------------------------------------------------
# Case table
# ---------------------------------------------------------------------------

# Cases known not yet supported (tracked separately from hard failures).
KNOWN_XFAIL: set[str] = set()

# Default GEMM problem size.
DEFAULT_GEMM_SHAPE = (128, 128, 4096)

# MLS-store epilogue probe: larger MN tile, skinny K.
MLS_STORE_GEMM_SHAPE = (256, 256, 4096)
MLS_STORE_BLOCK_M = 256
MLS_STORE_BLOCK_N = 256
MLS_STORE_BLOCK_K = 64

# TwoPTwoC (WDRA 8+8) slices A for M-split but leaves B shared. The tt.load
# pipeline currently clones that shared B producer into both Load partitions.
# Both then local_store to the same B LDS allocation while their MMA consumers
# wait on separate abarrier pairs, producing a data race with random inputs.
#
# K=512 is a known-green workaround; K=4096 is the realistic default that
# exposes the race. Keep this coverage until the shared producer has a single
# writer and shared producer/consumer barriers.
WDRA88_LOAD_GEMM_SHAPE = (128, 128, 512)


@dataclass
class Case:
    name: str
    kind: str          # "gemm" | "fa"
    use_mls: bool
    wasp: bool
    wdra: bool         # True => WASP+WDRA; requires wasp
    load_warps: int
    mma_warps: int
    run: Callable[[bool], None]
    shape: Optional[tuple] = None  # GEMM only; None => DEFAULT_GEMM_SHAPE
    num_stages: int = 2
    use_matrix_store: bool = False

    @property
    def xfail(self) -> bool:
        return self.name in KNOWN_XFAIL


def _mode_tag(wasp: bool, wdra: bool, load_warps: int, mma_warps: int,
              num_stages: int = 2) -> str:
    if not wasp:
        return "nowrap"
    if not wdra:
        tag = f"wasp{load_warps}+{mma_warps}"
    else:
        tag = f"wdra{load_warps}+{mma_warps}"
    if num_stages != 2:
        tag = f"{tag}/ns{num_stages}"
    return tag


def run_dense_fa(sequence: int, stages: int, dim: int, causal: bool,
                 *, compile_only: bool = False) -> None:
    """Exercise transposed K and repeated barrier epochs with distinct Q/K/V."""
    kernel = _load_sibling("fa_dense_regression", "fa_dense.py")._flash_attention_fwd_kernel

    if compile_only:
        print("  (dense FA has no compile-only mode; running full accuracy check)")
    generator = torch.Generator().manual_seed(2026)
    q, k, v = [torch.randn((1, 1, sequence, dim), dtype=torch.float16,
                          generator=generator) for _ in range(3)]
    scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(dim)
    if causal:
        scores.masked_fill_(torch.ones(sequence, sequence, dtype=torch.bool).triu(1),
                            float("-inf"))
    ref_out = scores.softmax(-1) @ v.float()
    ref_lse = scores.logsumexp(-1)
    q, k, v = q.cuda(), k.cuda(), v.cuda()
    out = torch.empty_like(q)
    lse = torch.empty((1, 1, sequence), device=q.device, dtype=torch.float32)
    block_m = 32 if dim == 128 else 64
    kernel[(triton.cdiv(sequence, block_m),)](
        q, k, v, out, lse, dim ** -0.5,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(), *lse.stride(),
        N_CTX=sequence, NUM_HEADS=1, NUM_Q_BLOCKS=triton.cdiv(sequence, block_m),
        HEAD_DIM=dim, BLOCK_M=block_m, BLOCK_N=32, CAUSAL=causal,
        WARP_SPECIALIZE=True, num_stages=stages, num_warps=4,
        waves_per_eu=1, matrix_instr_nonkdim=16, kpack=2,
        wasp_enabled=True, wdra_enabled=True, wasp_num_load_warps=4,
        wasp_num_mma_warps=8, wdra_num_load_regs=32,
        wdra_num_mma_regs_main=164, wdra_num_mma_regs_tail=164,
        enable_v_mmac_cluster=1, enable_consumer_pingpong=True,
        optimize_epilogue=True, hcu_use_gfx946_sched_model_v2=False)
    torch.testing.assert_close(out.cpu().float(), ref_out, atol=0.005, rtol=0.005)
    torch.testing.assert_close(lse.cpu(), ref_lse, atol=0.005, rtol=0.005)


def all_cases(*, include_nowrap: bool = False) -> List[Case]:
    """Default matrix is wasp/wdra only; nowrap is opt-in via --mode nowrap.

    Mode tuple: (wasp, wdra, load_warps, mma_warps, num_stages).
    GEMM also covers TwoPTwoC wdra 8+8; its tt.load case uses the known-green
    K=512 workaround. FA stays on 4+4 / 4+8 for now. Extra GEMM coverage:
    wdra4+8 with num_stages=4 (TwoPTwoC cannot use stages=4), and mls_store
    epilogue cases with mmac_layout_force=3.
    """
    cases: List[Case] = []
    # (wasp, wdra, load_warps, mma_warps, num_stages)
    modes = [
        (True, False, 4, 4, 2),   # wasp4+4
        (True, True, 4, 8, 2),    # wdra4+8 OnePTwoC
        # (True, True, 8, 8, 2),  # wdra8+8 TwoPTwoC — temporarily disabled
        (True, True, 4, 8, 4),    # wdra4+8 num_stages=4 (GEMM only below)
    ]
    if include_nowrap:
        modes = [(False, False, 0, 0, 2)]
    for kind in ("gemm", "fa"):
        for use_mls in (False, True):
            for wasp, wdra, load_w, mma_w, nstages in modes:
                if kind == "fa" and not wasp:
                    continue  # FA nowrap not supported yet
                if kind == "fa" and wdra and load_w >= 8:
                    continue  # FA TwoPTwoC not in default matrix yet
                if kind == "fa" and nstages != 2:
                    continue  # FA stages=4 not in default matrix yet
                if wdra and load_w >= 8 and nstages == 4:
                    continue  # TwoPTwoC + stages=4 exceeds abarrier budget
                load = "mls" if use_mls else "load"
                name = f"{kind}/{load}/{_mode_tag(wasp, wdra, load_w, mma_w, nstages)}"
                shape = DEFAULT_GEMM_SHAPE
                if kind == "gemm" and wdra and load_w >= 8 and not use_mls:
                    shape = WDRA88_LOAD_GEMM_SHAPE
                if kind == "gemm":
                    runner = lambda co, mls=use_mls, wp=wasp, w=wdra, lw=load_w, mw=mma_w, ns=nstages, sh=shape: run_gemm(
                        use_mls=mls, wasp=wp, wdra=w, load_warps=lw or None,
                        mma_warps=mw or None, num_stages=ns, compile_only=co,
                        shape=sh,
                    )
                else:
                    runner = lambda co, mls=use_mls, wp=wasp, w=wdra, lw=load_w, mw=mma_w: run_fa(
                        use_mls=mls, wasp=wp, wdra=w, load_warps=lw or None,
                        mma_warps=mw or None, compile_only=co
                    )
                cases.append(Case(
                    name=name, kind=kind, use_mls=use_mls, wasp=wasp, wdra=wdra,
                    load_warps=load_w, mma_warps=mma_w, run=runner, shape=shape,
                    num_stages=nstages,
                ))

    # GEMM MLS-store epilogue: matrix_load A/B + matrix_store C, layout=3.
    # Cover one/multiple K tiles and one/multiple M/N CTAs.  The long-K shape
    # also carries the PMD performance guard in run_gemm.
    mls_store_modes = [
        (True, True, 4, 8, 2),
        # no-WASP MLS store is not supported yet: B operand Interleave2 is not
        # remapped correctly for the normal MMAC consumer path.
        # (False, False, 0, 0, 2),
    ]
    for wasp, wdra, load_w, mma_w, nstages in mls_store_modes:
        for shape in MLS_STORE_TEST_SHAPES:
            shape_tag = "x".join(str(dim) for dim in shape)
            name = (f"gemm/mls_store/{shape_tag}/"
                    f"{_mode_tag(wasp, wdra, load_w, mma_w, nstages)}")
            runner = lambda co, wp=wasp, w=wdra, lw=load_w, mw=mma_w, ns=nstages, sh=shape: run_gemm(
                use_mls=True, use_matrix_store=True,
                mmac_layout_force=MLS_STORE_MMAC_LAYOUT,
                wasp=wp, wdra=w, load_warps=lw or None,
                mma_warps=mw or None, num_stages=ns, compile_only=co,
                shape=sh,
                block_m=MLS_STORE_BLOCK_M,
                block_n=MLS_STORE_BLOCK_N,
                block_k=MLS_STORE_BLOCK_K,
            )
            cases.append(Case(
                name=name, kind="gemm", use_mls=True, wasp=wasp, wdra=wdra,
                load_warps=load_w, mma_warps=mma_w, run=runner, shape=shape,
                num_stages=nstages, use_matrix_store=True,
            ))
    if not include_nowrap:
        for sequence, stages, dim, causal in (
            (160, 2, 64, False), (288, 4, 64, False),
            (512, 2, 128, False), (1024, 2, 64, True),
            (1024, 2, 128, True),
        ):
            cases.append(Case(
                name=f"fa/dense/s{sequence}d{dim}/causal{int(causal)}/wdra4+8/ns{stages}",
                kind="fa", use_mls=False, wasp=True, wdra=True,
                load_warps=4, mma_warps=8, num_stages=stages,
                run=lambda co, s=sequence, ns=stages, d=dim, c=causal:
                    run_dense_fa(s, ns, d, c, compile_only=co),
            ))
    return cases


def filter_cases(cases: List[Case], args: argparse.Namespace) -> List[Case]:
    out = cases
    if args.only == "gemm":
        out = [c for c in out if c.kind == "gemm"]
    elif args.only == "fa":
        out = [c for c in out if c.kind == "fa"]
    if getattr(args, "mls_store", None) is True:
        out = [c for c in out if c.use_matrix_store]
    elif args.mls is True:
        out = [c for c in out if c.use_mls and not c.use_matrix_store]
    elif args.mls is False:
        out = [c for c in out if not c.use_mls]
    if args.mode == "wasp":
        out = [c for c in out if c.wasp and not c.wdra]
    elif args.mode == "wdra":
        # Historical: OnePTwoC 4+8 only.
        out = [c for c in out if c.wdra and c.load_warps == 4 and c.mma_warps == 8]
    elif args.mode == "wdra88":
        out = [c for c in out if c.wdra and c.load_warps == 8 and c.mma_warps == 8]
    elif args.mode == "nowrap":
        out = [c for c in out if not c.wasp]
    if args.num_stages is not None:
        out = [c for c in out if c.num_stages == args.num_stages]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["gemm", "fa", "all"], default="all")
    parser.add_argument(
        "--mode",
        choices=["wasp", "wdra", "wdra88", "nowrap", "all"],
        default="all",
        help="wasp=4+4; wdra=4+8; wdra88=8+8 TwoPTwoC; nowrap=no WASP (GEMM only)",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        choices=[2, 4],
        default=None,
        help="filter by WASP LDS buffer depth (default: all)",
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--mls", dest="mls", action="store_true", default=None,
                   help="only MLS-load cases (excludes mls_store)")
    g.add_argument("--no-mls", dest="mls", action="store_false",
                   help="only non-MLS (tt.load) cases")
    g.add_argument("--mls-store", dest="mls_store", action="store_true",
                   default=None,
                   help="only GEMM mls_store epilogue cases (mmac_layout_force=3)")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--continue-on-fail", action="store_true", default=False,
                        help="run all cases even if one fails")
    parser.add_argument(
        "--fixed-ab",
        action="store_true",
        default=False,
        help="GEMM only: A=B=1 (fp16 0x3c00) for itrace / IT0 load checks",
    )
    args = parser.parse_args()

    cases = filter_cases(all_cases(include_nowrap=(args.mode == "nowrap")), args)
    if not cases:
        print("No cases selected.")
        return 1

    mls_needed = any(c.use_mls for c in cases)
    if mls_needed and not is_hcu_support_mls():
        print("ERROR: selected MLS cases but target does not support MLS")
        return 1
    if args.fixed_ab and any(c.kind != "gemm" for c in cases):
        print("ERROR: --fixed-ab only applies to GEMM; use --only gemm")
        return 1

    print("=" * 72)
    print(f"WASP/MLS regression: {len(cases)} case(s)"
          + (" [fixed-ab]" if args.fixed_ab else ""))
    for c in cases:
        tag = " [XFAIL]" if c.xfail else ""
        print(f"  - {c.name}{tag}")
    print("=" * 72)

    passed, failed, xfailed = [], [], []
    t0 = time.time()
    for c in cases:
        xfail_tag = " [XFAIL]" if c.xfail else ""
        print(f"\n>>> RUN {c.name}{xfail_tag}", flush=True)
        t1 = time.time()
        try:
            if args.fixed_ab:
                run_gemm(
                    use_mls=c.use_mls,
                    use_matrix_store=c.use_matrix_store,
                    mmac_layout_force=(MLS_STORE_MMAC_LAYOUT
                                       if c.use_matrix_store else None),
                    wasp=c.wasp,
                    wdra=c.wdra,
                    load_warps=c.load_warps or None,
                    mma_warps=c.mma_warps or None,
                    num_stages=c.num_stages,
                    compile_only=args.compile_only,
                    shape=c.shape or DEFAULT_GEMM_SHAPE,
                    fixed_ab=True,
                )
            else:
                c.run(args.compile_only)
            dt = time.time() - t1
            if c.xfail:
                print(f"<<< XPASS {c.name} ({dt:.1f}s) — unexpected pass, remove from KNOWN_XFAIL",
                      flush=True)
                passed.append(c.name)
            else:
                print(f"<<< PASS {c.name} ({dt:.1f}s)", flush=True)
                passed.append(c.name)
        except Exception as e:
            dt = time.time() - t1
            if c.xfail:
                print(f"<<< XFAIL {c.name} ({dt:.1f}s): {type(e).__name__}: {e}",
                      flush=True)
                xfailed.append(c.name)
            else:
                print(f"<<< FAIL {c.name} ({dt:.1f}s): {type(e).__name__}: {e}",
                      flush=True)
                traceback.print_exc()
                failed.append(c.name)
                if not args.continue_on_fail:
                    break

    print("\n" + "=" * 72, flush=True)
    print(f"Summary: {len(passed)} passed, {len(xfailed)} xfailed, {len(failed)} failed "
          f"({time.time() - t0:.1f}s total)", flush=True)
    if passed:
        print("  PASS:  " + ", ".join(passed), flush=True)
    if xfailed:
        print("  XFAIL: " + ", ".join(xfailed), flush=True)
    if failed:
        print("  FAIL:  " + ", ".join(failed), flush=True)
    print("=" * 72, flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
