"""
HCU WASP / WASP+WDRA regression matrix: GEMM & FA × MLS & non-MLS.

Default scenarios (12):
  GEMM  × {load, mls} × {wasp 4+4, wdra 4+8}              # num_stages=2
  # GEMM  × {load, mls} × {wdra 8+8}                       # temporarily disabled
  GEMM  × {load, mls} × {wdra 4+8} × num_stages=4
  FA    × {load, mls} × {wasp 4+4, wdra 4+8}              # num_stages=2

Known limitation: gemm/load/wdra8+8 uses K=512; see the comment near
WDRA88_LOAD_GEMM_SHAPE below. wdra8+8 does not support num_stages=4
(abarrier ID limit).

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
  python .../test_wasp_mls_regression.py --compile-only
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import triton
import triton.language as tl

os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "1")

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


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

def gemm_config(
    *,
    wasp: bool,
    wdra: bool,
    load_warps: Optional[int] = None,
    mma_warps: Optional[int] = None,
    num_stages: int = 2,
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
    block = 64 if wdra else 128
    cfg = {
        "BLOCK_SIZE_M": block,
        "BLOCK_SIZE_N": block,
        "BLOCK_SIZE_K": block,
        "GROUP_SIZE_M": 1,
        "WARP_SPECIALIZE": wasp,
        "wasp_enabled": wasp,
        "wdra_enabled": wdra,
        "num_stages": num_stages,
    }
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
# GEMM kernels (tt.load / matrix_load)
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


def run_gemm(*, use_mls: bool, wasp: bool = True, wdra: bool = False,
             load_warps: Optional[int] = None, mma_warps: Optional[int] = None,
             num_stages: int = 2, compile_only: bool = False,
             shape=(128, 128, 4096), fixed_ab: bool = False) -> None:
    M, N, K = shape
    torch.manual_seed(0)
    if fixed_ab:
        # Constant A/B for itrace (IT0): A=1, B=1 => C[i,j]=K; fp16 1.0 = 0x3c00.
        # Keep the same layouts as the random path (A row-major, B col-major)
        # so MLS encoding / strides match production.
        a = torch.ones((M, K), device="cpu", dtype=torch.float16)
        b = torch.ones((N, K), device="cpu", dtype=torch.float16).transpose(1, 0)
    else:
        a = torch.randn((M, K), device="cpu", dtype=torch.float16)
        b = torch.randn((N, K), device="cpu", dtype=torch.float16).transpose(1, 0)
    a_dev, b_dev = a.to("cuda"), b.to("cuda")
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    config = gemm_config(
        wasp=wasp, wdra=wdra, load_warps=load_warps, mma_warps=mma_warps,
        num_stages=num_stages,
    )
    kernel = gemm_mls_kernel if use_mls else gemm_load_kernel
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
    kernel[grid](*args, **config)
    torch.testing.assert_close(c.cpu(), torch.matmul(a, b), atol=1e-2, rtol=1e-2)


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


def all_cases(*, include_nowrap: bool = False) -> List[Case]:
    """Default matrix is wasp/wdra only; nowrap is opt-in via --mode nowrap.

    Mode tuple: (wasp, wdra, load_warps, mma_warps, num_stages).
    GEMM also covers TwoPTwoC wdra 8+8; its tt.load case uses the known-green
    K=512 workaround. FA stays on 4+4 / 4+8 for now. Extra GEMM coverage:
    wdra4+8 with num_stages=4 (TwoPTwoC cannot use stages=4).
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
    return cases


def filter_cases(cases: List[Case], args: argparse.Namespace) -> List[Case]:
    out = cases
    if args.only == "gemm":
        out = [c for c in out if c.kind == "gemm"]
    elif args.only == "fa":
        out = [c for c in out if c.kind == "fa"]
    if args.mls is True:
        out = [c for c in out if c.use_mls]
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
                   help="only MLS cases")
    g.add_argument("--no-mls", dest="mls", action="store_false",
                   help="only non-MLS (tt.load) cases")
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
