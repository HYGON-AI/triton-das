"""
HCU WASP / WASP+WDRA regression matrix: GEMM & FA × MLS & non-MLS.

Scenarios (8):
  GEMM  × {load, mls} × {wasp 4+4, wdra 4+8}
  FA    × {load, mls} × {wasp 4+4, wdra 4+8}

Usage (from triton-staging root):
  OPTIMIZE_EPILOGUE=1 GPU_DFLAGS=\"['StatLog', 'SQAbar',]\" \\
    PYTHONPATH=python TRITON_BACKENDS_IN_TREE=1 \\
    python third_party/hcu/test/test_wasp_mls_regression.py

  # subset / filters
  python .../test_wasp_mls_regression.py --only gemm
  python .../test_wasp_mls_regression.py --only fa --mls
  python .../test_wasp_mls_regression.py --mode wasp   # 4+4 only
  python .../test_wasp_mls_regression.py --mode wdra   # 4+8 only
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

def wasp_config_gemm(*, wdra: bool) -> dict:
    # WDRA MLS currently may retain both full and sliced LDS buffers; use
    # smaller tiles so a doubled footprint still fits in 128KB.
    block = 64 if wdra else 128
    cfg = {
        "BLOCK_SIZE_M": block,
        "BLOCK_SIZE_N": block,
        "BLOCK_SIZE_K": block,
        "GROUP_SIZE_M": 1,
        "wasp_enabled": True,
        "wasp_num_load_warps": 4,
        "wasp_num_mma_warps": 8 if wdra else 4,
        "wdra_enabled": wdra,
    }
    if wdra:
        cfg.update(
            {
                "wdra_num_load_regs": 88,
                "wdra_num_mma_regs_main": 144,
                "wdra_num_mma_regs_tail": 140,
            }
        )
    return cfg


def wasp_config_fa(*, wdra: bool) -> dict:
    # Same LDS pressure concern as GEMM when Q also uses MLS under WDRA.
    block = 32 if wdra else 64
    cfg = {
        "BLOCK_M": block,
        "BLOCK_N": block,
        "pre_load_v": False,
        "num_stages": 1,
        "wasp_enabled": True,
        "wasp_num_load_warps": 4,
        "wasp_num_mma_warps": 8 if wdra else 4,
        "wdra_enabled": wdra,
    }
    if wdra:
        cfg.update(
            {
                "wdra_num_load_regs": 52,
                "wdra_num_mma_regs_main": 160,
                "wdra_num_mma_regs_tail": 160,
            }
        )
    return cfg


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
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=True):
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
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K), warp_specialize=True):
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


def run_gemm(*, use_mls: bool, wdra: bool, compile_only: bool = False,
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
    config = wasp_config_gemm(wdra=wdra)
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


def run_fa(*, use_mls: bool, wdra: bool, compile_only: bool = False) -> None:
    if compile_only:
        # FA path has no compile-only hook; skip with note.
        print("  (fa has no compile-only mode; running full accuracy check)")
    fa = _load_sibling("hcu_fa_regression", "fa.py")
    Z, Q_H, K_H, N_CTX, D_HEAD = 1, 2, 2, 128, 128
    config = wasp_config_fa(wdra=wdra)
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


@dataclass
class Case:
    name: str
    kind: str          # "gemm" | "fa"
    use_mls: bool
    wdra: bool         # False => wasp-only 4+4; True => wasp+wdra 4+8
    run: Callable[[bool], None]

    @property
    def xfail(self) -> bool:
        return self.name in KNOWN_XFAIL


def all_cases() -> List[Case]:
    cases: List[Case] = []
    for kind in ("gemm", "fa"):
        for use_mls in (False, True):
            for wdra in (False, True):
                mode = "wdra4+8" if wdra else "wasp4+4"
                load = "mls" if use_mls else "load"
                name = f"{kind}/{load}/{mode}"
                if kind == "gemm":
                    runner = lambda co, mls=use_mls, w=wdra: run_gemm(
                        use_mls=mls, wdra=w, compile_only=co
                    )
                else:
                    runner = lambda co, mls=use_mls, w=wdra: run_fa(
                        use_mls=mls, wdra=w, compile_only=co
                    )
                cases.append(Case(name=name, kind=kind, use_mls=use_mls,
                                  wdra=wdra, run=runner))
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
        out = [c for c in out if not c.wdra]
    elif args.mode == "wdra":
        out = [c for c in out if c.wdra]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["gemm", "fa", "all"], default="all")
    parser.add_argument("--mode", choices=["wasp", "wdra", "all"], default="all",
                        help="wasp=4+4 no WDRA; wdra=4+8 WASP+WDRA")
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

    cases = filter_cases(all_cases(), args)
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
                    wdra=c.wdra,
                    compile_only=args.compile_only,
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
