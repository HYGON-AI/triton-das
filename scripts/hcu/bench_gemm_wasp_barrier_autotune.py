#!/usr/bin/env python3

"""Benchmark GEMM across WASP/WDRA modes and barrier / MMAC-cluster knobs.

Autotune knobs (passed as HIPOptions / kernel kwargs):
  1. empty_arrive_after_mmac — empty abarrier after mmac + SchedBarrier fence
  2. enable_v_mmac_cluster   — 0 off | 1 cluster (+ A/B & before-mmac SchedBarrier)
                               | 2 cluster + cross-region analysis
  3. mode                    — nowrap | wasp44 | wdra48 | wdra88

Usage (from triton-staging root):

  # Sweep all modes × empty-arrive × cluster levels
  python scripts/hcu/bench_gemm_wasp_barrier_autotune.py --autotune

  # Single point
  python scripts/hcu/bench_gemm_wasp_barrier_autotune.py \\
    --mode wdra88 --empty-arrive-after-mmac --enable-v-mmac-cluster 1 \\
    --m 768 --n 1024 --k 512 --bm 128 --bn 128 --bk 64

  # Restrict mode / knobs
  python scripts/hcu/bench_gemm_wasp_barrier_autotune.py --autotune \\
    --modes nowrap,wasp44,wdra48,wdra88 \\
    --empty-arrive-after-mmac both \\
    --enable-v-mmac-cluster all
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "third_party/hcu/test/test_wasp_mls_regression.py"
TRITON_CACHE = Path.home() / ".triton" / "cache"

DEFAULT_M = DEFAULT_N = DEFAULT_K = 512
DEFAULT_BM, DEFAULT_BN, DEFAULT_BK = 128, 128, 64
DEFAULT_LOAD_REGS, DEFAULT_MMA_MAIN, DEFAULT_MMA_TAIL = 88, 144, 140
#DEFAULT_GPU_DFLAGS = "['StatLog', 'TT', 'Perf']"
# SQAbar/SQEbar: PMD traces which async/ebarrier is stuck on hang.
#DEFAULT_GPU_DFLAGS = "['StatLog', 'SQAbar', 'SQEbar']"
DEFAULT_GPU_DFLAGS = "['StatLog']"

MODES = ("nowrap", "wasp44", "wdra48", "wdra88")


def setup_env() -> None:
    os.chdir(ROOT)
    os.environ.setdefault("OPTIMIZE_EPILOGUE", "1")
    os.environ.setdefault("GPU_DFLAGS", DEFAULT_GPU_DFLAGS)
    os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
    os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "1")
    py = str(ROOT / "python")
    cur = os.environ.get("PYTHONPATH", "")
    parts = [p for p in cur.split(os.pathsep) if p]
    prefix = [py, str(ROOT)]
    os.environ["PYTHONPATH"] = os.pathsep.join(prefix + [p for p in parts if p not in prefix])
    for p in reversed(prefix):
        if p not in sys.path:
            sys.path.insert(0, p)


def wipe_m5out() -> None:
    for d in [ROOT / "m5out", ROOT / "scripts/hcu/m5out", Path("/xukang/code/m5out")]:
        if d.exists():
            shutil.rmtree(d)
    (ROOT / "m5out").mkdir(parents=True, exist_ok=True)


def wipe_triton_cache() -> None:
    if TRITON_CACHE.exists():
        shutil.rmtree(TRITON_CACHE)
    TRITON_CACHE.mkdir(parents=True, exist_ok=True)


def ensure_gpu_dflags() -> None:
    #required = ("StatLog", "TT", "Perf")
    #required = ("StatLog", "SQAbar", "SQEbar")
    required = ("StatLog",)
    raw = os.environ.get("GPU_DFLAGS", DEFAULT_GPU_DFLAGS).strip()
    flags: list[str] = []
    if raw.startswith("[") and raw.endswith("]"):
        for tok in raw[1:-1].split(","):
            t = tok.strip().strip("'\"")
            if t:
                flags.append(t)
    for f in required:
        if f not in flags:
            flags.append(f)
    os.environ["GPU_DFLAGS"] = "[" + ", ".join(f"'{f}'" for f in flags) + "]"


def load_reg():
    spec = importlib.util.spec_from_file_location("test_wasp_mls_regression", REG)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def best_stats():
    best = None
    for root in [ROOT / "m5out", ROOT / "scripts/hcu/m5out", Path("/xukang/code/m5out")]:
        if not root.is_dir():
            continue
        for p in root.rglob("stats.txt"):
            text = p.read_text(errors="replace")
            m = re.search(r"^totalGcSimCycles\s+(\d+)", text, re.M)
            f = re.search(r"^gcFrequency\s+([0-9.eE+-]+)", text, re.M)
            if not m:
                continue
            cyc = int(m.group(1))
            freq = float(f.group(1)) if f else None
            if best is None or cyc > best[0]:
                best = (cyc, freq, p)
    if best is None:
        return None, None, None
    return best[2], best[0], best[1]


def tops(m: int, n: int, k: int, cycles: int, freq_ghz: float) -> float:
    flops = 2 * m * n * k
    t_sec = cycles / (freq_ghz * 1e9)
    return flops / t_sec / 1e12


def parse_bool_choice(s: str) -> List[bool]:
    s = s.strip().lower()
    if s in ("both", "all", "sweep"):
        return [False, True]
    if s in ("1", "true", "yes", "on"):
        return [True]
    if s in ("0", "false", "no", "off"):
        return [False]
    raise argparse.ArgumentTypeError(
        f"expected true/false/both, got {s!r}"
    )


def parse_cluster_choice(s: str) -> List[int]:
    """Parse enable_v_mmac_cluster: 0|1|2|all (or comma-separated)."""
    s = s.strip().lower()
    if s in ("both", "all", "sweep"):
        return [0, 1, 2]
    if s in ("false", "no", "off"):
        return [0]
    if s in ("true", "yes", "on", "cluster"):
        return [1]
    if s in ("analysis", "region", "cross-region", "cross_region"):
        return [2]
    out: List[int] = []
    for tok in s.split(","):
        t = tok.strip()
        if not t:
            continue
        v = int(t)
        if v not in (0, 1, 2):
            raise argparse.ArgumentTypeError(
                f"enable_v_mmac_cluster values must be 0/1/2; got {v}")
        out.append(v)
    if not out:
        raise argparse.ArgumentTypeError(
            f"expected 0|1|2|all (or comma list), got {s!r}")
    return out


def mode_config(mode: str, *, load_regs: int, mma_main: int, mma_tail: int) -> dict:
    if mode == "nowrap":
        return {
            "wasp_enabled": False,
            "wdra_enabled": False,
            "WARP_SPECIALIZE": False,
            "num_warps": 4,
        }
    if mode == "wasp44":
        return {
            "wasp_enabled": True,
            "wdra_enabled": False,
            "WARP_SPECIALIZE": True,
            "wasp_num_load_warps": 4,
            "wasp_num_mma_warps": 4,
        }
    if mode == "wdra48":
        return {
            "wasp_enabled": True,
            "wdra_enabled": True,
            "WARP_SPECIALIZE": True,
            "wasp_num_load_warps": 4,
            "wasp_num_mma_warps": 8,
            "wdra_num_load_regs": load_regs,
            "wdra_num_mma_regs_main": mma_main,
            "wdra_num_mma_regs_tail": mma_tail,
        }
    if mode == "wdra88":
        return {
            "wasp_enabled": True,
            "wdra_enabled": True,
            "WARP_SPECIALIZE": True,
            "wasp_num_load_warps": 8,
            "wasp_num_mma_warps": 8,
            "wdra_num_load_regs": load_regs,
            "wdra_num_mma_regs_main": mma_main,
            "wdra_num_mma_regs_tail": mma_tail,
        }
    raise ValueError(f"unknown mode {mode}")


def run_one(
    *,
    mod,
    mode: str,
    empty_after: bool,
    enable_v_mmac_cluster: int,
    m: int,
    n: int,
    k: int,
    bm: int,
    bn: int,
    bk: int,
    load_regs: int,
    mma_main: int,
    mma_tail: int,
    check: bool,
    compile_only: bool,
    wipe: bool,
) -> Tuple[str, Optional[float], Optional[int]]:
    import torch
    import triton

    if wipe:
        wipe_m5out()

    label = (
        f"{mode} empty_after={int(empty_after)} "
        f"mmac_cluster={int(enable_v_mmac_cluster)} "
        f"{m}x{n}x{k} MN{bm}_K{bk}"
    )
    print(f"=== {label} ===", flush=True)

    torch.manual_seed(0)
    a = torch.randn((m, k), device="cpu", dtype=torch.float16)
    b = torch.randn((n, k), device="cpu", dtype=torch.float16).transpose(1, 0)
    a_dev, b_dev = a.to("cuda"), b.to("cuda")
    c = torch.empty((m, n), device="cuda", dtype=torch.float16)

    cfg = mode_config(mode, load_regs=load_regs, mma_main=mma_main, mma_tail=mma_tail)
    cfg.update(
        {
            "BLOCK_SIZE_M": bm,
            "BLOCK_SIZE_N": bn,
            "BLOCK_SIZE_K": bk,
            "GROUP_SIZE_M": 1,
            "empty_arrive_after_mmac": empty_after,
            "enable_v_mmac_cluster": int(enable_v_mmac_cluster),
        }
    )

    kernel = mod.gemm_load_kernel
    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_SIZE_M"]) * triton.cdiv(n, META["BLOCK_SIZE_N"]),
    )
    kargs = (
        a_dev, b_dev, c, m, n, k,
        a_dev.stride(0), a_dev.stride(1),
        b_dev.stride(0), b_dev.stride(1),
        c.stride(0), c.stride(1),
    )

    t0 = time.time()
    if compile_only:
        kernel.warmup(*kargs, grid=grid, **cfg)
        print(f"compile-only done in {time.time()-t0:.1f}s", flush=True)
        return label, None, None

    kernel[grid](*kargs, **cfg)
    torch.cuda.synchronize()
    dt = time.time() - t0

    if check:
        torch.testing.assert_close(c.cpu(), torch.matmul(a, b), atol=1e-2, rtol=1e-2)
        golden = "PASS"
    else:
        golden = "SKIP"

    path, cycles, freq = best_stats()
    if cycles is None or freq is None:
        print(f"wall_s={dt:.1f} golden={golden} WARN: no m5out stats", flush=True)
        return label, None, None

    p = tops(m, n, k, cycles, freq)
    print(
        f"RESULT\t{label}\tcycles={cycles}\tfreq_ghz={freq}\tTOPS={p:.3f}\t"
        f"wall_s={dt:.1f}\tgolden={golden}\tstats={path}",
        flush=True,
    )
    return label, p, cycles


def expand_modes(s: str) -> List[str]:
    out = []
    for tok in s.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t not in MODES:
            raise argparse.ArgumentTypeError(f"unknown mode {t!r}; choose from {MODES}")
        out.append(t)
    if not out:
        raise argparse.ArgumentTypeError("empty --modes")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--autotune", action="store_true",
                    help="Sweep modes × empty-arrive × mmac-cluster levels")
    ap.add_argument("--mode", default="wdra88", choices=MODES,
                    help="Single-run mode (ignored with --autotune unless --modes unset)")
    ap.add_argument("--modes", type=expand_modes, default=None,
                    help="Comma-separated modes for --autotune (default: all)")
    ap.add_argument("--empty-arrive-after-mmac", nargs="?", const="true",
                    default="false", type=str,
                    help="true|false|both (default false; with --autotune default both)")
    ap.add_argument("--enable-v-mmac-cluster", nargs="?", const="1",
                    default="0", type=str,
                    help="0|1|2|all (default 0; with --autotune default all). "
                         "1 enables cluster + SchedBarriers; 2 adds cross-region analysis")
    ap.add_argument("--m", type=int, default=DEFAULT_M)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--bm", type=int, default=DEFAULT_BM)
    ap.add_argument("--bn", type=int, default=DEFAULT_BN)
    ap.add_argument("--bk", type=int, default=DEFAULT_BK)
    ap.add_argument("--load-regs", type=int, default=DEFAULT_LOAD_REGS)
    ap.add_argument("--mma-regs-main", type=int, default=DEFAULT_MMA_MAIN)
    ap.add_argument("--mma-regs-tail", type=int, default=DEFAULT_MMA_TAIL)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--keep-cache", action="store_true")
    ap.add_argument("--no-wipe", action="store_true")
    args = ap.parse_args()

    setup_env()
    ensure_gpu_dflags()
    if not args.keep_cache:
        wipe_triton_cache()

    # With --autotune, default knobs to full sweep unless user overrode.
    empty_arg = args.empty_arrive_after_mmac
    cluster_arg = args.enable_v_mmac_cluster
    if args.autotune:
        if empty_arg == "false" and "--empty-arrive-after-mmac" not in sys.argv:
            empty_arg = "both"
        if cluster_arg == "0" and "--enable-v-mmac-cluster" not in sys.argv:
            cluster_arg = "all"

    modes = args.modes if args.modes is not None else (
        list(MODES) if args.autotune else [args.mode]
    )
    empty_choices = parse_bool_choice(empty_arg)
    cluster_choices = parse_cluster_choice(cluster_arg)

    combos = list(itertools.product(modes, empty_choices, cluster_choices))
    print(
        f"grid={args.m}/{args.bm}*{args.n}/{args.bn} "
        f"combos={len(combos)} GPU_DFLAGS={os.environ.get('GPU_DFLAGS')!r}",
        flush=True,
    )

    mod = load_reg()
    results = []
    for i, (mode, empty_a, cluster) in enumerate(combos):
        # Wipe between runs so stats belong to this combo.
        wipe = not args.no_wipe
        if i == 0 and wipe:
            wipe_m5out()
        label, p, cyc = run_one(
            mod=mod,
            mode=mode,
            empty_after=empty_a,
            enable_v_mmac_cluster=cluster,
            m=args.m,
            n=args.n,
            k=args.k,
            bm=args.bm,
            bn=args.bn,
            bk=args.bk,
            load_regs=args.load_regs,
            mma_main=args.mma_regs_main,
            mma_tail=args.mma_regs_tail,
            check=args.check,
            compile_only=args.compile_only,
            wipe=wipe and i > 0,
        )
        results.append((label, p, cyc))

    print("=== SUMMARY ===", flush=True)
    best = None
    for label, p, cyc in results:
        print(f"{label}\tTOPS={p}\tcycles={cyc}", flush=True)
        if p is not None and (best is None or p > best[1]):
            best = (label, p, cyc)
    if best:
        print(f"BEST\t{best[0]}\tTOPS={best[1]:.3f}\tcycles={best[2]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
