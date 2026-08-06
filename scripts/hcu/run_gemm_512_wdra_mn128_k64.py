#!/usr/bin/env python3
"""Run fixed 512³ non-MLS WDRA GEMM with BLOCK_M=N=128, BLOCK_K=64.

Best MN/K combo from the 512 WDRA 12-wave block sweep (≈19.02 TOPS on PMD).
Also supports 16-wave (2P+2C) via --load-warps 8 --mma-warps 8.

Usage (from triton-staging root):

  # 12-wave (default)
  python scripts/hcu/run_gemm_512_wdra_mn128_k64.py

  # 16-wave + default VGPR (88/144/140)
  python scripts/hcu/run_gemm_512_wdra_mn128_k64.py --load-warps 8 --mma-warps 8

  # 16-wave + custom VGPR / blocks
  python scripts/hcu/run_gemm_512_wdra_mn128_k64.py \\
    --load-warps 8 --mma-warps 8 \\
    --bm 128 --bn 128 --bk 64 \\
    --load-regs 88 --mma-regs-main 144 --mma-regs-tail 140

Options:
  --check          enable CPU golden assert_close (off by default)
  --keep-cache     keep ~/.triton/cache (cleared by default)
  --no-wipe        keep existing m5out
  --compile-only   warmup / compile only
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "third_party/hcu/test/test_wasp_mls_regression.py"
TRITON_CACHE = Path.home() / ".triton" / "cache"

# Defaults: prior 12-wave sweep winner
DEFAULT_M = DEFAULT_N = DEFAULT_K = 512
DEFAULT_BM, DEFAULT_BN, DEFAULT_BK = 128, 128, 64
DEFAULT_LOAD_REGS, DEFAULT_MMA_MAIN, DEFAULT_MMA_TAIL = 88, 144, 140
DEFAULT_GPU_DFLAGS = "['StatLog', 'TT', 'Perf']"


def setup_env() -> None:
    os.chdir(ROOT)
    os.environ.setdefault("OPTIMIZE_EPILOGUE", "1")
    os.environ.setdefault("GPU_DFLAGS", DEFAULT_GPU_DFLAGS)
    os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
    os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "1")
    # Prefer in-tree python package.
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
        print(f"cleared {TRITON_CACHE}", flush=True)
    TRITON_CACHE.mkdir(parents=True, exist_ok=True)


def ensure_gpu_dflags() -> None:
    """Ensure StatLog/TT/Perf are present in GPU_DFLAGS."""
    required = ("StatLog", "TT", "Perf")
    raw = os.environ.get("GPU_DFLAGS", DEFAULT_GPU_DFLAGS).strip()
    # Parse simple "['A', 'B']" lists; fall back to default on anything else.
    flags: list[str] = []
    if raw.startswith("[") and raw.endswith("]"):
        for tok in raw[1:-1].split(","):
            t = tok.strip().strip("'\"")
            if t:
                flags.append(t)
    else:
        flags = []
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
    best = None  # (cycles, freq, path)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="Enable torch.matmul golden check (off by default)")
    ap.add_argument("--keep-cache", action="store_true",
                    help="Keep ~/.triton/cache (cleared by default)")
    ap.add_argument("--no-wipe", action="store_true",
                    help="Do not wipe m5out before run")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--m", type=int, default=DEFAULT_M, help="GEMM M (default 512)")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="GEMM N (default 512)")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="GEMM K (default 512)")
    ap.add_argument("--bm", type=int, default=DEFAULT_BM)
    ap.add_argument("--bn", type=int, default=DEFAULT_BN)
    ap.add_argument("--bk", type=int, default=DEFAULT_BK)
    ap.add_argument("--load-warps", type=int, default=4, choices=[4, 8],
                    help="4 → 12-wave 1P2C; 8 → 16-wave 2P2C (requires --mma-warps 8)")
    ap.add_argument("--mma-warps", type=int, default=8, choices=[4, 8])
    ap.add_argument("--load-regs", type=int, default=DEFAULT_LOAD_REGS)
    ap.add_argument("--mma-regs-main", type=int, default=DEFAULT_MMA_MAIN)
    ap.add_argument("--mma-regs-tail", type=int, default=DEFAULT_MMA_TAIL)
    args = ap.parse_args()

    if args.load_warps == 8 and args.mma_warps != 8:
        ap.error("16-wave 2P+2C requires --mma-warps 8")

    setup_env()
    ensure_gpu_dflags()

    if not args.keep_cache:
        wipe_triton_cache()
    if not args.no_wipe:
        wipe_m5out()

    import torch
    import triton

    mod = load_reg()
    M, N, K = args.m, args.n, args.k
    waves = 16 if args.load_warps == 8 else 12
    label = (
        f"gemm/load/wdra{args.load_warps}+{args.mma_warps} "
        f"{M}x{N}x{K} MN{args.bm}_K{args.bk} "
        f"L{args.load_regs}/M{args.mma_regs_main}/T{args.mma_regs_tail}"
    )
    print(
        f"=== {label} ({waves}-wave) ===\n"
        f"shape={M}x{N}x{K}  BLOCK_M={args.bm} BLOCK_N={args.bn} BLOCK_K={args.bk}\n"
        f"load_warps={args.load_warps} mma_warps={args.mma_warps}  "
        f"regs load/main/tail={args.load_regs}/{args.mma_regs_main}/{args.mma_regs_tail}\n"
        f"GPU_DFLAGS={os.environ.get('GPU_DFLAGS')!r}  "
        f"OPTIMIZE_EPILOGUE={os.environ.get('OPTIMIZE_EPILOGUE')!r}",
        flush=True,
    )

    torch.manual_seed(0)
    a = torch.randn((M, K), device="cpu", dtype=torch.float16)
    b = torch.randn((N, K), device="cpu", dtype=torch.float16).transpose(1, 0)
    a_dev, b_dev = a.to("cuda"), b.to("cuda")
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    config = mod.gemm_config(wasp=True, wdra=True)
    config["BLOCK_SIZE_M"] = args.bm
    config["BLOCK_SIZE_N"] = args.bn
    config["BLOCK_SIZE_K"] = args.bk
    config["wasp_num_load_warps"] = args.load_warps
    config["wasp_num_mma_warps"] = args.mma_warps
    config["wdra_num_load_regs"] = args.load_regs
    config["wdra_num_mma_regs_main"] = args.mma_regs_main
    config["wdra_num_mma_regs_tail"] = args.mma_regs_tail

    kernel = mod.gemm_load_kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    kargs = (
        a_dev, b_dev, c, M, N, K,
        a_dev.stride(0), a_dev.stride(1),
        b_dev.stride(0), b_dev.stride(1),
        c.stride(0), c.stride(1),
    )

    t0 = time.time()
    if args.compile_only:
        kernel.warmup(*kargs, grid=grid, **config)
        print(f"compile-only done in {time.time()-t0:.1f}s", flush=True)
        return 0

    kernel[grid](*kargs, **config)
    torch.cuda.synchronize()
    dt = time.time() - t0

    if args.check:
        torch.testing.assert_close(
            c.cpu(), torch.matmul(a, b), atol=1e-2, rtol=1e-2
        )
        check = "PASS"
    else:
        check = "SKIP"

    path, cycles, freq = best_stats()
    print(f"wall_s={dt:.1f}  golden={check}", flush=True)
    if cycles is None or freq is None:
        print("WARN: no m5out stats.txt found (need GPU_DFLAGS StatLog on PMD)",
              flush=True)
        return 1

    p = tops(M, N, K, cycles, freq)
    print(
        f"RESULT\t{label}\t"
        f"cycles={cycles}\tfreq_ghz={freq}\tTOPS={p:.3f}\tstats={path}",
        flush=True,
    )
    if args.load_warps == 4 and args.bm == 128 and args.bk == 64:
        print(
            f"ref (12-wave sweep): cycles≈30222  TOPS≈19.52  "
            f"(MN128_K64 on PMD)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
