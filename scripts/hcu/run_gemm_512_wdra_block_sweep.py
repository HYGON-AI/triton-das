#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Sweep BLOCK_M/N/K for 512³ GEMM load+WDRA; pick lowest totalGcSimCycles.

Default grid (MN square, K independent):
  MN ∈ {32, 64, 128}
  K  ∈ {32, 64, 128, 256}

Usage:
  cd /xukang/code/triton-staging
  # 12-wave (default load=4, mma=8)
  python scripts/hcu/run_gemm_512_wdra_block_sweep.py

  # 16-wave 2P+2C
  python scripts/hcu/run_gemm_512_wdra_block_sweep.py \\
    --load-warps 8 --mma-warps 8 \\
    --out /tmp/gc_cycles_512_wdra16_mnk_sweep.txt
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "third_party/hcu/test/test_wasp_mls_regression.py"
TRITON_CACHE = Path.home() / ".triton" / "cache"
DEFAULT_GPU_DFLAGS = "['StatLog', 'TT', 'Perf']"
SHAPE = (512, 512, 512)

DEFAULT_MN_LIST = (32, 64, 128)
DEFAULT_K_LIST = (32, 64, 128, 256)


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
    os.environ["PYTHONPATH"] = os.pathsep.join(
        prefix + [p for p in parts if p not in prefix]
    )
    for p in reversed(prefix):
        if p not in sys.path:
            sys.path.insert(0, p)


def ensure_gpu_dflags() -> None:
    # StatLog is enough for totalGcSimCycles; do not force TT (huge ttrace / slow).
    required = ("StatLog", "Perf")
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


def wipe_m5out() -> None:
    for d in [ROOT / "m5out", ROOT / "scripts/hcu/m5out", Path("/xukang/code/m5out")]:
        if d.exists():
            shutil.rmtree(d)
    (ROOT / "m5out").mkdir(parents=True, exist_ok=True)


def wipe_triton_cache() -> None:
    if TRITON_CACHE.exists():
        shutil.rmtree(TRITON_CACHE)
    TRITON_CACHE.mkdir(parents=True, exist_ok=True)


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
            # Prefer the largest cycle count among packets (kernel vs tiny setup).
            if best is None or cyc > best[0]:
                best = (cyc, freq, p)
    if best is None:
        return None, None, None
    return best[2], best[0], best[1]


def tops(m: int, n: int, k: int, cycles: int, freq_ghz: float) -> float:
    return (2 * m * n * k) / (cycles / (freq_ghz * 1e9)) / 1e12


def run_worker(args: argparse.Namespace) -> int:
    setup_env()
    ensure_gpu_dflags()
    wipe_triton_cache()
    wipe_m5out()

    import torch
    import triton

    mod = load_reg()
    M, N, K = SHAPE
    bm, bn, bk = args.bm, args.bn, args.bk
    waves = 16 if args.load_warps == 8 else 12
    label = f"load/wdra{args.load_warps}+{args.mma_warps}/MN{bm}_K{bk}"
    print(
        f"=== {label} ({waves}-wave) === shape={M}x{N}x{K} "
        f"BLOCK_M={bm} BLOCK_N={bn} BLOCK_K={bk} "
        f"regs={args.load_regs}/{args.mma_regs_main}/{args.mma_regs_tail} "
        f"GPU_DFLAGS={os.environ['GPU_DFLAGS']!r}",
        flush=True,
    )

    torch.manual_seed(0)
    a = torch.randn((M, K), device="cpu", dtype=torch.float16)
    b = torch.randn((N, K), device="cpu", dtype=torch.float16).transpose(1, 0)
    a_dev, b_dev = a.to("cuda"), b.to("cuda")
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    config = mod.gemm_config(wasp=True, wdra=True)
    config["BLOCK_SIZE_M"] = bm
    config["BLOCK_SIZE_N"] = bn
    config["BLOCK_SIZE_K"] = bk
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
        a_dev,
        b_dev,
        c,
        M,
        N,
        K,
        a_dev.stride(0),
        a_dev.stride(1),
        b_dev.stride(0),
        b_dev.stride(1),
        c.stride(0),
        c.stride(1),
    )
    kernel[grid](*kargs, **config)
    torch.cuda.synchronize()

    path, cycles, freq = best_stats()
    print(f"RESULT\t{cycles}\t{freq}\t{path}", flush=True)
    return 0 if cycles is not None else 1


def run_parent(args: argparse.Namespace) -> int:
    if args.load_warps == 8 and args.mma_warps != 8:
        print("error: 16-wave 2P+2C requires --mma-warps 8", flush=True)
        return 2
    setup_env()
    ensure_gpu_dflags()
    out = Path(args.out)
    out.write_text("")
    script = str(Path(__file__).resolve())
    env = os.environ.copy()
    env.setdefault("OPTIMIZE_EPILOGUE", "1")
    env["GPU_DFLAGS"] = os.environ["GPU_DFLAGS"]
    env.setdefault("TRITON_BACKENDS_IN_TREE", "1")
    env["PYTHONPATH"] = f"{ROOT / 'python'}:{ROOT}"

    mn_list = tuple(int(x) for x in args.mn.split(",") if x.strip())
    k_list = tuple(int(x) for x in args.k.split(",") if x.strip())
    combos = [(mn, mn, k) for mn in mn_list for k in k_list]
    rows = []
    waves = 16 if args.load_warps == 8 else 12
    print(
        f"load+WDRA {waves}-wave block sweep "
        f"(load={args.load_warps} mma={args.mma_warps}): "
        f"MN={mn_list} K={k_list} → {len(combos)} configs → {out}",
        flush=True,
    )

    for bm, bn, bk in combos:
        label = f"load/wdra{args.load_warps}+{args.mma_warps}/MN{bm}_K{bk}"
        print(f"======== {label} ========", flush=True)
        cmd = [
            sys.executable,
            "-u",
            script,
            "--worker",
            "--bm",
            str(bm),
            "--bn",
            str(bn),
            "--bk",
            str(bk),
            "--load-warps",
            str(args.load_warps),
            "--mma-warps",
            str(args.mma_warps),
            "--load-regs",
            str(args.load_regs),
            "--mma-regs-main",
            str(args.mma_regs_main),
            "--mma-regs-tail",
            str(args.mma_regs_tail),
        ]
        t0 = time.time()
        proc = subprocess.run(
            cmd, cwd=str(ROOT), env=env, capture_output=True, text=True
        )
        dt = time.time() - t0
        sys.stdout.write(proc.stdout)
        sys.stdout.flush()
        if proc.returncode != 0 and proc.stderr:
            sys.stderr.write(proc.stderr[-3000:])
            sys.stderr.flush()

        cycles = freq = None
        status = "PASS" if proc.returncode == 0 else f"FAIL:exit={proc.returncode}"
        for line in proc.stdout.splitlines():
            if "RESULT\t" not in line:
                continue
            line = line[line.index("RESULT\t") :]
            parts = line.split("\t")
            try:
                cycles = int(parts[1]) if parts[1] not in ("None", "") else None
                freq = float(parts[2]) if parts[2] not in ("None", "") else None
            except ValueError:
                pass
        if cycles is not None:
            status = "PASS"
        elif proc.returncode == 0:
            status = "FAIL:no-stats"

        tval = tops(*SHAPE, cycles, freq) if cycles and freq else None
        print(
            f"status={status} time={dt:.1f}s cycles={cycles} freq={freq} "
            f"TOPS={tval:.3f}"
            if tval
            else f"status={status} time={dt:.1f}s cycles={cycles} freq={freq}",
            flush=True,
        )
        row = f"{label}\t{bm}\t{bn}\t{bk}\t{cycles}\t{freq}\t{status}\t{dt:.1f}"
        if tval is not None:
            row += f"\t{tval:.3f}"
        rows.append((cycles if cycles is not None else 10**18, row, label, tval))
        with out.open("a") as f:
            f.write(row + "\n")

    print("===== DONE =====", flush=True)
    print(out.read_text(), flush=True)

    ok = [(c, lab, t) for c, _, lab, t in rows if c < 10**18]
    if not ok:
        print("BEST: none succeeded", flush=True)
        return 1
    ok.sort(key=lambda x: x[0])
    best_c, best_lab, best_t = ok[0]
    print(
        f"BEST\t{best_lab}\tcycles={best_c}\tTOPS={best_t:.3f}",
        flush=True,
    )
    print("----- ranked (lowest cycles first) -----", flush=True)
    for c, lab, t in ok:
        print(f"  {lab}\tcycles={c}\tTOPS={t:.3f}", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--worker", action="store_true")
    p.add_argument("--bm", type=int, default=64)
    p.add_argument("--bn", type=int, default=64)
    p.add_argument("--bk", type=int, default=64)
    p.add_argument("--load-warps", type=int, default=4, choices=[4, 8])
    p.add_argument("--mma-warps", type=int, default=8, choices=[4, 8])
    p.add_argument("--load-regs", type=int, default=88)
    p.add_argument("--mma-regs-main", type=int, default=144)
    p.add_argument("--mma-regs-tail", type=int, default=140)
    p.add_argument(
        "--mn",
        default=",".join(str(x) for x in DEFAULT_MN_LIST),
        help="Comma-separated square MN block sizes",
    )
    p.add_argument(
        "--k",
        default=",".join(str(x) for x in DEFAULT_K_LIST),
        help="Comma-separated BLOCK_K values",
    )
    p.add_argument(
        "--out",
        default="/tmp/gc_cycles_512_wdra_mnk_sweep.txt",
    )
    args = p.parse_args()
    if args.worker:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
