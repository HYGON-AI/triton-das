#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Nightly-vs-local Triton perf compare via aiter benches.

Default: sync+install aiter, download matching nightly whl, compare vs local.
Skip aiter sync/install only when --aiter-dir (or $AITER_ROOT) is set.
Skip wheel download only when --nightly-whl is set.

Usage:
  ./scripts/hcu/run_perf_regression.sh
  python3 scripts/hcu/perf_report.py --aiter-dir /path/to/aiter --nightly-whl /path/to.whl
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

HCU_DIR = Path(__file__).resolve().parent
REPO_ROOT = HCU_DIR.parents[1]
DEFAULT_AITER_DIR = REPO_ROOT / "third_party" / "aiter"
AITER_REPO_URL = "ssh://git@10.16.1.204:10022/dcutoolkit/deeplearing/aiter.git"
_BENCH_DIR = "op_tests/op_benchmarks/triton"

DEFAULT_NIGHTLY_INDEX = os.environ.get(
    "NIGHTLY_TRITON_INDEX",
    "http://10.16.1.201:9929/nightly/dtk2604/+simple/triton/",
)
DEFAULT_NIGHTLY_VERSION_PREFIX = os.environ.get("NIGHTLY_TRITON_VERSION_PREFIX", "3.6")
DEFAULT_LLVM_ROOT = Path(
    os.environ.get("TRITON_LLVM_ROOT", "/root/.triton/llvm/llvm-60942316-rocky-x64")
)
DEFAULT_AITER_REF = os.environ.get("AITER_REF", "main")
DEFAULT_LOCAL_VERSION = os.environ.get("TRITON_LOCAL_VERSION_BASE", "3.6.0")

# ---------------------------------------------------------------------------
# Operator registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchJob:
    script: str
    bench_cwd: str
    args: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    result_kind: str = "triton_bench_csv"
    use_driver: bool = False
    driver_patch: dict = field(default_factory=dict)
    driver_entry: str = "main"


_GEMM_SHAPES = [
    (512, 1280, 8192),
    (1024, 1280, 8192),
    (4096, 1280, 8192),
    (4096, 4096, 4096),
]
_MOE_SHAPES = [
    (128, 256, 1024, 8, 2),
    (512, 256, 1024, 8, 2),
    (1024, 256, 7168, 8, 2),
]
_MOE_QUANT_ARGS: list[tuple[list[str], dict]] = [
    ([], {"quant_mode": "bf16", "dtype": "fp16"}),
    (["-int8_w8a16"], {"quant_mode": "int8_w8a16", "dtype": "fp16"}),
    (["-fp8_w8a8", "-dtype", "bf16"], {"quant_mode": "fp8_w8a8", "dtype": "fp16"}),
]
_FA_FILTER_CASES = [
    {"batch": 1, "q_heads": 32, "k_heads": 8, "head_dim": 128, "seq_q": 512, "seq_k": 512, "causal": True, "dtype": "fp16"},
    {"batch": 1, "q_heads": 32, "k_heads": 8, "head_dim": 128, "seq_q": 1024, "seq_k": 1024, "causal": True, "dtype": "fp16"},
    {"batch": 1, "q_heads": 32, "k_heads": 8, "head_dim": 128, "seq_q": 2048, "seq_k": 2048, "causal": True, "dtype": "fp16"},
]
_FA_ATTENTION_CASES = [
    (1, 32, 8, [512], [512], 128),
    (1, 32, 8, [1024], [1024], 128),
    (1, 32, 8, [2048], [2048], 128),
]
OPERATOR_COLUMNS: dict[str, list[str]] = {
    "gemm_a16w16": ["M", "N", "K", "dtype", "layout"],
    "flash_attention_forward": ["batch", "q_heads", "k_heads", "head_dim", "seq_q", "seq_k", "dtype", "causal"],
    "fused_moe": ["M", "N", "K", "E", "top_k", "dtype", "quant_mode"],
}


def _gemm_jobs() -> list[BenchJob]:
    jobs: list[BenchJob] = []
    for m, n, k in _GEMM_SHAPES:
        for dtype in ("bf16", "fp16"):
            jobs.append(
                BenchJob(
                    script="bench_gemm_a16w16.py",
                    bench_cwd=_BENCH_DIR,
                    args=["--shape", str(m), str(n), str(k), "--layout", "TN", "--metric", "throughput"],
                    params={"M": m, "N": n, "K": k, "layout": "TN", "dtype": dtype},
                    use_driver=True,
                )
            )
    return jobs


def _flash_attention_jobs() -> list[BenchJob]:
    return [
        BenchJob(
            script="bench_flash_attention_forward.py",
            bench_cwd=_BENCH_DIR,
            result_kind="fa_bench_csv",
            use_driver=True,
            driver_patch={"attention_test_cases": _FA_ATTENTION_CASES},
            driver_entry="bench_op_fwd",
        )
    ]


def _moe_jobs() -> list[BenchJob]:
    jobs: list[BenchJob] = []
    for m, n, k, e, top_k in _MOE_SHAPES:
        for extra_args, params in _MOE_QUANT_ARGS:
            jobs.append(
                BenchJob(
                    script="bench_moe.py",
                    bench_cwd=_BENCH_DIR,
                    args=[*extra_args, "-dtype", params["dtype"]],
                    params={"M": m, "N": n, "K": k, "E": e, "top_k": top_k, **params},
                    use_driver=True,
                    driver_patch={"moe_perf_model_cases_list": [(m, n, k, e, top_k)]},
                )
            )
    return jobs


OPERATOR_JOBS: dict[str, list[BenchJob]] = {
    "gemm_a16w16": _gemm_jobs(),
    "flash_attention_forward": _flash_attention_jobs(),
    "fused_moe": _moe_jobs(),
}


def list_operators() -> list[str]:
    return sorted(OPERATOR_JOBS)


def jobs_for_operator(operator: str) -> list[BenchJob]:
    try:
        return list(OPERATOR_JOBS[operator])
    except KeyError as e:
        raise KeyError(f"Unknown operator {operator!r}; known: {', '.join(list_operators())}") from e


# ---------------------------------------------------------------------------
# aiter sync
# ---------------------------------------------------------------------------

def _rmtree_force(path: Path) -> None:
    """NFS-safe remove used before aiter setup.py recreates aiter_meta."""
    if not path.exists():
        return
    # Rename first when possible; plain rmtree often fails on this NFS mount.
    bak = path.with_name(f"{path.name}.bak.{os.getpid()}")
    try:
        path.rename(bak)
        path = bak
    except OSError:
        pass
    subprocess.run(["rm", "-rf", str(path)], check=False)


def _aiter_git(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "safe.directory=*", *args],
        cwd=str(cwd),
        check=check,
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE")},
    )


def _aiter_has_git(aiter_dir: Path) -> bool:
    return (aiter_dir / ".git").is_dir() or (aiter_dir / ".git").is_file()


def _aiter_commit(aiter_dir: Path) -> str:
    """Return full HEAD sha for an aiter checkout, or empty string."""
    if not _aiter_has_git(aiter_dir):
        return ""
    r = _aiter_git("rev-parse", "HEAD", cwd=aiter_dir, check=False)
    return (r.stdout or "").strip() if r.returncode == 0 else ""


def install_aiter(aiter_dir: Path) -> None:
    """Run `python setup.py develop` (aiter README develop install)."""
    _rmtree_force(aiter_dir / "aiter_meta")
    env = os.environ.copy()
    env.setdefault("GPU_ARCHS", "gfx936;gfx938")
    print(f"Installing aiter (python setup.py develop) in {aiter_dir}", file=sys.stderr)
    subprocess.run(
        [sys.executable, "setup.py", "develop"],
        cwd=aiter_dir,
        env=env,
        check=True,
    )


def sync_aiter(
    aiter_dir: Path,
    *,
    ref: str = "main",
    init_submodules: bool = True,
    install: bool = True,
    git_update: bool = True,
) -> Path:
    """Clone/update aiter per its README: submodule update --init, then setup.py develop."""
    aiter_dir = aiter_dir.resolve()
    if not _aiter_has_git(aiter_dir):
        aiter_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "-c", "safe.directory=*", "clone", "--recursive", AITER_REPO_URL, str(aiter_dir)],
            check=True,
        )
        git_update = False
    elif git_update:
        _aiter_git("fetch", "origin", cwd=aiter_dir)
        _aiter_git("checkout", ref, cwd=aiter_dir)
        _aiter_git("pull", "--ff-only", "origin", ref, cwd=aiter_dir)
    if init_submodules:
        _aiter_git("submodule", "sync", cwd=aiter_dir)
        _aiter_git("submodule", "update", "--init", "--recursive", cwd=aiter_dir)
    if install:
        install_aiter(aiter_dir)
    return aiter_dir


# ---------------------------------------------------------------------------
# Bench driver (subprocess entry: perf_report.py --_bench-driver ...)
# ---------------------------------------------------------------------------

# HCU gemm tuned defaults (kept in triton; aiter may not ship BW200B-GEMM-A16W16.json).
_HCU_GEMM_A16W16_CONFIG = {
    "any": {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 4,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 32,
        "cache_modifier": None,
        "kpack": 1,
    }
}


def _apply_aiter_compat(mod, module_name: str) -> None:
    """Runtime shims for aiter benches; never edit aiter sources."""
    if module_name == "bench_gemm_a16w16":
        if hasattr(mod, "generate_gemm_a16w16_inputs"):
            orig = mod.generate_gemm_a16w16_inputs

            def _compat(*args, **kwargs):
                out = orig(*args, **kwargs)
                # Helper returns (x, w, out_dtype, y); bench unpacks (x, w, y).
                if isinstance(out, tuple) and len(out) == 4:
                    x, w, _out_dtype, y = out
                    return x, w, y
                return out

            mod.generate_gemm_a16w16_inputs = _compat

        # Inject HCU gemm config in-process when aiter lacks device JSON.
        import aiter.ops.triton.gemm_a16w16 as gemm_mod

        def _get_config(M: int, N: int, K: int):
            cfg = _HCU_GEMM_A16W16_CONFIG
            if M < 128 and "small" in cfg:
                return cfg["small"]
            return cfg["any"]

        gemm_mod._get_config = _get_config
        if hasattr(gemm_mod._get_config, "cache_clear"):
            gemm_mod._get_config.cache_clear()

    elif module_name == "bench_moe" and hasattr(mod, "triton_moe"):
        # aiter bench_moe still calls the pre-sorted_weights signature.
        orig = mod.triton_moe

        def _moe_compat(
            A,
            B,
            C,
            A_scale,
            B_scale,
            B_zp,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            compute_type,
            **kwargs,
        ):
            return orig(
                A,
                B,
                C,
                A_scale,
                B_scale,
                B_zp,
                topk_weights,
                topk_ids,
                sorted_token_ids,
                None,  # sorted_weights
                expert_ids,
                num_tokens_post_padded,
                mul_routed_weight,
                top_k,
                compute_type,
                **kwargs,
            )

        mod.triton_moe = _moe_compat


def _bench_driver_main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Run aiter bench with optional patches")
    p.add_argument("--_bench-driver", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--module", required=True)
    p.add_argument("--entry", default="main")
    p.add_argument("--patch", default="{}")
    p.add_argument("bench_args", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)

    patch = json.loads(args.patch)
    # JSON turns tuples into lists; aiter bench_moe does `t + (dtype, group_size)`.
    if "moe_perf_model_cases_list" in patch:
        patch["moe_perf_model_cases_list"] = [
            tuple(row) for row in patch["moe_perf_model_cases_list"]
        ]
    mod = importlib.import_module(args.module)
    _apply_aiter_compat(mod, args.module)
    for name, value in patch.items():
        setattr(mod, name, value)

    bench_args = list(args.bench_args)
    if bench_args and bench_args[0] == "--":
        bench_args = bench_args[1:]

    entry = getattr(mod, args.entry)
    if bench_args:
        old_argv = sys.argv
        try:
            sys.argv = [f"{args.module}.py", *bench_args]
            entry()
        finally:
            sys.argv = old_argv
    else:
        entry()
    return 0


# ---------------------------------------------------------------------------
# Parse aiter bench CSV output
# ---------------------------------------------------------------------------

def _latency_from_gemm_tflops(row: dict) -> float | None:
    try:
        m, n, k = int(float(row["M"])), int(float(row["N"])), int(float(row["K"]))
        tflops = float(row["Triton"])
    except (KeyError, TypeError, ValueError):
        return None
    if tflops <= 0:
        return None
    return 2.0 * m * n * k / (tflops * 1e9) * 1000.0


def _moe_tflops(m: int, n: int, k: int, top_k: int, latency_ms: float) -> float | None:
    if latency_ms <= 0:
        return None
    return 2.0 * m * top_k * k * n / latency_ms * 1e-9


def _parse_seq_lens(raw: str) -> int:
    nums = re.findall(r"\d+", raw or "")
    return int(nums[0]) if nums else 0


def _latency_from_fa_tflops(
    batch: int, q_heads: int, seq_q: int, seq_k: int, head_dim: int, causal: bool, tflops: float
) -> float | None:
    if tflops <= 0:
        return None
    total_flops = batch * 2 * seq_q * seq_k * head_dim * 2 * q_heads
    if causal:
        total_flops *= 0.5
    return total_flops / (tflops * 1e9) * 1000.0


def parse_triton_bench_csv(path: Path, *, params: dict, operator: str) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"kernel": operator, "status": "error", "error": f"empty bench csv: {path}", "args": params}

    row = rows[0]
    if operator == "gemm_a16w16":
        latency_ms = _latency_from_gemm_tflops(row)
        try:
            tflops = float(row["Triton"])
        except (KeyError, TypeError, ValueError):
            tflops = None
        if latency_ms is None or tflops is None:
            return {"kernel": operator, "status": "error", "error": f"failed to parse gemm csv: {path}", "args": params}
        return {
            "kernel": operator, "status": "ok", "latency_ms": latency_ms, "value": tflops, "metric": "tflops",
            "args": {
                "M": int(float(row.get("M", params.get("M", 0)))),
                "N": int(float(row.get("N", params.get("N", 0)))),
                "K": int(float(row.get("K", params.get("K", 0)))),
                "layout": row.get("layout", params.get("layout", "")),
                "dtype": params.get("dtype", ""),
            },
        }

    if operator == "fused_moe":
        try:
            latency_ms = float(row["Time (ms)"])
        except (KeyError, TypeError, ValueError):
            return {"kernel": operator, "status": "error", "error": f"failed to parse moe csv: {path}", "args": params}
        m = int(float(row.get("M", params["M"])))
        n = int(float(row.get("N", params["N"])))
        k = int(float(row.get("K", params["K"])))
        top_k = int(float(row.get("top_k", params["top_k"])))
        tflops = _moe_tflops(m, n, k, top_k, latency_ms)
        if tflops is None:
            return {"kernel": operator, "status": "error", "error": "invalid moe latency", "args": params}
        return {
            "kernel": operator, "status": "ok", "latency_ms": latency_ms, "value": tflops, "metric": "tflops",
            "args": {
                "M": m, "N": n, "K": k,
                "E": int(float(row.get("E", params["E"]))),
                "top_k": top_k,
                "dtype": params.get("dtype", ""),
                "quant_mode": params.get("quant_mode", ""),
            },
        }

    return {"kernel": operator, "status": "error", "error": f"unsupported operator: {operator}", "args": params}


def find_triton_bench_csv(work_dir: Path) -> Path | None:
    candidates = sorted(work_dir.glob("*.csv"))
    if not candidates:
        return None
    return candidates[0] if len(candidates) == 1 else max(candidates, key=lambda p: p.stat().st_mtime)


def parse_fa_bench_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results: list[dict] = []
    for row in rows:
        batch = int(row["batch_size"])
        q_heads = int(row["q_heads"])
        k_heads = int(row["k_heads"])
        head_dim = int(row["head_dim"])
        seq_q = _parse_seq_lens(row.get("seq_lens_q", ""))
        seq_k = _parse_seq_lens(row.get("seq_lens_k", ""))
        causal = str(row.get("causal", "True")).lower() in ("true", "1", "yes")
        use_fp8 = str(row.get("use_fp8", "False")).lower() in ("true", "1", "yes")
        case_args = {
            "batch": batch, "q_heads": q_heads, "k_heads": k_heads, "head_dim": head_dim,
            "seq_q": seq_q, "seq_k": seq_k, "dtype": "fp8" if use_fp8 else "fp16", "causal": causal,
        }

        matched = any(all(case_args.get(k) == fc.get(k) for k in fc) for fc in _FA_FILTER_CASES)
        if not matched:
            continue

        status = "ok" if row.get("test_passed", "").lower() in ("true", "1", "yes") else "error"
        try:
            tflops = float(row["TFLOPS"])
        except (KeyError, TypeError, ValueError):
            tflops = 0.0

        latency_ms = 0.0
        if status == "ok":
            lat = _latency_from_fa_tflops(batch, q_heads, seq_q, seq_k, head_dim, causal, tflops)
            if lat is None:
                status = "error"
            else:
                latency_ms = lat

        results.append({
            "kernel": "flash_attention_forward", "status": status,
            "latency_ms": latency_ms, "value": tflops, "metric": "tflops", "args": case_args,
            **({"error": "benchmark failed"} if status == "error" else {}),
        })
    return results


def find_fa_bench_csv(work_dir: Path) -> Path | None:
    matches = sorted(work_dir.glob("attention_test_results_*.csv"))
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _set_gpu(gpu: int | None) -> None:
    if gpu is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu)


def _auto_select_gpu() -> None:
    if os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    probe = "import torch\nx=torch.empty(1,device='cuda')\ntorch.cuda.synchronize()\n"
    for gpu_id in range(16):
        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            r = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            continue
        if r.returncode == 0:
            os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"Auto-selected HIP_VISIBLE_DEVICES={gpu_id}", file=sys.stderr)
            return
    print("ERROR: no healthy GPU found. Set HIP_VISIBLE_DEVICES.", file=sys.stderr)
    sys.exit(2)


def _probe_device_info() -> dict[str, str]:
    code = """
import json, triton, torch
from triton.runtime.driver import driver
info = {
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
    "arch": driver.active.get_current_target().arch if torch.cuda.is_available() else "unknown",
    "triton": getattr(triton, "__version__", "?"),
}
print(json.dumps(info))
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=os.environ.copy())
    if r.returncode != 0:
        return {"device": "unknown", "arch": "unknown", "triton": "?"}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"device": "unknown", "arch": "unknown", "triton": "?"}


def _bench_env(aiter_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    bench_dir = aiter_root / "op_tests" / "op_benchmarks" / "triton"
    parts = [str(aiter_root), str(aiter_root / "op_tests"), str(bench_dir)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(parts)
    return env


def _run_job(job: BenchJob, *, operator: str, aiter_root: Path, env: dict[str, str]) -> list[dict]:
    work_dir = Path(tempfile.mkdtemp(prefix=f"aiter_bench_{operator}_"))
    try:
        if job.use_driver:
            cmd = [
                sys.executable, str(HCU_DIR / "perf_report.py"), "--_bench-driver",
                "--module", Path(job.script).stem,
                "--entry", job.driver_entry,
                "--patch", json.dumps(job.driver_patch),
                "--", *job.args,
            ]
        else:
            cmd = [sys.executable, str(aiter_root / job.bench_cwd / job.script), *job.args]

        proc = subprocess.run(cmd, cwd=work_dir, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            return [{"kernel": operator, "status": "error",
                     "error": err[-2000:] if err else f"exit {proc.returncode}", "args": job.params}]

        if job.result_kind == "fa_bench_csv":
            csv_path = find_fa_bench_csv(work_dir)
            if csv_path is None:
                return [{"kernel": operator, "status": "error", "error": "FA bench csv not found", "args": job.params}]
            return parse_fa_bench_csv(csv_path)

        csv_path = find_triton_bench_csv(work_dir)
        if csv_path is None:
            return [{"kernel": operator, "status": "error", "error": "triton bench csv not found", "args": job.params}]
        return [parse_triton_bench_csv(csv_path, params=job.params, operator=operator)]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _case_id(case: dict) -> str:
    return ",".join(f"{k}={case[k]}" for k in sorted(case))


def write_csv(results: list, path: Path, meta: dict, *, operator: str) -> None:
    arg_keys = [k for k in OPERATOR_COLUMNS.get(operator, []) if any(k in r.get("args", {}) for r in results)]
    metrics = {r.get("metric") for r in results if r.get("metric")}
    throughput_col = next(iter(metrics)) if len(metrics) == 1 else "throughput"
    fieldnames = [*arg_keys, "latency_ms", throughput_col, "status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row: dict = {"status": r["status"]}
            if r["status"] == "ok":
                row["latency_ms"] = f"{r['latency_ms']:.6f}"
                row[throughput_col] = f"{r['value']:.3f}"
            else:
                row["latency_ms"] = row[throughput_col] = ""
            for k in arg_keys:
                row[k] = r.get("args", {}).get(k, "")
            w.writerow(row)
        f.write("\n")
        for k, v in meta.items():
            f.write(f"# {k}={v}\n")
        f.write("# status: ok=benchmark succeeded, error=kernel failed\n")


def print_report(report: dict) -> None:
    s = report["summary"]
    print(f"\n# Triton Perf Report  ({report['device']}, {report['triton']})")
    print(f"OK {s['ok']}/{s['total']}, errors {s['error']}, {s['duration_s']}s\n")
    print(f"{'Kernel':<22} {'Case':<46} {'ms':>8} {'Metric':>12}")
    print("-" * 100)
    for r in report["results"]:
        cid = _case_id(r.get("args", {}))
        if r["status"] != "ok":
            print(f"{r['kernel']:<22} {cid:<46} ERROR: {r.get('error', '')}")
            continue
        print(f"{r['kernel']:<22} {cid:<46} {r['latency_ms']:8.3f} {r['value']:8.3f} {r['metric']:<4}")


# ---------------------------------------------------------------------------
# Nightly wheel resolve / install / compare
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NightlyWheel:
    name: str
    url: str
    version: str
    timestamp: str
    githash: str
    py_tag: str
    manylinux: bool


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run git with safe.directory=* so NFS / shared worktrees work."""
    return subprocess.run(
        ["git", "-c", "safe.directory=*", *args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE")},
    )


def _http_get(url: str, *, dest: Path | None = None, timeout_s: int = 120) -> bytes | Path:
    """Fetch URL via curl (more reliable than urllib on this mirror)."""
    if dest is None:
        r = subprocess.run(
            ["curl", "-sfL", "--retry", "3", "--retry-delay", "1", "--max-time", str(timeout_s), url],
            capture_output=True,
            check=False,
        )
        if r.returncode != 0:
            raise RuntimeError(f"GET failed ({r.returncode}): {url}\n{r.stderr[-500:]!r}")
        return r.stdout
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            "curl", "-fL", "--retry", "3", "--retry-delay", "2",
            "--max-time", str(timeout_s),
            "-o", str(dest), url,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"download failed ({r.returncode}): {url}\n{r.stderr[-800:]}")
    return dest


def _py_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def list_nightly_wheels(index_url: str) -> list[NightlyWheel]:
    """Parse Pep 503 simple index; filenames come from href (anchor text may truncate)."""
    raw = _http_get(index_url)
    html = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else Path(raw).read_text()
    hrefs = re.findall(r'<a href="([^"]+\.whl)(?:#[^"]*)?"', html, flags=re.I)
    wheels: list[NightlyWheel] = []
    name_re = re.compile(
        r"triton-(?P<ver>\d+\.\d+\.\d+)\+.*\.(?P<ts>\d{10})\.g(?P<g>[0-9a-f]+)"
        r"-(?P<py>cp\d+)-",
        re.I,
    )
    for href in hrefs:
        path = href.split("#", 1)[0]
        name = path.rsplit("/", 1)[-1]
        m = name_re.search(name)
        if not m:
            continue
        wheels.append(
            NightlyWheel(
                name=name,
                url=urllib_parse_urljoin(index_url, path),
                version=m.group("ver"),
                timestamp=m.group("ts"),
                githash=m.group("g").lower(),
                py_tag=m.group("py").lower(),
                manylinux="manylinux" in name.lower(),
            )
        )
    return wheels


def urllib_parse_urljoin(base: str, url: str) -> str:
    from urllib.parse import urljoin

    return urljoin(base, url)


def resolve_nightly_wheel(
    *,
    index_url: str,
    version_prefix: str,
    repo: Path = REPO_ROOT,
    max_commits: int = 500,
    prefer_manylinux: bool = True,
) -> tuple[NightlyWheel, str, int]:
    """Pick the newest wheel whose .gXXXXXX matches HEAD or an ancestor commit.

    Walks `git log` from HEAD. Prefer manylinux + current CPython tag.
    Returns (wheel, matched_full_sha, depth_from_head).
    """
    py = _py_tag()
    wheels = list_nightly_wheels(index_url)
    cands = [
        w for w in wheels
        if w.version.startswith(version_prefix) and w.py_tag == py
    ]
    if prefer_manylinux:
        many = [w for w in cands if w.manylinux]
        if many:
            cands = many
    if not cands:
        raise RuntimeError(
            f"no wheels for version_prefix={version_prefix!r} py={py} at {index_url}"
        )

    # newest timestamp wins per githash
    by_g: dict[str, NightlyWheel] = {}
    for w in sorted(cands, key=lambda x: x.timestamp, reverse=True):
        by_g.setdefault(w.githash, w)

    log = _git("log", f"--format=%H", f"-{max_commits}", cwd=repo)
    if log.returncode != 0:
        raise RuntimeError(f"git log failed in {repo}: {log.stderr.strip()}")
    shas = [s.strip() for s in log.stdout.splitlines() if s.strip()]
    if not shas:
        raise RuntimeError(f"empty git history in {repo}")

    for depth, sha in enumerate(shas):
        for g, w in by_g.items():
            if sha.startswith(g) or g.startswith(sha[: len(g)]):
                return w, sha, depth

    newest = max(cands, key=lambda w: w.timestamp)
    raise RuntimeError(
        f"no nightly wheel matches the last {len(shas)} commits from HEAD={shas[0][:12]}; "
        f"newest published {newest.name} (.g{newest.githash})"
    )


def clear_triton_caches() -> None:
    home = Path.home()
    for p in (
        home / ".triton" / "cache",
        Path("/tmp") / "triton_cache",
        Path(os.environ.get("TRITON_CACHE_DIR", "")) if os.environ.get("TRITON_CACHE_DIR") else None,
    ):
        if p and str(p) not in ("", ".") and p.exists():
            print(f"Clearing Triton cache: {p}", file=sys.stderr)
            _rmtree_force(p)


def _purge_site_triton() -> None:
    """Remove leftover site-packages/triton so editable install is not shadowed."""
    code = (
        "import glob,os,site,sys\n"
        "paths=list(site.getsitepackages())+[site.getusersitepackages()]\n"
        "for base in paths:\n"
        "  if not base: continue\n"
        "  for pat in ('triton','triton-*.dist-info','triton-*.egg-info','~riton*'):\n"
        "    for p in glob.glob(os.path.join(base, pat)):\n"
        "      print(p)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    for line in (r.stdout or "").splitlines():
        p = Path(line.strip())
        if not p.exists():
            continue
        print(f"Removing {p}", file=sys.stderr)
        if p.is_dir():
            _rmtree_force(p)
        else:
            p.unlink(missing_ok=True)


def install_nightly_wheel(wheel_path: Path) -> None:
    print(f"Installing nightly wheel: {wheel_path.name}", file=sys.stderr)
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "triton"],
        check=False,
        capture_output=True,
    )
    _purge_site_triton()
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", str(wheel_path)],
        check=True,
    )
    clear_triton_caches()


def _git_safe_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Env that makes nested git (setuptools_scm / pip metadata) accept this tree.

    Does not touch `git config --global`; uses a temporary GIT_CONFIG_GLOBAL and
    SETUPTOOLS_SCM_PRETEND_VERSION so metadata gen can proceed without scm git.
    """
    env = dict(base or os.environ)
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    cfg = Path(tempfile.gettempdir()) / "triton_perf_git_safe.conf"
    if not cfg.is_file():
        cfg.write_text("[safe]\n\tdirectory = *\n", encoding="utf-8")
    env["GIT_CONFIG_GLOBAL"] = str(cfg)
    try:
        n = int(env.get("GIT_CONFIG_COUNT", "0"))
    except ValueError:
        n = 0
    env[f"GIT_CONFIG_KEY_{n}"] = "safe.directory"
    env[f"GIT_CONFIG_VALUE_{n}"] = "*"
    env["GIT_CONFIG_COUNT"] = str(n + 1)
    head = _git("rev-parse", "--short=8", "HEAD", cwd=REPO_ROOT)
    if head.returncode == 0 and head.stdout.strip():
        ver = f"{DEFAULT_LOCAL_VERSION}+git{head.stdout.strip()}"
        env["SETUPTOOLS_SCM_PRETEND_VERSION"] = ver
        env.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TRITON", ver)
    return env


def install_local_editable(*, llvm_root: Path = DEFAULT_LLVM_ROOT) -> None:
    """Reinstall triton from this checkout (editable / develop, no build isolation)."""
    py_dir = REPO_ROOT / "python"
    if not (py_dir / "setup.py").is_file() and not (py_dir / "pyproject.toml").is_file():
        raise RuntimeError(f"local triton python package not found under {py_dir}")
    llvm_root = llvm_root.resolve()
    env = _git_safe_env()
    if llvm_root.is_dir():
        env["LLVM_INCLUDE_DIRS"] = str(llvm_root / "include")
        env["LLVM_LIBRARY_DIR"] = str(llvm_root / "lib")
        env["LLVM_SYSPATH"] = str(llvm_root)
        env.setdefault("LLVM_DIR", str(llvm_root / "lib" / "cmake" / "llvm"))
    print(f"Reinstalling local editable triton from {py_dir}", file=sys.stderr)
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "triton"],
        check=False,
        capture_output=True,
        env=env,
    )
    _purge_site_triton()

    def _probe() -> tuple[bool, str]:
        r = subprocess.run(
            [
                sys.executable, "-c",
                "import triton, pathlib; p=pathlib.Path(triton.__file__).resolve(); "
                f"ok=str(p).startswith({str(py_dir.resolve())!r}); "
                "print(('OK' if ok else 'BAD'), triton.__version__, p); "
                "raise SystemExit(0 if ok else 1)",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        msg = (r.stdout or r.stderr or "").strip()
        return r.returncode == 0, msg

    # setup.py develop often prints scm/git errors on this NFS worktree but still
    # installs; treat import-from-checkout as the success criterion.
    subprocess.run(
        [sys.executable, "setup.py", "develop", "--no-deps"],
        cwd=str(py_dir),
        env=env,
        check=False,
    )
    ok, msg = _probe()
    if not ok:
        # Last-resort: site .pth so the built tree is importable without metadata.
        try:
            import site as _site

            for base in _site.getsitepackages():
                pth = Path(base) / "triton-local-checkout.pth"
                pth.write_text(str(py_dir.resolve()) + "\n", encoding="utf-8")
                print(f"Wrote {pth}", file=sys.stderr)
                break
        except Exception as e:
            print(f"pth fallback failed: {e}", file=sys.stderr)
        ok, msg = _probe()
    if not ok:
        raise RuntimeError(f"local triton not importable from {py_dir}: {msg}")
    print(f"Local triton: {msg}", file=sys.stderr)
    clear_triton_caches()


def run_operator_suite(
    *,
    operators: list[str],
    aiter_dir: Path,
    out_dir: Path,
    generated_at: str,
    label: str = "",
) -> dict:
    """Run selected operators and write per-kernel CSVs under out_dir."""
    t0 = time.time()
    device_info = _probe_device_info()
    env = _bench_env(aiter_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list = []
    prefix = f"[{label}] " if label else ""

    for operator in operators:
        kresults: list = []
        for job in jobs_for_operator(operator):
            rows = _run_job(job, operator=operator, aiter_root=aiter_dir, env=env)
            kresults.extend(rows)
            print(
                f"{prefix}{operator}: {job.params} -> {rows[0]['status'] if rows else 'error'}",
                file=sys.stderr,
            )
        results.extend(kresults)
        path = out_dir / f"{operator}.csv"
        write_csv(
            kresults,
            path,
            {
                "generated_at": generated_at,
                "label": label,
                "triton": device_info["triton"],
                "device": device_info["device"],
                "arch": device_info["arch"],
                "aiter_dir": str(aiter_dir),
                "kernel": operator,
                "cases": len(kresults),
                "ok": sum(r["status"] == "ok" for r in kresults),
                "error": sum(r["status"] == "error" for r in kresults),
            },
            operator=operator,
        )
        print(f"{prefix}Wrote {path}")

    summary = {
        "total": len(results),
        "ok": sum(r["status"] == "ok" for r in results),
        "error": sum(r["status"] == "error" for r in results),
        "duration_s": round(time.time() - t0, 2),
    }
    return {
        "generated_at": generated_at,
        **device_info,
        "label": label,
        "aiter_dir": str(aiter_dir),
        "results": results,
        "summary": summary,
    }


def _result_key(r: dict) -> tuple:
    return (r.get("kernel", ""), _case_id(r.get("args", {})))


def build_compare_rows(nightly: dict, local: dict) -> list[dict]:
    """Join nightly vs local rows; positive delta_pct_ms means local slower."""
    base = {_result_key(r): r for r in nightly.get("results", [])}
    loc = {_result_key(r): r for r in local.get("results", [])}
    keys = sorted(set(base) | set(loc))
    rows: list[dict] = []
    for key in keys:
        nb, lb = base.get(key), loc.get(key)
        kernel, case = key
        row: dict = {
            "kernel": kernel,
            "case": case,
            "nightly_status": (nb or {}).get("status", "missing"),
            "local_status": (lb or {}).get("status", "missing"),
            "nightly_latency_ms": "",
            "local_latency_ms": "",
            "delta_pct_ms": "",
            "nightly_tflops": "",
            "local_tflops": "",
            "delta_pct_tflops": "",
            "metric": (lb or nb or {}).get("metric", ""),
        }
        if nb and nb.get("status") == "ok":
            row["nightly_latency_ms"] = f"{nb['latency_ms']:.3f}"
            row["nightly_tflops"] = f"{nb['value']:.3f}"
        if lb and lb.get("status") == "ok":
            row["local_latency_ms"] = f"{lb['latency_ms']:.3f}"
            row["local_tflops"] = f"{lb['value']:.3f}"
        if (
            nb and lb
            and nb.get("status") == "ok"
            and lb.get("status") == "ok"
            and nb.get("latency_ms")
        ):
            d_ms = (lb["latency_ms"] - nb["latency_ms"]) / nb["latency_ms"] * 100
            d_tf = (lb["value"] - nb["value"]) / nb["value"] * 100 if nb.get("value") else 0.0
            row["delta_pct_ms"] = f"{d_ms:+.2f}%"
            row["delta_pct_tflops"] = f"{d_tf:+.2f}%"
            row["_delta_ms"] = d_ms
            row["_delta_tf"] = d_tf
        rows.append(row)
    return rows


def write_compare_markdown(
    rows: list[dict],
    path: Path,
    *,
    meta: dict | None = None,
    nightly: dict | None = None,
    local: dict | None = None,
) -> None:
    """Write the final nightly vs local report as Markdown tables."""
    nightly = nightly or {}
    local = local or {}
    meta = meta or {}
    path.parent.mkdir(parents=True, exist_ok=True)

    ok = sum(
        1 for r in rows
        if r["nightly_status"] == "ok" and r["local_status"] == "ok"
    )
    err = len(rows) - ok
    by_kernel: dict[str, list[dict]] = {}
    for r in rows:
        by_kernel.setdefault(r["kernel"], []).append(r)

    lines: list[str] = []
    lines.append("# Triton Perf Compare: Nightly vs Local")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| generated_at | {meta.get('generated_at', nightly.get('generated_at', ''))} |")
    lines.append(f"| device | {local.get('device') or nightly.get('device', '')} |")
    lines.append(f"| arch | {local.get('arch') or nightly.get('arch', '')} |")
    lines.append(f"| nightly triton | {nightly.get('triton', '')} |")
    lines.append(f"| local triton | {local.get('triton', '')} |")
    if meta.get("wheel_name"):
        lines.append(f"| nightly wheel | `{meta['wheel_name']}` |")
    if meta.get("matched_commit"):
        lines.append(f"| matched commit | `{meta['matched_commit'][:12]}` |")
    if meta.get("head_commit"):
        lines.append(f"| local HEAD | `{str(meta['head_commit'])[:12]}` |")
    if meta.get("aiter_dir"):
        lines.append(f"| aiter | `{meta['aiter_dir']}` |")
    if meta.get("aiter_commit"):
        lines.append(f"| aiter commit | `{meta['aiter_commit']}` |")
    elif meta.get("aiter_sha"):
        lines.append(f"| aiter commit | `{meta['aiter_sha']}` |")
    lines.append(f"| cases | {ok} ok / {err} error (of {len(rows)}) |")
    lines.append("")
    lines.append(
        "> **Δms%**: `(local - nightly) / nightly`; positive ⇒ local slower.  \n"
        "> **Δtflops%**: positive ⇒ local higher throughput."
    )
    lines.append("")

    for kernel in sorted(by_kernel):
        krows = by_kernel[kernel]
        lines.append(f"## {kernel}")
        lines.append("")
        lines.append(
            "| Case | Nightly ms | Local ms | Δms% | Nightly TFLOPS | Local TFLOPS | Δtflops% | Status |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for r in krows:
            if r["nightly_status"] != "ok" or r["local_status"] != "ok":
                status = f"{r['nightly_status']}/{r['local_status']}"
                lines.append(
                    f"| `{r['case']}` | {r['nightly_latency_ms'] or '—'} | "
                    f"{r['local_latency_ms'] or '—'} | — | "
                    f"{r['nightly_tflops'] or '—'} | {r['local_tflops'] or '—'} | — | {status} |"
                )
                continue
            lines.append(
                f"| `{r['case']}` | {r['nightly_latency_ms']} | {r['local_latency_ms']} | "
                f"{r['delta_pct_ms']} | {r['nightly_tflops']} | {r['local_tflops']} | "
                f"{r['delta_pct_tflops']} | ok |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_compare(rows: list[dict], *, nightly_label: str, local_label: str) -> None:
    print(f"\n# Nightly vs Local  ({nightly_label}  vs  {local_label})")
    print(f"{'Kernel':<22} {'Case':<46} {'nightly_ms':>10} {'local_ms':>10} {'Δms%':>8} {'Δtf%':>8}")
    print("-" * 110)
    for row in rows:
        if row["nightly_status"] != "ok" or row["local_status"] != "ok":
            print(
                f"{row['kernel']:<22} {row['case']:<46} "
                f"{row['nightly_status']}/{row['local_status']}"
            )
            continue
        print(
            f"{row['kernel']:<22} {row['case']:<46} "
            f"{float(row['nightly_latency_ms']):10.3f} "
            f"{float(row['local_latency_ms']):10.3f} "
            f"{row['delta_pct_ms']:>8} {row['delta_pct_tflops']:>8}"
        )


def resolve_or_use_wheel(args: argparse.Namespace, out_dir: Path) -> tuple[Path, dict]:
    """Return wheel path + meta. Always re-download unless --nightly-whl is set."""
    head = _git("rev-parse", "HEAD", cwd=REPO_ROOT)
    head_sha = (head.stdout or "").strip()
    if args.nightly_whl:
        wheel_path = Path(args.nightly_whl).resolve()
        if not wheel_path.is_file():
            raise FileNotFoundError(f"--nightly-whl not found: {wheel_path}")
        print(f"Using provided wheel {wheel_path}", file=sys.stderr)
        meta = {
            "wheel_path": str(wheel_path),
            "wheel_name": wheel_path.name,
            "head_commit": head_sha,
            "source": "provided",
        }
        return wheel_path, meta

    print(f"Resolving nightly wheel from {args.nightly_index}", file=sys.stderr)
    wheel, matched_sha, depth = resolve_nightly_wheel(
        index_url=args.nightly_index,
        version_prefix=args.nightly_version_prefix,
        repo=REPO_ROOT,
    )
    print(
        f"Matched wheel {wheel.name} -> commit {matched_sha[:12]} "
        f"(depth={depth} from HEAD {head_sha[:12]})",
        file=sys.stderr,
    )
    wheel_path = out_dir / "wheels" / wheel.name
    print(f"Downloading {wheel.url}", file=sys.stderr)
    _http_get(wheel.url, dest=wheel_path, timeout_s=600)
    meta = {
        "nightly_index": args.nightly_index,
        "wheel_name": wheel.name,
        "wheel_url": wheel.url,
        "wheel_path": str(wheel_path),
        "wheel_githash": wheel.githash,
        "matched_commit": matched_sha,
        "match_depth": depth,
        "head_commit": head_sha,
        "source": "downloaded",
    }
    return wheel_path, meta


def run_compare_nightly(args: argparse.Namespace, *, aiter_dir: Path, out_dir: Path) -> int:
    generated_at = datetime.now(timezone.utc).isoformat()
    wheel_path, wheel_meta = resolve_or_use_wheel(args, out_dir)
    head_sha = wheel_meta.get("head_commit", "")
    githash = wheel_meta.get("wheel_githash") or Path(wheel_path).name

    aiter_commit = _aiter_commit(aiter_dir)
    meta = {
        **wheel_meta,
        "aiter_dir": str(aiter_dir),
        "aiter_commit": aiter_commit,
        "generated_at": generated_at,
    }
    (out_dir / "compare_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if aiter_commit:
        print(f"aiter commit {aiter_commit[:12]} ({aiter_dir})", file=sys.stderr)

    install_nightly_wheel(wheel_path)
    nightly_report = run_operator_suite(
        operators=args.operators,
        aiter_dir=aiter_dir,
        out_dir=out_dir / "nightly",
        generated_at=generated_at,
        label="nightly",
    )
    (out_dir / "nightly" / "report.json").write_text(
        json.dumps(nightly_report, indent=2), encoding="utf-8"
    )
    print_report(nightly_report)

    install_local_editable(llvm_root=Path(args.llvm_root))
    local_report = run_operator_suite(
        operators=args.operators,
        aiter_dir=aiter_dir,
        out_dir=out_dir / "local",
        generated_at=generated_at,
        label="local",
    )
    (out_dir / "local" / "report.json").write_text(
        json.dumps(local_report, indent=2), encoding="utf-8"
    )
    print_report(local_report)

    compare_path = out_dir / "compare.md"
    rows = build_compare_rows(nightly_report, local_report)
    write_compare_markdown(
        rows,
        compare_path,
        meta=meta,
        nightly=nightly_report,
        local=local_report,
    )
    print(f"Wrote {compare_path}")
    print_compare(
        rows,
        nightly_label=f"nightly {githash} ({nightly_report.get('triton')})",
        local_label=f"local {head_sha[:12]} ({local_report.get('triton')})",
    )

    json_path = args.json or (out_dir / "compare.json")
    payload = {
        "meta": meta,
        "nightly": nightly_report,
        "local": local_report,
        "compare": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")

    if args.fail_on_error and (
        nightly_report["summary"]["error"] or local_report["summary"]["error"]
    ):
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Nightly-vs-local Triton perf compare via aiter benches",
    )
    p.add_argument("--_bench-driver", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("-g", "--gpu", type=int, help="physical GPU id (HIP_VISIBLE_DEVICES)")
    p.add_argument("-o", "--output", type=Path, help="output directory")
    p.add_argument(
        "--aiter-dir",
        type=Path,
        default=None,
        help="use this aiter tree as-is (skip git sync + setup.py develop). "
             "Also accepted via $AITER_ROOT. Default: third_party/aiter with forced sync/install.",
    )
    p.add_argument("--aiter-ref", default=DEFAULT_AITER_REF)
    p.add_argument(
        "--nightly-whl",
        type=Path,
        default=None,
        help="use this local triton .whl (skip resolve + download)",
    )
    p.add_argument("--operators", nargs="+", choices=list_operators(), default=list_operators())
    p.add_argument("--json", type=Path)
    p.add_argument("--fail-on-error", action="store_true")
    p.add_argument(
        "--resolve-nightly-only",
        action="store_true",
        help="only resolve/print the matching nightly wheel URL and exit",
    )
    p.add_argument("--nightly-index", default=DEFAULT_NIGHTLY_INDEX)
    p.add_argument(
        "--nightly-version-prefix",
        default=DEFAULT_NIGHTLY_VERSION_PREFIX,
        help="filter wheels whose version starts with this prefix (default: 3.6)",
    )
    p.add_argument(
        "--llvm-root",
        type=Path,
        default=DEFAULT_LLVM_ROOT,
        help="LLVM root for local editable reinstall",
    )
    p.add_argument("--module")
    p.add_argument("--entry", default="main")
    p.add_argument("--patch", default="{}")
    p.add_argument("bench_args", nargs=argparse.REMAINDER)
    args = p.parse_args()

    if args._bench_driver:
        return _bench_driver_main(sys.argv[1:])

    # Treat $AITER_ROOT like an explicit --aiter-dir (skip sync/install).
    aiter_specified = args.aiter_dir is not None or bool(os.environ.get("AITER_ROOT"))
    if args.aiter_dir is not None:
        aiter_dir = args.aiter_dir.resolve()
    elif os.environ.get("AITER_ROOT"):
        aiter_dir = Path(os.environ["AITER_ROOT"]).resolve()
    else:
        aiter_dir = DEFAULT_AITER_DIR.resolve()

    if args.resolve_nightly_only:
        if args.nightly_whl:
            print(json.dumps({"wheel_path": str(Path(args.nightly_whl).resolve())}, indent=2))
            return 0
        wheel, sha, depth = resolve_nightly_wheel(
            index_url=args.nightly_index,
            version_prefix=args.nightly_version_prefix,
            repo=REPO_ROOT,
        )
        head = _git("rev-parse", "HEAD", cwd=REPO_ROOT)
        print(json.dumps({
            "head": (head.stdout or "").strip(),
            "matched_commit": sha,
            "match_depth": depth,
            "wheel": wheel.name,
            "url": wheel.url,
            "githash": wheel.githash,
            "timestamp": wheel.timestamp,
            "version": wheel.version,
        }, indent=2))
        return 0

    _set_gpu(args.gpu)
    if args.gpu is None:
        _auto_select_gpu()

    if aiter_specified:
        if not _aiter_has_git(aiter_dir) and not (aiter_dir / "setup.py").is_file():
            print(f"aiter not found at {aiter_dir}", file=sys.stderr)
            return 2
        print(f"Using provided aiter at {aiter_dir} (skip sync/install)", file=sys.stderr)
    else:
        print(f"Syncing+installing aiter -> {aiter_dir} ({args.aiter_ref})", file=sys.stderr)
        sync_aiter(aiter_dir, ref=args.aiter_ref, install=True)

    out_dir = args.output
    if out_dir is None:
        stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        out_dir = REPO_ROOT / "perf_reports" / f"compare_{stamp}"
    elif out_dir.suffix in {".csv", ".md", ".json"}:
        out_dir = out_dir.with_suffix("")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output={out_dir}", file=sys.stderr)

    return run_compare_nightly(args, aiter_dir=aiter_dir, out_dir=out_dir)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
