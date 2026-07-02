"""HCU regression: compile GEMM with WASP+WDRA and verify key MLIR properties."""

import os
import subprocess
import sys

import pytest
import torch

os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "1")

TRITON_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../.."))
LIT_RUNNER = os.path.join(TRITON_ROOT, "test/TritonGPU/amd/run_hcu_wasp_pass.py")
LIT_INPUT = os.path.join(TRITON_ROOT, "test/TritonGPU/amd/Inputs/hcu-gemm-wasp-wdra-pre.mlir")


def _run_wasp_pass(load_warps: int, mma_warps: int) -> str:
    proc = subprocess.run(
        [
            sys.executable,
            LIT_RUNNER,
            LIT_INPUT,
            "--wdra-enabled",
            f"--wasp-num-load-warps={load_warps}",
            f"--wasp-num-mma-warps={mma_warps}",
        ],
        text=True,
        capture_output=True,
        cwd=TRITON_ROOT,
        env={**os.environ, "PYTHONPATH": os.path.join(TRITON_ROOT, "python")},
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.mark.parametrize(
    "load_warps,mma_warps,total_warps,registers,partitions",
    [
        (4, 4, 8, "array<i32: 88, 88>", ["partition0", "partition1"]),
        (4, 8, 12, "array<i32: 88, 88, 88>", ["partition0", "partition1", "partition2"]),
    ],
)
def test_hcu_gemm_wasp_wdra_mlir(load_warps, mma_warps, total_warps, registers, partitions):
    mlir = _run_wasp_pass(load_warps, mma_warps)
    assert "ttg.warp_specialize" in mlir
    assert f'"ttg.total-num-warps" = {total_warps} : i32' in mlir
    assert f"requestedRegisters = {registers}" in mlir
    assert "rocdl.hcu.s.abarrier.init" in mlir
    assert "rocdl.hcu.s.abarrier.try.wait" in mlir
    for partition in partitions:
        assert partition in mlir


def test_hcu_gemm_wasp_wdra_compile_only():
    from matmul_wasp import matmul

    a = torch.randn((128, 128), device="cpu", dtype=torch.float16)
    b = torch.randn((128, 128), device="cpu", dtype=torch.float16).transpose(1, 0)
    matmul(a.to("cuda"), b.to("cuda"), compile_only=True)
