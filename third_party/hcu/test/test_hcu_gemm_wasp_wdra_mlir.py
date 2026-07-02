"""HCU regression: verify GEMM WASP+WDRA lowering IR properties."""

import os

import pytest

TRITON_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../.."))
INPUTS = os.path.join(TRITON_ROOT, "test/TritonGPU/amd/Inputs")


def _read(name: str) -> str:
    with open(os.path.join(INPUTS, name)) as f:
        return f.read()


@pytest.mark.parametrize(
    "golden,total_warps,registers,partitions",
    [
        (
            "hcu-gemm-wasp-wdra-post-44.ttgir",
            8,
            "array<i32: 88, 88>",
            ["partition0", "partition1"],
        ),
        (
            "hcu-gemm-wasp-wdra-post-48.ttgir",
            12,
            "array<i32: 88, 88, 88>",
            ["partition0", "partition1", "partition2"],
        ),
    ],
)
def test_hcu_gemm_wasp_wdra_mlir(golden, total_warps, registers, partitions):
    mlir = _read(golden)
    assert "ttg.warp_specialize" in mlir
    assert f'"ttg.total-num-warps" = {total_warps} : i32' in mlir
    assert f"requestedRegisters = {registers}" in mlir
    assert "rocdl.hcu.s.abarrier.init" in mlir
    assert "rocdl.hcu.s.abarrier.try.wait" in mlir
    for partition in partitions:
        assert partition in mlir
