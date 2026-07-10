"""HCU regression: verify GEMM WASP / WASP+WDRA lowering IR properties."""

import os

import pytest

TRITON_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../.."))
INPUTS = os.path.join(TRITON_ROOT, "test/TritonGPU/amd/Inputs")


def _read(name: str) -> str:
    with open(os.path.join(INPUTS, name)) as f:
        return f.read()


@pytest.mark.parametrize(
    "golden,total_warps,registers,starts,partitions,partition_warps",
    [
        # WASP-only
        (
            "hcu-gemm-wasp-only-post-44.ttgir",
            8,
            "array<i32: 88, 88>",
            "array<i32: 0, 4>",
            ["partition0", "partition1"],
            {"partition0": 4, "partition1": 4},
        ),
        (
            "hcu-gemm-wasp-only-post-48.ttgir",
            12,
            "array<i32: 88, 88>",
            "array<i32: 0, 8>",
            ["partition0", "partition1"],
            {"partition0": 8, "partition1": 4},
        ),
        # WASP+WDRA
        (
            "hcu-gemm-wasp-wdra-post-44.ttgir",
            8,
            "array<i32: 88, 88>",
            "array<i32: 0, 4>",
            ["partition0", "partition1"],
            {"partition0": 4, "partition1": 4},
        ),
        (
            "hcu-gemm-wasp-wdra-post-48.ttgir",
            12,
            "array<i32: 88, 88, 88>",
            "array<i32: 0, 4, 8>",
            ["partition0", "partition1", "partition2"],
            {"partition0": 4, "partition1": 4, "partition2": 4},
        ),
    ],
)
def test_hcu_gemm_wasp_wdra_mlir(golden, total_warps, registers, starts, partitions,
                                 partition_warps):
    mlir = _read(golden)
    assert "ttg.warp_specialize" in mlir
    assert f'"ttg.total-num-warps" = {total_warps} : i32' in mlir
    assert f"requestedRegisters = {registers}" in mlir
    assert f"warpGroupStartIds = {starts}" in mlir
    assert "rocdl.hcu.s.abarrier.init" in mlir
    assert "rocdl.hcu.s.abarrier.try.wait" in mlir
    assert "ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16" in mlir
    for partition in partitions:
        assert partition in mlir
        assert f"{partition}" in mlir
        # Match "partitionN(...) num_warps(W)"
        needle = f"num_warps({partition_warps[partition]})"
        # Ensure the expected num_warps appears on the partition header line.
        found = False
        for line in mlir.splitlines():
            if partition in line and "num_warps(" in line:
                assert needle in line, f"{partition} expected {needle}, got: {line.strip()}"
                found = True
                break
        assert found, f"missing num_warps header for {partition}"
    if "partition2" not in partitions:
        assert "partition2" not in mlir
