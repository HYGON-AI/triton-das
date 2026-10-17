# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""CPU-side IR checks for HCU-only buffer-to-LDS preparation and conversion."""

import pytest
import ast
import re
from pathlib import Path

from triton._C.libtriton import ir, passes, hcu, amd
from buffer_lds_test_utils import run_pass, _runtime_k_gemm
from triton.backends.compiler import GPUTarget
from triton.backends.hcu.compiler import HIPBackend, HIPOptions, _validate_prefetch_distances


def add_scheduled_buffer_to_lds(pm):
    """Exercise copy conversion followed by pipeline synchronization."""
    hcu.passes.ttgpuir.add_convert_buffer_lds_copies(pm)
    hcu.passes.ttgpuir.add_schedule_buffer_to_lds(pm)


def with_buffer_to_lds(source, enabled, arch="gfx936", contract=True):
    # The placement encoding owns wrap; its shared_linear component alone
    # describes ordinary, unwrapped storage.
    match = re.search(r'#shared = #ttg.shared_linear<\{offset = (\[.*\])\}, alignment = 16>', source)
    if contract and match:
        bases = ast.literal_eval(match.group(1))
        is_a = bases[4] == [1, 0]
        geometry = ("rows = 256, rowBytes = 32, waves = 8, copyBytesPerLane = 16, "
                    "rowsPerChunk = 8, wrapCount = 2, wrapStepBytes = 16" if is_a else
                    "rows = 16, rowBytes = 512, waves = 8, copyBytesPerLane = 16, "
                    "rowsPerChunk = 1, wrapCount = 2, wrapStepBytes = 64")
        aliases = (
            '#contract_mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 2], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1, tilesPerWarp = [1, 2]}>\n'
            f'#contract_dot = #ttg.dot_op<{{opIdx = {0 if is_a else 1}, parent = #contract_mma, kWidth = 4}}>\n')
        placement = ('#placement = #ttg.hcu_buffer_lds_shared<{linearComponent = #shared, '
                     'elementBytes = 2, contiguousDim = 1, ' + geometry + '}>\n')
        source = aliases + source.replace('module attributes {', placement + 'module attributes {', 1)
        source = source.replace('#shared, #smem', '#placement, #smem')
    return source.replace("module attributes {",
                          f'module attributes {{"ttg.target" = "hip:{arch}", hcu.buffer_to_lds = {str(enabled).lower()}, ')


def allocation_contract_fixture(body):
    source = '''
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [64, 0], [128, 0], [8, 0], [16, 0], [32, 0]]}, alignment = 16>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func public @read(%slot: i32, %cond: i1) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<4x256x16xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<4x256x16xf16, #shared, #smem, mutable>
    %av = ttg.memdesc_index %a[%slot] : !ttg.memdesc<4x256x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x16xf16, #shared, #smem, mutable>
    %bv = ttg.memdesc_index %b[%slot] : !ttg.memdesc<4x256x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x16xf16, #shared, #smem, mutable>
    BODY
    tt.return
  }
}
'''.replace("BODY", body)
    source = with_buffer_to_lds(source, True)
    return source


@pytest.mark.parametrize("group", [3, 16])
def test_invalid_planned_wave_group(tmp_path, capfd, group):
    source = allocation_contract_fixture("").replace(
        "wrapStepBytes = 16}", f"wrapStepBytes = 16, wavesPerRowGroup = {group}}}")
    with pytest.raises(RuntimeError):
        run_pass(tmp_path, source, passes.common.add_canonicalizer)
    assert "HCU buffer-to-LDS wave group must be zero or a power of two" in capfd.readouterr().err


@pytest.mark.parametrize("replacement", ["#shared", "#different"])
def test_slot_cannot_change_placement_encoding(tmp_path, capfd, replacement):
    source = allocation_contract_fixture("")
    source = source.replace("module attributes {",
        "#different = #ttg.hcu_buffer_lds_shared<{linearComponent = #shared, "
        "elementBytes = 2, contiguousDim = 1, rows = 256, rowBytes = 32, "
        "waves = 8, copyBytesPerLane = 16, rowsPerChunk = 8, "
        "wrapCount = 2, wrapStepBytes = 32}>\nmodule attributes {", 1)
    source = source.replace("-> !ttg.memdesc<256x16xf16, #placement",
                            f"-> !ttg.memdesc<256x16xf16, {replacement}")
    with pytest.raises(RuntimeError):
        run_pass(tmp_path, source, passes.common.add_canonicalizer)
    assert "must preserve HCU buffer-to-LDS placement" in capfd.readouterr().err


@pytest.mark.parametrize("view", ["memdesc_trans", "memdesc_reshape"])
def test_view_cannot_discard_wrapped_layout(tmp_path, capfd, view):
    attrs = " {order = array<i32: 1, 0>}" if view == "memdesc_trans" else ""
    body = (f"%view = ttg.{view} %av{attrs} : "
            "!ttg.memdesc<256x16xf16, #shared, #smem, mutable> -> "
            "!ttg.memdesc<16x256xf16, #plain, #smem, mutable>")
    source = allocation_contract_fixture(body).replace("module attributes {",
        "#plain = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>\nmodule attributes {", 1)
    with pytest.raises(RuntimeError):
        run_pass(tmp_path, source, passes.common.add_canonicalizer)
    if view == "memdesc_trans":
        assert "transpose of HCU buffer-to-LDS placement is unsupported" in capfd.readouterr().err


def test_partial_tile_placement_rejected(tmp_path, capfd):
    source = allocation_contract_fixture("").replace(
        "!ttg.memdesc<256x16xf16, #placement", "!ttg.memdesc<128x16xf16, #placement")
    with pytest.raises(RuntimeError):
        run_pass(tmp_path, source, passes.common.add_canonicalizer)
    assert "complete fp16/bf16/fp8 backing tile" in capfd.readouterr().err


def test_wrong_linear_component_rejected(tmp_path, capfd):
    source = allocation_contract_fixture("").replace("[8, 0], [16, 0]", "[16, 0], [8, 0]")
    with pytest.raises(RuntimeError):
        run_pass(tmp_path, source, passes.common.add_canonicalizer)
    assert "linear component does not match" in capfd.readouterr().err


def test_allocation_recreation_preserves_placement_in_type(tmp_path):
    source = allocation_contract_fixture("")
    prefix = source[:source.index("module attributes {")]
    source = prefix + '''
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func public @init(%input: tensor<256x16xf16, #blocked>) -> !ttg.memdesc<256x16xf16, #placement, #smem, mutable> {
    %cvt = ttg.convert_layout %input : tensor<256x16xf16, #blocked> -> tensor<256x16xf16, #contract_dot>
    %a = ttg.local_alloc %cvt : (tensor<256x16xf16, #contract_dot>) -> !ttg.memdesc<256x16xf16, #placement, #smem, mutable>
    tt.return %a : !ttg.memdesc<256x16xf16, #placement, #smem, mutable>
  }
}
'''
    result = run_pass(tmp_path, source, passes.common.add_canonicalizer)
    assert "ttg.convert_layout" not in result
    assert "#ttg.hcu_buffer_lds_shared" in result
    assert "hcu.buffer_to_lds.contract" not in result
    assert "hcu.buffer_to_lds.access" not in result


@pytest.mark.parametrize("field,value", [("wrapCount = 2", "wrapCount = 3"),
                                        ("rows = 256", "rows = 128")])
def test_invalid_placement_encoding(tmp_path, capfd, field, value):
    source = allocation_contract_fixture("").replace(field, value)
    with pytest.raises(RuntimeError):
        run_pass(tmp_path, source, passes.common.add_canonicalizer)
    assert "HCU" in capfd.readouterr().err


@pytest.mark.parametrize("contract,llvm_lowering", [(False, False), (True, False), (True, True)])
def test_wrapped_async_copy_requires_wrap_aware_producer(tmp_path, capfd, contract, llvm_lowering):
    source = '''
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [64, 0], [128, 0], [8, 0], [16, 0], [32, 0]]}, alignment = 16>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @copy(%p: tensor<256x16x!tt.ptr<f16>, #blocked>, %dst: !ttg.memdesc<256x16xf16, #shared, #smem, mutable>) {
    %token = ttg.async_copy_global_to_local %p, %dst : tensor<256x16x!tt.ptr<f16>, #blocked> -> <256x16xf16, #shared, #smem, mutable>
    tt.return
  }
}
'''
    source = with_buffer_to_lds(source, True, contract=contract)
    def lower(pm):
        if llvm_lowering:
            amd.passes.ttgpuir.add_allocate_shared_memory(pm)
            hcu.passes.ttgpuir.add_to_llvmir(pm, "gfx936", False)
        else:
            add_scheduled_buffer_to_lds(pm)
    if contract:
        with pytest.raises(RuntimeError):
            run_pass(tmp_path, source, lower)
        expected = ("failed to legalize operation 'ttg.async_copy_global_to_local'" if llvm_lowering
                    else "requires a buffer_load_to_local producer")
        assert expected in capfd.readouterr().err
    else:
        result = run_pass(tmp_path, source, lower)
        assert ("async_copy_global_to_local" in result) is not llvm_lowering


@pytest.mark.parametrize("dtype", ["f16", "f32"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("arch", ["gfx936", "gfx938"])
def test_prepare_is_opt_in_and_type_checked(tmp_path, monkeypatch, dtype, enabled, arch):
    # Removed experimental flags must not override module-local policy.
    monkeypatch.setenv("TRITON_HCU_ROCBLAS_A_LAYOUT", "1")

    source = '''
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @load(%p: tensor<256x16x!tt.ptr<DTYPE>, #blocked>) -> tensor<256x16xDTYPE, #blocked> {
    %v = tt.load %p : tensor<256x16x!tt.ptr<DTYPE>, #blocked>
    tt.return %v : tensor<256x16xDTYPE, #blocked>
  }
}
'''.replace("DTYPE", dtype)
    source = with_buffer_to_lds(source, enabled, arch)
    result = run_pass(tmp_path, source, lambda pm:
                      hcu.passes.ttgpuir.add_accelerate_matmul(
                          pm, arch, 0, 1, -1))
    assert "sizePerThread = [1, 8]" not in result


def test_common_coalesce_ignores_hcu_environment(tmp_path, monkeypatch):
    source = '''
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @load(%p: tensor<256x16x!tt.ptr<f16>, #blocked>) -> tensor<256x16xf16, #blocked> {
    %v = tt.load %p : tensor<256x16x!tt.ptr<f16>, #blocked>
    tt.return %v : tensor<256x16xf16, #blocked>
  }
}
'''
    monkeypatch.setenv("TRITON_HCU_ROCBLAS_A_LAYOUT", "0")
    baseline = run_pass(tmp_path, source, passes.ttgpuir.add_coalesce)
    monkeypatch.setenv("TRITON_HCU_ROCBLAS_A_LAYOUT", "1")
    assert run_pass(tmp_path, source, passes.ttgpuir.add_coalesce) == baseline


@pytest.mark.parametrize("legacy_attribute", [False, True])
def test_panel_selection_ignores_legacy_async_attribute(
        tmp_path, monkeypatch, legacy_attribute):
    monkeypatch.setenv("TRITON_HCU_ENABLE_DS_READ_M", "1")
    source = '''
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #blocked, kWidth = 0}>
#b = #ttg.dot_op<{opIdx = 1, parent = #blocked, kWidth = 0}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32, hcu.async_copy = LEGACY} {
  tt.func @gemm(%pa: tensor<16x16x!tt.ptr<f16>, #blocked>, %pb: tensor<16x32x!tt.ptr<f16>, #blocked>) -> tensor<16x32xf32, #blocked> {
    %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>, #blocked>
    %b = tt.load %pb : tensor<16x32x!tt.ptr<f16>, #blocked>
    %zero = arith.constant dense<0.0> : tensor<16x32xf32, #blocked>
    %ad = ttg.convert_layout %a : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #a>
    %bd = ttg.convert_layout %b : tensor<16x32xf16, #blocked> -> tensor<16x32xf16, #b>
    %out = tt.dot %ad, %bd, %zero : tensor<16x16xf16, #a> * tensor<16x32xf16, #b> -> tensor<16x32xf32, #blocked>
    tt.return %out : tensor<16x32xf32, #blocked>
  }
}
'''.replace("LEGACY", str(legacy_attribute).lower())
    result = run_pass(tmp_path, source, lambda pm:
                      hcu.passes.ttgpuir.add_accelerate_matmul(
                          pm, "gfx936", 16, 1, -1))
    assert "#ttg.shared_linear" in result


@pytest.mark.parametrize("arch", ["gfx936", "gfx938"])
@pytest.mark.parametrize("operand", [0, 1])
@pytest.mark.parametrize("dtype", ["f16", "bf16", "f8E4M3FN", "f8E5M2"])
def test_normalized_non_k_matrix_read_for_both_operands(tmp_path, capfd, operand, arch, dtype):
    # The same K16/K32 x non-K32 physical placement and fragment, with logical
    # axes swapped for A. No policy flag is needed once the contract exists.
    bits = 8 if dtype.startswith("f8") else 16
    element_bytes, k, k_width = bits // 8, 256 // bits, 64 // bits
    basis = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16],
             [1, 0], [2, 0], [4, 0], [8, 0]]
    if bits == 8:
        basis.append([16, 0])
    if operand == 0:
        basis = [list(reversed(v)) for v in basis]
    shape = f"32x{k}" if operand == 0 else f"{k}x32"
    tiles = "[2, 1]" if operand == 0 else "[1, 2]"
    register_bits = 4 if bits == 8 else 3
    source = f'''
#mma = #ttg.amd_mfma<{{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16, {k}], isTransposed = false, mmacLayout = 1, tilesPerWarp = {tiles}}}>
#dot = #ttg.dot_op<{{opIdx = {operand}, parent = #mma, kWidth = {k_width}}}>
#producer = #ttg.linear<{{register = {basis[:register_bits]}, lane = {basis[register_bits:]}, warp = [], block = []}}>
#shared = #ttg.shared_linear<{{offset = {basis}}}, alignment = 16>
#smem = #ttg.shared_memory
#placement = #ttg.hcu_buffer_lds_shared<{{linearComponent = #shared, elementBytes = {element_bytes}, contiguousDim = {operand}, rows = {k}, rowBytes = {32 * element_bytes}, waves = 1, copyBytesPerLane = 16, rowsPerChunk = 1, wrapCount = 1, wrapStepBytes = 0}}>
module attributes {{"ttg.target" = "hip:gfx936", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32}} {{
  tt.func public @read(%src: !ttg.memdesc<{shape}xf16, #placement, #smem, mutable>, %out: tensor<{shape}x!tt.ptr<f16>, #dot>) {{
    %v = ttg.local_load %src : !ttg.memdesc<{shape}xf16, #placement, #smem, mutable> -> tensor<{shape}xf16, #dot>
    tt.store %out, %v : tensor<{shape}x!tt.ptr<f16>, #dot>
    tt.return
  }}
}}
'''
    source = source.replace("hip:gfx936", f"hip:{arch}").replace("xf16", f"x{dtype}").replace("ptr<f16>", f"ptr<{dtype}>")
    lower = lambda pm: hcu.passes.ttgpuir.add_to_llvmir(pm, arch, False)
    result = run_pass(tmp_path, source, lower)
    assert f"llvm.hcu.ds.read.m32x{k}.b{bits}" in result
    assert "ttg.local_load" not in result


@pytest.mark.parametrize("arch", ["gfx936", "gfx938", "gfx92a", "gfx946"])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_ordinary_lds_rotation_requires_contract(tmp_path, capfd, dtype, arch):
    # Identical SharedLinear encodings do not imply wrapped placement.
    # Only the HCU contract-backed local-load pattern may add rotation.
    source = """
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 2], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1, tilesPerWarp = [1, 2]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [64, 0], [128, 0], [8, 0], [16, 0], [32, 0]]}, alignment = 16>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func public @read(%src: !ttg.memdesc<256x16xf16, #shared, #smem, mutable>, %out: tensor<256x16x!tt.ptr<f16>, #dot>) {
    %v = ttg.local_load %src : !ttg.memdesc<256x16xf16, #shared, #smem, mutable> -> tensor<256x16xf16, #dot>
    tt.store %out, %v : tensor<256x16x!tt.ptr<f16>, #dot>
    tt.return
  }
}
"""
    def lower(contract):
        ir_source = with_buffer_to_lds(source, False, arch=arch, contract=contract)
        ir_source = ir_source.replace("f16", dtype)
        return run_pass(tmp_path, ir_source, lambda pm:
                        hcu.passes.ttgpuir.add_to_llvmir(pm, arch, False))

    ordinary, wrapped = lower(False), lower(True)
    for result in (ordinary, wrapped):
        assert "ttg.local_load" not in result
        assert "llvm.load" in result
        assert "llvm.hcu.ds.read.m32x16.b16" not in result
    # Producer-wave division and modulo rotation are absent from the ordinary
    # address path. The contract is sufficient without a module policy flag.
    assert wrapped.count("llvm.udiv") > ordinary.count("llvm.udiv")
    assert wrapped.count("llvm.and") > ordinary.count("llvm.and")


@pytest.mark.parametrize("extra_use", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("operand", ["A", "B"])
@pytest.mark.parametrize("contract", [False, True])
@pytest.mark.parametrize("synchronization", ["conversion-only", "standalone", "pipeline"])
def test_direct_store_preserves_other_users(tmp_path, monkeypatch, capfd, extra_use, enabled, operand, contract, synchronization):
    monkeypatch.setenv("TRITON_HCU_ROCBLAS_A_BUFFER_TO_LDS", "1")

    source = '''
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [64, 0], [128, 0], [8, 0], [16, 0], [32, 0]]}, alignment = 16>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @copy(%p: !tt.ptr<f16>, %offset: tensor<256x16xi32, #blocked>, %stride: i32, %dst: !ttg.memdesc<256x16xf16, #shared, #smem, mutable>) RETURN_TYPE {
    %v = amdg.buffer_load %p[%offset] stride = %stride : tensor<256x16xf16, #blocked>
    ttg.local_store %v, %dst : tensor<256x16xf16, #blocked> -> !ttg.memdesc<256x16xf16, #shared, #smem, mutable>
    RETURN
  }
}
'''
    source = source.replace("RETURN_TYPE", "-> tensor<256x16xf16, #blocked>" if extra_use else "")
    source = source.replace("RETURN", "tt.return %v : tensor<256x16xf16, #blocked>" if extra_use else "tt.return")
    if operand == "B":
        source = source.replace("256x16", "16x256")
        source = source.replace(
            "[[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [64, 0], [128, 0], [8, 0], [16, 0], [32, 0]]",
            "[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [8, 0], [1, 0], [2, 0], [4, 0]]")
    source = with_buffer_to_lds(source, enabled, contract=contract)
    def add_copies(pm):
        hcu.passes.ttgpuir.add_convert_buffer_lds_copies(pm)
        if synchronization == "standalone":
            hcu.passes.ttgpuir.add_synchronize_buffer_lds_copies(pm)
        elif synchronization == "pipeline":
            hcu.passes.ttgpuir.add_schedule_buffer_to_lds(pm)

    if contract and extra_use:
        # Preserving a generic writer is unsafe while reads apply M0 wrap.
        with pytest.raises(RuntimeError):
            run_pass(tmp_path, source, add_copies)
        assert "buffer-to-LDS" in capfd.readouterr().err
        return
    result = run_pass(tmp_path, source, add_copies)
    assert ("amdg.buffer_load_to_local" in result) == (contract and not extra_use)
    if synchronization == "conversion-only":
        assert "ttg.async_commit_group" not in result
        assert "ttg.async_wait" not in result
        assert "ttg.local_barrier" not in result
    elif contract and not extra_use:
        assert result.count("ttg.async_commit_group") == 1
        assert "ttg.async_wait" in result
        assert "num = 0" in result
        if synchronization == "standalone":
            # Synchronization is immediately after submission, not at return.
            assert re.search(
                r"amdg.buffer_load_to_local[^\n]*\n\s*%\w+ = ttg.async_commit_group"
                r"[^\n]*\n\s*%\w+ = ttg.async_wait[^\n]*\n\s*ttg.local_barrier",
                result)
    if extra_use:
        assert "ttg.local_store" in result


@pytest.mark.parametrize("enabled", [False, True])
def test_wrapped_initialized_allocation_rejects_generic_writer(tmp_path, capfd, enabled):
    source = '''
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [64, 0], [128, 0], [8, 0], [16, 0], [32, 0]]}, alignment = 16>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @copy(%v: tensor<256x16xf16, #blocked>) -> tensor<256x16xf16, #blocked> {
    %a = ttg.local_alloc %v : (tensor<256x16xf16, #blocked>) -> !ttg.memdesc<256x16xf16, #shared, #smem>
    %r = ttg.local_load %a : !ttg.memdesc<256x16xf16, #shared, #smem> -> tensor<256x16xf16, #blocked>
    tt.return %r : tensor<256x16xf16, #blocked>
  }
}
'''
    source = with_buffer_to_lds(source, enabled)
    with pytest.raises(RuntimeError):
        run_pass(tmp_path, source, add_scheduled_buffer_to_lds)
    assert "buffer-to-LDS wrap requires" in capfd.readouterr().err


@pytest.mark.parametrize("arch", ["gfx936", "gfx938", "gfx92a", "gfx946"])
def test_buffer_lds_copy_lowering_boundary(tmp_path, arch):
    # Bypass the pipeline driver and verify that a selected BufferLds copy is
    # consumed by the target lowering on every supported architecture.
    prefix = allocation_contract_fixture("").split("module attributes {")[0]
    source = prefix + '''
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.target" = "hip:gfx936", "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @copy(%p: !tt.ptr<f16>, %offset: tensor<256x16xi32, #blocked>, %stride: i32, %dst: !ttg.memdesc<256x16xf16, #placement, #smem, mutable>) {
    %v = amdg.buffer_load %p[%offset] stride = %stride : tensor<256x16xf16, #blocked>
    ttg.local_store %v, %dst : tensor<256x16xf16, #blocked> -> !ttg.memdesc<256x16xf16, #placement, #smem, mutable>
    tt.return
  }
}
'''
    source = run_pass(tmp_path, source, add_scheduled_buffer_to_lds)
    assert "amdg.buffer_load_to_local" in source
    # This boundary test bypasses the normal axis-info/vectorization pass;
    # provide the proven 8xf16 width that production lowering infers.
    source = source.replace("contiguity = 0", "contiguity = 8")
    source = source.replace("hip:gfx936", f"hip:{arch}")
    def lower(pm):
        amd.passes.ttgpuir.add_allocate_shared_memory(pm)
        hcu.passes.ttgpuir.add_to_llvmir(pm, arch, False)
    result = run_pass(tmp_path, source, lower)
    assert "amdg.buffer_load_to_local" not in result
    assert "raw.ptr.buffer.load.async.lds" in result
    # This fixture bypasses ConvertToBufferOps, so its incoming stride remains
    # dynamic. The common descriptor builder consumes it without a planned
    # structured-address reconstruction.
    assert "llvm.trunc %arg2 : i32 to i16" in result


@pytest.mark.parametrize("vary", ["pointer", "stride", "initial_pointer",
                                  "initial_offsets", "global_write",
                                  "initial_mask", "none"])
@pytest.mark.parametrize("store_after_loop", [False, True])
def test_loop_producer_requires_invariant_descriptor(tmp_path, capfd, vary, store_after_loop):
    source = '''
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [64, 0], [128, 0], [8, 0], [16, 0], [32, 0]]}, alignment = 16>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @copy(%p: !tt.ptr<f16>, %other_p: !tt.ptr<f16>, %offset: tensor<256x16xi32, #blocked>, %other_offset: tensor<256x16xi32, #blocked>, %mask: tensor<256x16xi1, #blocked>, %stride: i32, %n: i32, %dst: !ttg.memdesc<256x16xf16, #shared, #smem, mutable>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %step = arith.constant dense<8> : tensor<256x16xi32, #blocked>
    %initial = amdg.buffer_load %p[%offset] stride = %stride : tensor<256x16xf16, #blocked>
    %r:2 = scf.for %iv = %zero to %n step %one iter_args(%current = %initial, %off = %offset) -> (tensor<256x16xf16, #blocked>, tensor<256x16xi32, #blocked>) : i32 {
      %p_next = tt.addptr %p, %iv : !tt.ptr<f16>, i32
      %s_next = arith.addi %stride, %iv : i32
      %off_next = arith.addi %off, %step : tensor<256x16xi32, #blocked>
      %future = amdg.buffer_load POINTER[%off_next] stride = STRIDE : tensor<256x16xf16, #blocked>
      INSIDE_STORE
      scf.yield %future, %off_next : tensor<256x16xf16, #blocked>, tensor<256x16xi32, #blocked>
    }
    OUTSIDE_STORE
    tt.return
  }
}
'''
    store = "ttg.local_store VALUE, %dst : tensor<256x16xf16, #blocked> -> !ttg.memdesc<256x16xf16, #shared, #smem, mutable>"
    source = source.replace("POINTER", "%p_next" if vary == "pointer" else "%p")
    source = source.replace("STRIDE", "%s_next" if vary == "stride" else "%stride")
    source = source.replace("INSIDE_STORE", "" if store_after_loop else store.replace("VALUE", "%current"))
    source = source.replace("OUTSIDE_STORE", store.replace("VALUE", "%r#0") if store_after_loop else "")
    if vary == "initial_pointer":
        source = source.replace("%initial = amdg.buffer_load %p[", "%initial = amdg.buffer_load %other_p[")
    if vary == "initial_offsets":
        source = source.replace("%initial = amdg.buffer_load %p[%offset]", "%initial = amdg.buffer_load %p[%other_offset]")
    if vary == "initial_mask":
        source = source.replace("%initial = amdg.buffer_load %p[%offset]", "%initial = amdg.buffer_load %p[%offset], %mask")
    if vary == "global_write":
        source = source.replace("%p_next =", "%zero_f = arith.constant 0.0 : f16\n      tt.store %p, %zero_f : !tt.ptr<f16>\n      %p_next =")
    source = with_buffer_to_lds(source, True)
    if vary in ("none", "initial_mask"):
        result = run_pass(tmp_path, source, add_scheduled_buffer_to_lds)
        assert "amdg.buffer_load_to_local" in result
        assert "ttg.local_store" not in result
        if vary == "initial_mask":
            assert "arith.select" in result
            # Canonicalization may reverse slt(0, n) into sgt(n, 0).
            assert "arith.cmpi" in result
        return
    with pytest.raises(RuntimeError):
        run_pass(tmp_path, source, add_scheduled_buffer_to_lds)
    assert "buffer-to-LDS wrap requires" in capfd.readouterr().err


def test_buffer_to_lds_option_is_cache_keyed(monkeypatch):
    from triton.backends.hcu.compiler import HIPOptions
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.setenv("TRITON_HIP_USE_ASYNC_COPY", "0")
    base = HIPOptions(arch="gfx936", num_warps=8, use_block_pingpong=True)
    enabled = HIPOptions(arch="gfx936", num_warps=8, use_block_pingpong=True,
                         use_async_copy=True)
    assert not base.use_async_copy
    assert base.hash() != enabled.hash()


def test_async_copy_does_not_require_matrix_read(monkeypatch):
    from triton.backends.hcu.compiler import HIPOptions
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.setenv("TRITON_HIP_USE_ASYNC_COPY", "0")
    monkeypatch.setenv("TRITON_HCU_ENABLE_DS_READ_M", "0")
    assert HIPOptions(arch="gfx936", num_warps=8, use_block_pingpong=True,
                      use_async_copy=True).use_async_copy


@pytest.mark.parametrize("environment", [False, True])
@pytest.mark.parametrize("explicit", [None, False, True])
def test_async_copy_option_overrides_environment(monkeypatch, environment, explicit):
    from triton.backends.compiler import GPUTarget
    from triton.backends.hcu.compiler import HIPBackend, HIPOptions
    monkeypatch.setenv("TRITON_HIP_USE_ASYNC_COPY", str(int(environment)))
    # MR !442 resolves the legacy environment in parse_options, not in the
    # HIPOptions dataclass. Exercise the actual compiler entry point.
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    options = backend.parse_options({"use_async_copy": explicit})
    expected = environment if explicit is None else explicit
    assert options.use_async_copy is expected
    monkeypatch.setenv("TRITON_HIP_USE_ASYNC_COPY", str(int(not environment)))
    assert options.use_async_copy is expected
    assert options.hash() == HIPOptions(arch="gfx936", use_async_copy=expected).hash()


@pytest.mark.parametrize("generic", [False, True])
@pytest.mark.parametrize("operand", ["A", "B"])
def test_wrapped_storage_supports_each_consumer_layout(tmp_path, capfd, generic, operand):
    shape = "256x16" if operand == "A" else "16x256"
    basis = ("[[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [64, 0], [128, 0], [8, 0], [16, 0], [32, 0]]"
             if operand == "A" else
             "[[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [8, 0], [1, 0], [2, 0], [4, 0]]")
    encoding = '#blocked' if generic else '#dot'
    source = f'''
#blocked = #ttg.blocked<{{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}}>
#mma = #ttg.amd_mfma<{{version = 3, warpsPerCTA = [4, 2], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1, tilesPerWarp = [1, 2]}}>
#dot = #ttg.dot_op<{{opIdx = {0 if operand == 'A' else 1}, parent = #mma, kWidth = 4}}>
#shared = #ttg.shared_linear<{{offset = {basis}}}, alignment = 16>
#smem = #ttg.shared_memory
module attributes {{"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32}} {{
  tt.func @read(%src: !ttg.memdesc<{shape}xf16, #shared, #smem, mutable>, %out: tensor<{shape}x!tt.ptr<f16>, {encoding}>) {{
    %v = ttg.local_load %src : !ttg.memdesc<{shape}xf16, #shared, #smem, mutable> -> tensor<{shape}xf16, {encoding}>
    tt.store %out, %v : tensor<{shape}x!tt.ptr<f16>, {encoding}>
    tt.return
  }}
}}
'''
    source = with_buffer_to_lds(source, True)
    def lower(pm):
        add_scheduled_buffer_to_lds(pm)
        hcu.passes.ttgpuir.add_to_llvmir(pm, "gfx936", False)
    result = run_pass(tmp_path, source, lower)
    assert "ttg.local_load" not in result
    assert "llvm." in result


@pytest.mark.parametrize("override", [dict(arch="gfx938"),
                                     dict(num_ctas=2), dict(use_block_pingpong=False),
                                     dict(utc_warmup=True)])
def test_async_copy_accepts_non_wrapped_modes(monkeypatch, override):
    from triton.backends.hcu.compiler import HIPOptions
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    options = dict(arch="gfx936", num_warps=8, use_block_pingpong=True,
                   use_async_copy=True)
    options.update(override)
    assert HIPOptions(**options).use_async_copy


@pytest.mark.parametrize("waves", [1, 2, 4, 8, 16])
def test_buffer_to_lds_accepts_wave_counts(monkeypatch, waves):
    from triton.backends.hcu.compiler import HIPOptions
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    options = HIPOptions(arch="gfx936", num_warps=waves,
                         use_block_pingpong=True, use_async_copy=True)
    assert options.num_warps == waves


def test_buffer_to_lds_target_profiles_are_shared_with_backend_selection():
    from triton.backends.hcu.compiler import supports_buffer_lds
    assert all(supports_buffer_lds(arch)
               for arch in ("gfx936", "gfx938", "gfx92a", "gfx946"))
    assert not supports_buffer_lds("gfx948")


@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("integrated", [False, True])
def test_default_pingpong_keeps_two_slots(tmp_path, monkeypatch, enabled, integrated):
    source = (Path(__file__).resolve().parents[3] /
              "test/TritonGPU/hcu/block-pingpong-prepare.mlir").read_text()
    # Exercise the measured architecture as well as the fixture's gfx938.
    source = source.replace("hip:gfx938", "hip:gfx936")
    if enabled is not None:
        # A request without an actual operand contract must not select the
        # four-generation policy, even on gfx936.
        source = with_buffer_to_lds(source, enabled)
        # with_buffer_to_lds adds the target; remove the fixture's duplicate.
        source = source.replace('ttg.target = "hip:gfx936", ', '')
    monkeypatch.setenv("TRITON_HCU_ROCBLAS_FOUR_PREFETCH", "1")
    def pipeline(pm):
        hcu.passes.ttgpuir.add_prepare_block_pingpong(pm)
        if integrated:
            amd.passes.ttgpuir.add_schedule_loops(pm, 2)
            amd.passes.ttgpuir.add_pipeline(pm, False, False)
            passes.common.add_canonicalizer(pm)
            hcu.passes.ttgpuir.add_block_pingpong(pm)

    result = run_pass(tmp_path, source, pipeline)
    assert 'tt.num_stages = 3 : i32' in result
    if integrated:
        assert 'hcu.block_pingpong.static_buffers' in result
    else:
        assert 'ttg.pipeline.hcu_num_buffers = 2 : i32' in result
    assert 'ttg.pipeline.hcu_num_buffers = 4 : i32' not in result
    assert 'tt.num_stages = 6 : i32' not in result


@pytest.mark.parametrize("bad_backedge", [False, True])
def test_four_resident_slots_require_matching_backedges(tmp_path, capfd, bad_backedge):
    aliases = '''
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [64, 0], [128, 0], [8, 0], [16, 0], [32, 0]]}, alignment = 16>
#smem = #ttg.shared_memory
'''
    tile = "!ttg.memdesc<256x16xf16, #shared, #smem, mutable>"
    allocation = "!ttg.memdesc<4x256x16xf16, #shared, #smem, mutable>"
    tensor = "tensor<256x16xf16, #blocked>"
    lines = [aliases, 'module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {',
             f'tt.func @ring(%limit: i32) -> {tensor} {{']
    lines += [f'%c{i} = arith.constant {i} : i32' for i in range(5)]
    for operand in ("a", "b"):
        lines.append(f'%{operand} = ttg.local_alloc : () -> {allocation}')
        for slot in range(4):
            lines.append(f'%init_{operand}{slot} = ttg.memdesc_index %{operand}[%c{slot}] : {allocation} -> {tile}')
    names = [f'{operand}{slot}' for operand in ('a', 'b') for slot in range(4)]
    init = ', '.join(f'%{name} = %init_{name}' for name in names)
    types = ', '.join(['i32'] + [tile] * 8)
    lines.append(f'%r:9 = scf.for %iv = %c0 to %limit step %c4 iter_args(%ring = %c3, {init}) -> ({types}) : i32 {{')
    for slot in range(4):
        previous = '%ring' if slot == 0 else f'%idx{slot - 1}'
        lines += [f'%inc{slot} = arith.addi {previous}, %c1 : i32',
                  f'%cmp{slot} = arith.cmpi slt, %inc{slot}, %c4 : i32',
                  f'%idx{slot} = arith.select %cmp{slot}, %inc{slot}, %c0 : i32']
        for operand in ('a', 'b'):
            name = f'{operand}{slot}'
            lines += [f'%hot_{name} = ttg.memdesc_index %{operand}[%idx{slot}] : {allocation} -> {tile}',
                      f'%v_{name} = ttg.local_load %{name} : {tile} -> {tensor}',
                      f'ttg.local_store %v_{name}, %hot_{name} : {tensor} -> {tile}']
    yields = names.copy()
    if bad_backedge:
        yields[0], yields[1] = yields[1], yields[0]
    lines += ['scf.yield %idx3, ' + ', '.join(f'%hot_{name}' for name in yields) + f' : {types}',
              '} {hcu.block_pingpong.applied, tt.num_stages = 6 : i32}',
              f'%va = ttg.local_load %r#1 : {tile} -> {tensor}',
              f'%vb = ttg.local_load %r#5 : {tile} -> {tensor}',
              f'%sum = arith.addf %va, %vb : {tensor}',
              f'tt.return %sum : {tensor}', '}', '}']
    source = with_buffer_to_lds('\n'.join(lines), True)
    if bad_backedge:
        with pytest.raises(RuntimeError):
            run_pass(tmp_path, source, hcu.passes.ttgpuir.add_block_pingpong)
        assert 'cannot prove the buffer-to-LDS ring' in capfd.readouterr().err
    else:
        result = run_pass(tmp_path, source, hcu.passes.ttgpuir.add_block_pingpong)
        assert 'hcu.block_pingpong.static_buffers' in result
        assert result.count('ttg.local_alloc') == 8
        assert 'memdesc<4x' not in result


def test_prefetch_hash():
    hashes = {HIPOptions(arch='gfx936', block_pingpong_async_global_prefetch=g,
                         block_pingpong_async_local_prefetch=r).hash()
              for g in range(2, 9) for r in range(1, g)}
    assert len(hashes) == 28


def test_automatic_depth_hash():
    automatic = HIPOptions(arch='gfx936')
    fixed = HIPOptions(arch='gfx936', block_pingpong_async_global_prefetch=4)
    assert automatic.hash() != fixed.hash()


@pytest.mark.parametrize('g,r', [
    (1, 1), (9, 1), (2, 2), (4, 0), (4, 4), (True, 1), (None, 8),
])
def test_invalid_prefetch_options(g, r):
    backend = HIPBackend(GPUTarget('hip', 'gfx936', 64))
    with pytest.raises(ValueError, match='planned prefetch'):
        backend.parse_options(dict(block_pingpong_async_global_prefetch=g,
                                   block_pingpong_async_local_prefetch=r))


def test_automatic_global_depth_does_not_imply_four_during_option_parsing():
    backend = HIPBackend(GPUTarget('hip', 'gfx936', 64))
    options = backend.parse_options(dict(block_pingpong_async_local_prefetch=4))
    assert options.block_pingpong_async_global_prefetch is None
    assert options.block_pingpong_async_local_prefetch == 4


def test_automatic_depth_is_validated_after_resolution():
    with pytest.raises(ValueError, match='LDS distance < global distance'):
        _validate_prefetch_distances(4, 4)


def test_copy_selection_is_independent_of_module_dot_count(tmp_path):
    source = (Path(__file__).resolve().parents[3] /
              "test/TritonGPU/hcu/block-pingpong-prepare.mlir").read_text()
    # Two independent copies of one small bf16 case suffice; do not run the
    # placement search over the fixture's unrelated large fp8 configurations.
    aliases = source[:source.index("module attributes {")]
    start = source.index("  tt.func public @bf16_bk64_256x128_single_a_candidate(")
    end = source.index("\n  }", start) + len("\n  }")
    function = source[start:end]
    source = (aliases + 'module attributes {ttg.target = "hip:gfx938", '
              '"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, '
              '"ttg.num-ctas" = 1 : i32} {\n' + function + '\n' +
              function.replace("@bf16_bk64_256x128_single_a_candidate", "@second") + '\n}')
    assert source.count("tt.dot ") > 1
    ordinary = run_pass(tmp_path, source,
                        hcu.passes.ttgpuir.add_select_buffer_lds_config)
    single = run_pass(
        tmp_path, aliases + 'module attributes {ttg.target = "hip:gfx938", '
        '"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, '
        '"ttg.num-ctas" = 1 : i32} {\n' + function + '\n}',
        hcu.passes.ttgpuir.add_select_buffer_lds_config)
    assert "#ttg.hcu_buffer_lds_shared" in ordinary
    assert ordinary.count("ttg.local_load") == 2 * single.count("ttg.local_load")


@pytest.mark.parametrize("arch", ["gfx936", "gfx938"])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("g,r,balanced", [(4, 1, True), (3, 1, False), (4, 2, False)])
def test_fragment_balance_precedes_buffer_lds_selection(tmp_path, arch, dtype,
                                                       g, r, balanced):
    source = '''
#a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#b = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1, tilesPerWarp = [1, 2]}>
#da = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#db = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
module attributes {ttg.target = "hip:ARCH", "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func public @gemm(
      %ap: tensor<256x16x!tt.ptr<DTYPE>, #a> {tt.contiguity = dense<[1, 8]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>},
      %bp: tensor<16x256x!tt.ptr<DTYPE>, #b> {tt.contiguity = dense<[1, 8]> : tensor<2xi32>, tt.divisibility = dense<[16, 16]> : tensor<2xi32>}) -> tensor<256x256xf32, #mma> {
    %zero = arith.constant dense<0.0> : tensor<256x256xf32, #mma>
    %av = tt.load %ap : tensor<256x16x!tt.ptr<DTYPE>, #a>
    %bv = tt.load %bp : tensor<16x256x!tt.ptr<DTYPE>, #b>
    %ad = ttg.convert_layout %av : tensor<256x16xDTYPE, #a> -> tensor<256x16xDTYPE, #da>
    %bd = ttg.convert_layout %bv : tensor<16x256xDTYPE, #b> -> tensor<16x256xDTYPE, #db>
    %c = tt.dot %ad, %bd, %zero : tensor<256x16xDTYPE, #da> * tensor<16x256xDTYPE, #db> -> tensor<256x256xf32, #mma>
    tt.return %c : tensor<256x256xf32, #mma>
  }
}
'''.replace("ARCH", arch).replace("DTYPE", dtype)
    selected = run_pass(tmp_path, source,
                        hcu.passes.ttgpuir.add_select_buffer_lds_config)
    assert "isTransposed = true" not in selected

    prepared = run_pass(
        tmp_path, source,
        lambda pm: hcu.passes.ttgpuir.add_prepare_block_pingpong_layout(pm, g, r))
    assert ("isTransposed = true" in prepared) == balanced
    assert "ttg.local_alloc" not in prepared
    if balanced:
        assert "warpsPerCTA = [4, 2]" in prepared
    selected = run_pass(tmp_path, prepared,
                        hcu.passes.ttgpuir.add_select_buffer_lds_config)
    assert "#ttg.hcu_buffer_lds_shared" in selected
    assert ("isTransposed = true" in selected) == balanced


# Complete pipeline compilation and runtime-bound preservation.

def test_pingpong_rematch_keeps_unsupported_ordinary_loop(tmp_path):
    source = '''
module attributes {"ttg.num-warps" = 8 : i32,
                   "ttg.threads-per-warp" = 64 : i32,
                   ttg.target = "hip:gfx936"} {
  tt.func @ordinary(%n: i32, %p: !tt.ptr<i32>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    scf.for %i = %zero to %n step %one : i32 {
      tt.store %p, %i : !tt.ptr<i32>
      scf.yield
    } {hcu.block_pingpong.buffer_load_stage3}
    tt.return
  }
}
'''
    result = run_pass(tmp_path, source, hcu.passes.ttgpuir.add_block_pingpong)
    assert "scf.for" in result
    assert "hcu.block_pingpong.buffer_load_stage3" not in result
    assert "hcu.block_pingpong.applied" not in result


@pytest.mark.parametrize("arch", ["gfx936", "gfx938"])
@pytest.mark.parametrize("column_b", [False, True])
@pytest.mark.parametrize("mmac_layout", [-1, 2, 4])
def test_runtime_k_complete_ttgir_pipeline(monkeypatch, arch, column_b,
                                          mmac_layout):
    from triton.compiler import ASTSource
    from triton._C.libtriton import ir

    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    backend = HIPBackend(GPUTarget("hip", arch, 64))
    options = backend.parse_options(dict(
        use_async_copy=True, use_block_pingpong=True, num_warps=8,
        num_stages=4, matrix_instr_nonkdim=16, optimize_epilogue=False,
        mmac_layout_force=mmac_layout))
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    # Pointer range and K alignment are explicit input contracts for this
    # IR-only test; the reduction bound itself remains a runtime argument.
    source = ASTSource(
        _runtime_k_gemm, dict(a="*fp16", b="*fp16", c="*fp16", K="i32"),
        constexprs=dict(COLUMN_B=column_b, BK=32 if column_b else 16),
        attrs={(i,): [["tt.divisibility", 16], ["tt.pointer_range", 32]]
               for i in range(3)} | {(3,): [["tt.divisibility", 16]]})
    module = source.make_ir(options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)
    metadata = {}
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    assert "applied_planned_async_schedule" not in metadata
    assert "block_pingpong_async_global_prefetch" not in metadata
    assert "hcu.planned_async" not in str(module)
    assert "hcu.block_pingpong.global_prefetch_distance" not in str(module)
    assert "hcu.block_pingpong.lds_prefetch_distance" not in str(module)
    # gfx936 lacks MMAC_LAYOUT and AccelerateMatmul already falls back to
    # LEGACY for unsupported force values. gfx938 must retain valid forced
    # interleaved layouts rather than selecting the balanced transpose.
    if column_b or (arch == "gfx938" and mmac_layout in (2, 4)):
        assert "isTransposed = true" not in str(module)


@pytest.mark.parametrize("arch", ["gfx936", "gfx938"])
def test_resolved_prefetch_depth_rejected_before_expansion(monkeypatch, capfd, arch):
    from triton.compiler import ASTSource
    from triton._C.libtriton import ir

    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    backend = HIPBackend(GPUTarget("hip", arch, 64))
    options = backend.parse_options(dict(
        use_async_copy=True, use_block_pingpong=True, num_warps=8,
        num_stages=4, matrix_instr_nonkdim=16, optimize_epilogue=False,
        mmac_layout_force=1, block_pingpong_async_local_prefetch=4))
    assert options.block_pingpong_async_global_prefetch is None
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(
        _runtime_k_gemm, dict(a="*fp16", b="*fp16", c="*fp16", K="i32"),
        constexprs=dict(COLUMN_B=False, BK=16),
        attrs={(i,): [["tt.divisibility", 16], ["tt.pointer_range", 32]]
               for i in range(3)} | {(3,): [["tt.divisibility", 16]]})
    module = source.make_ir(options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)
    module = backend.make_ttir(module, {}, options)
    with pytest.raises(RuntimeError):
        backend.make_ttgir(module, {}, options)
    diagnostic = capfd.readouterr().err
    assert "1 <= LDS distance < global distance" in diagnostic
    assert "fragment prefetch exceeds" not in diagnostic


@pytest.mark.parametrize("arch", ["gfx936", "gfx938"])
@pytest.mark.parametrize("dtype,bm,bn,bk,g", [
    ("i8", 128, 128, 32, 4),
    ("i8", 128, 256, 32, 4),
    ("i8", 128, 128, 64, 4),
    ("fp16", 128, 128, 64, 8),
    ("bf16", 128, 128, 64, 8),
])
def test_unsupported_copy_retains_generic_pipeline(monkeypatch, arch, dtype,
                                                  bm, bn, bk, g):
    # Verify this change's fallback boundary at TTGIR. Generic async backend
    # lowering is a separate contract (including existing int8 toolchain issues).
    from triton.compiler import ASTSource
    from triton._C.libtriton import ir
    from test_buffer_to_lds_gpu import direct_gemm

    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    backend = HIPBackend(GPUTarget("hip", arch, 64))
    options = backend.parse_options(dict(
        use_async_copy=True, use_block_pingpong=True, num_warps=4,
        num_stages=2, matrix_instr_nonkdim=16, optimize_epilogue=False,
        block_pingpong_async_global_prefetch=g))
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(
        direct_gemm,
        dict(a="*" + dtype, b="*" + dtype,
             c="*i32" if dtype == "i8" else "*" + dtype,
             m="i32", n="i32", k="i32"),
        constexprs=dict(BM=bm, BN=bn, BK=bk, REVERSE_A=False,
                        A_COLUMN_MAJOR=False, B_COLUMN_MAJOR=False,
                        BATCHED=False, AFFINE_COORDS=False, K_STEP_RANGE=False),
        attrs={(i,): [["tt.divisibility", 16], ["tt.pointer_range", 32]]
               for i in range(3)})
    module = source.make_ir(options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)
    metadata = {}
    module = backend.make_ttir(module, metadata, options)
    module = backend.make_ttgir(module, metadata, options)
    assert not metadata["uses_bank_conflict_aware_lds_placement"]
    assert "#ttg.hcu_buffer_lds_shared" not in str(module)
    assert "hcu.block_pingpong.static_buffers" not in str(module)
