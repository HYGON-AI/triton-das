# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Check every converted pair and store-vector half without a GPU.

Adapted from !561's truncf test for store-side fusion and typed ROCDL ops.
Canonicalization runs only after lowering, so input layout conversions remain
visible to the fusion step. PMD value coverage lives in the Gluon tests.
"""
import os
import re
import subprocess
import pytest
from triton._C import libtriton

SSA = r"%[\w.]+"

def _source(dtype, consumer, extent, tile):
    layout = "#mma"
    # Changing only K does not change the accumulator's linear layout. Keep
    # this convert_layout in the input: canonicalizing first would hide the
    # store-input fusion behavior that this test is exercising.
    converted = ""
    value = "%narrow"
    if consumer == "equivalent_layout":
        layout = "#mma_k8"
        converted = f"""
    %converted = ttg.convert_layout %narrow : tensor<{extent}x32x{dtype}, #mma> -> tensor<{extent}x32x{dtype}, #mma_k8>
"""
        value = "%converted"
    if consumer == "ordinary":
        use = f"""
    %ptr = tt.splat %base : !tt.ptr<{dtype}> -> tensor<{extent}x32x!tt.ptr<{dtype}>, #mma>
    tt.store %ptr, %narrow : tensor<{extent}x32x!tt.ptr<{dtype}>, #mma>
"""
    else:
        use = f"""
    amdg.matrix_store_from_reg {value}, %base, shape = [%rows, %cols], strides = [%stride, %one], tensorShape = [{extent}, 32], indices = [%zero, %zero]
      {{MlsStoreEncoding = #amdg<MlsStoreEncoding storeTile = [{tile[0]}, {tile[1]}], elemBitWidth = 16, interleaveKind = 1, version = 3, order = [1, 0], warpsPerCTA = [1, 1]>}}
      : tensor<{extent}x32x{dtype}, {layout}>, !tt.ptr<{dtype}>
"""
    return f"""
#mma = #ttg.amd_mfma<{{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}}>
#mma_k8 = #ttg.amd_mfma<{{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 8], isTransposed = false, mmacLayout = 3, tilesPerWarp = [1, 2]}}>
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx946"}} {{
  tt.func private @truncf_store(%input: tensor<{extent}x32xf32, #mma>, %base: !tt.ptr<{dtype}>) {{
    %zero = arith.constant 0 : i32
    %rows = arith.constant {extent} : i32
    %cols = arith.constant 32 : i32
    %stride = arith.constant 32 : i64
    %one = arith.constant 1 : i64
    %narrow = arith.truncf %input : tensor<{extent}x32xf32, #mma> to tensor<{extent}x32x{dtype}, #mma>
    {converted}
    {use}
    tt.return
  }}
}}
"""


def _lower(tmp_path, source):
    path = tmp_path / "truncf_store.ttgir"
    path.write_text(source)
    # A source-tree executable can be selected without installing its wheel.
    if os.environ.get("TRITON_OPT"):
        result = subprocess.run(
            [os.environ["TRITON_OPT"], str(path), "--allocate-shared-memory",
             "--convert-triton-hcu-to-llvm=arch=gfx946", "--canonicalize", "--cse"],
            check=True, capture_output=True, text=True,
        )
        return result.stdout
    if not hasattr(libtriton, "hcu"):
        pytest.skip("requires the HCU compiler backend, but no GPU")
    ir = libtriton.ir
    context = ir.context()
    ir.load_dialects(context)
    libtriton.hcu.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    pm = ir.pass_manager(context)
    libtriton.amd.passes.ttgpuir.add_allocate_shared_memory(pm)
    libtriton.hcu.passes.ttgpuir.add_to_llvmir(pm, "gfx946", False)
    libtriton.passes.common.add_canonicalizer(pm)
    libtriton.passes.common.add_cse(pm)
    pm.run(module, "truncf_matrix_store")
    return str(module)


def _llvm_ir(tmp_path, lowered):
    """Translate actual emitted LLVM dialect to LLVM IR, still without a GPU."""
    ir = libtriton.ir
    context = ir.context()
    ir.load_dialects(context)
    if hasattr(libtriton, "hcu"):
        libtriton.hcu.load_dialects(context)
    path = tmp_path / "truncf_store.llvm.mlir"
    path.write_text(lowered)
    module = ir.parse_mlir_module(str(path), context)
    pm = ir.pass_manager(context)
    libtriton.passes.common.add_canonicalizer(pm)
    libtriton.passes.common.add_cse(pm)
    pm.run(module, "truncf_matrix_store_cleanup")
    llvm_context = libtriton.llvm.context()
    return str(libtriton.llvm.to_module(module, llvm_context))



@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("extent", [32, 64])
@pytest.mark.parametrize("consumer,tile", [
    ("direct", (16, 32)), ("direct", (32, 32)),
    ("equivalent_layout", (16, 32)), ("equivalent_layout", (32, 32)),
    ("ordinary", None),
])
def test_truncf_matrix_store_pairing(tmp_path, dtype, extent, consumer, tile):
    lowered = _lower(tmp_path, _source(dtype, consumer, extent, tile))
    (tmp_path / "lowered.mlir").write_text(lowered)
    inputs = {value: int(index) for value, index in re.findall(
        rf"({SSA}) = llvm.extractvalue %arg0\[(\d+)\]", lowered)}
    calls = re.findall(
        rf"({SSA}) = rocdl.hcu.cvt.pk.{dtype}.f32 ({SSA}), ({SSA})", lowered)
    assert calls, lowered
    actual = [(inputs[a], inputs[b]) for _, a, b in calls]
    if consumer == "ordinary":
        expected = [(i, i + 1) for i in range(0, extent // 2, 2)]
    else:
        expected = [(0, 4), (1, 5), (2, 6), (3, 7),
                    (8, 12), (9, 13), (10, 14), (11, 15)]
        if extent == 64:
            expected += [(16, 20), (17, 21), (18, 22), (19, 23),
                         (24, 28), (25, 29), (26, 30), (27, 31)]
    assert actual == expected, lowered

    # Track SSA values through extraction and insertion: a packed pair's low
    # and high halves must reach the correct positions in each store vector.
    constants = {v: int(n) for v, n in re.findall(
        rf"({SSA}) = llvm.mlir.constant\((\d+) : i32\)", lowered)}
    pairs = {result: (inputs[a], inputs[b]) for result, a, b in calls}
    halves = {}
    vectors = {}
    stores = []
    for line in lowered.splitlines():
        m = re.search(rf"({SSA}) = llvm.extractelement ({SSA})\[({SSA}) : i32\]", line)
        if m and m[2] in pairs:
            halves[m[1]] = pairs[m[2]][constants[m[3]]]
        m = re.search(rf"({SSA}) = llvm.insertelement ({SSA}), ({SSA})\[({SSA}) : i32\]", line)
        if m and m[2] in halves:
            vector = dict(vectors.get(m[3], {}))
            vector[constants[m[4]]] = halves[m[2]]
            vectors[m[1]] = vector
        # The native store views these same bits as i32 VGPRs. Keep slot
        # numbering in 16-bit units while following this final bitcast.
        m = re.search(rf"({SSA}) = llvm.bitcast ({SSA}) :", line)
        if m and m[2] in vectors:
            vectors[m[1]] = vectors[m[2]]
        m = re.search(rf"rocdl.matrix.store[^ ]* ({SSA})", line)
        if m:
            stores.append(vectors[m[1]])
    if consumer != "ordinary":
        elems_per_store = tile[0] * tile[1] // 64
        assert len(stores) == extent // tile[0], lowered
        flattened = [v[i] for v in stores for i in range(elems_per_store)]
        assert flattened == [v for pair in expected for v in pair], lowered
        assert sorted(flattened) == list(range(extent // 2)), lowered
    else:
        assert not stores, lowered

    # The actual translator must preserve all pairings, not only parse the op.
    llvm_ir = _llvm_ir(tmp_path, lowered)
    (tmp_path / "translated.ll").write_text(llvm_ir)
    llvm_inputs = {v: int(i) for v, i in re.findall(
        rf"({SSA}) = extractvalue \{{ float(?:, float)* \}} {SSA}, (\d+)", llvm_ir)}
    operands = re.findall(
        rf"call <2 x (?:half|bfloat)> @llvm.hcu.cvt.pk.{dtype}.f32\(([^)]*)\)", llvm_ir)
    llvm_pairs = [tuple(llvm_inputs[v] for v in re.findall(rf"float ({SSA})", x))
                  for x in operands]
    assert llvm_pairs == expected, llvm_ir
    if tile is not None:
        insn = "32x16" if tile == (16, 32) else "32x32"
        assert len(re.findall(rf"call void @llvm.hcu.matrix.store.{insn}.b16\(", llvm_ir)) == extent // tile[0]
