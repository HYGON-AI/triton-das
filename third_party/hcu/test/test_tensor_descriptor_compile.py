# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Descriptor compatibility and automatic MLS selection, without a GPU."""
import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from test_scaled_compile import compile_to_llvm

TARGET = GPUTarget("hip", "gfx946", 64)


def assert_descriptor_stores_use_pointers(compiled):
    for stage in ("ttir", "ttgir"):
        assert "matrix_store" not in compiled.asm[stage]


@triton.jit
def descriptor_copy(A, C, ROWS: tl.constexpr, COLS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                    STRIDE: tl.constexpr, OM: tl.constexpr, ON: tl.constexpr, PAD: tl.constexpr):
    desc = tl.make_tensor_descriptor(A, [ROWS, COLS], [STRIDE, 1], [BM, BN], padding_option=PAD)
    value = desc.load([OM, ON])
    out = tl.make_tensor_descriptor(C, [BM, BN], [BN, 1], [BM, BN])
    out.store([0, 0], value)


@triton.jit
def descriptor_gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, TILES: tl.constexpr,
                    WARP_SPECIALIZE: tl.constexpr = False, BM: tl.constexpr = 64, BN: tl.constexpr = 64,
                    TRANS_B: tl.constexpr = False):
    a = tl.make_tensor_descriptor(A, [M, K], [K, 1], [BM, 64])
    if TRANS_B:
        b = tl.make_tensor_descriptor(B, [N, K], [K, 1], [BN, 64])
    else:
        b = tl.make_tensor_descriptor(B, [K, N], [N, 1], [64, BN])
    c = tl.make_tensor_descriptor(C, [M, N], [N, 1], [BM, BN])
    acc = tl.zeros((BM, BN), tl.float32)
    for i in tl.range(TILES, warp_specialize=WARP_SPECIALIZE):
        av = a.load([0, i * 64])
        if TRANS_B:
            bv = b.load([0, i * 64]).T
        else:
            bv = b.load([i * 64, 0])
        acc = tl.dot(av, bv, acc)
    c.store([0, 0], acc.to(C.dtype.element_ty))


def compile_copy(dtype="fp16", *, arch="gfx946", aligned=True, options=None, **kwargs):
    constants = dict(ROWS=64, COLS=64, BM=64, BN=64, STRIDE=64, OM=0, ON=0, PAD="zero")
    constants.update(kwargs)
    # kwargs deliberately override defaults, including descriptor dimensions.
    source = ASTSource(descriptor_copy, signature={"A": f"*{dtype}", "C": f"*{dtype}"}, constexprs=constants,
                       attrs={(i, ): [("tt.divisibility", 16 if aligned else 1)]
                              for i in range(2)})
    opts = dict(num_warps=4, num_stages=1)
    opts.update(options or {})
    return compile_to_llvm(source, GPUTarget("hip", arch, 64), opts)


@triton.jit
def descriptor_dot(A, B, C, ROWS: tl.constexpr, COLS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   STRIDE: tl.constexpr, OM: tl.constexpr, ON: tl.constexpr, PAD: tl.constexpr,
                   TRANS: tl.constexpr = False):
    desc = tl.make_tensor_descriptor(A, [ROWS, COLS], [STRIDE, 1], [BM, BN], padding_option=PAD)
    a = desc.load([OM, ON])
    if TRANS:
        a = a.T
    bk: tl.constexpr = BM if TRANS else BN
    m: tl.constexpr = BN if TRANS else BM
    offsets = tl.arange(0, bk)[:, None] * 64 + tl.arange(0, 64)[None, :]
    b = tl.load(B + offsets)
    out = tl.make_tensor_descriptor(C, [m, 64], [64, 1], [m, 64])
    out.store([0, 0], tl.dot(a, b).to(C.dtype.element_ty))


def compile_dot(dtype="fp16", *, arch="gfx946", aligned=True, options=None, **kwargs):
    constants = dict(ROWS=64, COLS=64, BM=64, BN=64, STRIDE=64, OM=0, ON=0, PAD="zero", TRANS=False)
    constants.update(kwargs)
    source = ASTSource(descriptor_dot, signature=dict(A=f"*{dtype}", B=f"*{dtype}", C="*fp16"),
                       constexprs=constants,
                       attrs={(i, ): [("tt.divisibility", 16 if aligned else 1)] for i in range(3)})
    opts = dict(num_warps=4, num_stages=1)
    opts.update(options or {})
    return compile_to_llvm(source, GPUTarget("hip", arch, 64), opts)


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "i8", "u8", "fp8e4nv", "fp8e5"])
@pytest.mark.parametrize("arch", ["gfx946", "gfx948"])
def test_standalone_descriptor_load_keeps_pointers(monkeypatch, dtype, arch):
    monkeypatch.setenv("TRITON_HCU_ENABLE_TENSOR_DESCRIPTOR_MLS", "0")
    original = compile_copy(dtype, arch=arch)
    monkeypatch.setenv("TRITON_HCU_ENABLE_TENSOR_DESCRIPTOR_MLS", "1")
    compiled = compile_copy(dtype, arch=arch)
    assert compiled.asm == original.asm
    assert "tt.matrix_load" not in compiled.asm["ttir"]
    assert "matrix_load_to_local" not in compiled.asm["ttgir"]


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "i8", "fp8e4nv", "fp8e5"])
@pytest.mark.parametrize("arch", ["gfx946", "gfx948"])
def test_descriptor_load_uses_mls(dtype, arch):
    compiled = compile_dot(dtype, arch=arch)
    assert "tt.matrix_load" in compiled.asm["ttir"]
    assert "amdg.matrix_load_to_local" in compiled.asm["ttgir"]
    assert "@llvm.hcu.matrix.load" in compiled.asm["llir"]
    assert_descriptor_stores_use_pointers(compiled)


@pytest.mark.parametrize("kwargs", [
    dict(OM=-1),
    dict(ON=-8),
    dict(ON=1),
    dict(STRIDE=65),
    dict(BM=8),
    dict(PAD="nan"),
    dict(OM=2**31 - 32),
])
def test_incompatible_load_keeps_pointer_semantics(kwargs):
    compiled = compile_dot(**kwargs)
    assert "tt.matrix_load" not in compiled.asm["ttir"]
    assert "matrix_load_to_local" not in compiled.asm["ttgir"]


# gfx942 gating is covered by the TTIR lit regression.
@pytest.mark.parametrize("dtype,arch,aligned", [
    ("fp32", "gfx946", True),
    ("fp16", "gfx938", True),
    ("fp16", "gfx92a", True),
    ("fp16", "gfx946", False),
])
def test_incompatible_type_target_alignment(dtype, arch, aligned):
    compiled = compile_dot(dtype, arch=arch, aligned=aligned)
    assert "tt.matrix_load" not in compiled.asm["ttir"]


@pytest.mark.parametrize("layout", [1, 2, 3, 4])
@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("tail", [False, True, "both"])
def test_descriptor_gemm(layout, stages, tail):
    source = ASTSource(
        descriptor_gemm, signature={"A": "*fp16", "B": "*fp16", "C": "*fp16"},
        constexprs=dict(M=60 if tail else 64, N=56 if tail == "both" else 64, K=96 if tail else 128, TILES=2,
                        WARP_SPECIALIZE=False, BM=64, BN=64, TRANS_B=False), attrs={(i, ): [("tt.divisibility", 16)]
                                                                                    for i in range(3)})
    compiled = compile_to_llvm(source, TARGET,
                               dict(num_warps=4, num_stages=stages, mmac_layout_force=layout, optimize_epilogue=True))
    assert compiled.asm["ttir"].count("tt.matrix_load") == 2
    assert "tt.matrix_store" not in compiled.asm["ttir"]
    assert "amdg.matrix_load_to_local" in compiled.asm["ttgir"]
    assert_descriptor_stores_use_pointers(compiled)


@pytest.mark.parametrize("om,on", [(64, 64), (128, 128)])
def test_positive_oob_retains_both_boundary_checks(om, on):
    compiled = compile_dot(OM=om, ON=on)
    assert "boundaryCheck = [0, 1]" in compiled.asm["ttir"]
    assert "amdg.matrix_load_to_local" in compiled.asm["ttgir"]


def test_proven_in_bounds_access_omits_boundary_checks():
    compiled = compile_dot()
    assert "boundaryCheck" not in compiled.asm["ttir"]


@pytest.mark.parametrize("value,enabled", [(None, True), ("1", True), ("true", True), ("on", True),
                                         ("0", False), ("false", False), ("off", False)])
def test_descriptor_mls_environment_switch(monkeypatch, value, enabled):
    from triton._C.libtriton import get_cache_invalidating_env_vars

    name = "TRITON_HCU_ENABLE_TENSOR_DESCRIPTOR_MLS"
    if value is None:
        monkeypatch.delenv(name, raising=False)
        assert name not in get_cache_invalidating_env_vars()
    else:
        monkeypatch.setenv(name, value)
        assert get_cache_invalidating_env_vars()[name] == str(enabled).lower()
    compiled = compile_dot()
    assert ("tt.matrix_load" in compiled.asm["ttir"]) == enabled
    assert "tt.matrix_store" not in compiled.asm["ttir"]
    assert ("amdg.matrix_load_to_local" in compiled.asm["ttgir"]) == enabled


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("bm,bn,warps", [(16, 32, 1), (64, 64, 8), (64, 64, 16)])
def test_descriptor_dot_warp_counts(dtype, bm, bn, warps):
    compiled = compile_dot(dtype, BM=bm, BN=bn, options=dict(num_warps=warps))
    assert "tt.matrix_load" in compiled.asm["ttir"]
    assert "amdg.matrix_load_to_local" in compiled.asm["ttgir"]
    assert_descriptor_stores_use_pointers(compiled)


@pytest.mark.parametrize("bm,bn", [(16, 32), (32, 64), (64, 128)])
def test_standalone_load_tiles_keep_pointers(bm, bn):
    compiled = compile_copy(BM=bm, BN=bn, STRIDE=128, COLS=128)
    assert "tt.matrix_load" not in compiled.asm["ttir"]
    assert "matrix_load_to_local" not in compiled.asm["ttgir"]


@triton.jit
def dynamic_descriptor_copy(A, B, C, rows, cols, offset, flag):
    desc = tl.make_tensor_descriptor(A, [rows, cols], [cols, 1], [64, 64])
    if flag:
        desc = tl.make_tensor_descriptor(B, [rows, cols], [cols, 1], [64, 64])
    value = desc.load([offset, 0])
    out = tl.make_tensor_descriptor(C, [64, 64], [64, 1], [64, 64])
    out.store([0, 0], value)


def test_control_flow_and_unknown_address_keep_generic_fallback():
    source = ASTSource(dynamic_descriptor_copy, signature=dict(A="*fp16", B="*fp16", C="*fp16", rows="i32", cols="i32",
                                                               offset="i32", flag="i1"))
    compiled = compile_to_llvm(source, TARGET, dict(num_warps=4, num_stages=1))
    assert "tensordesc" not in compiled.asm["ttir"]
    assert "tt.matrix_load" not in compiled.asm["ttir"]


@triton.jit
def ranked_descriptor_copy(A, C, RANK: tl.constexpr):
    if RANK == 1:
        desc = tl.make_tensor_descriptor(A, [64], [1], [64])
        out = tl.make_tensor_descriptor(C, [64], [1], [64])
        out.store([0], desc.load([0]))
    else:
        desc = tl.make_tensor_descriptor(A, [2, 16, 64], [1024, 64, 1], [2, 16, 64])
        out = tl.make_tensor_descriptor(C, [2, 16, 64], [1024, 64, 1], [2, 16, 64])
        out.store([0, 0, 0], desc.load([0, 0, 0]))


@pytest.mark.parametrize("rank", [1, 3])
def test_other_descriptor_ranks_keep_generic_fallback(rank):
    source = ASTSource(ranked_descriptor_copy, signature=dict(A="*fp16", C="*fp16"), constexprs=dict(RANK=rank))
    compiled = compile_to_llvm(source, TARGET, dict(num_warps=4, num_stages=1))
    assert "tensordesc" not in compiled.asm["ttir"]
    assert "matrix_load" not in compiled.asm["ttir"]
    assert "matrix_store" not in compiled.asm["ttir"]


def pipeline_options(mode):
    transposed = mode.endswith("-trans")
    mode = mode.removesuffix("-trans")
    options = dict(num_warps=4, num_stages=2, mmac_layout_force=3, optimize_epilogue=True)
    if mode == "async":
        options["use_async_copy"] = True
    else:
        options.update(wasp_wdra=False, wasp_partition_warps=(4, 4, 4))
        if mode == "wasp-single":
            options.update(wasp_wdra=True, wasp_partition_warps=(4, 4), wasp_partition_regs=(16, 240))
        elif mode in ("wasp", "wasp-tail"):
            options.update(wasp_wdra=True, wasp_partition_regs=(16, 244, 244), empty_arrive_after_mmac=False)
    if transposed:
        options.pop("wasp_partition_regs", None)
    return options


@pytest.mark.parametrize(
    "mode",
    ["async", "wasp-small", "wasp-single", "wasp", "wasp-tail", "wasp-single-trans", "wasp-trans", "wasp-tail-trans"])
@pytest.mark.parametrize("layout", [2, 3])
def test_descriptor_gemm_pipeline_modes(mode, layout):
    base = mode.removesuffix("-trans")
    block = 128 if base == "wasp-single" else 256 if base in ("wasp", "wasp-tail") else 64
    constants = dict(M=240 if base == "wasp-tail" else block, N=248 if base == "wasp-tail" else block,
                     K=96 if base == "wasp-tail" else 128, TILES=2, BM=block, BN=block, WARP_SPECIALIZE=base != "async",
                     TRANS_B=mode.endswith("-trans"))
    source = ASTSource(descriptor_gemm, signature={"A": "*fp16", "B": "*fp16", "C": "*fp16"}, constexprs=constants,
                       attrs={(i, ): [("tt.divisibility", 16)]
                              for i in range(3)})
    compiled = compile_to_llvm(source, TARGET, dict(pipeline_options(mode), mmac_layout_force=layout))
    assert compiled.asm["ttir"].count("tt.matrix_load") == 2
    assert "amdg.matrix_load_to_local" in compiled.asm["ttgir"]
    assert_descriptor_stores_use_pointers(compiled)


@triton.jit
def narrow_dot(A, C):
    desc = tl.make_tensor_descriptor(A, [64, 64], [64, 1], [64, 64])
    a = desc.load([0, 0])
    b = tl.full((64, 8), 1, tl.float16)
    result = tl.dot(a, b)
    offsets = tl.arange(0, 64)[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(C + offsets, result)


def test_unaccelerated_dot_load_falls_back():
    source = ASTSource(narrow_dot, signature=dict(A="*fp16", C="*fp32"), attrs={(0, ): [("tt.divisibility", 16)]})
    compiled = compile_to_llvm(source, TARGET, dict(num_warps=4, num_stages=1))
    assert "tt.matrix_load" not in compiled.asm["ttir"]
    assert "matrix_load_to_local" not in compiled.asm["ttgir"]


@triton.jit
def two_gemms(A, B, B2, C, C2):
    ad = tl.make_tensor_descriptor(A, [64, 64], [64, 1], [64, 64])
    bd = tl.make_tensor_descriptor(B, [64, 64], [64, 1], [64, 64])
    cd = tl.make_tensor_descriptor(C, [64, 64], [64, 1], [64, 64])
    c2d = tl.make_tensor_descriptor(C2, [64, 64], [64, 1], [64, 64])
    a, b = ad.load([0, 0]), bd.load([0, 0])
    offsets = tl.arange(0, 64)[:, None] * 64 + tl.arange(0, 64)[None, :]
    b2 = tl.load(B2 + offsets)
    cd.store([0, 0], tl.dot(a, b).to(tl.float16))
    c2d.store([0, 0], tl.dot(a, b2).to(tl.float16))


def test_descriptor_load_shared_by_two_gemms():
    source = ASTSource(two_gemms, signature={name: "*fp16"
                                             for name in ("A", "B", "B2", "C", "C2")}, attrs={(i, ):
                                                                                              [("tt.divisibility", 16)]
                                                                                              for i in range(5)})
    compiled = compile_to_llvm(source, TARGET,
                               dict(num_warps=4, num_stages=1, mmac_layout_force=3, optimize_epilogue=True))
    assert compiled.asm["ttir"].count("tt.matrix_load") == 2
    assert "tt.load" in compiled.asm["ttir"]  # The second GEMM's ordinary B load.
    assert_descriptor_stores_use_pointers(compiled)


@triton.jit
def shared_load_mixed_dots(A, C, D, NARROW_FIRST: tl.constexpr):
    desc = tl.make_tensor_descriptor(A, [64, 64], [64, 1], [64, 64])
    a = desc.load([0, 0])
    b = tl.full((64, 64), 1, tl.float16).to(A.dtype.element_ty)
    narrow_b = tl.full((64, 8), 1, tl.float16).to(A.dtype.element_ty)
    if NARROW_FIRST:
        narrow = tl.dot(a, narrow_b)
        wide = tl.dot(a, b)
    else:
        wide = tl.dot(a, b)
        narrow = tl.dot(a, narrow_b)
    out = tl.make_tensor_descriptor(C, [64, 64], [64, 1], [64, 64])
    out.store([0, 0], wide.to(C.dtype.element_ty))
    offsets = tl.arange(0, 64)[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(D + offsets, narrow.to(D.dtype.element_ty))


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "i8", "fp8e4nv", "fp8e5"])
@pytest.mark.parametrize("narrow_first", [False, True])
def test_shared_load_checks_all_dot_consumers(monkeypatch, dtype, narrow_first):
    source = ASTSource(
        shared_load_mixed_dots, signature=dict(A=f"*{dtype}", C="*fp16", D="*fp16"),
        constexprs=dict(NARROW_FIRST=narrow_first),
        attrs={(i, ): [("tt.divisibility", 16)] for i in range(3)})
    options = dict(num_warps=4, num_stages=1)
    monkeypatch.setenv("TRITON_HCU_ENABLE_TENSOR_DESCRIPTOR_MLS", "0")
    original = compile_to_llvm(source, TARGET, options)
    monkeypatch.setenv("TRITON_HCU_ENABLE_TENSOR_DESCRIPTOR_MLS", "1")
    compiled = compile_to_llvm(source, TARGET, options)
    assert compiled.asm == original.asm
    assert "tt.matrix_load" not in compiled.asm["ttir"]
    assert "matrix_load_to_local" not in compiled.asm["ttgir"]


@triton.jit
def descriptor_transpose_copy(A, C, ROWS: tl.constexpr, COLS: tl.constexpr, WRITE_INPUT: tl.constexpr = False):
    desc = tl.make_tensor_descriptor(A, [ROWS, COLS], [64, 1], [64, 64])
    value = desc.load([0, 0])
    if WRITE_INPUT:
        offsets = tl.arange(0, 64)[:, None] * 64 + tl.arange(0, 64)[None, :]
        tl.store(A + offsets, tl.full((64, 64), 0, A.dtype.element_ty))
    out = tl.make_tensor_descriptor(C, [64, 64], [64, 1], [64, 64])
    out.store([0, 0], value.T)


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "i8", "u8", "fp8e4nv", "fp8e5"])
@pytest.mark.parametrize("rows,cols", [(64, 64), (48, 64), (64, 48)])
def test_standalone_transpose_load_keeps_pointers(monkeypatch, dtype, rows, cols):
    source = ASTSource(descriptor_transpose_copy, signature=dict(A=f"*{dtype}", C=f"*{dtype}"),
                       constexprs=dict(ROWS=rows, COLS=cols, WRITE_INPUT=False), attrs={(i, ): [("tt.divisibility", 16)]
                                                                                        for i in range(2)})
    options = dict(num_warps=4, num_stages=1)
    monkeypatch.setenv("TRITON_HCU_ENABLE_TENSOR_DESCRIPTOR_MLS", "0")
    original = compile_to_llvm(source, TARGET, options)
    monkeypatch.setenv("TRITON_HCU_ENABLE_TENSOR_DESCRIPTOR_MLS", "1")
    compiled = compile_to_llvm(source, TARGET, options)
    assert compiled.asm == original.asm
    assert "tt.matrix_load" not in compiled.asm["ttir"]
    assert "matrix_load_to_local" not in compiled.asm["ttgir"]


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "i8", "fp8e4nv", "fp8e5"])
@pytest.mark.parametrize("rows,cols,checks", [(64, 64, None), (48, 64, "[1]"), (64, 48, "[0]")])
def test_dot_transpose_load_swaps_metadata_and_boundary_axes(dtype, rows, cols, checks):
    compiled = compile_dot(dtype, ROWS=rows, COLS=cols, TRANS=True)
    assert "tt.matrix_load" in compiled.asm["ttir"]
    assert "tt.trans " not in compiled.asm["ttir"]
    assert "amdg.matrix_load_to_local" in compiled.asm["ttgir"]
    if checks is not None:
        assert f"boundaryCheck = {checks}" in compiled.asm["ttir"]


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("block,layout,tail", [(128, 2, False), (128, 3, False), (128, 2, True), (128, 3, True),
                                               (64, 3, False)])
def test_descriptor_gemm_transposed_b(dtype, block, layout, tail):
    source = ASTSource(
        descriptor_gemm, signature=dict(A=f"*{dtype}", B=f"*{dtype}", C=f"*{dtype}"),
        constexprs=dict(M=block - 4 if tail else block, N=block - 8 if tail else block, K=96 if tail else 128, TILES=2,
                        BM=block, BN=block, WARP_SPECIALIZE=False, TRANS_B=True), attrs={(i, ):
                                                                                         [("tt.divisibility", 16)]
                                                                                         for i in range(3)})
    compiled = compile_to_llvm(source, TARGET,
                               dict(num_warps=4, num_stages=2, mmac_layout_force=layout, optimize_epilogue=True))
    assert compiled.asm["ttir"].count("tt.matrix_load") == 2
    assert "tt.trans " not in compiled.asm["ttir"]
    assert "amdg.matrix_load_to_local" in compiled.asm["ttgir"]
    assert_descriptor_stores_use_pointers(compiled)


@pytest.mark.parametrize("dtype", ["fp8e4nv", "fp8e5"])
@pytest.mark.parametrize("layout", [2, 3])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("tail", [False, True])
def test_fp8_descriptor_gemm(dtype, layout, transposed, tail):
    source = ASTSource(
        descriptor_gemm, signature=dict(A=f"*{dtype}", B=f"*{dtype}", C="*fp16"),
        constexprs=dict(M=60 if tail else 64, N=48 if tail else 64, K=96 if tail else 128, TILES=2, BM=64, BN=64,
                        WARP_SPECIALIZE=False, TRANS_B=transposed), attrs={(i, ): [("tt.divisibility", 16)]
                                                                           for i in range(3)})
    compiled = compile_to_llvm(source, TARGET, dict(num_warps=4, num_stages=2, mmac_layout_force=layout))
    assert compiled.asm["ttir"].count("tt.matrix_load") == 2
    assert "amdg.matrix_load_to_local" in compiled.asm["ttgir"]
    assert_descriptor_stores_use_pointers(compiled)
    b_loads = [line for line in compiled.asm["ttgir"].splitlines() if "MlsEncoding opIdx = 1," in line]
    assert b_loads
    assert all("alt2Kind = 0," in line for line in b_loads)


@triton.jit
def mixed_explicit_store(A, B, C, X, Y, Z):
    a = tl.matrix_load(A, shape=[64, 64], strides=[64, 1], block_shape=[64, 64], offsets=[0, 0])
    b = tl.matrix_load(B, shape=[64, 64], strides=[64, 1], block_shape=[64, 64], offsets=[0, 0])
    tl.matrix_store(C, tl.dot(a, b).to(tl.float16), shape=[64, 64], strides=[64, 1],
                    block_shape=[64, 64], offsets=[0, 0])
    xd = tl.make_tensor_descriptor(X, [64, 64], [64, 1], [64, 64])
    yd = tl.make_tensor_descriptor(Y, [64, 64], [64, 1], [64, 64])
    zd = tl.make_tensor_descriptor(Z, [64, 64], [64, 1], [64, 64])
    zd.store([0, 0], tl.dot(xd.load([0, 0]), yd.load([0, 0])).to(tl.float16))


def test_explicit_store_keeps_descriptor_lowering_unchanged(monkeypatch):
    # The existing FP16 store imposes Interleave2 on function-wide B loads;
    # automatically selecting the FP8 B load would fail MLS instruction lookup.
    source = ASTSource(
        mixed_explicit_store,
        signature=dict(A="*fp16", B="*fp16", C="*fp16", X="*fp8e4nv", Y="*fp8e4nv", Z="*fp16"),
        attrs={(i, ): [("tt.divisibility", 16)] for i in range(6)})
    options = dict(num_warps=4, num_stages=1, mmac_layout_force=3, optimize_epilogue=True)
    monkeypatch.setenv("TRITON_HCU_ENABLE_TENSOR_DESCRIPTOR_MLS", "0")
    original = compile_to_llvm(source, TARGET, options)
    monkeypatch.setenv("TRITON_HCU_ENABLE_TENSOR_DESCRIPTOR_MLS", "1")
    compiled = compile_to_llvm(source, TARGET, options)
    assert compiled.asm == original.asm
    assert compiled.asm["ttir"].count("tt.matrix_load") == 2
    assert compiled.asm["ttgir"].count("amdg.matrix_store_from_reg") == 1
